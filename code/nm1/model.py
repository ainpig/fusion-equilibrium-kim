"""The flux regressor, and the two input blocks that are rebuilt rather than cached.

One architecture is used for every ensemble member: a residual MLP from the input features to
whitened PCA coefficients of the flux map plus a two-output ``(q95, betaN)`` head.  Members
differ only in seed, loss weighting and which optional input block they carry.

Members are averaged in *coefficient* space.  The reconstruction is affine and all members share
one basis, so that is the same as averaging flux maps -- and unlike averaging weights, it is
meaningful across members with different architectures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(F.silu(self.fc1(self.norm(x)))))


class MetricRegressor(nn.Module):
    def __init__(self, n_features: int, n_components: int, width: int = 1024,
                 depth: int = 6, dropout: float = 0.02, scalar_scale: int = 0) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualMLPBlock(width, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.coeff_head = nn.Linear(width, n_components)
        self.scalar_head = nn.Linear(width, 2)
        self.scalar_scale = int(scalar_scale)              # 1: q95 / (B_t/Ip); 2: betaN / (1/(B_t Ip)) too
        if self.scalar_scale:
            # q95 scales exactly as B_t / Ip: the head predicts what is left after that scaling (its own
            # spread is half of q95's), so the network never has to build a ratio of two inputs to 1%
            # accuracy. betaN (p / (B_t Ip) with p itself growing with Ip^2) has no such fixed factor
            # and is left as it is. Buffers (filled by the trainer, stored in the state dict): the base
            # block's whitening of (bcoil, Ip), the mean/std of q95 / (B_t/Ip) and of betaN, and the
            # targets' own whitening.
            for name in ("sc_mu", "sc_sigma", "y_mu", "y_sigma", "t_mu", "t_sigma"):
                self.register_buffer(name, torch.zeros(2))
            self.register_buffer("sc_idx", torch.tensor([1, 20], dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(self.blocks(self.input(x)))
        s = self.scalar_head(h)
        if self.scalar_scale:
            raw = x[:, self.sc_idx].float() * self.sc_sigma + self.sc_mu          # bcoil, Ip (kA)
            bc = raw[:, 0].abs().clamp_min(5.0)
            ip = raw[:, 1].abs().clamp_min(50.0)
            u2 = 1.0e4 / (bc * ip) if self.scalar_scale >= 2 else torch.ones_like(bc)
            u = torch.stack([bc / ip, u2], dim=1)
            s = (u * (self.y_mu + self.y_sigma * s.float()) - self.t_mu) / self.t_sigma
        return self.coeff_head(h), s


def load_member(path: str | Path, device: torch.device | str = "cpu", *, keep_state: bool = True
                ) -> tuple[MetricRegressor, dict]:
    payload = torch.load(path, map_location=device if keep_state else "cpu", weights_only=False)
    model = MetricRegressor(**payload["model"])
    model.load_state_dict(payload["state_dict"])
    if not keep_state:
        # Inference only needs the model's parameters and preprocessing metadata.
        # Keeping the checkpoint copy on CUDA doubles ensemble weight memory.
        payload.pop("state_dict")
    model.to(device).eval()
    return model, payload


def temporal_offsets(payload: dict) -> list[int]:
    return [int(v) for v in (payload.get("temporal_offsets") or [])]


def apply_context(features: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Append the feature difference to frame ``t + off`` for ONE shot's contiguous frames.

    A shot is one parquet row, so clamping at the block edges keeps every offset inside the shot
    and no context ever leaks across a shot boundary.
    """
    if not offsets:
        return features
    n = len(features)
    parts = [features]
    for off in offsets:
        parts.append(features[np.clip(np.arange(n) + off, 0, n - 1)] - features)
    return np.concatenate(parts, axis=1)


def context_indices(shot: np.ndarray, offsets: list[int]) -> np.ndarray:
    """``(n, len(offsets))`` gather indices for the cached, shot-contiguous training frames."""
    n = len(shot)
    bounds = np.flatnonzero(np.diff(shot)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    seg = np.searchsorted(starts, np.arange(n), side="right") - 1
    lo, hi = starts[seg], ends[seg] - 1
    return np.stack([np.clip(np.arange(n) + off, lo, hi) for off in offsets], 1)


def vacuum_coefficients(currents: np.ndarray, green: np.ndarray, components: np.ndarray,
                        k: int) -> np.ndarray:
    """The analytic vacuum flux of the coil currents, in the basis the model predicts in.

    ``green`` is the per-column Green response flattened to ``(n_columns, 4225)`` and scaled to
    amps, so this is the exact linear coil response resolved onto the first ``k`` flux modes --
    information the model would otherwise have to rediscover from the raw currents.
    """
    if not k:
        return np.zeros((len(currents), 0), dtype=np.float32)
    m = green.astype(np.float64) @ components.astype(np.float64).T
    vac = currents[:, :m.shape[0]].astype(np.float64) @ m
    return vac[:, :k].astype(np.float32)


def green_amps(row, columns) -> np.ndarray:
    """``(n_columns, 4225)`` vacuum response per unit of the current column, in amps."""
    from nm1.features import green_basis
    _, basis = green_basis(row, columns)
    return basis.reshape(len(basis), -1) * 1000.0


def base_width(payload: dict) -> int:
    """How many raw feature columns this member takes, before context and the vacuum block.

    Derived from the checkpoint's own shape rather than trusted from a stored field, so a member
    can never be fed a block of the wrong width.
    """
    offsets = temporal_offsets(payload)
    total = (int(payload["model"]["n_features"]) - int(payload.get("vac_features") or 0)
             - int(payload.get("n_ext") or 0) - int(payload.get("n_ctx") or 0)
             - int(payload.get("n_cds") or 0) - int(payload.get("n_cdsg") or 0)
             - int(payload.get("n_seg") or 0))
    width, remainder = divmod(total, 1 + len(offsets))
    if remainder:
        raise ValueError(f"checkpoint input width {total} is not a multiple of "
                         f"{1 + len(offsets)} context blocks")
    stored = payload.get("n_base_features")
    if stored is not None and int(stored) != width:
        raise ValueError(f"checkpoint says {stored} base features but its shape implies {width}")
    return width


def needs_cdsg(payload: dict) -> bool:
    """A second-stage member: it reads the flux-coordinate diffusion block (cds_geo.py)."""
    return bool(int(payload.get("n_cdsg") or 0))


def member_inputs(base: np.ndarray, currents: np.ndarray, payload: dict,
                  green: np.ndarray | None = None, ext: np.ndarray | None = None,
                  cds: np.ndarray | None = None, cdsg: np.ndarray | None = None) -> np.ndarray:
    """Whitened model input for one shot: context block, then the vacuum block, then ``ext``.

    Whitening is float32 here and float32 in the trainer, so the model sees the same numbers at
    both ends.  ``base`` may be wider than the member wants; the extra columns are sliced off.
    The order matches the trainer, which appends the vacuum block inside the cache pass and the
    features2 block after it, then the shot context, then the current-diffusion block.
    """
    width = base_width(payload)
    if base.shape[1] < width:
        raise ValueError(f"member wants {width} features, was given {base.shape[1]}")
    x = (apply_context(base[:, :width], temporal_offsets(payload))
         - payload["feature_mean"]) / payload["feature_std"]
    k = int(payload.get("vac_features") or 0)
    if k:
        if green is None:
            raise ValueError("this member needs the vacuum block; pass the shot Green response")
        vac = vacuum_coefficients(currents, green, payload["pca_components"], k)
        x = np.column_stack([x, (vac - payload["vac_mean"]) / payload["vac_std"]])
    n_ext = int(payload.get("n_ext") or 0)
    if n_ext:
        if ext is None or ext.shape[1] != n_ext:
            raise ValueError(f"this member needs the {n_ext}-column features2 block")
        x = np.column_stack([x, (ext - payload["ext_mean"]) / payload["ext_std"]])
    n_ctx = int(payload.get("n_ctx") or 0)
    if n_ctx:
        # shot context is computed over the whole shot passed in: callers pass one shot at a time
        from nm1.features2 import context_block
        c = context_block(ext, base[:, 136])
        x = np.column_stack([x, (c - payload["ctx_mean"]) / payload["ctx_std"]])
    n_cds = int(payload.get("n_cds") or 0)
    if n_cds:
        # later block versions only append columns, so a member reads its own leading n_cds
        if cds is None or cds.shape[1] < n_cds:
            raise ValueError(f"this member needs the {n_cds}-column current-diffusion block")
        x = np.column_stack([x, (cds[:, :n_cds] - payload["cds_mean"]) / payload["cds_std"]])
    n_cdsg = int(payload.get("n_cdsg") or 0)
    if n_cdsg:
        if cdsg is None or cdsg.shape[1] < n_cdsg:
            raise ValueError(f"this member needs the {n_cdsg}-column flux-coordinate block")
        x = np.column_stack([x, (cdsg[:, :n_cdsg] - payload["cdsg_mean"])
                             / payload["cdsg_std"]])
    n_seg = int(payload.get("n_seg") or 0)
    if n_seg:
        # where the frame sits in its run of consecutive EFIT frames (per shot, from the times)
        from nm1.features2 import seg_block
        x = np.column_stack([x, (seg_block(base[:, 136]) - payload["seg_mean"])
                             / payload["seg_std"]])
    return np.ascontiguousarray(x, dtype=np.float32)


@torch.inference_mode()
def predict_shot(members, base: np.ndarray, currents: np.ndarray, green: np.ndarray,
                 device: torch.device, batch_size: int = 4096,
                 weights: np.ndarray | None = None, with_heads: bool = False,
                 ext: np.ndarray | None = None, cds: np.ndarray | None = None,
                 cdsg: np.ndarray | None = None, coeff_out: list | None = None):
    """Weighted ensemble prediction for one shot, averaged in coefficient space.

    Every member shares the basis, so the coefficients are averaged first and reconstructed once.
    ``psi`` comes back in float64: the flux term is 55% of the score and the corrections applied
    after this are multiplicative, so the accumulation stays in double.

    ``with_heads`` additionally returns each member's own ``(q95, betaN)`` as
    ``(frames, member, 2)``.  The scalar stack needs them, and taking them from this pass costs
    nothing -- running the members a second time for them would double the cost of a fold.
    """
    first = members[0][1]
    pca_mean = torch.from_numpy(first["pca_mean"]).double().to(device)
    components = torch.from_numpy(first["pca_components"]).double().to(device)
    w = (np.full(len(members), 1.0 / len(members)) if weights is None
         else np.asarray(weights, dtype=np.float64) / np.sum(weights))
    n = len(base)
    coeff = torch.zeros((n, components.shape[0]), dtype=torch.float64, device=device)
    scalars = torch.zeros((n, 2), dtype=torch.float64, device=device)
    heads = np.empty((n, len(members), 2), dtype=np.float64) if with_heads else None
    for index, ((model, payload), weight) in enumerate(zip(members, w)):
        offloaded = getattr(model, "_fusion_offload", False)
        if offloaded:
            model.to(device)
        x = member_inputs(base, currents, payload, green, ext, cds, cdsg)
        coeff_std = torch.from_numpy(payload["coeff_std"]).double().to(device)
        scalar_mean = torch.from_numpy(payload["scalar_mean"]).double().to(device)
        scalar_std = torch.from_numpy(payload["scalar_std"]).double().to(device)
        member_coeff = (torch.zeros((n, components.shape[0]), dtype=torch.float64, device=device)
                        if coeff_out is not None else None)
        for start in range(0, n, batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            coeff_z, scalar_z = model(xb)
            stop = start + len(xb)
            coeff[start:stop] += weight * coeff_z.double() * coeff_std
            if member_coeff is not None:
                member_coeff[start:stop] = coeff_z.double() * coeff_std
            member_scalar = scalar_z.double() * scalar_std + scalar_mean
            scalars[start:stop] += weight * member_scalar
            if with_heads:
                heads[start:stop, index] = member_scalar.cpu().numpy()
        if member_coeff is not None:
            coeff_out.append(member_coeff)
        if offloaded:
            model.to("cpu")
    psi = (pca_mean + coeff @ components).reshape(n, 65, 65)
    out = (psi.cpu().numpy(), scalars[:, 0].float().cpu().numpy(),
           scalars[:, 1].float().cpu().numpy())
    return (*out, heads) if with_heads else out
