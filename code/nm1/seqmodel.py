"""Whole-shot sequence members: every frame of a discharge sees every other frame.

The residual MLP maps one frame's inputs to one frame's flux.  But an equilibrium is a state with
a history -- ``li`` is set by resistive current diffusion over hundreds of milliseconds, the
vertical position by a feedback loop whose corrections show up in the coil currents *after* the
plasma moved -- and the reconstruction here is offline: the whole discharge's inputs exist before
the first frame is predicted.  So these members read the shot as a sequence, in both directions:

    per-frame MLP blocks -> bidirectional GRU over the shot -> per-frame MLP blocks -> heads

The GRU enters through a zero-initialised residual projection, so at initialisation the member
*is* the per-frame MLP and the recurrence only adds what it can justify.  Measured on the cache
surrogate against a per-frame model with identical blocks, data and schedule: S^ 0.98455 against
0.98264 on the `selection` half, with all seven scalar terms up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nm1.model import ResidualMLPBlock


class SeqRegressor(nn.Module):
    def __init__(self, n_features: int, n_components: int, width: int = 1024, pre: int = 3,
                 post: int = 3, hid: int = 384, layers: int = 2, dropout: float = 0.02) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
        self.pre = nn.ModuleList([ResidualMLPBlock(width, dropout) for _ in range(pre)])
        self.tnorm = nn.LayerNorm(width)
        self.rnn = nn.GRU(width, hid, num_layers=layers, batch_first=True, bidirectional=True)
        self.tproj = nn.Linear(2 * hid, width)
        nn.init.zeros_(self.tproj.weight)
        nn.init.zeros_(self.tproj.bias)
        self.post = nn.ModuleList([ResidualMLPBlock(width, dropout) for _ in range(post)])
        self.norm = nn.LayerNorm(width)
        self.coeff_head = nn.Linear(width, n_components)
        self.scalar_head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lens: torch.Tensor):
        """``x`` (B, L, F) padded shots, ``mask`` (B, L) valid frames, ``lens`` (B,) lengths."""
        h = self.inp(x)
        for block in self.pre:
            h = block(h)
        m = mask.unsqueeze(-1)
        # the recurrence runs in float32: fp16 over a 450-step shot is where the earlier
        # attempt at this produced NaN
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.tnorm(h.float()) * m.float()
            packed = nn.utils.rnn.pack_padded_sequence(z, lens.cpu(), batch_first=True,
                                                       enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                      total_length=x.shape[1])
            h = h + self.tproj(out).to(h.dtype)
        for block in self.post:
            h = block(h)
        h = self.norm(h)
        return self.coeff_head(h), self.scalar_head(h)


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of each frame's discharge time (seconds), projected to the model width.

    Attention is told how far apart two frames are in time (EFIT frames are not evenly spaced:
    limited frames were removed), periods from ~10 ms to ~20 s."""

    def __init__(self, width: int, n_freq: int = 16) -> None:
        super().__init__()
        self.register_buffer("freq", torch.logspace(np.log10(2 * np.pi / 20.0), np.log10(2 * np.pi / 0.01), n_freq))
        self.proj = nn.Linear(2 * n_freq, width)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t.float().unsqueeze(-1) * self.freq
        return self.proj(torch.cat([a.sin(), a.cos()], -1))


class TfmRegressor(nn.Module):
    """Whole-shot transformer member (``kind == "seq"``, ``arch == "tfm"``).

    per-frame MLP blocks -> + time encoding -> transformer encoder over the shot (both directions,
    padding masked) -> zero-initialised residual -> per-frame MLP blocks -> coefficient / scalar heads.
    At initialisation the member is the per-frame MLP; attention adds what it can justify.  Each frame
    reads every other frame of its discharge directly, where the recurrence of SeqRegressor passed
    information one step at a time.  Attention runs in float32 (long shots, fp16 softmax overflow).
    """

    def __init__(self, n_features: int, n_components: int, width: int = 512, pre: int = 2, post: int = 2,
                 layers: int = 4, heads: int = 8, ffn: int = 1024, dropout: float = 0.05,
                 attn_width: int = 0, layer_scale: float = 0.0, chan_tokens: int = 0, chan_layers: int = 2,
                 chan_width: int = 128, n_experts: int = 0) -> None:
        super().__init__()
        self.chan_tokens, self.n_experts = int(chan_tokens), int(n_experts)
        # attn_width > 0: a wide per-frame trunk (the per-frame members' 1024 x 6) with the attention run in a
        # narrower space between a down projection and the zero-initialised up projection
        aw = int(attn_width) or width
        self.inp = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
        self.pre = nn.ModuleList([ResidualMLPBlock(width, dropout) for _ in range(pre)])
        self.down = nn.Linear(width, aw) if aw != width else nn.Identity()
        self.time = TimeEncoding(aw)
        layer = nn.TransformerEncoderLayer(aw, heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(aw)
        self.tproj = nn.Linear(aw, width)
        if layer_scale > 0:
            # LayerScale (Touvron et al. 2021): the projection keeps its ordinary initialisation and a per-channel
            # gain starting at `layer_scale` multiplies it, so the attention path receives gradient from the first
            # step while adding almost nothing to the trunk (a zero projection passes no gradient to the encoder)
            self.gamma = nn.Parameter(torch.full((width,), float(layer_scale)))
        else:
            self.gamma = None
            nn.init.zeros_(self.tproj.weight)
            nn.init.zeros_(self.tproj.bias)
        self.post = nn.ModuleList([ResidualMLPBlock(width, dropout) for _ in range(post)])
        self.norm = nn.LayerNorm(width)
        self.coeff_head = nn.Linear(width, n_components)
        self.scalar_head = nn.Linear(width, 2)
        if self.chan_tokens:
            # a second attention axis: inside each frame, across groups of input columns (chan_tokens columns per
            # token, a learned position per group), pooled and added to the trunk through a LayerScale gain, so a
            # trained member extended with it starts (almost) as itself
            k = -(-n_features // self.chan_tokens)
            self.chan_pad = k * self.chan_tokens - n_features
            self.chan_embed = nn.Linear(self.chan_tokens, chan_width)
            self.chan_pos = nn.Parameter(torch.zeros(k, chan_width))
            clayer = nn.TransformerEncoderLayer(chan_width, 4, 2 * chan_width, dropout, activation="gelu",
                                                batch_first=True, norm_first=True)
            self.chan_encoder = nn.TransformerEncoder(clayer, chan_layers, enable_nested_tensor=False)
            self.chan_norm = nn.LayerNorm(chan_width)
            self.chan_proj = nn.Linear(chan_width, width)
            self.chan_gamma = nn.Parameter(torch.full((width,), 1e-4))
        if self.n_experts:
            # discharge-phase experts on the flux coefficients: coeff_head(h) + sum_e g_e(h) (D_e h + b_e), with the
            # expert deltas and the gate starting at zero, so the member starts exactly as itself
            self.moe_gate = nn.Linear(width, self.n_experts)
            nn.init.zeros_(self.moe_gate.weight); nn.init.zeros_(self.moe_gate.bias)
            self.moe_w = nn.Parameter(torch.zeros(self.n_experts, n_components, width))
            self.moe_b = nn.Parameter(torch.zeros(self.n_experts, n_components))

    def _chan(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xt = F.pad(x.float(), (0, self.chan_pad)).view(B * L, -1, self.chan_tokens)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.chan_encoder(self.chan_embed(xt) + self.chan_pos)
            return (self.chan_proj(self.chan_norm(z.mean(1))) * self.chan_gamma).view(B, L, -1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The shared trunk: per-frame blocks, attention over the shot, per-frame blocks, norm."""
        h = self.inp(x)
        if self.chan_tokens:
            h = h + self._chan(x).to(h.dtype)
        for block in self.pre:
            h = block(h)
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.down(h.float()) + self.time(t)
            if getattr(self, "grad_ckpt", False) and self.training and torch.is_grad_enabled():
                # gradient checkpointing (training only): each attention layer's activations are recomputed in the
                # backward pass instead of stored, so a deep stack fits the GPU; same values, more compute
                from torch.utils.checkpoint import checkpoint
                for lyr in self.encoder.layers:
                    z = checkpoint(lyr, z, None, ~mask, use_reentrant=False)
            else:
                z = self.encoder(z, src_key_padding_mask=~mask)
            a = self.tproj(self.tnorm(z))
            if self.gamma is not None:
                a = a * self.gamma
            h = h + a.to(h.dtype)
        for block in self.post:
            h = block(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor):
        """``x`` (B, L, F) padded shots, ``mask`` (B, L) valid frames, ``t`` (B, L) frame times [s]."""
        h = self.encode(x, mask, t)
        c = self.coeff_head(h)
        if self.n_experts:
            g = torch.softmax(self.moe_gate(h).float(), -1)                       # (B, L, E)
            # expert outputs first (B, L, E, C), then the gate: the same sum without the (B, L, C, W) intermediate
            # a left-to-right three-operand einsum builds (it overflowed the 8 GB GPU, 7x slower epochs)
            e = torch.einsum("blw,ecw->blec", h.float(), self.moe_w)
            d = (g.unsqueeze(-1) * e).sum(2) + g @ self.moe_b
            c = c + d.to(c.dtype)
        return c, self.scalar_head(h)


def tfm_layout(n_base: int, n_ext: int, n_cds: int, n_cdsg: int, n_ctx: int) -> dict[str, slice]:
    """Column ranges of the transformer input ``[base | dt | ext | cds | cdsg | ctx]``."""
    out, at = {}, 0
    for name, k in (("base", n_base), ("dt", 2), ("ext", n_ext), ("cds", n_cds), ("cdsg", n_cdsg), ("ctx", n_ctx)):
        out[name] = slice(at, at + k)
        at += k
    return out


@torch.no_grad()
def warm_from_member(model: TfmRegressor, member: dict, layout: dict[str, slice],
                     feature_mean: np.ndarray, feature_std: np.ndarray,
                     scalar_mean: np.ndarray, scalar_std: np.ndarray, coeff_std: np.ndarray) -> list[str]:
    """Copy a trained per-frame member (``model.MetricRegressor``, width W x (pre + post) blocks) into the
    transformer's per-frame trunk, so the transformer starts as that member exactly.

    The per-frame member is ``input -> blocks[0..d) -> norm -> heads``; the transformer is
    ``inp -> pre -> (zero-initialised attention residual) -> post -> norm -> heads``.  With
    ``pre + post == d`` and the same width the blocks map one to one, and the attention residual is zero at
    initialisation, so only the input layer needs care: the two models whiten their inputs with their own
    statistics and order the blocks differently.  Each column the member reads is re-expressed in the
    transformer's whitening (``W' = W s_t / s_m``, the offset folded into the bias); columns the member never
    read (the wider base block, the frame spacing, a second diffusion block) start at zero weight.  The scalar
    head is re-expressed the same way.  Returns the names of the transformer column blocks left at zero.
    """
    sd = member["state_dict"]
    depth = int(member["model"]["depth"])
    if int(member["model"].get("scalar_scale") or 0) or member.get("decoder") or member["model"].get("decoder"):
        raise ValueError("warm start needs a plain per-frame member (no scalar scaling, linear decoder)")
    if int(member.get("vac_features") or 0) or member.get("temporal_offsets") or int(member.get("n_seg") or 0):
        raise ValueError("warm start does not map the vacuum / temporal-offset / segment blocks")
    if depth != len(model.pre) + len(model.post) or sd["input.0.weight"].shape[0] != model.inp[0].out_features:
        raise ValueError(f"member is {sd['input.0.weight'].shape[0]} x {depth}; transformer trunk is "
                         f"{model.inp[0].out_features} x ({len(model.pre)} + {len(model.post)})")
    # the member's own columns, in member_inputs order: base, ext, ctx, cds, cdsg
    nb = int(member["n_base_features"])
    m_blocks = [("base", nb, member["feature_mean"], member["feature_std"])]
    for name in ("ext", "ctx", "cds", "cdsg"):
        k = int(member.get(f"n_{name}") or 0)
        if k:
            m_blocks.append((name, k, member[f"{name}_mean"], member[f"{name}_std"]))
    w_m = sd["input.0.weight"].double().cpu().numpy()
    b_m = sd["input.0.bias"].double().cpu().numpy()
    w_t = np.zeros((w_m.shape[0], len(feature_mean)))
    b_t = b_m.copy()
    fm, fs = np.asarray(feature_mean, np.float64), np.asarray(feature_std, np.float64)
    at, mapped = 0, set()
    for name, k, mu_m, sd_m in m_blocks:
        cols = layout[name]
        if cols.stop - cols.start < k:
            raise ValueError(f"the transformer's {name} block has {cols.stop - cols.start} columns, member reads {k}")
        jt = np.arange(cols.start, cols.start + k)
        wm = w_m[:, at:at + k]
        mu_m, sd_m = np.asarray(mu_m, np.float64)[:k], np.asarray(sd_m, np.float64)[:k]
        # member input  x_m = (raw - mu_m) / sd_m,  raw = x_t * fs + fm
        w_t[:, jt] = wm * (fs[jt] / sd_m)
        b_t += wm @ ((fm[jt] - mu_m) / sd_m)
        mapped.add(name)
        at += k
    if at != w_m.shape[1]:
        raise ValueError(f"mapped {at} of the member's {w_m.shape[1]} input columns")
    dev = model.inp[0].weight.device
    model.inp[0].weight.copy_(torch.as_tensor(w_t, dtype=torch.float32, device=dev))
    model.inp[0].bias.copy_(torch.as_tensor(b_t, dtype=torch.float32, device=dev))
    blocks = list(model.pre) + list(model.post)
    for i, block in enumerate(blocks):
        block.load_state_dict({k[len(f"blocks.{i}."):]: v for k, v in sd.items() if k.startswith(f"blocks.{i}.")})
    model.norm.load_state_dict({"weight": sd["norm.weight"], "bias": sd["norm.bias"]})
    if not np.allclose(member["coeff_std"], coeff_std):
        raise ValueError("the member and the transformer whiten the flux coefficients differently")
    model.coeff_head.load_state_dict({"weight": sd["coeff_head.weight"], "bias": sd["coeff_head.bias"]})
    # scalars: member  raw = s_m * sd_m + mu_m;  transformer  s_t = (raw - mu_t) / sd_t
    a = np.asarray(member["scalar_std"], np.float64) / np.asarray(scalar_std, np.float64)
    c = (np.asarray(member["scalar_mean"], np.float64) - np.asarray(scalar_mean, np.float64)) / np.asarray(scalar_std, np.float64)
    ws = sd["scalar_head.weight"].double().cpu().numpy() * a[:, None]
    bs = sd["scalar_head.bias"].double().cpu().numpy() * a + c
    model.scalar_head.weight.copy_(torch.as_tensor(ws, dtype=torch.float32, device=dev))
    model.scalar_head.bias.copy_(torch.as_tensor(bs, dtype=torch.float32, device=dev))
    return [n for n in layout if n not in mapped and layout[n].stop > layout[n].start]


def dt_block(times_ms: np.ndarray) -> np.ndarray:
    """``(T, 2)`` spacing to the previous and next EFIT frame, in seconds.

    ``efit_times`` is non-contiguous (limited frames were removed), so the recurrence is told how
    far apart its steps are.
    """
    t = np.asarray(times_ms, dtype=np.float64) / 1000.0
    out = np.full((len(t), 2), 0.02, dtype=np.float32)
    if len(t) > 1:
        d = np.diff(t)
        out[1:, 0] = d
        out[:-1, 1] = d
        out[0, 0] = d[0]
        out[-1, 1] = d[-1]
    return out


def seq_inputs(base: np.ndarray, ext: np.ndarray | None, times_ms: np.ndarray,
               payload: dict, cds: np.ndarray | None = None, cdsg: np.ndarray | None = None) -> np.ndarray:
    """Whitened ``[base | dt | ext | cds | cdsg | ctx]`` for one shot, exactly as the trainer assembled it.

    ``ctx`` (the per-frame members' shot-context block, ``features2.context_block``) is present only for
    members trained with ``--ctx``; earlier members have ``n_ctx`` 0 and read the same columns as before."""
    nb = int(payload["n_base_features"])
    parts = [base[:, :nb], dt_block(times_ms)]
    if int(payload.get("n_ext") or 0):
        if ext is None:
            raise ValueError("this sequence member needs the features2 block")
        parts.append(ext)
    for name, block in (("n_cds", cds), ("n_cdsg", cdsg)):
        k = int(payload.get(name) or 0)
        if k:
            if block is None:
                raise ValueError(f"this sequence member needs the {name[2:]} block")
            parts.append(block[:, :k])
    if int(payload.get("n_ctx") or 0):
        if ext is None:
            raise ValueError("the shot-context block is computed from the features2 block")
        from nm1.features2 import context_block
        parts.append(context_block(ext, base[:, 136]))
    x = (np.hstack(parts) - payload["feature_mean"]) / payload["feature_std"]
    return np.ascontiguousarray(x, dtype=np.float32)


def load_seq(path: str | Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = (TfmRegressor if payload.get("arch") == "tfm" else SeqRegressor)(**payload["model"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


@torch.inference_mode()
def predict_seq(model, payload, base, ext, times_ms, device, cds=None, cdsg=None
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(coeff (T, n_comp) float64, scalars (T, 2) float64)`` for one whole shot."""
    x = torch.from_numpy(seq_inputs(base, ext, times_ms, payload, cds, cdsg)).to(device)[None]
    T = x.shape[1]
    mask = torch.ones((1, T), dtype=torch.bool, device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                        enabled=torch.device(device).type == "cuda"):
        if payload.get("arch") == "tfm":
            t = torch.from_numpy(np.asarray(times_ms, dtype=np.float32) / 1000.0).to(device)[None]
            c, s = model(x, mask, t)
        else:
            c, s = model(x, mask, torch.tensor([T]))
    coeff = c[0].double().cpu().numpy() * payload["coeff_std"]
    scal = s[0].double().cpu().numpy() * payload["scalar_std"] + payload["scalar_mean"]
    return coeff, scal