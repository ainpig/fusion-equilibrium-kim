"""DIII-D prediction: raw parquet in, submission arrays out.

One shot at a time, so a full fold never has to fit in memory.  Each shot goes through

    ensemble (coefficient-space mean of every member)
      -> scalar stack        q95 / betaN as a fitted linear combination of the members' heads
      -> flux projection     amplitude and gauge pulled toward the psi specialist
      -> temporal filter     a near-identity symmetric smoother along the shot's own frames

and the three post-processing steps are skipped individually when the fitted file says they did
not earn their place.  Target columns are dropped from the row before any feature is built.
"""
from __future__ import annotations

import argparse
import os
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from nm1 import ARTIFACTS, OUTPUTS
from nm1.features import D3D_COILS, d3d_input_features, d3d_profile_block, read_row
from nm1.features2 import extended_features
from nm1.fetch import listing, local_path
from nm1.model import base_width, green_amps, load_member, predict_shot

MEMBERS = ARTIFACTS / "members.json"
STACK = ARTIFACTS / "stack.json"


class NpzStreamWriter:
    """Write .npy members straight into a zip, so the whole submission never sits in RAM."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.zip = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                                   allowZip64=True)

    def write(self, key: str, value: np.ndarray) -> None:
        with self.zip.open(key + ".npy", "w", force_zip64=True) as stream:
            np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)

    def close(self) -> None:
        self.zip.close()


def load_ensemble(manifest: Path = MEMBERS, device: torch.device | str = "cuda", *, offload=False):
    """``(members, weights, specialist, seq_members, seq_weights)`` from the manifest.

    Per-frame members and whole-shot sequence members (``payload["kind"] == "seq"``) are listed
    together; their weights are normalised jointly, so a manifest's weights mean the same thing
    whichever kind a member is.
    """
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    root = Path(manifest).parent
    raw = np.asarray(spec.get("weights") or [1.0] * len(spec["members"]), dtype=np.float64)
    raw = raw / raw.sum()
    members, weights, seq_members, seq_weights = [], [], [], []
    for rel, w in zip(spec["members"], raw):
        payload = torch.load(root / rel, map_location="cpu", weights_only=False)
        if payload.get("kind") != "seq" and seq_members:
            # per-member lists of a manifest (stage1_weights, weights_q95, groups) index the per-frame
            # members by position: sequence members go last so those positions stay the same
            raise ValueError("sequence members must come after every per-frame member in the manifest")
        if payload.get("kind") == "seq":
            # sequence members are small (tens of MB): they stay on the device even when the
            # per-frame members are offloaded to the CPU between shots
            from nm1.seqmodel import load_seq
            seq_members.append(load_seq(root / rel, device))
            seq_weights.append(w)
        else:
            member = load_member(root / rel, "cpu" if offload else device, keep_state=False)
            member[0]._fusion_offload = bool(offload)
            members.append(member)
            weights.append(w)
        del payload
    specialist = (load_member(root / spec["specialist"], "cpu" if offload else device, keep_state=False)
                  if spec.get("specialist") else None)
    if specialist is not None:
        specialist[0]._fusion_offload = bool(offload)
    return (members, np.asarray(weights), specialist, seq_members, np.asarray(seq_weights))


def scalar_weights(manifest: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Optional per-scalar member weights (``weights_q95`` / ``weights_betaN``) from a manifest.

    The flux is always the ``weights`` mean. A scalar head can be blended differently because it
    never touches the flux map: measured on all three held-out parts, the base and features2
    groups carry the better q95 and the current-diffusion group the better betaN.
    """
    spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
    out = []
    for key in ("weights_q95", "weights_betaN"):
        w = spec.get(key)
        if w is None:
            out.append(None)
            continue
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (len(spec["members"]),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"{key} must be one non-negative weight per member")
        out.append(w / w.sum())
    return out[0], out[1]


def projected_flux(base: np.ndarray, specialist: np.ndarray, amplitude_weight: float,
                   gauge_weight: float) -> np.ndarray:
    """Pull the flux map's amplitude and offset toward a psi-only specialist.

    Per frame this is a positive affine map around the frame mean, so the flux surfaces -- and
    therefore the boundary and every shape scalar -- are preserved exactly; only the level and
    the contrast move.  Frames where the amplitude would be non-positive keep the baseline.
    """
    b, s = np.asarray(base, float), np.asarray(specialist, float)
    mb, ms = b.mean((1, 2)), s.mean((1, 2))
    centered = b - mb[:, None, None]
    variance = np.mean(centered ** 2, (1, 2))
    amplitude = 1 + amplitude_weight * (
        np.mean(centered * (s - ms[:, None, None]), (1, 2)) / np.maximum(variance, 1e-20) - 1)
    safe = np.isfinite(amplitude) & (amplitude > 0) & (variance > 1e-12)
    out = (mb[:, None, None] + amplitude[:, None, None] * centered
           + gauge_weight * (ms - mb)[:, None, None])
    out[~safe] = b[~safe]
    return out


def temporal_filter(values: np.ndarray, strength: float) -> np.ndarray:
    """Symmetric second-difference smoother with the shot endpoints left alone."""
    if not strength:
        return values
    v = np.asarray(values, float)
    change = np.zeros_like(v)
    change[1:-1] = v[:-2] + v[2:] - 2 * v[1:-1]
    return v + strength * change


class Predictor:
    """The whole DIII-D inference chain, reusable by both the scorer and the submission build."""

    def __init__(self, manifest: Path = MEMBERS, stack: Path | None = STACK,
                 device: torch.device | str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stack = (json.loads(Path(stack).read_text(encoding="utf-8"))
                      if stack and Path(stack).exists() else {})
        self.member_paths = list(json.loads(Path(manifest).read_text(encoding="utf-8"))["members"])
        self._ramp = None
        (self.members, self.weights, self.specialist, self.seq_members,
         self.seq_weights) = load_ensemble(manifest, self.device,
            offload=bool(os.environ.get("NM1_OFFLOAD_MEMBERS") or
                                                    self.stack.get("inference", {}).get("offload_members", False)))
        self.w_q95, self.w_betaN = scalar_weights(manifest)
        n_frame = len(self.members)
        if self.seq_members:
            # sequence members sit at the end of the manifest (load_ensemble checks it): the per-frame
            # lists keep their first n_frame entries; q95 / betaN stay the per-frame members' blend
            self.member_paths = self.member_paths[:n_frame]
            for name in ("w_q95", "w_betaN"):
                w = getattr(self, name)
                if w is not None:
                    if np.any(w[n_frame:] != 0):
                        raise ValueError("per-scalar weights must be zero on sequence members")
                    setattr(self, name, w[:n_frame] / w[:n_frame].sum())
        self.shape_head = None
        self.li_head = None
        self.shape_group_indices = []
        self.shape_member_targets = None
        correction = self.stack.get("field_correction") or {}
        if correction.get("enabled"):
            from nm1.v12_shape import ShapePredictor
            spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
            encoders = dict(zip(spec["members"], self.members))
            self.shape_head = ShapePredictor(
                [ARTIFACTS / name for name in correction["members"]],
                self.device, encoders=encoders)
            # optional li specialists: their li replaces the heads' li in the correction targets
            self.li_head = (ShapePredictor([ARTIFACTS / name for name in correction["li_members"]],
                                           self.device, encoders=encoders)
                            if correction.get("li_members") else None)
            if correction.get("method") in ("mixture", "v13"):
                from nm1.v12_mix import GROUPS
                # v13 names its groups in the stack (all eight flux-producing groups)
                groups = correction.get("groups") or GROUPS
                index = {name: i for i, name in enumerate(spec["members"])}
                self.shape_group_indices = [[index[name] for name in spec["groups"][group]]
                                            for group in groups]
                # optional "seq_groups" {group: {"share": s, "members": [members_tfm/...]}}: the group's correction map
                # becomes (1 - s) x its MLP mean + s x the mean of its transformers (absent: MLPs only, as before)
                seq_rel = [m for m in spec["members"] if m.startswith("members_tfm/")]
                sg = spec.get("seq_groups") or {}
                self.shape_group_seq = [(float(sg[g]["share"]), [seq_rel.index(m) for m in sg[g]["members"]])
                                        if g in sg else None for g in groups]
        self._green: np.ndarray | None = None
        # The radial profile-shape block costs a re-interpolation of four Thomson profiles per
        # shot, so it is only built when some member actually declares it.
        every = self.members + ([self.specialist] if self.specialist else [])
        self.base_width = max([base_width(payload) for _, payload in every]
                              + [int(p["n_base_features"]) for _, p in self.seq_members])
        if self.shape_head is not None:
            self.base_width = max(272, self.base_width)
        # the features2 block is only built when some member declares it
        self.needs_ext = any(int(payload.get("n_ext") or 0)
                             for _, payload in every + self.seq_members)
        # likewise the current-diffusion simulation
        self.needs_cds = any(int(payload.get("n_cds") or 0) for _, payload in every)
        if self.shape_head is not None:
            self.needs_ext = self.needs_cds = True
        # second-stage members read the diffusion block on the first stage's flux coordinate
        from nm1.model import needs_cdsg
        self.stage2 = [i for i, (_, p) in enumerate(self.members) if needs_cdsg(p)]
        if self.shape_head is not None and not self.stage2:
            raise ValueError("The v12 shape head needs the two-stage diffusion inputs")
        self.w_stage1 = None
        if self.stage2:
            spec = json.loads(Path(manifest).read_text(encoding="utf-8"))
            w1 = np.asarray(spec.get("stage1_weights") or [], dtype=np.float64)
            if self.seq_members and len(w1) == len(self.members) + len(self.seq_members):
                w1 = w1[:len(self.members)]
            if w1.shape != (len(self.members),) or (w1[self.stage2] != 0).any() or w1.sum() <= 0:
                raise ValueError("a manifest with second-stage members needs `stage1_weights`: "
                                 "one weight per member, zero on the second-stage ones")
            if self.stack.get("scalar_stack") or self.stack.get("retrieval"):
                raise ValueError("two-stage prediction does not combine with the scalar stack or retrieval")
            for _, payload in self.seq_members:
                if not np.array_equal(payload["pca_components"], self.members[0][1]["pca_components"]):
                    raise ValueError("a sequence member must share the per-frame members' flux basis")
            self.w_stage1 = w1 / w1.sum()
        if self.seq_members and self.stack.get("scalar_stack"):
            raise ValueError("the scalar stack is fitted per per-frame member head; refit it "
                             "without sequence members or drop it")
        self.retrieval = None
        ret = self.stack.get("retrieval")
        if ret:
            from nm1.retrieval import Retrieval
            self.retrieval = Retrieval(Path(manifest).parent / ret["member"], ret["k"],
                                       self.device)
            ref_payload = (self.members or self.seq_members)[0][1]
            self._mean = ref_payload["pca_mean"].astype(np.float64)
            self._comps = ref_payload["pca_components"].astype(np.float64)

    def green(self, row) -> np.ndarray:
        # DIII-D coil geometry is identical on every shot; the Thomson chords are not, and are
        # not part of this map.
        if self._green is None:
            self._green = green_amps(row, D3D_COILS[:-1])
        return self._green

    def _two_stage(self, row, base, currents, green, times, ext, cds, batch_size):
        """First stage -> its flux map -> the flux-coordinate diffusion block -> second stage.

        Every member runs once; the first-stage flux is the `stage1_weights` blend of the
        first-stage members, the submitted flux the `weights` blend of all of them.
        """
        from nm1.cds_geo import features_from_psi
        idx1 = [i for i in range(len(self.members)) if i not in self.stage2]
        m1 = [self.members[i] for i in idx1]
        m2 = [self.members[i] for i in self.stage2]
        c1: list = []
        _, _, _, h1 = predict_shot(m1, base, currents, green, self.device, batch_size, None,
                                   with_heads=True, ext=ext, cds=cds, coeff_out=c1)
        first = self.members[0][1]
        mean = torch.from_numpy(first["pca_mean"]).double().to(self.device)
        comps = torch.from_numpy(first["pca_components"]).double().to(self.device)
        w1 = self.w_stage1[idx1]
        coeff1 = sum(float(w) * c for w, c in zip(w1, c1))
        psi1 = (mean + coeff1 @ comps).reshape(-1, 65, 65).cpu().numpy()
        cdsg = features_from_psi(row, times, psi1)
        self.last_blocks["cdsg"] = cdsg
        c2: list = []
        _, _, _, h2 = predict_shot(m2, base, currents, green, self.device, batch_size, None,
                                   with_heads=True, ext=ext, cds=cds, cdsg=cdsg, coeff_out=c2)
        coeffs = [None] * len(self.members)
        heads = np.empty((len(times), len(self.members), 2))
        for j, i in enumerate(idx1):
            coeffs[i] = c1[j]
            heads[:, i] = h1[:, j]
        for j, i in enumerate(self.stage2):
            coeffs[i] = c2[j]
            heads[:, i] = h2[:, j]
        w = self.weights / self.weights.sum()
        rb = self.stack.get("regime_blend") or {}
        if rb.get("enabled") and getattr(self, "_ramp", None) is not None:
            # ramp frames (Ip below a fraction of the shot's peak, as train.py's --ramp-weight defines them):
            # the named members' flux weights are multiplied by `gain`, then each frame is renormalised
            names = [Path(m).name for m in self.member_paths]
            boost = np.array([rb["gain"] if n in set(Path(x).name for x in rb["members"]) else 1.0 for n in names])
            W = np.where(self._ramp[:, None], w[None, :] * boost[None, :], w[None, :])
            W = W / W.sum(1, keepdims=True)
            Wt = torch.as_tensor(W, dtype=torch.float64, device=self.device)
            coeff = sum(Wt[:, i:i + 1] * c for i, c in enumerate(coeffs))
        else:
            coeff = sum(float(wi) * c for wi, c in zip(w, coeffs))
        if self.seq_members:
            # whole-shot members (nm1.seqmodel, e.g. the transformer) join the flux blend with their
            # manifest weights: load_ensemble normalises per-frame and sequence weights jointly, so the
            # per-frame blend above keeps the share 1 - sum(sequence weights).  They read the same
            # input blocks, the second-stage block included.  q95 / betaN stay the per-frame blend's.
            from nm1.seqmodel import predict_seq
            seq_c = []
            for (model, payload) in self.seq_members:
                c_s, _ = predict_seq(model, payload, base, ext, times, self.device, cds=cds, cdsg=cdsg)
                seq_c.append(torch.as_tensor(c_s, dtype=torch.float64, device=self.device))
            # a whole-shot member whose output is not finite on this shot (fp16 overflow on inputs far outside the
            # training range) is left out of this shot's blend and its weight returns to the per-frame members;
            # on shots where every member is finite the arithmetic below is exactly the original one
            ok = [bool(torch.isfinite(c).all()) for c in seq_c]
            self.seq_dropped = [k for k, f in enumerate(ok) if not f]
            if self.seq_dropped:
                print(f"  whole-shot members left out on this shot (non-finite output): {self.seq_dropped}", flush=True)
            ws_eff = np.where(ok, self.seq_weights, 0.0)
            coeff = coeff * (1.0 - float(np.sum(ws_eff)))
            for c_s, ws, f in zip(seq_c, ws_eff, ok):
                if f:
                    coeff = coeff + float(ws) * c_s
        psi = (mean + coeff @ comps).reshape(-1, 65, 65).cpu().numpy()
        q95 = heads[:, :, 0] @ (self.w_q95 if self.w_q95 is not None else w)
        betaN = heads[:, :, 1] @ (self.w_betaN if self.w_betaN is not None else w)
        shape_targets = None
        if self.shape_head is not None:
            if self.stack["field_correction"].get("keep_member_targets"):
                from nm1.v12_shape import raw_features
                individual = self.shape_head.predict_members_raw(raw_features(base,ext,cds,cdsg))
                self.shape_member_targets = np.stack(individual)
                shape_targets = sum(individual) / len(individual)
            else:
                shape_targets = self.shape_head.predict(base,ext,cds,cdsg)
            if self.li_head is not None:
                shape_targets = np.array(shape_targets, copy=True)
                shape_targets[:, 6] = np.asarray(self.li_head.predict(base,ext,cds,cdsg))[:, 6]
        shape_groups = None
        if self.shape_group_indices:
            group_maps = []
            gseq = getattr(self, "shape_group_seq", None) or [None] * len(self.shape_group_indices)
            for indices, gs in zip(self.shape_group_indices, gseq):
                c = sum(coeffs[i] for i in indices) / len(indices)
                if gs is not None and self.seq_members:
                    keep = [k for k in gs[1] if k not in self.seq_dropped]
                    if keep:
                        c = (1 - gs[0]) * c + gs[0] * sum(seq_c[k] for k in keep) / len(keep)
                g = (mean + c @ comps).reshape(-1, 65, 65).cpu().numpy()
                group_maps.append(g.astype(np.float32))
            shape_groups = np.stack(group_maps)
        return psi, q95, betaN, shape_targets, shape_groups

    def predict(self, row, batch_size: int = 4096) -> dict[str, np.ndarray]:
        self.shape_member_targets = None
        if self.stack.get("ip_sign_normalize") or os.environ.get("NM1_IP_SIGN_NORMALIZE") == "1":
            from nm1.features import sign_normalized
            row = sign_normalized(row)
        base, currents, times = d3d_input_features(row)
        if (self.stack.get("regime_blend") or {}).get("enabled"):
            from nm1.features import interp
            ip = np.abs(interp(np.asarray(row["magnetics_plasma_current_times"], float),
                               row["magnetics_plasma_current"], np.asarray(times, float)))
            frac = float(self.stack["regime_blend"].get("ramp_fraction", 0.9))
            self._ramp = ip < frac * max(float(np.nanmax(ip)), 1e-9)
        else:
            self._ramp = None
        if self.base_width > base.shape[1]:
            base = np.hstack([base, d3d_profile_block(row, times)])
        ext = extended_features(row, times) if self.needs_ext else None
        cds = None
        if self.needs_cds:
            from nm1.diffusion import diffusion_features
            cds = diffusion_features(row, times)
        # the input blocks of the last shot, for building pseudo-labelled cache rows ; no effect on outputs
        self.last_blocks = {"base": base, "currents": currents, "times": times, "ext": ext, "cds": cds}
        green = self.green(row)
        scalar_stack = self.stack.get("scalar_stack")
        shape_targets = None
        shape_groups = None
        w_mlp = float(np.sum(self.weights)) if len(self.members) else 0.0
        if self.stage2:
            psi, q95, betaN, shape_targets, shape_groups = self._two_stage(
                row, base, currents, green, times, ext, cds, batch_size)
        elif self.members:
            psi, q95, betaN, heads = predict_shot(self.members, base, currents, green,
                                                  self.device, batch_size, self.weights,
                                                  with_heads=True, ext=ext, cds=cds)
            psi = w_mlp * np.asarray(psi, dtype=np.float64)
            q95 = w_mlp * np.asarray(q95, dtype=np.float64)
            betaN = w_mlp * np.asarray(betaN, dtype=np.float64)
            if scalar_stack:
                q95, betaN = apply_scalar_stack(heads, scalar_stack)
            if self.w_q95 is not None:
                q95 = heads[:, :, 0] @ self.w_q95
            if self.w_betaN is not None:
                betaN = heads[:, :, 1] @ self.w_betaN
        else:
            psi = np.zeros((len(times), 65, 65))
            q95 = np.zeros(len(times))
            betaN = np.zeros(len(times))
        if self.seq_members and not self.stage2:      # two-stage manifests blend them in _two_stage
            from nm1.seqmodel import predict_seq
            outs = [predict_seq(model, payload, base, ext, times, self.device) for model, payload in self.seq_members]
            ok = [bool(np.isfinite(c).all() and np.isfinite(sc).all()) for c, sc in outs]
            if not all(ok):
                print(f"  whole-shot members left out on this shot (non-finite output): "
                      f"{[k for k, f in enumerate(ok) if not f]}", flush=True)
                keep = float(np.sum(np.where(ok, self.seq_weights, 0.0)))
                rest = 1.0 - float(np.sum(self.seq_weights))
                scale = 1.0 / (rest + keep) if rest + keep > 0 else 1.0
                psi = psi * scale                   # q95 / betaN from a scalar stack or head weights keep their own scale
                if not scalar_stack and self.w_q95 is None:
                    q95 = q95 * scale
                if not scalar_stack and self.w_betaN is None:
                    betaN = betaN * scale
            for (coeff, scal), w, f in zip(outs, self.seq_weights, ok):
                if not f:
                    continue
                if not all(ok):
                    w = w * scale
                recon = (payload["pca_mean"].astype(np.float64)
                         + coeff @ payload["pca_components"].astype(np.float64))
                psi = psi + w * recon.reshape(-1, 65, 65)
                q95 = q95 + w * scal[:, 0]
                betaN = betaN + w * scal[:, 1]
        if self.retrieval is not None:
            # borrow the nearest training equilibria's flux-surface shape, keep the level
            c_knn = self.retrieval.neighbour_coeff(base, currents, ext)
            psi_knn = (self._mean + c_knn @ self._comps).reshape(-1, 65, 65)
            w = float(self.stack["retrieval"]["weight"])
            psi = projected_flux((1 - w) * psi + w * psi_knn, psi, 1.0, 1.0)
        flux = self.stack.get("flux_projection")
        if flux and self.specialist is not None:
            other, _, _ = predict_shot([self.specialist], base, currents, green, self.device,
                                       batch_size, ext=ext, cds=cds)
            psi = projected_flux(psi, other, flux["amplitude_weight"], flux["gauge_weight"])
        temporal = self.stack.get("temporal")
        if temporal:
            psi = temporal_filter(psi, temporal["flux_strength"])
            q95 = temporal_filter(q95, temporal["scalar_strengths"][0])
            betaN = temporal_filter(betaN, temporal["scalar_strengths"][1])
        out = {"psirz": np.asarray(psi, dtype=np.float64),
               "q95": np.asarray(q95, dtype=np.float32),
               "betaN": np.asarray(betaN, dtype=np.float32),
               "times": times}
        for key in ("psirz", "q95", "betaN"):
            if not np.isfinite(out[key]).all():
                raise ValueError(f"non-finite {key}")
        if shape_targets is not None:
            if not np.isfinite(shape_targets).all():
                raise ValueError("non-finite shape targets")
            out["shape_targets"] = shape_targets
        if shape_groups is not None:
            if not np.isfinite(shape_groups).all():
                raise ValueError("non-finite shape flux groups")
            out["shape_groups"] = shape_groups
        if self.shape_member_targets is not None:
            out["shape_member_targets"] = self.shape_member_targets
        return out


def apply_scalar_stack(heads: np.ndarray, stack: dict) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(stack["weights"], dtype=np.float64)          # (n_members, 2)
    b = np.asarray(stack["intercept"], dtype=np.float64)        # (2,)
    value = np.einsum("fmj,mj->fj", heads, w) + b
    return value[:, 0].astype(np.float32), value[:, 1].astype(np.float32)


def post_pool(predictor: "Predictor"):
    """The per-shot post-processing of the deployed chain (li guard, field correction), as in ``build``.

    Returns ``(pool or None, extra(out) -> dict)``.  Used by held-out scoring (``nm1.score``) so that it
    scores what ``build`` writes; ``build`` itself keeps its own copy of this set-up unchanged.
    """
    guard_cfg = predictor.stack.get("li_guard") or {}
    correction_cfg = predictor.stack.get("field_correction") or {}
    guard = None
    v13 = correction_cfg.get("enabled") and correction_cfg.get("method") == "v13"
    if correction_cfg.get("enabled"):
        if not guard_cfg.get("enabled"):
            raise ValueError("v12 field correction expects the baseline li guard")
        params = {k: guard_cfg[k] for k in ("li_max", "knee", "slope") if k in guard_cfg}
        # NM1_CORR_WORKERS: fewer CPU correction workers on a small-RAM machine (each holds a few hundred MB and the
        # queue keeps 3 shots per worker); the shots are corrected independently, so the output does not change
        workers = int(os.environ.get("NM1_CORR_WORKERS") or correction_cfg.get("workers", 6))
        if v13:
            from nm1.v13_shape import V13CorrectionPool
            guard = V13CorrectionPool(correction_cfg.get("config") or {}, workers=workers,
                                      depth=max(12, 3 * workers), params=params,
                                      group_names=correction_cfg["groups"])
        else:
            from nm1.v12_shape import FieldCorrectionPool
            guard = FieldCorrectionPool(correction_cfg["strength"], workers=workers,
                params=params, method=correction_cfg.get("method", "legacy"))
    elif guard_cfg.get("enabled"):
        from nm1.guard import ShotGuard
        guard = ShotGuard(workers=int(guard_cfg.get("workers", 6)),
                          params={k: guard_cfg[k] for k in ("li_max", "knee", "slope") if k in guard_cfg})

    def extra(out):
        if guard is None:
            return {}
        e = {"targets": out["shape_targets"]} if correction_cfg.get("enabled") else {}
        if "shape_groups" in out:
            e["groups"] = out["shape_groups"]
        if v13:
            e["times"] = np.asarray(out["times"], float) / 1000.0
        return e
    return guard, extra


def build(config: str = "diii_d_public_test", output: Path | None = None,
          manifest: Path = MEMBERS, stack: Path | None = STACK, prefetch: int = 4,
          flux_dtype: str = "float32", limit: int = 0) -> Path:
    """Stream one machine's public-test predictions into a submission .npz."""
    output = output or OUTPUTS / "submission" / f"{config}.npz"
    files = listing(config)
    if limit:
        files = files[:limit]
    paths = [local_path(f) for f in files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} {config} parquets are missing; "
                         f"run nm1.fetch --config {config}")
    # One writer per output file. Two runs interleaving .npy members into the same zip produce a
    # file that opens and is silently wrong, which is the worst failure mode available here.
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock")
    if lock.exists():
        holder = lock.read_text(encoding="utf-8").strip()
        raise SystemExit(f"{output.name} is already being written by pid {holder}. "
                         f"Stop it, or delete {lock} if that process is gone.")
    lock.write_text(str(os.getpid()), encoding="utf-8")

    predictor = Predictor(manifest, stack)
    print(f"{config}: {len(paths)} shots, {len(predictor.members)} members, "
          f"device={predictor.device}, pid={os.getpid()}", flush=True)
    writer = NpzStreamWriter(output)
    started, frames = time.perf_counter(), 0
    guard_cfg = predictor.stack.get("li_guard") or {}
    correction_cfg = predictor.stack.get("field_correction") or {}
    guard = None
    v13 = correction_cfg.get("enabled") and correction_cfg.get("method") == "v13"
    if correction_cfg.get("enabled"):
        if not guard_cfg.get("enabled"):
            raise ValueError("v12 field correction expects the baseline li guard")
        params = {k: guard_cfg[k] for k in ("li_max", "knee", "slope") if k in guard_cfg}
        # NM1_CORR_WORKERS: fewer CPU correction workers on a small-RAM machine (each holds a few hundred MB and the
        # queue keeps 3 shots per worker); the shots are corrected independently, so the output does not change
        workers = int(os.environ.get("NM1_CORR_WORKERS") or correction_cfg.get("workers", 6))
        if v13:
            from nm1.v13_shape import V13CorrectionPool
            guard = V13CorrectionPool(correction_cfg.get("config") or {}, workers=workers,
                                      depth=max(12, 3 * workers), params=params,
                                      group_names=correction_cfg["groups"])
        else:
            from nm1.v12_shape import FieldCorrectionPool
            guard = FieldCorrectionPool(correction_cfg["strength"], workers=workers,
                params=params, method=correction_cfg.get("method", "legacy"))
    elif guard_cfg.get("enabled"):
        from nm1.guard import ShotGuard
        guard = ShotGuard(workers=int(guard_cfg.get("workers", 6)),
                          params={k: guard_cfg[k] for k in ("li_max", "knee", "slope")
                                  if k in guard_cfg})

    def emit(key, psi, scalars):
        writer.write(f"shot_{key:04d}_psirz", np.asarray(psi).astype(flux_dtype))
        writer.write(f"shot_{key:04d}_q95", scalars[0])
        writer.write(f"shot_{key:04d}_betaN", scalars[1])

    try:
        # A bounded read-ahead window, not pool.map: one DIII-D row holds ~480 k magnetics
        # samples per channel (tens of MB), and map would submit all 874 at once and keep every
        # completed row alive while inference -- far slower than parsing -- worked through them.
        # NM1_SKIP_SHOTS (resume after an out-of-memory kill, nm1.salvage): a json list of shot indices that an
        # earlier run of the same manifest already finished; they are not predicted here and are stitched in later
        skip = set(json.loads(Path(os.environ["NM1_SKIP_SHOTS"]).read_text())) if os.environ.get("NM1_SKIP_SHOTS") else set()
        todo = [i for i in range(len(paths)) if i not in skip]
        with ThreadPoolExecutor(max_workers=prefetch) as pool:
            ahead = {n: pool.submit(read_row, paths[todo[n]], True)
                     for n in range(min(prefetch, len(todo)))}
            for n, i in enumerate(todo):
                row = ahead.pop(n).result()
                if n + prefetch < len(todo):
                    ahead[n + prefetch] = pool.submit(read_row, paths[todo[n + prefetch]], True)
                out = predictor.predict(row)
                bump = predictor.stack.get("interior_bump")
                if bump and bump.get("enabled"):
                    from nm1.interior import apply_shot
                    out["psirz"] = apply_shot(out["psirz"], row, out["times"], bump)
                grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
                grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
                del row
                if guard is not None:
                    extra = ({"targets": out["shape_targets"]}
                             if correction_cfg.get("enabled") else {})
                    if "shape_groups" in out:
                        extra["groups"] = out["shape_groups"]
                    if v13:
                        extra["times"] = np.asarray(out["times"], float) / 1000.0
                    for key, psi_g, sc in guard.submit(
                            i, out["psirz"], grid_r, grid_z,
                            (out["q95"], out["betaN"]), **extra):
                        emit(key, psi_g, sc)
                else:
                    emit(i, out["psirz"], (out["q95"], out["betaN"]))
                frames += len(out["q95"])
                if (n + 1) % 25 == 0 or n + 1 == len(todo):
                    el = time.perf_counter() - started
                    print(f"  {n + 1}/{len(todo)} shots, {frames:,} frames, "
                          f"{output.stat().st_size / 1e6:.0f} MB, {el / 60:.1f} min "
                          f"(~{el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min left)", flush=True)
        if guard is not None:
            for key, psi_g, sc in guard.drain():
                emit(key, psi_g, sc)
            guard_notes = [n for n in guard.notes if n.get("status") != "field_correction"]
            print(f"li guard: {len(guard_notes)} frame(s) outside the screened range; "
                  f"{sum(1 for n in guard.notes if n.get('status') == 'guarded')} relabelled",
                  flush=True)
            if correction_cfg.get("enabled"):
                print(f"shape corrections: {sum(n.get('status') == 'field_correction' for n in guard.notes)} shots",
                      flush=True)
    finally:
        writer.close()
        lock.unlink(missing_ok=True)
    output.with_suffix(".json").write_text(json.dumps(
        {"config": config, "shots": len(paths), "frames": frames, "flux_dtype": flux_dtype,
         "members": json.loads(Path(manifest).read_text(encoding="utf-8"))["members"],
         "stack": predictor.stack,
         "li_guard": (guard.notes if guard is not None else None)}, indent=2, default=float),
        encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e9:.2f} GB)", flush=True)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="diii_d_public_test")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--manifest", type=Path, default=MEMBERS)
    ap.add_argument("--stack", type=Path, default=STACK)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--flux-dtype", default="float32", choices=("float16", "float32", "float64"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.config, a.output, a.manifest, a.stack, a.prefetch, a.flux_dtype, a.limit)


if __name__ == "__main__":
    main()