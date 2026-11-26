"""Train one ensemble member against a differentiable surrogate of the competition score.

A plain PCA-coefficient MSE weights mode m by ``var(c_m)``, which spans four orders of magnitude,
so every mode past ~16 is trained on almost nothing -- while the Consistency term, 20% of the
score, lives entirely in modes 0-63.  The loss here is built from the metric instead:

    L = 0.55*(1 - R2_psi) + 0.20*(1 - Consistency) + 0.15*(1 - R2_qb)

with the Consistency deficit expanded to first order through the measured mode Jacobian (see
``jacobian.py``).  ``D_LCFS`` (0.10) is not in the loss: it is carried by the same boundary modes
as the shape scalars.

The whole coefficient cache sits on the GPU, so an epoch over 1.4 M frames takes seconds.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from nm1 import ARTIFACTS
from nm1.cache import BASIS, CACHE
from nm1.jacobian import OUT as JACOBIAN
from nm1.model import MetricRegressor, context_indices

W_PSI, W_QB, W_CONS = 0.55, 0.15, 0.20


def _with_context(feat: np.ndarray, shot: np.ndarray, offsets: list[int]) -> np.ndarray:
    """Base block plus one difference block per offset, written straight into the output.

    Concatenating a list of blocks would hold two copies of a multi-GB array at once.
    """
    if not offsets:
        return feat
    idx = context_indices(shot, offsets)
    n, d = feat.shape
    out = np.empty((n, d * (1 + len(offsets))), dtype=feat.dtype)
    out[:, :d] = feat
    for k in range(len(offsets)):
        np.subtract(feat[idx[:, k]], feat, out=out[:, d * (k + 1):d * (k + 2)])
    return out


def _stats(block: np.ndarray, chunk: int = 262_144) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and std, accumulated in float64 chunks.

    float64 is needed per chunk even though the block is float32: the density-derived columns run
    to ~1e19 and their squares overflow float32.
    """
    total = np.zeros(block.shape[1], dtype=np.float64)
    for a in range(0, len(block), chunk):
        total += block[a:a + chunk].astype(np.float64).sum(0)
    mean = total / len(block)
    var = np.zeros(block.shape[1], dtype=np.float64)
    for a in range(0, len(block), chunk):
        d = block[a:a + chunk].astype(np.float64) - mean
        var += (d * d).sum(0)
    return mean.astype(np.float32), np.maximum(np.sqrt(var / len(block)), 1e-6).astype(np.float32)


class _MirrorPlan:
    """How a mirrored frame is read off the whitened input matrix (nm1/mirror.py).

    Blocks are registered in the order they are appended to the matrix. Column ``j`` of the
    image is ``sign[j] * frame[:, perm[j]]``, or the constant ``fill_val[j]`` where ``fill[j]``.
    """

    def __init__(self) -> None:
        self.perm, self.sign, self.fill, self.fill_val = [], [], [], []
        self.width = 0

    def add(self, perm, sign=None, fill=None, fill_val=None) -> None:
        n = len(perm)
        self.perm.append(np.asarray(perm, dtype=np.int64) + self.width)
        self.sign.append(np.ones(n) if sign is None else np.asarray(sign, dtype=np.float64))
        self.fill.append(np.zeros(n, bool) if fill is None else np.asarray(fill, dtype=bool))
        self.fill_val.append(np.zeros(n) if fill_val is None else np.asarray(fill_val, np.float64))
        self.width += n

    def tensors(self, device):
        cat = np.concatenate
        return (torch.from_numpy(cat(self.perm)).to(device),
                torch.from_numpy(cat(self.sign).astype(np.float32)).to(device),
                torch.from_numpy(cat(self.fill)).to(device),
                torch.from_numpy(cat(self.fill_val).astype(np.float32)).to(device))


def build_weights(scheme: str, coeff_std: np.ndarray, jacobian: Path, n_comp: int,
                  scalar_weights: np.ndarray | None = None):
    """``(psi_weight, cons_weight, gram)`` per mode, in whitened-coefficient units.

    `scalar_weights` rescales the seven scored scalars inside the Consistency surrogate before
    they are summed. All ones reproduces the metric's own first-order weighting, which is the
    default and what every deployed member used.

    Why it exists: the summed weight is each mode's *sensitivity*, and the seven scalars do not
    contribute equally to it -- `volume` carries 6.07 of it and `li` 1.90, the lowest. But the
    metric averages the seven R-squareds equally, and `li` is the one furthest from its own
    ceiling (0.976 against 0.9996). Whether shifting loss weight toward it buys more than it
    costs the others is an empirical question, not a derivation, so it is an option and off by
    default.
    """
    v_tot = float((coeff_std.astype(np.float64) ** 2).sum())
    w_psi = coeff_std.astype(np.float64) ** 2 / v_tot
    w_cons = np.zeros(n_comp)
    if scheme == "variance":                            # the plain coefficient-MSE objective
        return w_psi, w_cons, None
    z = np.load(jacobian, allow_pickle=True)
    contrib = z["contrib"].astype(np.float64)
    if scalar_weights is not None:
        sw = np.asarray(scalar_weights, dtype=np.float64)
        if sw.shape != (contrib.shape[0],) or not np.all(sw >= 0) or not sw.any():
            raise SystemExit(f"scalar_weights must be {contrib.shape[0]} non-negative numbers")
        sw = sw * (len(sw) / sw.sum())          # keep the overall scale of the term
        contrib = contrib * sw[:, None]
    w = contrib.sum(axis=0)
    k = min(len(w), n_comp)
    w_cons[:k] = w[:k] / 7.0                            # the metric averages the seven equally
    if scheme == "jacobian":
        return w_psi, w_cons, None
    if scheme != "gram":
        raise SystemExit(f"unknown weight scheme {scheme}")
    if scalar_weights is None:
        G = z["gram"].astype(np.float64)
    elif "gram_per_scalar" in z.files:
        G = np.einsum("j,jmn->mn", sw, z["gram_per_scalar"].astype(np.float64))
    else:
        raise SystemExit("scalar_weights with --weights gram needs `gram_per_scalar`, which this "
                         "Jacobian artifact predates; re-run `nm1.jacobian`")
    # Clip to the PSD cone: the finite-difference estimate can carry tiny negative eigenvalues,
    # and a loss that is negative along some error direction would be maximised, not minimised.
    ev, V = np.linalg.eigh(G)
    if int((ev < 0).sum()):
        print(f"  gram: clipped {int((ev < 0).sum())} negative eigenvalue(s), min {ev.min():.3e}")
        G = (V * np.clip(ev, 0.0, None)) @ V.T
    full = np.zeros((n_comp, n_comp))
    kk = min(len(G), n_comp)
    full[:kk, :kk] = G[:kk, :kk] / 7.0
    w_cons[:kk] = np.diag(full)[:kk]
    return w_psi, w_cons, full


def train(output: Path, cache: Path = CACHE, basis: Path = BASIS, jacobian: Path = JACOBIAN,
          weights: str = "gram", epochs: int = 70, batch_size: int = 8192, width: int = 1024,
          depth: int = 6, dropout: float = 0.02, lr: float = 2e-3, weight_decay: float = 1e-5,
          warmup: int = 300, seed: int = 1234, val_frames: int = 60_000,
          vac_features: int = 0, ema: float = 0.0, temporal_context: str = "",
          loss_weights: str = "", select: str = "S_hat", n_features: int = 140,
          scalar_weights: str = "", extra_train: bool = False,
          grad_weight: float = 0.0, ext: bool = False, axis_weight: float = 0.0,
          axis_exclude: bool = False, init: Path | None = None, emul_weight: float = 0.0,
          lcfs_weight: float = 0.0, emulator: Path | None = None, ctx: bool = False,
          cds: bool = False, cds_columns: int = 0, ramp_weight: float = 1.0,
          cdsg: bool = False, cdsg_columns: int = 0, folds: int = 0, fold: int = 0,
          all_data: bool = False, save_epoch: int = 0, save_source: str = "raw",
          gs_weight: float = 0.0, gs_ramp: int = 10, gs_psi_max: float = 0.9,
          seg: bool = False, mirror: bool = False, mirror_vac: bool = True,
          mirror_dropout: float = 0.0, dilate_lo: float = 0.0, dilate_hi: float = 0.0,
          dilate_p: float = 0.5, dilate_cons_scale: bool = True,
          scalar_scale: int = 0) -> Path:
    """``init`` continues a trained member (same inputs, same whitening, same architecture).

    ``emul_weight`` in [0, 1] moves the Consistency term from the averaged Gram form to the scorer
    emulator (``emulator.py``) frame by frame; ``lcfs_weight`` adds the emulator's LCFS Hausdorff
    at that multiple of D_LCFS's metric weight.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    init_payload = None
    if init is not None:
        # Everything that fixes the member's inputs comes from the checkpoint, so the continued
        # network reads exactly the numbers it was trained on.
        init_payload = torch.load(init, map_location="cpu", weights_only=False)
        temporal_context = ",".join(str(v) for v in (init_payload.get("temporal_offsets") or []))
        vac_features = int(init_payload.get("vac_features") or 0)
        n_features = int(init_payload.get("n_base_features") or 140)
        ext = bool(int(init_payload.get("n_ext") or 0))
        ctx = bool(int(init_payload.get("n_ctx") or 0))
        cds = bool(int(init_payload.get("n_cds") or 0))
        cdsg = bool(int(init_payload.get("n_cdsg") or 0))
        seg = bool(int(init_payload.get("n_seg") or 0))
        width = int(init_payload["model"]["width"])
        depth = int(init_payload["model"]["depth"])
        dropout = float(init_payload["model"]["dropout"])
        print(f"continuing {init} (epoch {init_payload.get('epoch')}): {width}x{depth}, "
              f"{n_features} base, context '{temporal_context}', vac {vac_features}, "
              f"ext {ext}", flush=True)

    state = torch.load(basis, map_location="cpu", weights_only=False)
    coeff_std = state["coeff_std"]
    n_comp = len(coeff_std)
    offsets = [int(v) for v in temporal_context.split(",") if v.strip()]
    if mirror and (vac_features or offsets or n_features != 140
                   or extra_train or all_data or folds or axis_weight or gs_weight
                   or (ramp_weight and ramp_weight != 1.0) or init is not None):
        raise SystemExit("--mirror covers the base 140 block plus the ext/ctx/cds/cdsg/seg blocks "
                         "(nm1/mirror.py); not vacuum, history offsets, extra/all-data, "
                         "folds, the axis/GS terms or --init")
    plan = _MirrorPlan() if mirror else None
    dplan = None
    if dilate_hi:
        if not 0 < dilate_lo < 1 < dilate_hi:
            raise SystemExit("--dilate LO,HI needs 0 < LO < 1 < HI")
        # --all-data is allowed: the augmentation acts on a batch row (inputs, coefficients, scalars) and does not
        # depend on which shots are in training; the final run trains the dilation group on every labelled shot
        if (vac_features or offsets or n_features != 140 or mirror or seg or init is not None
                or extra_train or folds):
            raise SystemExit("--dilate covers the base 140 block plus ext/ctx/cds/cdsg "
                             "(nm1/dilation.py); not vacuum, history offsets, mirror, seg, --init "
                             "or the extra-train and fold modes")
        from nm1.dilation import Plan as DilationPlan
        dplan = DilationPlan()

    print("loading the coefficient cache to the GPU ...", flush=True)
    t0 = time.perf_counter()
    with h5py.File(cache, "r") as h5:
        tr, va = h5["train"], h5["val"]
        stored = tr["features"].shape[1]
        if n_features > stored:
            raise SystemExit(f"the cache has {stored} feature columns, not {n_features}")

        # `extra_train` moves the calibration third of the held-out shots into training and picks
        # epochs on `selection` instead. Calibration was only ever an epoch-selection pool and the
        # stack's fitting set, and `selection` can carry both; `evaluation` is untouched either
        # way. It is 6,188 -> 6,604 shots, and 417 shots the model has never seen is the one lever
        # here that adds information rather than rearranging it.
        # The halves come from `score.py`, which defines them, rather than being re-derived here
        # from the cache's own ids -- those two disagree by two unparseable shots, and slicing each
        # independently put 33 evaluation shots into training the first time this was tried.
        from nm1.score import held_out_ids
        vshot_all = va["shot"][:]
        vids = np.unique(vshot_all)
        cal_ids = np.array(sorted(held_out_ids("calibration")), dtype=vids.dtype)
        sel_ids = np.array(sorted(held_out_ids("selection")), dtype=vids.dtype)
        cal_mask = np.isin(vshot_all, cal_ids)
        val_ids = sel_ids if (extra_train or all_data) else cal_ids
        # `add_mask`: the held-out rows that join the training rows
        add_mask = None
        if extra_train and all_data:
            raise SystemExit("--extra-train and --all-data are alternatives")
        if extra_train:
            ev = held_out_ids("evaluation")
            bad = ev & set(cal_ids.tolist())
            if bad:
                raise SystemExit(f"{len(bad)} evaluation shots would enter training -- refusing")
            add_mask = cal_mask
        if all_data:
            # every labelled shot: no held-out frame is left to pick an epoch on, so the epoch
            # (and raw/EMA) must be given -- the one the same member picked on `selection`
            # when trained with --extra-train
            if save_epoch <= 0:
                raise SystemExit("--all-data needs --save-epoch: nothing is held out to pick one")
            add_mask = np.ones(len(vshot_all), dtype=bool)

        tr_shot = tr["shot"][:]
        tr_feat = tr["features"][:, :n_features]
        if add_mask is not None:
            tr_feat = np.concatenate([tr_feat, va["features"][:, :n_features][add_mask]])
            tr_shot = np.concatenate([tr_shot, vshot_all[add_mask]])
            print(f"  {'all_data' if all_data else 'extra_train'}: "
                  f"+{int(add_mask.sum()):,} held-out frames "
                  f"({len(np.unique(vshot_all[add_mask]))} shots)"
                  + (f", checkpoint at epoch {save_epoch} [{save_source}]" if all_data
                     else ", epochs selected on `selection`"), flush=True)
        frame_w = None
        if ramp_weight and ramp_weight != 1.0:
            # Ip over its own shot's maximum, from the base block's instantaneous Ip column; the
            # frames below 90% of it (ramp-up, ramp-down, the end of a discharge) carry half of
            # li's residual on held-out shots, so they are up-weighted in every loss term.
            ip = np.abs(tr_feat[:, 20].astype(np.float64))
            b_ = np.flatnonzero(np.diff(tr_shot)) + 1
            st_ = np.concatenate([[0], b_])
            en_ = np.concatenate([b_, [len(tr_shot)]])
            peak = np.empty_like(ip)
            for s_, e_ in zip(st_, en_):
                peak[s_:e_] = max(float(ip[s_:e_].max()), 1.0)
            frame_w = np.where(ip / peak < 0.9, float(ramp_weight), 1.0).astype(np.float32)
            print(f"  ramp weight x{ramp_weight:g} on {float((frame_w != 1).mean()):.1%} of the "
                  f"training frames", flush=True)
        ftr = _with_context(tr_feat, tr_shot, offsets)
        del tr_feat
        if init_payload is not None:
            fmean, fstd = init_payload["feature_mean"], init_payload["feature_std"]
        else:
            fmean, fstd = _stats(ftr)
            if plan is not None:
                # statistics of the frames and their mirror images together, so a mirrored frame
                # is exactly the signed permutation of the whitened real one
                from nm1.mirror import base_perm, mirror_stats
                bperm, bsign = base_perm(ftr.shape[1])
                fmean, fstd = mirror_stats(fmean, fstd, bperm, bsign)
                plan.add(bperm, bsign)
            if dplan is not None:
                from nm1.dilation import base_rules
                dplan.add(fmean, fstd, base_rules(len(fmean)))
        ftr -= fmean                                    # in place: a whitened copy is another GB
        ftr /= fstd
        # Every input block is whitened on the host and the member's input matrix goes to the GPU
        # once, at the end. Concatenating blocks on the GPU held the old and the new matrix at
        # the same time -- ~11 GB for a history member with both new blocks -- and on a 16 GB
        # card shared by two lanes that spilled into WDDM paging and stretched one member to
        # two and a half hours.
        blocks_tr = [ftr]
        del ftr
        ctr_np = tr["coeff"][:]
        scalars_tr = tr["scalars"][:]
        if add_mask is not None:
            ctr_np = np.concatenate([ctr_np, va["coeff"][:][add_mask]])
            scalars_tr = np.concatenate([scalars_tr, va["scalars"][:][add_mask]])
        if mirror:
            # the up-down mirror image of every training frame (nm1/mirror.py): the flipped
            # flux map's coefficients, with the exact vacuum-flux correction for the coil set's
            # few-mm asymmetry; q95 and betaN do not change
            from nm1 import mirror as mir
            A_m, b_m, E_m = mir.flux_flip_maps(state["pca_mean"], state["pca_components"],
                                               h5["green"][:])
            ctr_np = np.concatenate([ctr_np, mir.mirror_coefficients(
                ctr_np, tr["currents"][:], A_m, b_m, E_m if mirror_vac else None)])
            scalars_tr = np.concatenate([scalars_tr, scalars_tr])
            print(f"  mirror: {len(ctr_np):,} training rows (real + up-down images), "
                  f"vacuum correction {'on' if mirror_vac else 'off'}", flush=True)
        cur_bi = None
        if scalar_scale:
            if init_payload is not None:
                raise SystemExit("--scalar-scale does not combine with --init")
            cur_bi = tr["currents"][:][:, [1, 20]]                    # bcoil, Ip of every training row
            if add_mask is not None:
                cur_bi = np.concatenate([cur_bi, va["currents"][:][add_mask][:, [1, 20]]])
            if mirror:
                cur_bi = np.concatenate([cur_bi, cur_bi])            # an image keeps Ip and bcoil
        ctr = torch.from_numpy(ctr_np).to(device)
        del ctr_np
        if init_payload is not None:
            smean, sstd = init_payload["scalar_mean"], init_payload["scalar_std"]
        else:
            smean = np.nanmean(scalars_tr, axis=0).astype(np.float32)
            sstd = np.maximum(np.nanstd(scalars_tr, axis=0), 1e-4).astype(np.float32)
        str_ = torch.from_numpy(scalars_tr).to(device)

        # Epochs are selected on half the validation shots only, so the shots the official-proxy
        # scorer later reports on are never the ones a checkpoint was chosen against.
        vshot = vshot_all
        pool = np.flatnonzero(np.isin(vshot, val_ids))
        nval = min(val_frames, len(pool))
        sel = pool[np.linspace(0, len(pool) - 1, nval).astype(np.int64)]
        print(f"  selection pool: {len(val_ids)}/{len(vids)} validation shots, "
              f"{len(pool):,} frames", flush=True)
        xva_full = _with_context(va["features"][:, :n_features], vshot, offsets)
        blocks_va = [((xva_full[sel] - fmean) / fstd).astype(np.float32)]
        del xva_full
        cva = torch.from_numpy(va["coeff"][:][sel]).to(device)
        sva = torch.from_numpy(va["scalars"][:][sel]).to(device)

        vac_stats: dict = {"vac_features": 0}
        if vac_features:
            comps = state["pca_components"]
            m = h5["green"][:].astype(np.float64) @ comps.astype(np.float64).T
            def vac_of(currents: np.ndarray) -> np.ndarray:
                return (currents[:, :m.shape[0]].astype(np.float64) @ m)[:, :vac_features]
            currents_tr = tr["currents"][:]
            if add_mask is not None:
                currents_tr = np.concatenate([currents_tr, va["currents"][:][add_mask]])
            vtr = vac_of(currents_tr)
            if init_payload is not None:
                vmean, vstd = init_payload["vac_mean"], init_payload["vac_std"]
            else:
                vmean = vtr.mean(0).astype(np.float32)
                vstd = np.maximum(vtr.std(0), 1e-6).astype(np.float32)
            blocks_tr.append(((vtr - vmean) / vstd).astype(np.float32))
            del vtr
            vva = vac_of(va["currents"][:][sel])
            blocks_va.append(((vva - vmean) / vstd).astype(np.float32))
            vac_stats = {"vac_features": vac_features, "vac_mean": vmean, "vac_std": vstd}
            print(f"  appended {vac_features} whitened vacuum-flux coefficients", flush=True)

    # The second feature block (nm1/features2.py): full-rate coil-current windows and
    # position-aware Thomson profiles, row-aligned with the coefficient cache by cache2.py.
    ext_stats: dict = {"n_ext": 0}
    if ext:
        from nm1.cache2 import CACHE2
        with h5py.File(CACHE2, "r") as h5e:
            etr = h5e["train"]["ext"][:]
            eva_all = h5e["val"]["ext"][:]
        if add_mask is not None:
            etr = np.concatenate([etr, eva_all[add_mask]])
        if init_payload is not None:
            emean, estd = init_payload["ext_mean"], init_payload["ext_std"]
        else:
            emean, estd = _stats(etr)
            estd = np.maximum(estd, 1e-3).astype(np.float32)   # near-constant columns must not explode
            if plan is not None:
                from nm1.features2 import UNSEEN
                from nm1.mirror import ext_perm, mirror_stats
                eperm, esign, efill = ext_perm()
                emean, estd = mirror_stats(emean, estd, eperm, esign)
                estd = np.maximum(estd, 1e-3).astype(np.float32)
                plan.add(eperm, esign, efill, (UNSEEN - emean) / estd)
            if dplan is not None:
                from nm1.dilation import ext_rules
                dplan.add(emean, estd, ext_rules())
        etr -= emean
        etr /= estd
        blocks_tr.append(etr)
        del etr
        blocks_va.append(((eva_all[sel] - emean) / estd).astype(np.float32))
        del eva_all
        ext_stats = {"n_ext": int(emean.shape[0]), "ext_mean": emean, "ext_std": estd}
        print(f"  appended {emean.shape[0]} extended features", flush=True)
        if ctx:
            # shot context (features2.context_block), per shot on the time-ordered frames
            from nm1.features2 import context_block
            with h5py.File(cache, "r") as h5c:
                t_tr = h5c["train"]["features"][:, 136]
                t_va = h5c["val"]["features"][:, 136]
            with h5py.File(CACHE2, "r") as h5e:
                e_tr_raw = h5e["train"]["ext"][:]
                e_va_raw = h5e["val"]["ext"][:]
            if add_mask is not None:
                # whole held-out shots, appended in cache order like every other block, so
                # the per-shot context sees each of them complete
                t_tr = np.concatenate([t_tr, t_va[add_mask]])
                e_tr_raw = np.concatenate([e_tr_raw, e_va_raw[add_mask]])

            def ctx_of(e_raw, t_s, shot_ids):
                out = np.empty((len(e_raw), 0), np.float32)
                b = np.flatnonzero(np.diff(shot_ids)) + 1
                st = np.concatenate([[0], b])
                en = np.concatenate([b, [len(shot_ids)]])
                parts = [context_block(e_raw[s:e], t_s[s:e]) for s, e in zip(st, en)]
                return np.concatenate(parts) if parts else out

            ctr_ = ctx_of(e_tr_raw, t_tr, tr_shot)
            cva_ = ctx_of(e_va_raw, t_va, vshot_all)
            del e_tr_raw, e_va_raw
            if init_payload is not None:
                cmean, cstd_ = init_payload["ctx_mean"], init_payload["ctx_std"]
            else:
                cmean, cstd_ = _stats(ctr_)
                cstd_ = np.maximum(cstd_, 1e-3).astype(np.float32)
                if plan is not None:
                    plan.add(np.arange(len(cmean)))       # Ip- and Te-history columns: even
                if dplan is not None:
                    from nm1.dilation import ctx_rules
                    dplan.add(cmean, cstd_, ctx_rules(len(cmean)))
            blocks_tr.append(((ctr_ - cmean) / cstd_).astype(np.float32))
            blocks_va.append(((cva_[sel] - cmean) / cstd_).astype(np.float32))
            del ctr_, cva_
            ext_stats.update({"n_ctx": int(cmean.shape[0]), "ctx_mean": cmean, "ctx_std": cstd_})
            print(f"  appended {cmean.shape[0]} shot-context features", flush=True)

    # The current-diffusion block (nm1/diffusion.py), row-aligned with the coefficient cache.
    if cds:
        from nm1.diffusion import CDS_CACHE
        with h5py.File(CDS_CACHE, "r") as h5d:
            ncol = (int(init_payload["n_cds"]) if init_payload is not None
                    else (cds_columns or h5d["train"]["cds"].shape[1]))
            dtr = h5d["train"]["cds"][:, :ncol]
            dva_all = h5d["val"]["cds"][:, :ncol]
        if add_mask is not None:
            dtr = np.concatenate([dtr, dva_all[add_mask]])
        if init_payload is not None:
            dmean, dstd = init_payload["cds_mean"], init_payload["cds_std"]
        else:
            dmean, dstd = _stats(dtr)
            dstd = np.maximum(dstd, 1e-3).astype(np.float32)
            if plan is not None:
                plan.add(np.arange(len(dmean)))           # fixed symmetric ellipse: even
            if dplan is not None:
                from nm1.dilation import cds_rules
                dplan.add(dmean, dstd, cds_rules(len(dmean)))
        dtr -= dmean
        dtr /= dstd
        blocks_tr.append(dtr)
        del dtr
        blocks_va.append(((dva_all[sel] - dmean) / dstd).astype(np.float32))
        del dva_all
        ext_stats.update({"n_cds": int(dmean.shape[0]), "cds_mean": dmean, "cds_std": dstd})
        print(f"  appended {dmean.shape[0]} current-diffusion features", flush=True)

    # The same block with Thomson on the equilibrium's flux coordinate (nm1/cds_geo.py):
    # the out-of-fold first stage's flux for training rows, v8's for the held-out ones.
    if cdsg:
        from nm1.cds_geo import CDSG_CACHE
        with h5py.File(CDSG_CACHE, "r") as h5g:
            ncol = (int(init_payload["n_cdsg"]) if init_payload is not None
                    else (cdsg_columns or h5g["train"]["cds"].shape[1]))
            gtr = h5g["train"]["cds"][:, :ncol]
            gva_all = h5g["val"]["cds"][:, :ncol]
        if add_mask is not None:
            # the held-out rows' block is on v8's own flux, which never saw those shots -- a
            # predicted equilibrium, like the out-of-fold one of the training rows
            gtr = np.concatenate([gtr, gva_all[add_mask]])
        if init_payload is not None:
            gmean, gstd = init_payload["cdsg_mean"], init_payload["cdsg_std"]
        else:
            gmean, gstd = _stats(gtr)
            gstd = np.maximum(gstd, 1e-3).astype(np.float32)
            if plan is not None:
                plan.add(np.arange(len(gmean)))           # sqrt(psi_N) of a mirrored map: even
            if dplan is not None:
                from nm1.dilation import cds_rules
                dplan.add(gmean, gstd, cds_rules(len(gmean)))
        gtr -= gmean
        gtr /= gstd
        blocks_tr.append(gtr)
        del gtr
        blocks_va.append(((gva_all[sel] - gmean) / gstd).astype(np.float32))
        del gva_all
        ext_stats.update({"n_cdsg": int(gmean.shape[0]), "cdsg_mean": gmean, "cdsg_std": gstd})
        print(f"  appended {gmean.shape[0]} flux-coordinate current-diffusion features", flush=True)

    # Where each frame sits in its run of consecutive EFIT frames (features2.seg_block): the
    # limited <-> diverted transitions, from the shipped `efit_times` alone. Appended last.
    if seg:
        from nm1.features2 import seg_of_shots
        with h5py.File(cache, "r") as h5s:
            t_tr_s = h5s["train"]["features"][:, 136].astype(np.float64)
            t_va_s = h5s["val"]["features"][:, 136].astype(np.float64)
        if add_mask is not None:
            t_tr_s = np.concatenate([t_tr_s, t_va_s[add_mask]])
        seg_tr = seg_of_shots(t_tr_s, tr_shot)
        seg_va = seg_of_shots(t_va_s, vshot_all)
        if init_payload is not None:
            sgmean, sgstd = init_payload["seg_mean"], init_payload["seg_std"]
        else:
            sgmean, sgstd = _stats(seg_tr)
            sgstd = np.maximum(sgstd, 1e-3).astype(np.float32)
            if plan is not None:
                plan.add(np.arange(len(sgmean)))          # position in the run: even
        blocks_tr.append(((seg_tr - sgmean) / sgstd).astype(np.float32))
        blocks_va.append(((seg_va[sel] - sgmean) / sgstd).astype(np.float32))
        ext_stats.update({"n_seg": int(sgmean.shape[0]), "seg_mean": sgmean, "seg_std": sgstd})
        print(f"  appended {sgmean.shape[0]} run-position features", flush=True)
        del seg_tr, seg_va, t_tr_s, t_va_s

    def _assemble(blocks: list[np.ndarray]) -> torch.Tensor:
        """One host matrix, filled block by block, then a single copy to the GPU."""
        out = np.empty((len(blocks[0]), sum(b.shape[1] for b in blocks)), dtype=np.float32)
        col = 0
        while blocks:
            b = blocks.pop(0)                  # released as soon as it has been copied
            out[:, col:col + b.shape[1]] = b
            col += b.shape[1]
            del b
        return torch.from_numpy(out).to(device)

    xtr = _assemble(blocks_tr)
    xva = _assemble(blocks_va)
    fill_t = fillv_t = None
    if plan is not None:
        # real rows, then their up-down images, in one half-precision matrix (the first layer runs
        # in fp16 under autocast anyway); built in chunks so the two never coexist in full
        assert plan.width == xtr.shape[1], (plan.width, xtr.shape)
        perm_t, sign_t, fill_t, fillv_t = plan.tensors(device)
        n_real, d_in = xtr.shape
        xfull = torch.empty((2 * n_real, d_in), dtype=torch.float16, device=device)
        for s in range(0, n_real, 262_144):
            blk = xtr[s:s + 262_144]
            xfull[s:s + len(blk)] = blk.half()
            xm = blk[:, perm_t] * sign_t
            xm[:, fill_t] = fillv_t[fill_t]
            xfull[n_real + s:n_real + s + len(blk)] = xm.half()
        del xtr, blk, xm
        xtr = xfull
        del xfull
        torch.cuda.empty_cache()
        print(f"  mirror: input matrix {tuple(xtr.shape)} half precision, "
              f"{int(fill_t.sum())} columns filled in the images", flush=True)
    print(f"  member input: {xtr.shape[1]} features; GPU "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    # The scorer's own axis functional, exact per frame (nm1/axis.py).
    axis = None
    if axis_weight:
        from nm1.axis import AxisHead, cached_targets
        ax_tr, _ = cached_targets()
        if add_mask is not None:
            raise SystemExit("--axis-weight with --extra-train/--all-data is not wired up")
        axis = {"head": AxisHead(state["pca_mean"], state["pca_components"], coeff_std, device),
                "iz": torch.from_numpy(ax_tr["iz"].astype(np.int64)).to(device),
                "ir": torch.from_numpy(ax_tr["ir"].astype(np.int64)).to(device),
                "R": torch.from_numpy(ax_tr["R_axis"].astype(np.float32)).to(device),
                "Z": torch.from_numpy(ax_tr["Z_axis"].astype(np.float32)).to(device),
                "var_R": float(np.var(ax_tr["R_axis"])), "var_Z": float(np.var(ax_tr["Z_axis"]))}
        print(f"  exact axis loss x{axis_weight:g} (std R {axis['var_R'] ** 0.5:.4f} m, "
              f"std Z {axis['var_Z'] ** 0.5:.4f} m)"
              + ("; R/Z axis removed from the Gram term" if axis_exclude else ""), flush=True)

    cstd_t = torch.from_numpy(coeff_std).to(device)
    ztr, zva = ctr / cstd_t, cva / cstd_t
    del ctr, cva
    fw_t = torch.from_numpy(frame_w).to(device) if frame_w is not None else None

    # The physics-informed term (nm1/pinn.py): the Grad-Shafranov residual of the predicted
    # flux at sampled points inside the true plasma, with the profiles refitted per frame.
    gs = None
    if gs_weight:
        from nm1.gradient import grid
        from nm1.pinn import GSFC_CACHE, delta_star_basis
        R_g, Z_g = grid()
        dsm, dsc = delta_star_basis(state["pca_mean"].astype(np.float64),
                                    state["pca_components"].astype(np.float64), R_g, Z_g)
        with h5py.File(GSFC_CACHE, "r") as h5g:
            gtr_, gva_ = h5g["train"], h5g["val"]
            parts = {k: gtr_[k][:] for k in ("idx", "psin", "ynorm", "valid")}
            if add_mask is not None:
                for k in parts:
                    parts[k] = np.concatenate([parts[k], gva_[k][:][add_mask]])
            vparts = {k: gva_[k][:][sel] for k in ("idx", "psin", "ynorm", "valid")}
        gs = {"DSm": torch.from_numpy(dsm).float().to(device),
              "DSw": torch.from_numpy(dsc * coeff_std[:, None]).float().to(device),
              "R2": torch.from_numpy(np.tile(R_g ** 2, len(Z_g))).float().to(device),
              **{k: torch.from_numpy(v).to(device) for k, v in parts.items()},
              **{"va_" + k: torch.from_numpy(v).to(device) for k, v in vparts.items()}}
        print(f"  physics term: Grad-Shafranov residual x{gs_weight:g} (ramped over "
              f"{gs_ramp} epochs), {parts['idx'].shape[1]} points/frame, "
              f"{float(parts['valid'].mean()):.1%} of the training frames usable", flush=True)
        del parts, vparts

    if folds:
        # out-of-fold first stage: train on every training shot outside fold `fold`
        if axis is not None:
            raise SystemExit("--folds with --axis-weight is not wired up")
        keep = torch.from_numpy((tr_shot % folds) != fold).to(device)
        xtr, ztr, str_ = xtr[keep], ztr[keep], str_[keep]
        if fw_t is not None:
            fw_t = fw_t[keep]
        if gs is not None:
            for k in ("idx", "psin", "ynorm", "valid"):
                gs[k] = gs[k][keep]
        print(f"  fold {fold}/{folds}: training on {int(keep.sum()):,} of {len(keep):,} rows",
              flush=True)
        del keep
        torch.cuda.empty_cache()
    smean_t, sstd_t = torch.from_numpy(smean).to(device), torch.from_numpy(sstd).to(device)
    strz, svaz = (str_ - smean_t) / sstd_t, (sva - smean_t) / sstd_t
    dil = dplan
    if dil is not None:
        assert dil.width == xtr.shape[1], (dil.width, xtr.shape)
        dil.finalize(device, state["pca_mean"], state["pca_components"], coeff_std)
        print(f"  dilation augmentation: lambda in [{dilate_lo:g}, {dilate_hi:g}] on "
              f"{dilate_p:.0%} of the rows", flush=True)
    n_train = len(xtr)
    # NM1_HOST_INPUT=wide (or 1): the training input matrix stays in host memory and each batch is gathered there and
    # copied to the GPU -- the same rows in the same order, so the result is unchanged; for members whose
    # activations and the input matrix do not fit an 8 GB card together
    import os
    _host = os.environ.get("NM1_HOST_INPUT", "")
    host_input = device.type == "cuda" and (_host == "1" or (_host == "wide" and width >= 1536))
    if host_input:
        xtr = xtr.cpu()
        torch.cuda.empty_cache()
        print(f"  input matrix kept in host memory ({xtr.numel() * xtr.element_size() / 1e9:.2f} GB)", flush=True)
    print(f"  train={n_train:,}  val={nval:,}  ({time.perf_counter() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB on GPU)", flush=True)

    lw = [float(v) for v in loss_weights.split(",") if v.strip()]
    if lw and len(lw) != 3:
        raise SystemExit("--loss-weights takes exactly 'psi,qb,cons'")
    l_psi, l_qb, l_cons = lw if lw else (W_PSI, W_QB, W_CONS)

    grad_gram = None
    if grad_weight:
        from nm1.gradient import OUT as GRAD_GRAM
        if not GRAD_GRAM.exists():
            raise SystemExit(f"{GRAD_GRAM} is missing; run `python -m nm1.gradient` first")
        g = np.load(GRAD_GRAM)["gram"].astype(np.float64)
        full = np.zeros((n_comp, n_comp))
        k = min(len(g), n_comp)
        full[:k, :k] = g[:k, :k]
        grad_gram = torch.from_numpy(full).float().to(device)
        print(f"  gradient term: weight {grad_weight:g} on a {k}-mode 1/R field-energy form",
              flush=True)
    sw = ([float(v) for v in scalar_weights.split(",") if v.strip()]
          if scalar_weights else None)
    w_psi_np, w_cons_np, gram_np = build_weights(weights, coeff_std, jacobian, n_comp, sw)
    # epoch selection always reads the full seven-scalar surrogate, whatever the loss carries
    w_cons_eval = torch.from_numpy(w_cons_np).float().to(device)
    gram_eval = torch.from_numpy(gram_np).float().to(device) if gram_np is not None else None
    if axis is not None and axis_exclude:
        # R_axis/Z_axis are carried exactly by the axis loss, so their share of the averaged
        # Jacobian surrogate is removed rather than counted twice.
        z = np.load(jacobian, allow_pickle=True)
        gj = z["gram_per_scalar"].astype(np.float64)
        kk = min(gj.shape[1], n_comp)
        if weights == "gram":
            g5 = gj[2:].sum(0)
            ev, V = np.linalg.eigh(g5)
            g5 = (V * np.clip(ev, 0.0, None)) @ V.T
            gram_np = np.zeros((n_comp, n_comp))
            gram_np[:kk, :kk] = g5[:kk, :kk] / 7.0
            w_cons_np = np.diag(gram_np).copy()
        else:
            contrib = z["contrib"].astype(np.float64)
            w_cons_np = np.zeros(n_comp)
            w_cons_np[:kk] = contrib[2:, :kk].sum(0) / 7.0
    if sw:
        from nm1.jacobian import CONS_SCALARS
        print("  scalar weights: "
              + ", ".join(f"{n}={v:g}" for n, v in zip(CONS_SCALARS, sw)), flush=True)
    w_psi = torch.from_numpy(w_psi_np).float().to(device)
    w_cons = torch.from_numpy(w_cons_np).float().to(device)
    gram = torch.from_numpy(gram_np).float().to(device) if gram_np is not None else None
    print(f"weighting '{weights}': psi-term sum={w_psi_np.sum():.4f}, "
          f"cons-term sum={w_cons_np.sum():.4f}; loss weights "
          f"psi={l_psi} qb={l_qb} cons={l_cons}", flush=True)

    model = MetricRegressor(xtr.shape[1], n_comp, width, depth, dropout,
                            scalar_scale=scalar_scale).to(device)
    if scalar_scale:
        # q95 ~ B_t / Ip hard-wired in the head (model.py): fill its constants
        bc_ = np.maximum(np.abs(cur_bi[:, 0].astype(np.float64)), 5.0)
        ip_ = np.maximum(np.abs(cur_bi[:, 1].astype(np.float64)), 50.0)
        y_ = scalars_tr / np.stack([bc_ / ip_, (1.0e4 / (bc_ * ip_)) if scalar_scale >= 2
                                    else np.ones_like(bc_)], axis=1)
        for name, val in (("sc_mu", fmean[[1, 20]]), ("sc_sigma", fstd[[1, 20]]),
                          ("y_mu", np.nanmean(y_, axis=0)), ("y_sigma", np.nanstd(y_, axis=0)),
                          ("t_mu", smean), ("t_sigma", sstd)):
            getattr(model, name).copy_(torch.as_tensor(np.asarray(val, dtype=np.float32)))
        print(f"  q95 head scaled by B_t/Ip: y mean {np.nanmean(y_, axis=0)}, "
              f"std {np.nanstd(y_, axis=0)}", flush=True)
    if init_payload is not None:
        model.load_state_dict(init_payload["state_dict"])

    emul = None
    if emul_weight or lcfs_weight:
        from nm1.emulator import OUT as EMUL, load as load_emulator
        emul, emul_payload = load_emulator(emulator or EMUL, device)
        emul_h = float(emul_payload["h_scale"])
        # when the exact axis loss is on, the emulator carries the other five scalars only
        emul_cols = slice(2, 7) if axis is not None else slice(0, 7)
        print(f"  emulator: Consistency x{emul_weight:g} per frame (Gram x{1 - emul_weight:g}), "
              f"LCFS x{lcfs_weight:g}; held-out fidelity corr "
              + " ".join(f"{k}={v:.2f}" for k, v in emul_payload["fidelity_corr"].items()),
              flush=True)
        if emul_weight:
            select = "S_em"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total = math.ceil(n_train / batch_size) * epochs

    sv = svaz.clone()
    sv[~torch.isfinite(sv)] = 0.0

    def cons_term(dz: torch.Tensor) -> torch.Tensor:
        if gram is None:
            return (w_cons * dz.pow(2)).sum(1).mean()
        return ((dz @ gram) * dz).sum(1).mean()

    def cons_term_eval(dz: torch.Tensor) -> torch.Tensor:
        if gram_eval is None:
            return (w_cons_eval * dz.pow(2)).sum(1).mean()
        return ((dz @ gram_eval) * dz).sum(1).mean()

    def grad_term(dz: torch.Tensor) -> torch.Tensor:
        """The 1/R field energy of the flux error -- what `li` integrates, left free until now."""
        return ((dz @ grad_gram) * dz).sum(1).mean()

    def gs_residual(zw: torch.Tensor, idx16, psin16, ynorm, valid, reduce: bool = True):
        """Relative Grad-Shafranov residual of the predicted flux (whitened coefficients `zw`):
        Delta* psi at each frame's points, minus its least-squares fit by R^2{1,x,x^2} and
        {1,x,x^2} (x = psi_N), over the true Delta* psi there. float32, no autocast."""
        ib = idx16.long()
        y = torch.gather(zw @ gs["DSw"], 1, ib) + gs["DSm"][ib]
        x = psin16.double()
        r2 = gs["R2"][ib].double()
        X = torch.stack([r2, r2 * x, r2 * x * x, torch.ones_like(x), x, x * x], dim=2)
        # only the core (psi_N < gs_psi_max) is asked to follow the quadratic profiles: an
        # H-mode edge carries a bootstrap-current peak they cannot represent
        w = (x < gs_psi_max).double()
        valid = valid & (w.sum(1) >= 12)
        Xw = X * w.unsqueeze(2)
        # least squares through the 6x6 normal equations, in float64 (a batched QR of 8192
        # 128x6 matrices costs five times the rest of the step on this GPU)
        G = Xw.transpose(1, 2) @ Xw + 1e-9 * torch.eye(6, dtype=X.dtype, device=X.device)
        beta = torch.linalg.solve(G, Xw.transpose(1, 2) @ (y.double() * w).unsqueeze(2))
        res = ((y.double() - (X @ beta).squeeze(2)) * w).float()
        rel = (res * res).sum(1) / ynorm.clamp_min(1e-12)
        if not reduce:
            return rel, valid
        return rel[valid].mean() if bool(valid.any()) else rel.sum() * 0.0

    def evaluate(net) -> dict:
        net.eval()
        with torch.inference_mode():
            zp, sp = [], []
            for i in range(0, len(xva), 16384):
                a, b = net(xva[i:i + 16384])
                zp.append(a.float())
                sp.append(b.float())
            zp, sp = torch.cat(zp), torch.cat(sp)
            dz = zp - zva
            rel = dz.std(0)
            r2_psi = 1.0 - float((w_psi * dz.pow(2)).sum(1).mean())
            cons_hat = 1.0 - float(cons_term_eval(dz))
            fin = torch.isfinite(svaz)
            ss_res = ((sp - sv).pow(2) * fin).sum(0)
            ss_tot = (((svaz - svaz.nanmean(0)).pow(2)) * fin).nan_to_num().sum(0)
            r2_qb = float((1 - ss_res / ss_tot.clamp_min(1e-9)).mean())
            out = {"rel64": float(rel[:64].mean()), "rel32": float(rel[:32].mean()),
                   "rel_all": float(rel.mean()), "r2_psi": r2_psi, "cons_hat": cons_hat,
                   "r2_qb": r2_qb,
                   "S_hat": W_PSI * r2_psi + W_QB * r2_qb + W_CONS * cons_hat + 0.10 * 0.97}
            if emul is not None:
                acc_c = acc_h = 0.0
                for i in range(0, len(dz), 32768):
                    dsn, hn = emul(zva[i:i + 32768], dz[i:i + 32768])
                    acc_c += float((dsn.pow(2).sum(1) / 7.0).sum())
                    acc_h += float((hn * emul_h).sum())
                out["cons_em"] = 1 - acc_c / len(dz)
                out["lcfs_em"] = acc_h / len(dz)
                out["S_em"] = (W_PSI * r2_psi + W_QB * r2_qb + W_CONS * out["cons_em"]
                               + 0.10 * (1 - out["lcfs_em"]))
            if gs is not None:
                tot = cnt = 0.0
                for i in range(0, len(zp), 8192):
                    rel_, ok_ = gs_residual(zp[i:i + 8192], gs["va_idx"][i:i + 8192],
                                            gs["va_psin"][i:i + 8192], gs["va_ynorm"][i:i + 8192],
                                            gs["va_valid"][i:i + 8192], reduce=False)
                    tot += float(rel_[ok_].sum())
                    cnt += float(ok_.sum())
                out["gs"] = tot / max(cnt, 1.0)
        net.train()
        return out

    ema_model = None
    if ema:
        import copy
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    best, best_epoch, best_tag, step = -1e9, 0, "raw", 0
    output.parent.mkdir(parents=True, exist_ok=True)
    if init_payload is not None:
        m0 = evaluate(model)
        print(f"epoch 000 (initial): r2psi={m0['r2_psi']:.6f} cons^={m0['cons_hat']:.5f} "
              f"r2qb={m0['r2_qb']:.5f} S^={m0['S_hat']:.5f} "
              + (f"cons_em={m0['cons_em']:.5f} lcfs_em={m0['lcfs_em']:.5f} S_em={m0['S_em']:.5f}"
                 if "S_em" in m0 else ""), flush=True)
    print(f"training {epochs} epochs x {math.ceil(n_train / batch_size)} steps", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        run = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, zb, sb = (xtr[idx.cpu()].to(device) if host_input else xtr[idx]), ztr[idx], strz[idx]
            lam_b = None
            if dil is not None:
                from nm1.dilation import sample_lambda
                lam_b = sample_lambda(len(idx), dilate_lo, dilate_hi, dilate_p, device)
                xb, zb, sb = dil.apply(xb, zb, sb, lam_b, smean_t, sstd_t)
            if mirror_dropout and fill_t is not None and bool(fill_t.any()):
                # real frames sometimes lose the columns a mirror image cannot have, so that
                # "those columns are missing" does not by itself say "this is an image"
                drop = torch.rand(len(idx), device=device) < mirror_dropout
                xb = torch.where(drop[:, None] & fill_t[None, :], fillv_t[None, :].to(xb.dtype), xb)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                zp, sp = model(xb)
                dz = (zp - zb).float()
                fin = torch.isfinite(sb)
                if fw_t is None:
                    if lam_b is not None and dilate_cons_scale:
                        # the scored shape functionals do not change with the scale of the flux, so an
                        # absolute coefficient error on a dilated frame is a relative error 1/lambda larger
                        per_c = ((dz @ gram) * dz).sum(1) if gram is not None else (w_cons * dz.pow(2)).sum(1)
                        cons_l = (per_c / lam_b.pow(2)).mean()
                    else:
                        cons_l = cons_term(dz)
                    loss = (l_psi * (w_psi * dz.pow(2)).sum(1).mean()
                            + l_cons * (1.0 - emul_weight) * cons_l
                            + l_qb * (F.mse_loss(sp.float()[fin], sb[fin]) if fin.any()
                                      else dz.sum() * 0))
                else:
                    # frame-weighted means of the same three terms
                    fb = fw_t[idx]
                    per = (l_psi * (w_psi * dz.pow(2)).sum(1)
                           + l_cons * (1.0 - emul_weight)
                           * ((w_cons * dz.pow(2)).sum(1) if gram is None
                              else ((dz @ gram) * dz).sum(1)))
                    se = torch.where(fin, (sp.float() - torch.nan_to_num(sb)).pow(2),
                                     torch.zeros_like(sb))
                    qb = (se * fb[:, None]).sum() / (fin.float() * fb[:, None]).sum().clamp_min(1)
                    loss = (per * fb).sum() / fb.sum() + l_qb * qb
                if grad_gram is not None:
                    loss = loss + grad_weight * grad_term(dz)
            if axis is not None:
                # float32 and outside autocast: the refinement divides small flux differences
                with torch.autocast(device_type=device.type, enabled=False):
                    pr, pz = axis["head"].predict(zp.float(), axis["iz"][idx], axis["ir"][idx])
                    la = ((pr - axis["R"][idx]).pow(2) / axis["var_R"]
                          + (pz - axis["Z"][idx]).pow(2) / axis["var_Z"]).mean()
                loss = loss + axis_weight * (l_cons / 7.0) * la
            if gs is not None:
                ramp = min(1.0, epoch / gs_ramp) if gs_ramp else 1.0
                with torch.autocast(device_type=device.type, enabled=False):
                    lg = gs_residual(zp.float(), gs["idx"][idx], gs["psin"][idx],
                                     gs["ynorm"][idx], gs["valid"][idx])
                loss = loss + gs_weight * ramp * lg
            if emul is not None:
                with torch.autocast(device_type=device.type, enabled=False):
                    dsn, hn = emul(zb, dz)
                    if emul_weight:
                        loss = loss + l_cons * emul_weight * (
                            dsn[:, emul_cols].pow(2).sum(1) / 7.0).mean()
                    if lcfs_weight:
                        loss = loss + lcfs_weight * 0.10 * (hn * emul_h).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            now = lr * min(1.0, step / max(1, warmup)) * (
                0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for pg in opt.param_groups:
                pg["lr"] = now
            if ema_model is not None:
                with torch.no_grad():
                    d = min(ema, (1 + step) / (10 + step))    # fast early, decaying to `ema`
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema_model.buffers(), model.buffers()):
                        be.copy_(bm)
            run += float(loss.detach()) * len(idx)
        m = evaluate(model)
        tag, chosen = "raw", model
        if ema_model is not None:
            me = evaluate(ema_model)
            if (save_source == "ema") if save_epoch else (me[select] > m[select]):
                m, tag, chosen = me, "ema", ema_model
        print(f"epoch {epoch:03d}: loss={run / n_train:.6g}  rel64={m['rel64']:.4f} "
              f"r2psi={m['r2_psi']:.6f} cons^={m['cons_hat']:.5f} r2qb={m['r2_qb']:.5f}  "
              f"S^={m['S_hat']:.5f} "
              + (f"cons_em={m['cons_em']:.5f} lcfs_em={m['lcfs_em']:.5f} S_em={m['S_em']:.5f} "
                 if "S_em" in m else "")
              + (f"gs={m['gs']:.2e} " if "gs" in m else "")
              + f"[{tag}] {time.perf_counter() - t0:.1f}s", flush=True)
        # with a given epoch the checkpoint is that epoch's, whatever the (in-sample, for
        # --all-data) monitor says, and training stops there: the schedule is still the full one
        keep = (epoch == save_epoch) if save_epoch else (m[select] > best)
        if keep:
            best, best_epoch, best_tag = m[select], epoch, tag
            torch.save({"state_dict": chosen.state_dict(),
                        "model": {"n_features": int(xtr.shape[1]), "n_components": n_comp,
                                  "width": width, "depth": depth, "dropout": dropout,
                                  **({"scalar_scale": int(scalar_scale)} if scalar_scale else {})},
                        "feature_mean": fmean, "feature_std": fstd,
                        "pca_mean": state["pca_mean"], "pca_components": state["pca_components"],
                        "coeff_std": coeff_std, "scalar_mean": smean, "scalar_std": sstd,
                        "val": m, "weights": weights, "epoch": epoch, "seed": seed,
                        "loss_weights": [l_psi, l_qb, l_cons], "source": tag,
                        "temporal_offsets": offsets, "n_base_features": n_features,
                        "scalar_weights": sw, "extra_train": bool(extra_train),
                        "grad_weight": float(grad_weight),
                        "axis_weight": float(axis_weight), "axis_exclude": bool(axis_exclude),
                        "init": str(init) if init else None, "emul_weight": float(emul_weight),
                        "lcfs_weight": float(lcfs_weight), "ramp_weight": float(ramp_weight),
                        "folds": int(folds), "fold": int(fold), "all_data": bool(all_data),
                        "save_epoch": int(save_epoch), "gs_weight": float(gs_weight),
                        "gs_psi_max": float(gs_psi_max),
                        **vac_stats, **ext_stats}, output)
        if save_epoch and epoch >= save_epoch:
            break
    if save_epoch and best_epoch != save_epoch:
        raise SystemExit(f"epoch {save_epoch} was never reached ({epochs} epochs)")
    print(f"best {select}={best:.5f} at epoch {best_epoch} [{best_tag}] -> {output}", flush=True)
    output.with_suffix(".json").write_text(json.dumps(
        {"best": best, "epoch": best_epoch, "source": best_tag,
         "selected_on": (f"given epoch {save_epoch} [{save_source}]" if save_epoch else select),
         "all_data": bool(all_data),
         "weights": weights, "seed": seed, "temporal_context": temporal_context,
         "vac_features": vac_features, "ema": ema, "loss_weights": loss_weights,
         "width": width, "depth": depth, "epochs": epochs,
         "n_base_features": n_features, "n_ext": ext_stats["n_ext"],
         "axis_weight": axis_weight, "axis_exclude": axis_exclude}, indent=2), encoding="utf-8")
    # Members are trained back to back in one process and their cached blocks differ in width by
    # 5x, so the allocator is emptied between them rather than left to fragment.
    del xtr, xva, ztr, zva, strz, svaz, sv, model, opt, ema_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ARTIFACTS / "members" / "member.pt")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--basis", type=Path, default=BASIS)
    ap.add_argument("--jacobian", type=Path, default=JACOBIAN)
    ap.add_argument("--weights", default="gram", choices=("gram", "jacobian", "variance"))
    ap.add_argument("--grad-weight", type=float, default=0.0,
                    help="weight on the 1/R flux-gradient (field-energy) term")
    ap.add_argument("--extra-train", action="store_true",
                    help="train on the calibration third too, select epochs on `selection`")
    ap.add_argument("--scalar-weights", default="",
                    help="seven comma-separated multipliers for the scored scalars, in "
                         "CONS_SCALARS order; needs --weights jacobian")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val-frames", type=int, default=60_000)
    ap.add_argument("--vac-features", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--temporal-context", default="")
    ap.add_argument("--loss-weights", default="")
    ap.add_argument("--select", default="S_hat", choices=("S_hat", "r2_qb", "r2_psi"))
    ap.add_argument("--n-features", type=int, default=140,
                    help="140 = the base block; 272 also takes the radial profile-shape block")
    ap.add_argument("--ext", action="store_true",
                    help="append the features2 block (coil windows, positioned Thomson)")
    ap.add_argument("--axis-weight", type=float, default=0.0,
                    help="exact per-frame magnetic-axis loss, in units of its metric weight")
    ap.add_argument("--axis-exclude", action="store_true",
                    help="remove R_axis/Z_axis from the Gram term when the exact loss is on")
    ap.add_argument("--init", type=Path, help="continue this trained member")
    ap.add_argument("--emul-weight", type=float, default=0.0,
                    help="share of the Consistency term taken by the scorer emulator")
    ap.add_argument("--lcfs-weight", type=float, default=0.0,
                    help="emulator LCFS Hausdorff, in units of D_LCFS's metric weight")
    ap.add_argument("--emulator", type=Path)
    ap.add_argument("--ctx", action="store_true",
                    help="with --ext: append the shot-context block (features2.context_block)")
    ap.add_argument("--cds", action="store_true",
                    help="append the current-diffusion simulation block (diffusion.py)")
    ap.add_argument("--cds-columns", type=int, default=0,
                    help="leading columns of the diffusion block to use (0 = all; 34 = v1)")
    ap.add_argument("--ramp-weight", type=float, default=1.0,
                    help="loss weight of frames below 90%% of their shot's peak Ip")
    ap.add_argument("--cdsg", action="store_true",
                    help="append the flux-coordinate current-diffusion block (cds_geo.py)")
    ap.add_argument("--cdsg-columns", type=int, default=0)
    ap.add_argument("--folds", type=int, default=0,
                    help="out-of-fold first stage: number of folds over training shots")
    ap.add_argument("--fold", type=int, default=0, help="the fold this member leaves out")
    ap.add_argument("--all-data", action="store_true",
                    help="train on every held-out shot too; needs --save-epoch")
    ap.add_argument("--save-epoch", type=int, default=0,
                    help="keep this epoch's weights and stop there (0 = best on the monitor)")
    ap.add_argument("--save-source", default="raw", choices=("raw", "ema"),
                    help="with --save-epoch: the raw or the EMA weights")
    ap.add_argument("--gs-weight", type=float, default=0.0,
                    help="physics-informed term: Grad-Shafranov residual weight (nm1/pinn.py)")
    ap.add_argument("--gs-ramp", type=int, default=10,
                    help="epochs over which the Grad-Shafranov weight ramps up from 0")
    ap.add_argument("--gs-psi-max", type=float, default=0.9,
                    help="the Grad-Shafranov term only covers psi_N below this (the core)")
    ap.add_argument("--mirror", action="store_true",
                    help="augment with the up-down mirror image of every training frame "
                         "(base-block members; nm1/mirror.py)")
    ap.add_argument("--no-mirror-vac", action="store_true",
                    help="with --mirror: skip the exact vacuum correction of the mirrored flux")
    ap.add_argument("--mirror-dropout", type=float, default=0.0,
                    help="with --mirror: probability that a training row loses the columns a "
                         "mirror image cannot have (the vertical Thomson chord block)")
    ap.add_argument("--dilate", default="",
                    help="LO,HI: exact dilation-symmetry augmentation (currents x lambda, flux x lambda, "
                         "pressure x lambda^2) with lambda log-uniform in [LO, HI] (base-block members; "
                         "nm1/dilation.py)")
    ap.add_argument("--dilate-p", type=float, default=0.5,
                    help="with --dilate: fraction of the rows that are dilated")
    ap.add_argument("--no-dilate-cons-scale", action="store_true",
                    help="with --dilate: do not divide a dilated row's Consistency term by lambda^2")
    ap.add_argument("--scalar-scale", action="store_true",
                    help="build the q95 head on the exact scaling q95 ~ B_t / Ip (model.MetricRegressor)")
    ap.add_argument("--scalar-scale-bn", action="store_true",
                    help="with --scalar-scale: betaN through 1 / (B_t Ip) as well")
    ap.add_argument("--seg", action="store_true",
                    help="append the run-position block (features2.seg_block): where a frame sits "
                         "between the removed limited stretches of `efit_times`")
    a = ap.parse_args()
    if a.save_source == "ema" and not a.ema:
        raise SystemExit("--save-source ema needs --ema")
    if a.save_epoch > a.epochs:
        raise SystemExit("--save-epoch is past --epochs")
    if a.ctx and not a.ext and not a.init:
        raise SystemExit("--ctx is computed from the features2 block; pass --ext too")
    train(a.output, a.cache, a.basis, a.jacobian, a.weights, a.epochs, a.batch_size, a.width,
          a.depth, a.dropout, a.lr, a.weight_decay, a.warmup, a.seed, a.val_frames,
          a.vac_features, a.ema, a.temporal_context, a.loss_weights, a.select, a.n_features,
          a.scalar_weights, a.extra_train, a.grad_weight, a.ext, a.axis_weight, a.axis_exclude,
          a.init, a.emul_weight, a.lcfs_weight, a.emulator, a.ctx, a.cds, a.cds_columns,
          a.ramp_weight, a.cdsg, a.cdsg_columns, a.folds, a.fold, a.all_data, a.save_epoch,
          a.save_source, a.gs_weight, a.gs_ramp, a.gs_psi_max, a.seg, a.mirror,
          not a.no_mirror_vac, a.mirror_dropout,
          *([float(v) for v in a.dilate.split(",")] if a.dilate else [0.0, 0.0]), a.dilate_p,
          not a.no_dilate_cons_scale, (2 if a.scalar_scale_bn else 1) if a.scalar_scale else 0)


if __name__ == "__main__":
    main()
