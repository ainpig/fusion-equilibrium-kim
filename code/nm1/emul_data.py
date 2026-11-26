"""Training data for a differentiable emulator of the scorer's contour functionals.

The Consistency surrogate in ``train.py`` is one quadratic form averaged over 640 frames, and the
scorer's contour-derived scalars are anything but uniform across frames: the LCFS level is set by
a bisection that stops either at the X-point or where the surface first touches the corpus
envelope mask, and which of the two happens -- and where -- changes from frame to frame.  Instead
of approximating that by hand, this samples it: for training frames, the scorer's own functionals
are evaluated on the true flux (reconstructed in the basis) and on perturbations of it drawn from
the measured error distribution of real ensemble members.  ``emulator.py`` then learns

    (true coefficients, perturbation) -> change in the seven scalars, and the LCFS Hausdorff

and that network is what a member can differentiate through.

Only ``diii_d_train`` training-split frames are sampled; no held-out frame is touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from nm1 import ARTIFACTS, DATA, ROOT
from nm1.cache import BASIS, CACHE

sys.path.insert(0, str(ROOT / "fusion_scoring"))
OUT = DATA / "emul_data.h5"
SCALES = (0.5, 1.0, 2.0)
_G: dict = {}


def _init(pca_mean, comps):
    sys.path.insert(0, str(ROOT / "fusion_scoring"))
    m = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    _G.update(R=m["grid_R"].astype(np.float64), Z=m["grid_Z"].astype(np.float64),
              mc=m["mask_coarse"].astype(bool), mean=pca_mean.astype(np.float64),
              comps=comps.astype(np.float64))
    _G["mf"] = _G["mc"].astype(np.float64)


def _derive(coeff: np.ndarray):
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    psi = (_G["mean"] + coeff.astype(np.float64) @ _G["comps"]).reshape(65, 65)
    C, _ = extract_lcfs_with_sign(psi, _G["R"], _G["Z"], "DIII-D", _G["mc"], _G["mf"],
                                  axis_sign=-1.0)
    v = derive_frame(psi, _G["R"], _G["Z"], "DIII-D", _G["mc"], _G["mf"], contour=C,
                     with_li=True, axis_sign=-1.0)
    names = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
    return np.array([v[k] for k in names], dtype=np.float64), C


def _job(item):
    """One frame: the truth and len(deltas) perturbations of it."""
    from contour import symmetric_hausdorff
    from lcfs import major_radius
    coeff, deltas = item
    s0, C0 = _derive(coeff)
    rgeo = major_radius(C0) if C0 is not None else np.nan
    outs = [np.concatenate([s0, [0.0]])]
    for d in deltas:
        s, C = _derive(coeff + d)
        if C is None or C0 is None:
            h = 1.0
        else:
            h = min(1.0, symmetric_hausdorff(C, C0) / rgeo)
        outs.append(np.concatenate([s, [h]]))
    return np.stack(outs)


def error_covariance(members_npz: Path | None = None) -> np.ndarray:
    """Covariance of single-member coefficient errors on held-out frames (calibration half).

    Read from cache-space predictions of trained members (``cachepred``); only the calibration
    half is used, which the trainer already reads for epoch selection.
    """
    from nm1.cachepred import PRED_DIR
    from nm1.score import held_out_ids
    with h5py.File(CACHE, "r") as h5:
        true = h5["val"]["coeff"][:].astype(np.float64)
        shot = h5["val"]["shot"][:]
    cal = np.isin(shot, np.array(sorted(held_out_ids("calibration"))))
    files = sorted(PRED_DIR.glob("*.npz")) if members_npz is None else [members_npz]
    covs = []
    for f in files:
        z = np.load(f)
        keys = [k for k in z.files if k.endswith("coeff")]
        for k in keys:
            err = z[k].astype(np.float64)[cal] - true[cal]
            covs.append(np.cov(err, rowvar=False))
    if not covs:
        raise SystemExit("no member predictions to take an error covariance from")
    return np.mean(covs, 0)


def build(n_frames: int = 100_000, workers: int = 8, seed: int = 7, cov_npz: Path | None = None,
          out: Path = OUT) -> Path:
    state = torch.load(BASIS, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)
    cov = error_covariance(cov_npz)
    ev, V = np.linalg.eigh(cov)
    L = V * np.sqrt(np.clip(ev, 0, None))
    with h5py.File(CACHE, "r") as h5:
        n = h5["train"]["coeff"].shape[0]
        idx = np.sort(rng.choice(n, min(n_frames, n), replace=False))
        coeff = h5["train"]["coeff"][idx].astype(np.float64)
    deltas = np.stack([(rng.standard_normal((len(idx), len(ev))) @ L.T) * s for s in SCALES], 1)
    print(f"emulator data: {len(idx):,} frames x {1 + len(SCALES)} versions, {workers} workers; "
          f"error rms per mode (first 8): {np.sqrt(np.diag(cov))[:8].round(5)}", flush=True)
    t0 = time.perf_counter()
    res = np.full((len(idx), 1 + len(SCALES), 8), np.nan)
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(state["pca_mean"], state["pca_components"])) as pool:
        for i, r in enumerate(pool.map(_job, zip(coeff, deltas), chunksize=16)):
            res[i] = r
            if (i + 1) % 5000 == 0:
                el = time.perf_counter() - t0
                print(f"  {i + 1:,}/{len(idx):,} frames, {el / 60:.1f} min "
                      f"(~{el / (i + 1) * (len(idx) - i - 1) / 60:.0f} min left)", flush=True)
    with h5py.File(out, "w") as o:
        o["idx"] = idx
        o["delta"] = deltas.astype(np.float32)
        o["scales"] = np.asarray(SCALES)
        o["out"] = res.astype(np.float64)
        o.attrs["names"] = "R_axis,Z_axis,kappa,tri_top,tri_bot,volume,li,hausdorff_over_rgeo"
    print(f"wrote {out} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cov-npz", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.frames, a.workers, a.seed, a.cov_npz, a.out)


if __name__ == "__main__":
    main()
