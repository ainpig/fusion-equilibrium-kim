"""Physics-informed (PINN) training term: the Grad-Shafranov residual of the predicted flux.

Inside the plasma an equilibrium obeys

    Delta* psi  =  R^2 A(psi_N) + B(psi_N)          (A = -mu0 p',  B = -F F')

with A and B functions of the normalised flux alone. The predicted flux is linear in the
predicted coefficients, and so is `Delta* psi` (`delta_star_basis`). For every cached frame this
module stores P points inside the true plasma (inside the LCFS, psi_N < 0.9) with their psi_N
(`build`). During training, `Delta*` of the prediction at those points is fitted by the six
functions R^2 {1, psi_N, psi_N^2} and {1, psi_N, psi_N^2}; what the fit leaves over, relative to
the true `Delta* psi` there, is the physics residual that `train.py --gs-weight` penalises. The
profiles are refitted per frame, so the term asks for *a* Grad-Shafranov equilibrium, not for
the true current density: it is a physics prior, not a second data term.

Measured before building it (held-out, `selection`): the EFIT flux leaves a relative residual of
1.9e-06 (median), its 256-mode PCA reconstruction 1.6e-05, the v8 ensemble 2.7e-05.

    python -m nm1.pinn                       # both splits -> data/d3d_gsfc.h5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from nm1 import CACHE_DATA

GSFC_CACHE = CACHE_DATA / "d3d_gsfc.h5"
P = 128                    # points per frame
PSI_MAX = 0.9              # stay off the edge, where the boundary discretisation dominates
_W: dict = {}


def delta_star(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2`` on the interior points (Z, R axes),
    central differences on the uniform grid; zero on the border (never sampled)."""
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    out = np.zeros(psi.shape, dtype=np.float64)
    c = psi[..., 1:-1, 1:-1]
    d2r = (psi[..., 1:-1, 2:] - 2 * c + psi[..., 1:-1, :-2]) / dR ** 2
    d1r = (psi[..., 1:-1, 2:] - psi[..., 1:-1, :-2]) / (2 * dR)
    d2z = (psi[..., 2:, 1:-1] - 2 * c + psi[..., :-2, 1:-1]) / dZ ** 2
    out[..., 1:-1, 1:-1] = d2r - d1r / R[1:-1] + d2z
    return out


def delta_star_basis(mean: np.ndarray, comps: np.ndarray, R: np.ndarray,
                     Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(Delta* mean, Delta* comps)`` flattened: shapes (G,) and (n_comp, G)."""
    n = len(R)
    dm = delta_star(mean.reshape(n, n), R, Z).ravel()
    dc = delta_star(comps.reshape(-1, n, n), R, Z).reshape(len(comps), -1)
    return dm, dc


def _init(basis_path: str) -> None:
    import torch
    root = Path(__file__).resolve().parents[1] / "fusion_scoring"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from nm1.gradient import grid
    from nm1.guard import box_mask
    st = torch.load(basis_path, map_location="cpu", weights_only=False)
    _W["mean"] = st["pca_mean"].astype(np.float64)
    _W["comps"] = st["pca_components"].astype(np.float64)
    R, Z = grid()
    _W["R"], _W["Z"] = R, Z
    _W["mask"] = box_mask(R, Z)
    rr, _ = np.meshgrid(np.arange(len(R)), np.arange(len(Z)))
    zz, _ = np.meshgrid(np.arange(len(Z)), np.arange(len(R)), indexing="ij")
    _W["pts"] = np.column_stack([zz.ravel(), rr.ravel()]).astype(float)   # (row, col)


def _frame(psi: np.ndarray, seed: int):
    """(idx, psin, ynorm) for one true flux map, or None when no closed surface is found."""
    from lcfs import _best_contour_at
    from matplotlib.path import Path as MPath
    from o_point import find_o_point
    from common import AXIS_SIGN
    R, Z, mask = _W["R"], _W["Z"], _W["mask"]
    phi = AXIS_SIGN["DIII-D"] * psi
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mask)
    axis = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo, hi = float(border.min()), axis - 1e-9
    mf = mask.astype(float)
    best = _best_contour_at(phi, hi, iz, ir, mf)
    if best is None or not hi > lo:
        return None
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        c = _best_contour_at(phi, mid, iz, ir, mf)
        if c is not None:
            hi, best = mid, c
        else:
            lo = mid
    span = axis - hi
    if span <= 0:
        return None
    psin = ((axis - phi) / span).ravel()
    inside = MPath(best).contains_points(_W["pts"])
    n = len(R)
    border_pt = np.zeros((n, n), bool)
    border_pt[[0, -1], :] = True
    border_pt[:, [0, -1]] = True
    ok = np.flatnonzero(inside & (psin < PSI_MAX) & (psin >= 0) & ~border_pt.ravel())
    if len(ok) < 16:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(ok, size=P, replace=len(ok) < P)
    y = delta_star(psi, R, Z).ravel()[idx]
    return idx.astype(np.int16), psin[idx].astype(np.float16), float((y ** 2).sum())


def _job(job):
    lo, coeff = job
    psi = (_W["mean"] + coeff.astype(np.float64) @ _W["comps"]).reshape(-1, 65, 65)
    idx = np.zeros((len(psi), P), np.int16)
    psin = np.zeros((len(psi), P), np.float16)
    ynorm = np.ones(len(psi), np.float32)
    valid = np.zeros(len(psi), bool)
    for k in range(len(psi)):
        try:
            r = _frame(psi[k], lo + k)
        except Exception:                          # noqa: BLE001 - an unusable frame
            r = None
        if r is not None:
            idx[k], psin[k], ynorm[k] = r
            valid[k] = True
    return lo, idx, psin, ynorm, valid


def build(out: Path = GSFC_CACHE, workers: int = 9, chunk: int = 2000) -> Path:
    """Sample points of every cached frame, on the true flux (its PCA reconstruction)."""
    from nm1.cache import BASIS, CACHE
    started = time.perf_counter()
    with h5py.File(CACHE, "r") as h5, h5py.File(out, "w") as o:
        for split in ("val", "train"):
            coeff = h5[split]["coeff"]
            n = coeff.shape[0]
            g = o.create_group(split)
            d_idx = g.create_dataset("idx", (n, P), dtype="i2")
            d_psin = g.create_dataset("psin", (n, P), dtype="f2")
            d_y = g.create_dataset("ynorm", (n,), dtype="f4")
            d_v = g.create_dataset("valid", (n,), dtype="bool")
            jobs = ((lo, np.asarray(coeff[lo:lo + chunk], np.float32))
                    for lo in range(0, n, chunk))
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(BASIS),)) as pool:
                for done, (lo, idx, psin, ynorm, valid) in enumerate(pool.map(_job, jobs), 1):
                    hi = lo + len(idx)
                    d_idx[lo:hi], d_psin[lo:hi], d_y[lo:hi], d_v[lo:hi] = idx, psin, ynorm, valid
                    if done % 50 == 0:
                        print(f"  {split}: {hi:,}/{n:,} frames, "
                              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
            print(f"  {split}: {int(d_v[:].sum()):,} of {n:,} frames usable", flush=True)
        o.attrs["points"] = P
        o.attrs["psi_max"] = PSI_MAX
    out.with_suffix(".json").write_text(json.dumps(
        {"points": P, "psi_max": PSI_MAX, "flux": "PCA reconstruction of the stored flux",
         "profiles": "R^2{1,x,x^2} + {1,x,x^2}, x = psi_N"}, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=GSFC_CACHE)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    build(a.out, a.workers)
