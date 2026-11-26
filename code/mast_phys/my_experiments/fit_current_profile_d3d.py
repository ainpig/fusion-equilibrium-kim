"""Fit the solver's current-profile shape on DIII-D labels alone, for zero-shot use on MAST.

Only alpha, beta and gamma are free, and they are fitted here on DIII-D, never on MAST: no MAST
example -- including the three labelled demo shots of the starter kit -- is read or used to select
them (Challenge 2 rule 3: no MAST equilibrium data may train, calibrate, validate or select a
component that produces MAST predictions). The fit runs on this script's default
150 DIII-D shots and gives alpha 0.5, beta 0.75, gamma 0.5.

The free-space Green basis cannot represent DIII-D's coil response (DIII-D has an iron transformer
core, MAST is air-core), so the solver is not checked on DIII-D through the vacuum field.

This route does not need the vacuum field at all. The Grad-Shafranov operator recovers the
toroidal current density from the flux map by differentiation alone:

    Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 = -mu0 R J_phi

so DIII-D's own stored psi gives J_phi directly, with no assumption about how its coils produced
it. Fitting

    J_phi(R, psi_N) = lambda (1 - psi_N^alpha)^beta [ (1-gamma) R/R0 + gamma R0/R ]

on that is a statement about plasma physics -- how current is distributed between the axis and the
boundary -- which is the part that should transfer between machines. lambda is per frame (it just
carries Ip), so only the three shape parameters are fitted, on DIII-D, and they are what crosses
to MAST.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from common import AXIS_SIGN  # noqa: E402
from my_experiments.fusion_pipeline import CACHE_ROOT, stack_psirz  # noqa: E402

MU0 = 4.0e-7 * np.pi


def gs_current(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """J_phi from Delta* psi, by second differences on the native grid. psi is (T, nZ, nR)."""
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    d2R = np.zeros_like(psi)
    d2Z = np.zeros_like(psi)
    d1R = np.zeros_like(psi)
    d2R[:, :, 1:-1] = (psi[:, :, 2:] - 2.0 * psi[:, :, 1:-1] + psi[:, :, :-2]) / (dR * dR)
    d2Z[:, 1:-1, :] = (psi[:, 2:, :] - 2.0 * psi[:, 1:-1, :] + psi[:, :-2, :]) / (dZ * dZ)
    d1R[:, :, 1:-1] = (psi[:, :, 2:] - psi[:, :, :-2]) / (2.0 * dR)
    star = d2R - d1R / R[None, None, :] + d2Z
    return -star / (MU0 * R[None, None, :])


def frame_psi_n(phi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                r_axis: float, z_axis: float,
                lcfs_r: np.ndarray, lcfs_z: np.ndarray):
    """psi_N in [0,1] from the released axis and LCFS, plus the in-plasma mask.

    Both come from the row's own labels rather than from a contour search, so the fit is not
    contaminated by whatever the extractor would have done with an imperfect map.
    """
    from matplotlib.path import Path as MplPath
    ia = int(np.argmin(np.abs(Z - z_axis)))
    ja = int(np.argmin(np.abs(R - r_axis)))
    phi_axis = float(phi[ia, ja])
    # boundary flux: phi sampled on the released LCFS polygon
    iz = np.clip(np.searchsorted(Z, lcfs_z) - 1, 0, len(Z) - 2)
    ir = np.clip(np.searchsorted(R, lcfs_r) - 1, 0, len(R) - 2)
    phi_b = float(np.median(phi[iz, ir]))
    span = phi_axis - phi_b
    if not np.isfinite(span) or abs(span) < 1e-9:
        return None, None
    psi_n = (phi_axis - phi) / span
    RR, ZZ = np.meshgrid(R, Z)
    poly = MplPath(np.column_stack([lcfs_r, lcfs_z]))
    inside = poly.contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(phi.shape)
    return psi_n, inside


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=CACHE_ROOT / "data" / "diii_d_train")
    ap.add_argument("--n-shots", type=int, default=150)
    ap.add_argument("--stride", type=int, default=7, help="keep every Nth frame of a shot")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "d3d_current_profile.json")
    args = ap.parse_args()

    files = sorted(args.src.glob("*.parquet"))
    rng = np.random.default_rng(args.seed)
    pick = [files[i] for i in np.sort(rng.choice(len(files), min(args.n_shots, len(files)),
                                                 replace=False))]
    cols = ["efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_r_axis", "efit_z_axis",
            "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z"]

    S_psi_n, S_R, S_J = [], [], []
    kept_frames = kept_shots = 0
    for n, p in enumerate(pick, 1):
        row = pd.read_parquet(p, columns=cols).iloc[0]
        psi = stack_psirz(row["efit_psirz"]).astype(np.float64)
        R = np.asarray(row["efit_grid_R"], dtype=np.float64)
        Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
        ra = np.asarray(row["efit_r_axis"], dtype=np.float64)
        za = np.asarray(row["efit_z_axis"], dtype=np.float64)
        cnt = np.asarray(row["efit_lcfs_n"], dtype=np.int32)
        J = gs_current(psi, R, Z)
        phi = AXIS_SIGN["DIII-D"] * psi          # axis becomes a maximum
        used = 0
        for k in range(0, len(psi), args.stride):
            if k >= len(ra) or k >= len(cnt):
                break
            m = int(cnt[k])
            if m < 12 or not (np.isfinite(ra[k]) and np.isfinite(za[k])):
                continue
            lr = np.asarray(row["efit_lcfs_r"][k], dtype=np.float64)[:m]
            lz = np.asarray(row["efit_lcfs_z"][k], dtype=np.float64)[:m]
            good = np.isfinite(lr) & np.isfinite(lz)
            if good.sum() < 12 or not np.isfinite(psi[k]).all():
                continue
            psi_n, inside = frame_psi_n(phi[k], R, Z, ra[k], za[k], lr[good], lz[good])
            if psi_n is None:
                continue
            # stay off the boundary layer: psi_N near 1 is where the finite differences and the
            # released contour disagree most, and it carries almost no current anyway
            sel = inside & (psi_n > 0.0) & (psi_n < 0.9)
            if sel.sum() < 100:
                continue
            RR = np.broadcast_to(R[None, :], psi_n.shape)
            S_psi_n.append(psi_n[sel]); S_R.append(RR[sel]); S_J.append(J[k][sel])
            used += 1
        kept_frames += used
        kept_shots += int(used > 0)
        if n % 25 == 0:
            print(f"  {n}/{len(pick)} shots, {kept_frames:,} frames kept", flush=True)

    x = np.concatenate(S_psi_n); r = np.concatenate(S_R); j = np.concatenate(S_J)
    # orient so the current is positive; the solver normalises the magnitude to Ip anyway
    if np.median(j) < 0:
        j = -j
    r0 = float(np.median(r))
    print(f"\nfitted on {kept_shots} DIII-D shots, {kept_frames:,} frames, {len(j):,} grid points")
    print(f"R0 = {r0:.3f} m,  J median = {np.median(j):.3e} A/m^2")

    best = None
    for alpha, beta, gamma in itertools.product(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5, 3.0, 4.0],
            [0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]):
        shape = ((1.0 - x ** alpha).clip(0.0) ** beta) * ((1.0 - gamma) * r / r0 + gamma * r0 / r)
        denom = float(shape @ shape)
        if denom <= 0:
            continue
        lam = float(shape @ j) / denom              # one global lambda; scale is free anyway
        resid = j - lam * shape
        r2 = 1.0 - float(resid @ resid) / float(((j - j.mean()) ** 2).sum())
        if best is None or r2 > best["r2"]:
            best = {"alpha": alpha, "beta": beta, "gamma": gamma, "lam": lam, "r2": round(r2, 5)}
    print(f"\nbest profile on DIII-D: alpha={best['alpha']} beta={best['beta']} "
          f"gamma={best['gamma']}   R2 of J_phi = {best['r2']}")

    out = {"alpha": best["alpha"], "beta": best["beta"], "gamma": best["gamma"],
           "fit_r2_J": best["r2"], "r0_d3d": r0,
           "n_shots": kept_shots, "n_frames": kept_frames, "n_points": int(len(j)),
           "fitted_on": "DIII-D efit_psirz via the Grad-Shafranov operator; no MAST data"}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
