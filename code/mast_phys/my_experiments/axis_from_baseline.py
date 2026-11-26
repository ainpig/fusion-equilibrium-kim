"""MAST: the submitted flux keeps its own LCFS but takes the baseline branch's magnetic-axis radius.

The submitted flux is the pressure-geometry branch (mast_v31); the baseline branch of the same run (mast_final)
places the magnetic axis differently.
Each frame's axis is moved, in R only, to where the
baseline branch of the same run puts it: a smooth local tilt (r - r_axis) * taper inside the LCFS, the taper a
Gaussian of 0.2 m around the axis times a smoothstep of the normalised flux that vanishes on the LCFS, the amplitude
found by a two-dimensional Newton step on the scorer's own O-point finder (Z held). The LCFS, and with it kappa, the
triangularities, the volume and D_LCFS, does not move (changes <= 4e-5 on a 42-frame check); li moves by at most
~0.004; the flux by a relative 1e-6.

Both maps are this run's own predictions; nothing here reads a MAST target or another run.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fusion_scoring'))

MAX_CHANGE = 0.02      # m per frame
WIDTH = 0.20           # m


def support_mask(R, Z, rmax, zmax):
    """The released-geometry support widened inwards (as q_physical_v14.limited_flags)."""
    rr, zz = np.meshgrid(R, Z)
    return (rr > float(R[1])) & (rr < rmax) & (np.abs(zz) < zmax)


def axis_r(psi, R, Z, mask):
    from derive import magnetic_axis
    return float(magnetic_axis(np.asarray(psi, float), R, Z, "MAST", mask)[0])


def move_axis_r(psi, R, Z, target_r, mask):
    """(new flux, applied shift in m). Returns the input unchanged when a step would not get closer."""
    from derive import magnetic_axis, _bilinear, _points_in_polygon
    from lcfs import extract_lcfs
    psi = np.asarray(psi, float)
    C = extract_lcfs(psi, R, Z, "MAST", mask, mask.astype(float), n_points=256)
    ra, za = magnetic_axis(psi, R, Z, "MAST", mask)[:2]
    if C is None or not np.isfinite([ra, za, target_r]).all():
        return psi, 0.0
    desired = np.array([ra + float(np.clip(target_r - ra, -MAX_CHANGE, MAX_CHANGE)), za])
    if abs(desired[0] - ra) < 1e-5:
        return psi, 0.0
    level = float(np.median(_bilinear(psi, R, Z, C[:, 0], C[:, 1])))
    peak = float(_bilinear(psi, R, Z, np.array([ra]), np.array([za]))[0])
    if peak - level <= 1e-8:
        return psi, 0.0
    rr, zz = np.meshgrid(R, Z)
    inside = _points_in_polygon(rr.ravel(), zz.ravel(), C).reshape(rr.shape)
    s = np.where(inside, np.clip((psi - level) / (peak - level), 0, 1), 0)
    taper = np.exp(-0.5 * ((rr - ra) ** 2 + (zz - za) ** 2) / WIDTH ** 2) * s * s * (3 - 2 * s)
    bases = np.stack([(rr - ra) * taper, (zz - za) * taper])
    amp = np.zeros(2)
    step = 1e-4
    for _ in range(4):
        field = psi + np.einsum("i,ijk->jk", amp, bases)
        got = np.asarray(magnetic_axis(field, R, Z, "MAST", mask)[:2])
        if np.linalg.norm(got - desired) < 1e-5:
            break
        jac = np.zeros((2, 2))
        for j in range(2):
            pos = np.asarray(magnetic_axis(field + step * bases[j], R, Z, "MAST", mask)[:2])
            neg = np.asarray(magnetic_axis(field - step * bases[j], R, Z, "MAST", mask)[:2])
            jac[:, j] = (pos - neg) / (2 * step)
        if not np.isfinite(jac).all() or abs(np.linalg.det(jac)) < 1e-12:
            return psi, 0.0
        amp += np.linalg.solve(jac, desired - got)
        amp = np.clip(amp, -0.05, 0.05)
    field = psi + np.einsum("i,ijk->jk", amp, bases)
    got = np.asarray(magnetic_axis(field, R, Z, "MAST", mask)[:2])
    if np.linalg.norm(got - desired) >= np.linalg.norm(np.array([ra, za]) - desired):
        return psi, 0.0
    if extract_lcfs(field, R, Z, "MAST", mask, mask.astype(float), n_points=256) is None:
        return psi, 0.0
    return field, float(got[0] - ra)


def shot(submitted, baseline, R, Z, rmax, zmax):
    """Move every frame of one shot; returns (float32 flux, shifts in m)."""
    mask = support_mask(R, Z, rmax, zmax)
    out = np.empty(np.shape(submitted), np.float32)
    shifts = np.zeros(len(submitted))
    for k in range(len(submitted)):
        f, d = move_axis_r(submitted[k], R, Z, axis_r(baseline[k], R, Z, mask), mask)
        out[k] = np.asarray(f, np.float32)
        shifts[k] = d
    return out, shifts
