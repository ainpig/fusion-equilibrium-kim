"""Boundary-locked interior bump for v6.

The correction is zero on an outer band of normalized flux and on a neighborhood
of the magnetic axis, so the LCFS contour and the axis stencil stay put. Only
internal inductance and interior pixels can move. v6 leaves it disabled until a
coefficient beats v3 on shots that did not choose it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCORING = Path(__file__).resolve().parents[1] / "fusion_scoring"
if str(_SCORING) not in sys.path:
    sys.path.insert(0, str(_SCORING))

from common import AXIS_SIGN  # noqa: E402
from derive import _bilinear, _phi  # noqa: E402
from lcfs import _best_contour_at  # noqa: E402
from o_point import find_o_point  # noqa: E402
from contour import grid_to_physical, resample_closed  # noqa: E402


def lcfs_level(phi, R, Z, n_iter=16, n_points=256):
    if not np.isfinite(phi).all():
        phi = np.nan_to_num(phi, nan=np.nanmin(phi) - 1.0)
    _, _, iz, ir, _ = find_o_point(phi, R, Z, None)
    axis_val = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo, hi = float(np.min(border)), axis_val - 1e-9
    if not hi > lo:
        return None
    best = _best_contour_at(phi, hi, iz, ir, None)
    if best is None:
        return None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        found = _best_contour_at(phi, mid, iz, ir, None)
        if found is not None:
            hi, best = mid, found
        else:
            lo = mid
    contour = resample_closed(grid_to_physical(best, R, Z), n_points)
    return float(hi), contour, int(iz), int(ir), axis_val


def prepare(psi, R, Z, machine="DIII-D", margin=0.22, freeze=2, n_iter=16):
    phi = _phi(psi, machine)
    found = lcfs_level(phi, R, Z, n_iter=n_iter)
    if found is None:
        return None
    level, contour, iz, ir, axis_val = found
    span = float(axis_val - level)
    if span <= 1e-8 or not 0.0 <= margin < 0.9:
        return None
    s = np.clip((phi - level) / span, 0.0, 1.0)
    u = np.clip((s - margin) / (1.0 - margin), 0.0, 1.0)
    basis = u * (1.0 - u)
    if freeze > 0:
        yy, xx = np.ogrid[: phi.shape[0], : phi.shape[1]]
        basis = np.array(basis, copy=True)
        basis[np.maximum(np.abs(yy - iz), np.abs(xx - ir)) <= freeze] = 0.0
    return {"phi": phi, "basis": basis, "span": span, "level": level, "axis": axis_val,
            "contour": contour, "sign": AXIS_SIGN[machine]}


def remap(prep, alpha):
    return prep["phi"] + float(alpha) * prep["basis"] * prep["span"]


def _core_chord(row, times):
    from nm1.features2 import _align_time, _chord_indices
    te = np.stack([np.asarray(p, float) for p in row["thomson_core_Te"]])
    ne = np.stack([np.asarray(p, float) for p in row["thomson_core_ne"]])
    idx = _chord_indices(row, "TS_core", slice(0, te.shape[1]))
    r = np.asarray(row["thomson_chord_R"], float)[idx]
    z = np.asarray(row["thomson_chord_Z"], float)[idx]
    c = min(len(r), len(z), te.shape[1], ne.shape[1])
    t_ts = np.asarray(row["thomson_core_times"], float)
    te, _ = _align_time(t_ts, te[:, :c], times)
    ne, _ = _align_time(t_ts, ne[:, :c], times)
    return te, ne, r[:c], z[:c]


def _peaking(values, s):
    core = (s > 0.62) & (s < 0.96)
    edge = (s > 0.08) & (s < 0.38)
    vc = values[core]
    ve = values[edge]
    vc = vc[np.isfinite(vc) & (vc > 0)]
    ve = ve[np.isfinite(ve) & (ve > 0)]
    if len(vc) < 3 or len(ve) < 3:
        return np.nan
    return float(np.log(vc.mean()) - np.log(ve.mean()))


def mismatch(prep, R, Z, te, ne, cr, cz):
    span = float(prep["axis"] - prep["level"])
    if span <= 1e-8:
        return np.nan
    dz, dr = np.gradient(prep["phi"], Z, R)
    grad = np.sqrt(dr * dr + dz * dz)
    s = (_bilinear(prep["phi"], R, Z, cr, cz) - prep["level"]) / span
    g = _bilinear(grad, R, Z, cr, cz)
    p = np.asarray(te, float) * np.asarray(ne, float)
    ok = np.isfinite(s) & np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    ok &= (s > 0.05) & (s < 0.98)
    if int(ok.sum()) < 8:
        return np.nan
    return _peaking(p[ok], s[ok]) - _peaking(g[ok], s[ok])


def apply_shot(psi, row, times, spec):
    """Return flux with one shot-level amplitude, or the input if the chord is missing."""
    if not spec.get("enabled") or float(spec.get("scale") or 0.0) == 0.0:
        return psi
    try:
        te, ne, cr, cz = _core_chord(row, times)
    except Exception:  # noqa: BLE001 - a ragged chord keeps the v3 map
        return psi
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    margin = float(spec.get("margin", 0.22))
    freeze = int(spec.get("freeze", 2))
    preps, values = [], []
    for k in range(len(psi)):
        prep = prepare(psi[k], R, Z, margin=margin, freeze=freeze)
        preps.append(prep)
        if prep is None:
            values.append(np.nan)
            continue
        values.append(mismatch(prep, R, Z, te[k], ne[k], cr, cz))
    finite = np.array([v for v in values if np.isfinite(v)], float)
    if len(finite) < 8:
        return psi
    mu, sd = float(spec["mu"]), float(spec["sd"] or 1.0)
    alpha = float(spec["scale"]) * (float(finite.mean()) - mu) / sd
    alpha = float(np.clip(alpha, -float(spec.get("clip", 0.12)), float(spec.get("clip", 0.12))))
    if alpha == 0.0:
        return psi
    out = np.array(psi, dtype=np.float64, copy=True)
    for k, prep in enumerate(preps):
        if prep is None:
            continue
        out[k] = prep["sign"] * remap(prep, alpha)
    return out
