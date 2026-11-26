"""v14: equal-temperature pairs at the plasma edge, where the hot plasma seen by Thomson ends.

The v15 constraints pair equal-temperature points between 25% and 85% of the peak temperature, so
the seed solve sees the core flux surfaces but not where the hot plasma ends. That leaves the
inboard side free. On MAST inputs (no label read), a part of the frames put the inboard midplane boundary on the numerical support edge (R ~ 0.17-0.18 m,
the solenoid surface plus half a cell) or inside 0.2 m, while the Thomson chord on those frames
still measures ~40 eV, a separatrix-like temperature, at its innermost channel (R = 0.235 m). The
outboard boundary of the same frames sits on the Te ~ 20-40 eV point within 1 cm, and the inboard
boundary does too on the frames where the chord sees the inboard edge.

Isothermal surfaces are flux surfaces inside the separatrix, at the edge as in the core. This module
adds equal-temperature pairs between the chord ends and 25% of the peak: `EDGE_LEVELS` levels spaced
geometrically from just above the hotter of the two chord-end temperatures (or `FLOOR_EV`, below
which the scrape-off layer is not a flux-surface quantity) up to the lowest v15 level. When the
inboard chord end is itself hot, the lowest pair puts the flux surface through the innermost channel
on its outboard partner of the same temperature, which bounds how far inside the chord the boundary
can be. The pairs enter the same isotherm residual as the core pairs, weighted `WEIGHT` each (the
residual is a weighted mean over pairs).

Constants (label-free, from DIII-D training shots and the Thomson chord geometry):
  FLOOR_EV = 30   -- DIII-D train shots: Te at the EFIT LCFS is 38 eV median, IQR 25-59 eV
                     (150 shots); a pair below ~30 eV is likely in the SOL
  SPAN     = 1.05 -- start 5% above the chord-end temperature so the inboard root is bracketed
  EDGE_LEVELS = 3, WEIGHT = 2.0 -- three edge surfaces, each counted twice a core pair
No MAST equilibrium, label or score is read here: only the shot's own Thomson input columns.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import isotonic_regression

FLOOR_EV = 30.0
SPAN = 1.05
EDGE_LEVELS = 3
WEIGHT = 2.0
TOP_FRACTION = 0.25          # the lowest v15 core level, as a fraction of the peak


def edge_pairs_frame(r, v, centre):
    """Equal-temperature (inboard R, outboard R) pairs between the chord ends and 25% of the peak."""
    v = np.asarray(v, float)
    good = np.isfinite(v) & (v > 0)
    if good.sum() < 12:
        return []
    idx = np.flatnonzero(good)
    a, b = idx[0], idx[-1]
    rr, vv = np.asarray(r, float)[a:b+1], v[a:b+1]
    # the cleaned profile is interpolated between valid channels, so the span [a, b] is positive;
    # smoothing stays inside it (mode 'nearest'), unlike a filter run over the zero-padded profile
    s = gaussian_filter1d(vv, 2, mode='nearest')
    split = int(np.argmin(np.abs(rr - centre)))
    if split < 3 or split > len(rr) - 4:
        return []
    left = isotonic_regression(s[:split+1], increasing=True).x
    right = isotonic_regression(s[split:], increasing=False).x
    top = min(left[-1], right[0])
    lo = max(left[0], right[-1], FLOOR_EV) * SPAN
    hi = TOP_FRACTION * top
    if not (hi > lo * 1.02):
        return []
    levels = np.geomspace(lo, hi, EDGE_LEVELS + 1)[:-1]
    out = []
    for value in levels:
        if value <= max(left[0], right[-1]) or value >= top:
            continue
        rin = float(np.interp(value, left, rr[:split+1]))
        rout = float(np.interp(value, right[::-1], rr[split:][::-1]))
        if rout - rin > 0.05:
            out.append((rin, rout))
    return out


def append_edge_pairs(r, te, counts, kin, constraints):
    """The v15 (pairs, valid, pressure) with `EDGE_LEVELS` weighted edge pairs appended per frame."""
    pairs, valid, pressure = constraints
    n = len(te)
    extra = np.zeros((n, EDGE_LEVELS, 2), np.float32)
    extra_valid = np.zeros((n, EDGE_LEVELS), np.float32)
    for i in range(n):
        if counts[i] < 12:
            continue
        found = edge_pairs_frame(r, te[i], float(kin[i][0]))
        for j, (rin, rout) in enumerate(found[:EDGE_LEVELS]):
            extra[i, j] = (rin, rout)
            extra_valid[i, j] = WEIGHT
    return (np.concatenate([pairs, extra], axis=1), np.concatenate([valid, extra_valid], axis=1),
            pressure)


def prepare_kinetic(row, times, exponent):
    """mast_inputs_v15.prepare_kinetic plus the edge pairs (same kinetic geometry, same pressure)."""
    from my_experiments.mast_inputs_v15 import prepare_kinetic as core, profiles
    kin, constraints = core(row, times, exponent)
    r, te, _, counts = profiles(row, times)
    return kin, append_edge_pairs(r, te, counts, kin, constraints)


def enabled():
    import os
    return os.environ.get('FUSION_MAST_EDGE_PAIRS', '0') == '1'
