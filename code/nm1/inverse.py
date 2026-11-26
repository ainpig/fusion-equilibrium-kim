"""Thomson anchors of a frame, used by ``nm1.mast.frame_inputs``.

The submitted MAST leg (``nm1.mast_phys``, ``mast_phys/``) does not use the development-phase anchored-profile
solver that also lived in this module; only the time-smoothing helper of the final MAST file imports
``nm1.mast``, and that path reads coil currents and Ip alone, never the anchors.  The anchors are kept so that
``frame_inputs`` returns the same structure.

Anchors per frame: (R of the Te maximum of the 3-point smoothed profile with a parabolic sub-cell shift,
R of the outboard 100 eV crossing), NaN where a frame has too few valid points.
"""
from __future__ import annotations

import numpy as np

EDGE_EV = 100.0


def _peak(r: np.ndarray, te: np.ndarray) -> float:
    s = np.convolve(te, np.ones(3) / 3.0, mode="same")
    ip = int(np.argmax(s))
    if 0 < ip < len(s) - 1:
        a, b, c = s[ip - 1], s[ip], s[ip + 1]
        den = a - 2 * b + c
        dr = float(np.median(np.diff(r))) if len(r) > 1 else 0.0
        shift = 0.0 if den >= -1e-9 else float(np.clip(0.5 * (a - c) / den, -1, 1))
        return float(r[ip] + shift * dr)
    return float(r[ip])


def _edge(r: np.ndarray, te: np.ndarray, r_pk: float) -> float:
    out = r >= r_pk
    ro, to = r[out], te[out]
    below = np.nonzero(to < EDGE_EV)[0]
    if len(ro) < 2 or not len(below) or below[0] == 0:
        return np.nan
    k = below[0]
    t0, t1, r0, r1 = to[k - 1], to[k], ro[k - 1], ro[k]
    return float(r0 + (EDGE_EV - t0) * (r1 - r0) / (t1 - t0)) if t1 != t0 else float(r1)


def frame_anchors(Rt: np.ndarray, TE: np.ndarray) -> np.ndarray:
    """(T, 2) array of (R_peak, R_edge) per frame from Thomson radii ``Rt`` and temperatures ``TE`` (T, n)."""
    Rt = np.asarray(Rt, float)
    TE = np.atleast_2d(np.asarray(TE, float))
    out = np.full((len(TE), 2), np.nan)
    for k, te in enumerate(TE):
        r = Rt if Rt.ndim == 1 else Rt[k]
        ok = np.isfinite(r) & np.isfinite(te) & (te > 0)
        if ok.sum() < 5:
            continue
        o = np.argsort(r[ok])
        rr, tt = r[ok][o], te[ok][o]
        pk = _peak(rr, tt)
        out[k] = pk, _edge(rr, tt, pk)
    return out
