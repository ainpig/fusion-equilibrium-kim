"""The li cliff guard: keep every predicted frame's internal inductance inside the scored range.

The scorer maps a derived scalar outside its plausible range to NaN and then *mean-substitutes*
it (``derive._clamp_to_nan``; for ``li`` the range is 0.1 - 5.0). For a frame whose true ``li``
is 4.5 -- the terminal frames of a current ramp-down reach 3-5 -- a prediction of 5.05 therefore
scores as if it had said 1.0: one such frame costs ~1e-4 of the composite on a 200-shot fold,
more than most of what separates two ensembles. Measured on held-out shots, single members lose
1e-4 to 4e-4 of S this way, and the deployed v3 ensemble two frames of the calibration third.

So each predicted frame's ``li`` is computed with the scorer's own functionals, and a frame above
``LI_MAX`` gets its flux relabelled *inside the LCFS only*:

    u = phi - phi_LCFS  (u >= 0 inside),    u' = u - (alpha/2) u^2 / u_axis

a monotonic map with ``u'(0) = 0`` and ``du'/du = 1`` on the boundary. Every contour keeps its
place, so the LCFS, the axis, kappa, both triangularities, the volume and ``D_LCFS`` are
unchanged; the boundary field and the Ampere loop integral are unchanged; only the interior
poloidal field falls, by ``1 - alpha u/u_axis`` -- which lowers ``li`` to the target. ``alpha`` is
found by bisection per frame. A frame the guard cannot evaluate is left as it was.

Mask: the scorer's envelope file is not read at inference (no pre-existing NPZ is), so the O-point
and LCFS searches here run inside a fixed rectangle covering DIII-D's vessel. For diverted
frames the X-point sets the LCFS either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCORING = Path(__file__).resolve().parents[1] / "fusion_scoring"
if str(_SCORING) not in sys.path:
    sys.path.insert(0, str(_SCORING))

from common import AXIS_SIGN  # noqa: E402
from derive import _points_in_polygon, internal_inductance  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402
from o_point import find_o_point  # noqa: E402

LI_MAX = 4.9
LI_MIN = 0.12
R_BOX = (1.00, 2.45)          # DIII-D vessel, generously
Z_BOX = (-1.45, 1.45)


def box_mask(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    rr, zz = np.meshgrid(np.asarray(R, float), np.asarray(Z, float))
    return (rr >= R_BOX[0]) & (rr <= R_BOX[1]) & (zz >= Z_BOX[0]) & (zz <= Z_BOX[1])


def _relabel(psi: np.ndarray, sign: float, level: float, axis_val: float, inside: np.ndarray,
             alpha: float) -> np.ndarray:
    phi = sign * psi
    u = phi - level
    span = axis_val - level
    u2 = np.where(inside & (u > 0), u - 0.5 * alpha * u * u / span, u)
    return (level + u2) / sign


SCREEN_MARGIN = 0.3           # coarse li this far below the first touched value is left alone


def li_target(li: float, li_max: float = LI_MAX, knee: float | None = None,
              slope: float = 1.0) -> float:
    """The li a frame is relabelled to. Without a knee (v7) only the cliff is removed.

    With a knee (v8), the tail above it is shrunk: held-out frames predicted at li 4.0-5.0 have a
    scored truth ~0.35-0.55 lower on average -- the truth is
    capped at 5.0 by the scorer and the extreme predictions regress toward the mean.
    """
    t = li if knee is None or li <= knee else knee + slope * (li - knee)
    return min(t, li_max)


def guard_frame(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, mask: np.ndarray,
                machine: str = "DIII-D", li_max: float = LI_MAX, knee: float | None = None,
                slope: float = 1.0) -> tuple[np.ndarray, dict]:
    """Return ``(psi, info)``; ``psi`` is unchanged unless its li is above the touched range."""
    psi = np.asarray(psi, dtype=np.float64)
    sign = AXIS_SIGN[machine]
    mf = mask.astype(np.float64)
    first = li_max if knee is None else min(knee, li_max)
    # screen: a coarser LCFS (14 bisections, 192 points) is enough to tell a frame far from the
    # touched range, which is nearly every frame, at a third of the cost
    coarse = extract_lcfs(psi, R, Z, machine, mask, mf, n_iter=14, n_points=192)
    if coarse is not None:
        li0 = internal_inductance(psi, R, Z, machine, coarse)
        if np.isfinite(li0) and LI_MIN <= li0 <= first - SCREEN_MARGIN:
            return psi, {"status": "ok", "li": li0}
    contour = extract_lcfs(psi, R, Z, machine, mask, mf, n_points=512)
    if contour is None:
        return psi, {"status": "no_lcfs"}
    li = internal_inductance(psi, R, Z, machine, contour)
    if not np.isfinite(li) or LI_MIN <= li <= first:
        return psi, {"status": "ok", "li": li}
    if li < LI_MIN:
        return psi, {"status": "low", "li": li}
    target = li_target(li, li_max, knee, slope)
    if li <= target + 1e-3:
        return psi, {"status": "ok", "li": li}
    phi = sign * psi
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mask)
    axis_val = float(phi[iz, ir])
    # the LCFS level: the flux on the contour, read back bilinearly
    from derive import _bilinear
    level = float(np.median(_bilinear(phi, np.asarray(R, float), np.asarray(Z, float),
                                      contour[:, 0], contour[:, 1])))
    if not axis_val > level:
        return psi, {"status": "no_span", "li": li}
    rr, zz = np.meshgrid(np.asarray(R, float), np.asarray(Z, float))
    inside = _points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(rr.shape)
    lo, hi = 0.0, 0.95
    out, got = psi, li
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        cand = _relabel(psi, sign, level, axis_val, inside, mid)
        c2 = extract_lcfs(cand, R, Z, machine, mask, mf, n_points=512)
        if c2 is None:
            hi = mid
            continue
        v = internal_inductance(cand, R, Z, machine, c2)
        if not np.isfinite(v):
            hi = mid
            continue
        if v > target:
            lo = mid
        else:
            hi = mid
            out, got = cand, v
    if got > min(li_max, target + 0.05):
        return psi, {"status": "unfixed", "li": li}
    return out, {"status": "guarded", "li": li, "target": target, "li_new": got, "alpha": hi}


def guard_shot(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, machine: str = "DIII-D",
               params: dict | None = None) -> tuple[np.ndarray, list[dict]]:
    p = params or {}
    kw = {"li_max": float(p.get("li_max", LI_MAX)),
          "knee": (float(p["knee"]) if p.get("knee") is not None else None),
          "slope": float(p.get("slope", 1.0))}
    mask = box_mask(R, Z)
    out = np.array(psi, dtype=np.float64, copy=True)
    notes = []
    for k in range(len(out)):
        out[k], info = guard_frame(out[k], R, Z, mask, machine, **kw)
        if info["status"] not in ("ok",):
            notes.append({"frame": k, **info})
    return out, notes


def _init_worker() -> None:
    """Guard workers run under the same read policy as the parent run (FUSION_RUN_ROOT)."""
    import os
    if os.environ.get("FUSION_RUN_ROOT"):
        from nm1.input_policy import install
        install()


def _job(args):
    psi, R, Z, params = args
    return guard_shot(psi, R, Z, params=params)


class ShotGuard:
    """Guards shots in a process pool while the GPU predicts the next ones; results in order."""

    def __init__(self, workers: int = 6, depth: int = 12, params: dict | None = None):
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(workers, initializer=_init_worker)
        self.depth = depth
        self.params = dict(params or {})
        self.pending: list = []
        self.notes: list[dict] = []

    def submit(self, key, psi, R, Z, payload) -> list:
        """Queue one shot; returns the (key, guarded psi, payload) of shots that are finished."""
        self.pending.append((key, self.pool.submit(_job, (psi, R, Z, self.params)), payload))
        done = []
        while self.pending and (len(self.pending) > self.depth or self.pending[0][1].done()):
            k, fut, pl = self.pending.pop(0)
            psi_g, notes = fut.result()
            self.notes.extend({"shot": k, **n} for n in notes)
            done.append((k, psi_g, pl))
        return done

    def drain(self) -> list:
        done = []
        while self.pending:
            k, fut, pl = self.pending.pop(0)
            psi_g, notes = fut.result()
            self.notes.extend({"shot": k, **n} for n in notes)
            done.append((k, psi_g, pl))
        self.pool.shutdown()
        return done