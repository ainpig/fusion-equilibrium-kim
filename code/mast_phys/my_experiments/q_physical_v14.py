"""v14 MAST q95: frames solved as inboard-limited do not set their own shape factor.

v11 (q_physical_v11.py) rebuilt q95 as y * F / Ip with the shape factor y regularized towards the shot's
flat-top value, and frames whose readout came from an unresolved geometry take the interpolated y.
v14 adds one kind of unresolved geometry: a frame whose X-point separatrix, traced with the inner
numerical support removed, reaches inside that support (the solenoid surface). Every frame of the
challenge is diverted, so such a frame is an artefact of the seed solve (19-25% of the frames of a
fresh 48-shot run, almost all in the fast ramp-up). Its separatrix reaches into the solenoid, its minor
radius is too large and so is its shape factor: at equal time in the discharge the limited frames have
y 15-20% above the diverted frames of the same window (t = 120-300 ms), 60% above at 80-90% of the peak
current (fresh 48-shot run, no label read). Those frames now take y interpolated from the diverted
frames of the same shot, and the flat-top reference is the median over diverted flat-top frames. Same
knee / slope / cap, running median and B_t / Ip scaling as v11.

Nothing here reads a MAST target or score.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fusion_scoring'))
from my_experiments import q_physical_v11 as qp

LIMIT_TOL = 0.003      # m: a separatrix passing this far inside the support edge rests on it


def limited_flags(psi, r, z, support):
    """True where the frame's separatrix, traced on a support widened inwards, reaches inside rmin."""
    from lcfs import extract_lcfs
    rmin, rmax, zmax = support
    rr, zz = np.meshgrid(r, z)
    wide = (rr > float(r[1])) & (rr < rmax) & (abs(zz) < zmax)
    out = np.zeros(len(psi), bool)
    for k, p in enumerate(psi):
        c = extract_lcfs(np.asarray(p, float), r, z, "MAST", wide, wide.astype(float), n_points=128)
        out[k] = c is not None and float(c[:, 0].min()) < rmin - LIMIT_TOL
    return out


def flat_top_reference(q95, ip, fvac, limited):
    keep = ~np.asarray(limited, bool)
    if not keep.any():
        return float('nan')
    return qp.flat_top_reference(np.asarray(q95, float)[keep], np.asarray(ip, float)[keep],
                                 np.asarray(fvac, float)[keep])


def regularize(q95, ip, fvac, y_reference, limited):
    """q_physical_v11.regularize with the limited frames treated as unresolved geometry."""
    q95 = np.asarray(q95, float)
    ip = np.asarray(ip, float)
    fvac = np.asarray(fvac, float)
    n = len(q95)
    limited = np.asarray(limited, bool)
    ok = qp.resolved(q95, ip, fvac) & ~limited
    y = qp.shape_factor(q95, ip, fvac)
    y_new = np.full(n, np.nan)
    y_new[ok] = y_reference * qp.compress(y[ok] / y_reference)
    if ok.any():
        idx = np.arange(n)
        y_new = np.interp(idx, idx[ok], y_new[ok])
    else:
        y_new[:] = y_reference
    y_new = qp._running_median(y_new, qp.MEDIAN_WINDOW)
    ip_eff = np.maximum(np.where(np.isfinite(ip), ip, qp.MIN_IP), qp.MIN_IP)
    f_eff = np.where(np.isfinite(fvac) & (fvac > qp.MIN_F), fvac,
                     np.nanmedian(fvac[fvac > qp.MIN_F]) if (fvac > qp.MIN_F).any() else 0.4)
    q_new = np.clip(y_new * f_eff / (ip_eff / 1.0e6), qp.Q_MIN, qp.Q_MAX)
    record = dict(reference=float(y_reference), frames=int(n), resolved=int(ok.sum()),
                  limited=int(limited.sum()), q_mean_before=float(np.nanmean(q95)),
                  q_mean_after=float(q_new.mean()))
    return q_new, record
