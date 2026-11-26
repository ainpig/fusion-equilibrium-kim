"""Physical regularization of the MAST q95 readout: q95 = y * F / Ip with a slowly varying shape factor y.

For a flux surface of fixed shape the safety factor scales exactly as the toroidal-field function over
the plasma current, q95 = y * F / Ip, where y = q95 * Ip / F depends only on the shape and current
profile. The per-frame readout (`pipeline_v23.prepare`: the median of the Ampere, axis-shape and
elliptical estimators) is accurate on the solved flat-top, but where the solved geometry is not
trustworthy -- the first frames of the ramp, where the equilibrium solver puts the X-point at the edge
of its numerical domain, and frames whose geometry did not resolve and got the constant DIII-D prior --
its shape factor jumps by a factor of 3-10 and q95 comes out at 40-110 for a plasma that cannot have
one. A pooled R^2 is a ratio of sums of squares, so a handful of such frames outweigh the other
thousands (a fresh 8-shot run: q95 up to 110, standard deviation 10 on a flat-top value of 5-6).

The shape of a discharge changes over tens of milliseconds; a factor of 3 in y within a few frames is
not physics. So the shape factor is compressed (knee / slope / cap) towards the shot's own flat-top
value (the median of y over the frames at >= 75% of the peak current), frames without a resolved
geometry take the interpolated value, isolated spikes are removed by a short running median, and q95
is rebuilt from the exact B_t/Ip scaling. Nothing here reads a MAST target or a MAST-derived number:
the knee and cap are shape-physics (a plasma cannot change its elongation-and-aspect-ratio factor by
2x between neighbouring frames of one discharge) and the reference is the shot's own solved flat-top.
"""
import numpy as np

FALLBACK_Q = 5.213475704193115      # the constant prior `mast_stages` gives frames whose geometry did not resolve
# The shape factor relative to the shot's flat-top value, r = y / y_ref, is left alone up to KNEE_HIGH (the solved
# flat-top varies by about +-15% frame to frame), compressed with SLOPE beyond the knee and capped at CAP_HIGH: a
# plasma's (a^2 (1 + kappa^2) / R) x aspect-ratio factor does not double within a few frames of the same
# discharge (same knee / slope / cap form as the DIII-D li guard). Every pathology seen in the solved geometry
# inflates y, so nothing is pulled up from below except by the floor CAP_LOW; a single-frame dip is taken out by
# the running median.
KNEE_HIGH = 1.2
SLOPE = 0.3
CAP_HIGH = 1.4
CAP_LOW = 0.4
FLAT_FRACTION = 0.75                # flat-top frames: Ip at least this fraction of the shot's peak current
MIN_FLAT = 5
MIN_IP = 5.0e4                      # A; below this the 1/Ip scaling is not used as-is
MIN_F = 0.05                        # T m
MEDIAN_WINDOW = 5
Q_MIN, Q_MAX = 2.0, 40.0            # kink limit / no diverted MAST discharge frame lives above this


def shape_factor(q95, ip, fvac):
    """y = q95 * Ip[MA] / F[T m] (dimensionless up to the units)."""
    return np.asarray(q95, float) * (np.asarray(ip, float) / 1.0e6) / np.maximum(np.asarray(fvac, float), 1e-12)


def resolved(q95, ip, fvac):
    """Frames whose readout came from a solved geometry (not the constant prior, not NaN, usable Ip and F)."""
    q95 = np.asarray(q95, float)
    return (np.isfinite(q95) & (q95 > 0) & (np.abs(q95 - FALLBACK_Q) > 1e-3)
            & (np.asarray(ip, float) > MIN_IP) & (np.asarray(fvac, float) > MIN_F))


def flat_top_reference(q95, ip, fvac):
    """Median shape factor over the shot's flat-top frames (all resolved frames if there are too few); NaN if none."""
    ok = resolved(q95, ip, fvac)
    if not ok.any():
        return float('nan')
    ip = np.asarray(ip, float)
    y = shape_factor(q95, ip, fvac)
    flat = ok & (ip >= FLAT_FRACTION * ip[ok].max())
    sel = flat if flat.sum() >= MIN_FLAT else ok
    return float(np.median(y[sel]))


def _running_median(x, window):
    if window <= 1 or len(x) < 3:
        return x.copy()
    half = window // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.array([np.median(padded[i:i + window]) for i in range(len(x))])


def compress(ratio):
    """Knee / slope / cap map of the shape-factor ratio r = y / y_reference."""
    r = np.asarray(ratio, float)
    high = np.minimum(KNEE_HIGH + SLOPE * (r - KNEE_HIGH), CAP_HIGH)
    return np.where(r > KNEE_HIGH, high, np.maximum(r, CAP_LOW))


def regularize(q95, ip, fvac, y_reference):
    """The regularized q95 of one shot (same length, float64) and a small record of what was changed.

    `y_reference` is the shot's flat-top shape factor, or, for a shot with no resolved frame, the median
    over the other shots of the same run.
    """
    q95 = np.asarray(q95, float)
    ip = np.asarray(ip, float)
    fvac = np.asarray(fvac, float)
    n = len(q95)
    ok = resolved(q95, ip, fvac)
    y = shape_factor(q95, ip, fvac)
    y_new = np.full(n, np.nan)
    y_new[ok] = y_reference * compress(y[ok] / y_reference)
    if ok.any():
        idx = np.arange(n)
        y_new = np.interp(idx, idx[ok], y_new[ok])
    else:
        y_new[:] = y_reference
    y_new = _running_median(y_new, MEDIAN_WINDOW)
    ip_eff = np.maximum(np.where(np.isfinite(ip), ip, MIN_IP), MIN_IP)
    f_eff = np.where(np.isfinite(fvac) & (fvac > MIN_F), fvac, np.nanmedian(fvac[fvac > MIN_F]) if (fvac > MIN_F).any() else 0.4)
    q_new = np.clip(y_new * f_eff / (ip_eff / 1.0e6), Q_MIN, Q_MAX)
    record = dict(reference=float(y_reference), frames=int(n), resolved=int(ok.sum()),
                  compressed_high=int((ok & (y > KNEE_HIGH * y_reference)).sum()),
                  floored_low=int((ok & (y < CAP_LOW * y_reference)).sum()),
                  q_max_before=float(np.nanmax(q95)), q_max_after=float(q_new.max()),
                  q_mean_before=float(np.nanmean(q95)), q_mean_after=float(q_new.mean()))
    return q_new, record
