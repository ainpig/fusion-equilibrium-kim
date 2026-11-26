"""The second feature block: what the base 140 throws away.

The base block reads every coil current as ONE interpolated sample at the EFIT time, and every
Thomson profile as quantiles over its channels -- which discards where each channel is.  Both
losses are measurable:

* coil currents on DIII-D carry 0.4-1.6 kA-turn of sample-to-sample noise (power-supply ripple and
  digitiser steps) on signals whose whole in-shot spread is 1.5-50 kA-turn, so a single sample is a
  noisy estimate of the current the plasma actually sees through the vessel's few-ms filter;
* Te and ne are flux functions, so a profile *as a function of chord position* is a direct
  measurement of flux-surface geometry.  The core system is a vertical chord at R = 1.94 m and the
  point where Te falls to ~50 eV tracks EFIT's LCFS crossing of that chord to ~9 mm (MAD, 8,551
  frames); the tangential system is a horizontal chord just below the midplane whose Te peak sits
  near the magnetic axis.  Chord positions move from shot to shot, so channel *index* is not a
  position.

Only test-legal columns are read.  Nothing here touches ``efit_*`` except ``efit_times`` (the
prediction timestamps), which the public test splits ship.
"""
from __future__ import annotations

import numpy as np

from nm1.data_fixes import fix_d3d_ip_times
from nm1.features import D3D_COILS

# fixed spatial grids the profiles are resampled onto
CORE_Z = np.concatenate([np.arange(0.0, 0.52, 0.04), np.arange(0.52, 0.8651, 0.01)])
TAN_R = np.arange(1.66, 2.0801, 0.02)
UNSEEN = -0.5                      # value for a grid point outside the chord's coverage

TE_LEVELS = (30.0, 60.0, 120.0, 250.0, 500.0, 1000.0)       # eV
NE_LEVELS = (0.3, 0.7, 1.5, 3.0)                            # 1e19 m^-3
TAN_FRACS = (0.9, 0.7, 0.5)
MAP_FRACS = (0.8, 0.6, 0.4, 0.2)

WINDOWS = (("m2", -2.0, 2.0), ("m10", -10.0, 10.0), ("p20", -30.0, -10.0), ("f20", 10.0, 30.0))


# ----------------------------------------------------------------------------------------------
# coil currents: window statistics from the full-rate signal
# ----------------------------------------------------------------------------------------------
def _window_stats(t: np.ndarray, v: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """``(T, 6)``: m2, m10-m2, p20-m2, f20-m2, slope over +-5 ms [kA/s], rms about that line."""
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok].astype(np.float64), v[ok].astype(np.float64)
    out = np.zeros((len(centers), 6), dtype=np.float64)
    if len(t) < 2:
        return out.astype(np.float32)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    ts = t / 1000.0                                   # seconds, keeps the moment sums small
    z = np.zeros(1)
    c1 = np.concatenate([z, np.cumsum(v)])
    ct = np.concatenate([z, np.cumsum(ts)])
    ctt = np.concatenate([z, np.cumsum(ts * ts)])
    ctv = np.concatenate([z, np.cumsum(ts * v)])
    cvv = np.concatenate([z, np.cumsum(v * v)])
    point = np.interp(centers, t, v)

    def bounds(lo, hi):
        i0 = np.searchsorted(t, centers + lo, side="left")
        i1 = np.searchsorted(t, centers + hi, side="right")
        return i0, i1

    means = {}
    for name, lo, hi in WINDOWS:
        i0, i1 = bounds(lo, hi)
        n = i1 - i0
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (c1[i1] - c1[i0]) / n
        # an empty window falls back to the signal at the window centre
        fallback = np.interp(centers + 0.5 * (lo + hi), t, v)
        means[name] = np.where(n > 0, m, fallback)
    i0, i1 = bounds(-5.0, 5.0)
    n = (i1 - i0).astype(np.float64)
    S1, St, Stt = c1[i1] - c1[i0], ct[i1] - ct[i0], ctt[i1] - ctt[i0]
    Stv, Svv = ctv[i1] - ctv[i0], cvv[i1] - cvv[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * Stt - St * St
        slope = np.where((n >= 3) & (den > 0), (n * Stv - St * S1) / den, 0.0)
        icpt = np.where(n > 0, (S1 - slope * St) / n, point)
        sse = (Svv - 2 * icpt * S1 - 2 * slope * Stv + n * icpt ** 2
               + 2 * icpt * slope * St + slope ** 2 * Stt)
        rms = np.where(n >= 3, np.sqrt(np.clip(sse, 0, None) / np.maximum(n, 1)), 0.0)
    out[:, 0] = means["m2"]
    out[:, 1] = means["m10"] - means["m2"]
    out[:, 2] = means["p20"] - means["m2"]
    out[:, 3] = means["f20"] - means["m2"]
    out[:, 4] = slope
    out[:, 5] = rms
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def coil_window_block(row, target: np.ndarray) -> np.ndarray:
    """``(T, 21*6)`` window statistics of the 20 coil currents and Ip."""
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_time = fix_d3d_ip_times(row)
    cols = []
    for name in D3D_COILS:
        t = ip_time if name == "magnetics_plasma_current" else shared
        cols.append(_window_stats(t, np.asarray(row[name], dtype=np.float64),
                                  target.astype(np.float64)))
    return np.hstack(cols)


# ----------------------------------------------------------------------------------------------
# Thomson: time alignment, then position-aware resampling
# ----------------------------------------------------------------------------------------------
def _stack(value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2:
            return arr
    except (TypeError, ValueError):
        pass
    rows = [np.asarray(x, dtype=np.float64) for x in value]
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _align_time(times: np.ndarray, prof: np.ndarray, target: np.ndarray,
                max_dt: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel linear interpolation that never blends a valid sample with an invalid one.

    Invalid (<=0 or non-finite) readings are Thomson's "no signal".  A frame gets the blend of
    its two bracketing laser pulses when both are valid, the nearer one alone when only that one
    is valid and within ``max_dt``, and 0 otherwise.  Returns ``(values, dt_to_nearest_pulse)``.
    """
    n = min(len(times), len(prof))
    times, prof = np.asarray(times[:n], np.float64), prof[:n]
    good_t = np.isfinite(times)
    times, prof = times[good_t], prof[good_t]
    T, C = len(target), prof.shape[1] if prof.ndim == 2 else 0
    if len(times) == 0 or C == 0:
        return np.zeros((T, C)), np.full(T, 1e3)
    order = np.argsort(times, kind="stable")
    times, prof = times[order], prof[order]
    valid = np.isfinite(prof) & (prof > 0)
    val = np.where(valid, prof, 0.0)
    k1 = np.clip(np.searchsorted(times, target, side="left"), 0, len(times) - 1)
    k0 = np.clip(k1 - 1, 0, len(times) - 1)
    t0, t1 = times[k0], times[k1]
    d0, d1 = np.abs(target - t0), np.abs(t1 - target)
    span = np.where(t1 > t0, t1 - t0, 1.0)
    w1 = np.clip((target - t0) / span, 0.0, 1.0)[:, None]
    v0, v1 = val[k0], val[k1]
    ok0, ok1 = valid[k0], valid[k1]
    near0 = (d0 <= max_dt)[:, None]
    near1 = (d1 <= max_dt)[:, None]
    both = ok0 & ok1 & (near0 | near1) & ((t1 - t0) <= 3 * max_dt)[:, None]
    out = np.where(both, (1 - w1) * v0 + w1 * v1, 0.0)
    only0 = ok0 & ~both & near0 & (~ok1 | (d0[:, None] <= d1[:, None]))
    only1 = ok1 & ~both & ~only0 & near1
    out = np.where(only0, v0, out)
    out = np.where(only1, v1, out)
    return out, np.minimum(d0, d1)


def _fill_missing(pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Inside the observed plasma, a zero channel is a dead channel -- interpolate it.

    ``pos`` ascending.  Zeros beyond the outermost valid channel (in the +pos direction) are
    real "no plasma" and stay 0.
    """
    out = vals.copy()
    for i in range(vals.shape[0]):
        v = vals[i]
        ok = v > 0
        if ok.sum() < 2:
            continue
        idx = np.flatnonzero(ok)
        inner = np.arange(idx[0], idx[-1] + 1)
        hole = inner[~ok[inner]]
        if len(hole):
            out[i, hole] = np.interp(pos[hole], pos[ok], v[ok])
    return out


def _resample(pos: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    lo, hi = pos.min(), pos.max()
    out = np.empty((vals.shape[0], len(grid)))
    for i in range(vals.shape[0]):
        out[i] = np.interp(grid, pos, vals[i])
    out[:, (grid < lo - 1e-6) | (grid > hi + 1e-6)] = np.nan
    return out


def _crossing_up(pos: np.ndarray, v: np.ndarray, level: float) -> float:
    """Outermost position (largest ``pos``) where the profile falls through ``level``."""
    above = np.flatnonzero(v >= level)
    if not len(above):
        return pos[0]                         # below the level everywhere we can see
    i = above.max()
    if i == len(v) - 1:
        return pos[-1]                        # still above it at the edge of coverage
    return pos[i] + (level - v[i]) * (pos[i + 1] - pos[i]) / (v[i + 1] - v[i])


def _chord_indices(row, prefix: str, fallback: slice) -> np.ndarray:
    try:
        names = np.asarray(row["thomson_chord_name"]).astype(str)
        idx = np.flatnonzero(np.char.startswith(names, prefix))
        if len(idx):
            return idx
    except Exception:                               # noqa: BLE001
        pass
    return np.arange(fallback.start, fallback.stop)


N_CORE = len(CORE_Z) * 2 + len(TE_LEVELS) + len(NE_LEVELS) + 6 + len(MAP_FRACS)
N_TAN = len(TAN_R) * 2 + 4 + 2 * len(TAN_FRACS) + 2


def core_block(row, target: np.ndarray, tan_te_max: np.ndarray | None) -> np.ndarray:
    """Vertical chord at R ~ 1.94 m: Te/ne on a fixed Z grid plus edge-crossing heights."""
    T = len(target)
    out = np.zeros((T, N_CORE), dtype=np.float64)
    try:
        times = np.asarray(row["thomson_core_times"], dtype=np.float64)
        te = _stack(row["thomson_core_Te"])
        ne = _stack(row["thomson_core_ne"]) / 1e19
        cidx = _chord_indices(row, "TS_core", slice(0, te.shape[1]))
        Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)[cidx]
        C = min(len(Z), te.shape[1], ne.shape[1])
        Z, te, ne = Z[:C], te[:, :C], ne[:, :C]
        o = np.argsort(Z, kind="stable")
        Z = Z[o]
        te_t, dt = _align_time(times, te[:, o], target)
        ne_t, _ = _align_time(times, ne[:, o], target)
    except Exception:                               # noqa: BLE001 - a missing subsystem
        out[:, :] = UNSEEN
        return out.astype(np.float32)
    te_f = _fill_missing(Z, te_t)
    ne_f = _fill_missing(Z, ne_t)
    g_te = np.log1p(_resample(Z, te_f, CORE_Z) / 100.0)
    g_ne = _resample(Z, ne_f, CORE_Z)
    k = len(CORE_Z)
    out[:, :k] = np.nan_to_num(g_te, nan=UNSEEN)
    out[:, k:2 * k] = np.nan_to_num(g_ne, nan=UNSEEN)
    col = 2 * k
    for i in range(T):
        v = te_f[i]
        has = (v > 0).sum() >= 3
        for j, lev in enumerate(TE_LEVELS):
            out[i, col + j] = _crossing_up(Z, v, lev) if has else UNSEEN
        w = ne_f[i]
        hasn = (w > 0).sum() >= 3
        for j, lev in enumerate(NE_LEVELS):
            out[i, col + len(TE_LEVELS) + j] = _crossing_up(Z, w, lev) if hasn else UNSEEN
    col += len(TE_LEVELS) + len(NE_LEVELS)
    valid = te_f > 0
    any_valid = valid.any(axis=1)
    last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    out[:, col] = np.where(any_valid, Z[last], UNSEEN)
    out[:, col + 1] = Z.max()
    out[:, col + 2] = Z.min()
    out[:, col + 3] = np.minimum(dt, 50.0) / 10.0
    out[:, col + 4] = np.log1p(te_f[:, 0] / 100.0)                 # lowest channel, near midplane
    out[:, col + 5] = valid.mean(axis=1)
    col += 6
    # where on this chord Te equals a fraction of the tangential (near-axis) peak: points on the
    # same flux surface as the corresponding tangential crossing
    for j, frac in enumerate(MAP_FRACS):
        for i in range(T):
            if tan_te_max is None or not np.isfinite(tan_te_max[i]) or tan_te_max[i] <= 0 \
                    or (te_f[i] > 0).sum() < 3:
                out[i, col + j] = UNSEEN
                continue
            lev = frac * tan_te_max[i]
            out[i, col + j] = _crossing_up(Z, te_f[i], lev)
    return np.nan_to_num(out, nan=UNSEEN).astype(np.float32)


def tangential_block(row, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal chord just below the midplane: Te/ne on a fixed R grid, peak and crossings."""
    T = len(target)
    out = np.zeros((T, N_TAN), dtype=np.float64)
    te_max = np.full(T, np.nan)
    try:
        times = np.asarray(row["thomson_edge_times"], dtype=np.float64)
        te = _stack(row["thomson_edge_Te"])
        ne = _stack(row["thomson_edge_ne"]) / 1e19
        cidx = _chord_indices(row, "TS_tangential", slice(44, 44 + te.shape[1]))
        R = np.asarray(row["thomson_chord_R"], dtype=np.float64)[cidx]
        C = min(len(R), te.shape[1], ne.shape[1])
        R, te, ne = R[:C], te[:, :C], ne[:, :C]
        o = np.argsort(R, kind="stable")
        R = R[o]
        te_t, dt = _align_time(times, te[:, o], target)
        ne_t, _ = _align_time(times, ne[:, o], target)
    except Exception:                               # noqa: BLE001
        out[:, :] = UNSEEN
        return out.astype(np.float32), te_max
    # dead channels anywhere inside the covered span are interpolated (the whole chord is inside
    # the plasma on DIII-D, 1.67-2.06 m against an outboard LCFS near 2.25 m)
    te_f = te_t.copy()
    ne_f = ne_t.copy()
    for arr in (te_f, ne_f):
        for i in range(T):
            ok = arr[i] > 0
            if 2 <= ok.sum() < len(ok):
                arr[i, ~ok] = np.interp(R[~ok], R[ok], arr[i, ok])
    k = len(TAN_R)
    out[:, :k] = np.nan_to_num(np.log1p(_resample(R, te_f, TAN_R) / 100.0), nan=UNSEEN)
    out[:, k:2 * k] = np.nan_to_num(_resample(R, ne_f, TAN_R), nan=UNSEEN)
    col = 2 * k
    for i in range(T):
        v = te_f[i]
        if (v > 0).sum() < 3:
            out[i, col:col + 4 + 2 * len(TAN_FRACS)] = UNSEEN
            continue
        j = int(np.argmax(v))
        rp = R[j]
        if 0 < j < len(v) - 1:
            a, b, c = v[j - 1], v[j], v[j + 1]
            den = a - 2 * b + c
            if den < 0:
                # non-uniform spacing: fit the parabola through the three points
                x = R[j - 1:j + 2]
                A = np.vstack([x ** 2, x, np.ones(3)]).T
                try:
                    p2, p1, _ = np.linalg.solve(A, v[j - 1:j + 2])
                    if p2 < 0:
                        rp = float(np.clip(-p1 / (2 * p2), R[j - 1], R[j + 1]))
                except np.linalg.LinAlgError:
                    pass
        te_max[i] = v.max()
        top = v >= 0.7 * v.max()
        out[i, col] = rp
        out[i, col + 1] = float((R[top] * v[top]).sum() / v[top].sum())
        out[i, col + 2] = np.log1p(v.max() / 100.0)
        out[i, col + 3] = float((v[0] - v[-1]) / max(v.max(), 1.0))
        for f, frac in enumerate(TAN_FRACS):
            lev = frac * v.max()
            outer = v[j:]
            out[i, col + 4 + f] = _crossing_up(R[j:], outer, lev) if len(outer) > 1 else R[-1]
            inner_v = v[:j + 1][::-1]
            inner_r = R[:j + 1][::-1]
            # mirror so the same "outermost crossing" routine walks inward
            out[i, col + 4 + len(TAN_FRACS) + f] = (
                -_crossing_up(-inner_r, inner_v, lev) if len(inner_v) > 1 else R[0])
    col += 4 + 2 * len(TAN_FRACS)
    out[:, col] = np.minimum(dt, 50.0) / 10.0
    out[:, col + 1] = (te_f > 0).mean(axis=1)
    return np.nan_to_num(out, nan=UNSEEN).astype(np.float32), te_max


def extended_features(row, target: np.ndarray) -> np.ndarray:
    """``(T, N_EXT)`` second block for one DIII-D shot, at the given EFIT times."""
    target = np.asarray(target, dtype=np.float64)
    coil = coil_window_block(row, target)
    tan, te_max = tangential_block(row, target)
    core = core_block(row, target, te_max)
    return np.hstack([coil, core, tan]).astype(np.float32)


N_EXT = len(D3D_COILS) * 6 + N_CORE + N_TAN

# ----------------------------------------------------------------------------------------------
# shot context: where in the discharge a frame sits, and how the current got there
# ----------------------------------------------------------------------------------------------
# `li` is the scalar with the heaviest tail: on held-out shots its fifty worst frames carry a
# third of its residual, and they sit in the first and last few percent of a discharge -- current
# ramp-up and ramp-down, where li runs to 2-4 because the edge current has decayed faster than the
# core's.  That is history, and a frame's own inputs do not carry it.  These columns do, from the
# test-legal Ip trace and the kinetic block alone.
CTX_LAGS = (0.1, 0.25, 0.5, 1.0)          # s, past
CTX_LEADS = (0.1, 0.25)                   # s, future (offline reconstruction: the shot is known)
IP_COL = len(D3D_COILS) * 6 - 6           # ext column of the Ip +-2 ms mean (kA)
TEMAX_COL = len(D3D_COILS) * 6 + N_CORE + 2 * len(TAN_R) + 2   # log1p(peak tangential Te / 100)
N_CTX = 2 + len(CTX_LAGS) + len(CTX_LEADS) + 4 + 2 + 2


def context_block(ext: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """``(T, N_CTX)`` shot-context features for ONE shot's frames (time-ordered)."""
    t = np.asarray(times_s, dtype=np.float64)
    ip = np.abs(ext[:, IP_COL].astype(np.float64))
    te = ext[:, TEMAX_COL].astype(np.float64)
    T = len(t)
    out = np.zeros((T, N_CTX), dtype=np.float64)
    if T == 0:
        return out.astype(np.float32)
    ipmax = max(float(ip.max()), 1.0)
    frac = ip / ipmax

    def at(x, when):
        return np.interp(when, t, x, left=x[0], right=x[-1])

    col = 0
    out[:, col] = frac
    out[:, col + 1] = ipmax / 1000.0                                   # MA, scenario scale
    col += 2
    for lag in CTX_LAGS:
        out[:, col] = (at(ip, t - lag) - ip) / ipmax
        col += 1
    for lead in CTX_LEADS:
        out[:, col] = (at(ip, t + lead) - ip) / ipmax
        col += 1
    flat = np.flatnonzero(frac > 0.9)
    t_ft0, t_ft1 = (t[flat[0]], t[flat[-1]]) if len(flat) else (t[0], t[-1])
    out[:, col] = np.clip(t - t_ft0, -5, 10)
    out[:, col + 1] = np.clip(t_ft1 - t, -5, 10)
    out[:, col + 2] = np.clip(t - t[0], 0, 15)
    out[:, col + 3] = np.clip(t[-1] - t, 0, 15)
    col += 4
    out[:, col] = at(te, t - 0.2) - te
    out[:, col + 1] = at(te, t - 0.5) - te
    col += 2
    # how much of the current is still to come or already gone, integrated over the shot
    out[:, col] = np.interp(t, t, np.concatenate([[0.0], np.cumsum(
        0.5 * (frac[1:] + frac[:-1]) * np.diff(t))])) if T > 1 else 0.0
    out[:, col + 1] = (out[-1, col] - out[:, col]) if T > 1 else 0.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# where a frame sits in its run of consecutive EFIT frames
# ----------------------------------------------------------------------------------------------
N_SEG = 8
SEG_STEP = 0.02          # s: DIII-D EFIT frames are exactly 20 ms apart in every shot
SEG_BREAK = 0.03         # s: a longer step means frames were removed in between


def seg_block(times_s: np.ndarray) -> np.ndarray:
    """``(T, N_SEG)``: how far a frame is from the ends of its run of consecutive EFIT frames.

    The challenge keeps diverted frames only ("limited frames have been removed from `efit_times`"),
    and `efit_times` is a shipped input. A step longer than 20 ms therefore marks a stretch of
    limited plasma (or a failed reconstruction), and the frames next to it -- like the first and
    last frames of a shot -- sit at a limited <-> diverted transition: the flux surface through the
    x-point is barely the boundary, the plasma is small and near the wall, and `li` runs to 3-5 in
    the ramp-down. Frames within 3 of a run's end are 13% of the held-out frames and carry 41% of
    the Consistency loss of the v10 ensemble; the network had no way to know it was at one.

    Only local quantities: distances to the two ends of the run (a decaying and a linear form),
    whether the run is the shot's first or last, and how long the removed stretches on each side
    are (log). Nothing that identifies a shot.
    """
    t = np.asarray(times_s, dtype=np.float64)
    T = len(t)
    out = np.zeros((T, N_SEG), dtype=np.float64)
    if T == 0:
        return out.astype(np.float32)
    brk = np.concatenate([[True], np.diff(t) > SEG_BREAK])
    run = np.cumsum(brk) - 1
    n_run = int(run[-1]) + 1
    idx = np.arange(T)
    first = np.flatnonzero(brk)
    last = np.concatenate([first[1:], [T]]) - 1
    since = idx - first[run]
    until = last[run] - idx
    # removed frames before a run's first frame / after its last frame, in nominal steps
    gap_before = np.zeros(n_run)
    gap_after = np.zeros(n_run)
    for k in range(1, n_run):
        g = (t[first[k]] - t[last[k - 1]]) / SEG_STEP - 1.0
        gap_before[k] = g
        gap_after[k - 1] = g
    out[:, 0] = np.exp(-since / 3.0)
    out[:, 1] = np.exp(-until / 3.0)
    out[:, 2] = np.minimum(since, 40) / 40.0
    out[:, 3] = np.minimum(until, 40) / 40.0
    out[:, 4] = (run == 0)
    out[:, 5] = (run == n_run - 1)
    out[:, 6] = np.log1p(np.minimum(gap_before[run], 200.0)) / 3.0
    out[:, 7] = np.log1p(np.minimum(gap_after[run], 200.0)) / 3.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def seg_of_shots(times_s: np.ndarray, shot_ids: np.ndarray) -> np.ndarray:
    """`seg_block` over cache rows: shots are contiguous runs of `shot_ids`."""
    n = len(shot_ids)
    out = np.empty((n, N_SEG), dtype=np.float32)
    b = np.flatnonzero(np.diff(shot_ids)) + 1
    st = np.concatenate([[0], b])
    en = np.concatenate([b, [n]])
    for s, e in zip(st, en):
        out[s:e] = seg_block(times_s[s:e])
    return out
