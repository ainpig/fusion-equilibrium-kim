"""Feature construction from a raw challenge parquet row.

Only columns that exist on the public test splits are read.  No ``efit_*`` target and no
``magnetics_dsep`` enters a feature, and ``read_row(..., drop_targets=True)`` removes them from
the row entirely so an inference path cannot reach one by accident.

DIII-D feature vector, 140 values per EFIT frame: 21 coil/plasma currents, their time
derivatives, their 20 ms and 100 ms causal changes, 4 coil aggregates, 48 Thomson profile
summaries, 4 clock terms.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import ellipe, ellipk

from nm1.data_fixes import fix_d3d_ip_times

D3D_COILS = [
    "magnetics_ECOILA", "magnetics_bcoil",
    "magnetics_F1A", "magnetics_F1B", "magnetics_F2A", "magnetics_F2B",
    "magnetics_F3A", "magnetics_F3B", "magnetics_F4A", "magnetics_F4B",
    "magnetics_F5A", "magnetics_F5B", "magnetics_F6A", "magnetics_F6B",
    "magnetics_F7A", "magnetics_F7B", "magnetics_F8A", "magnetics_F8B",
    "magnetics_F9A", "magnetics_F9B", "magnetics_plasma_current",
]

MAST_COILS = [
    "magnetics_p2l_current", "magnetics_p2u_current",
    "magnetics_p3l_current", "magnetics_p3u_current",
    "magnetics_p4l_current", "magnetics_p4u_current",
    "magnetics_p5l_current", "magnetics_p5u_current",
    "magnetics_p6l_current", "magnetics_p6u_current",
    "magnetics_sol_current", "magnetics_tf_current",
    "magnetics_efps_current", "magnetics_plasma_current",
]

TARGET_COLUMNS = ("efit_psirz", "efit_q95", "efit_beta_n", "magnetics_dsep")

_GREEN_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def read_row(path, drop_targets: bool = False) -> pd.Series:
    """One shot as a pandas row.  ``drop_targets`` enforces inference-time input isolation."""
    if drop_targets:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(path).schema_arrow.names
        keep = [k for k in names if k.startswith(("coil_", "thomson_")) or
                (k.startswith("magnetics_") and "dsep" not in k) or
                k in ("efit_times", "efit_grid_R", "efit_grid_Z")]
        return pd.read_parquet(path, columns=keep).iloc[0]
    return pd.read_parquet(path).iloc[0]


def stack_2d(value: Any, dtype=np.float32) -> np.ndarray:
    """Convert arrow/pandas nested list data to a dense ``(time, channel)`` array."""
    try:
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 2:
            return arr
    except (TypeError, ValueError):
        pass
    return np.stack([np.asarray(x, dtype=dtype) for x in value])


def stack_psirz(value: Any) -> np.ndarray:
    """Convert nested flux maps to dense ``(T, 65, 65)`` float32."""
    try:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3:
            return arr
    except (TypeError, ValueError):
        pass
    return np.stack([np.stack([np.asarray(r, dtype=np.float32) for r in frame]) for frame in value])


def interp(times: Any, values: Any, target: np.ndarray) -> np.ndarray:
    """NaN-aware, zero-order-safe linear interpolation onto ``target``."""
    t = np.asarray(times, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    valid = np.isfinite(t) & np.isfinite(v)
    if valid.sum() == 0:
        return np.zeros(len(target), dtype=np.float32)
    if valid.sum() == 1:
        return np.full(len(target), v[valid][0], dtype=np.float32)
    return np.interp(target, t[valid], v[valid],
                     left=v[valid][0], right=v[valid][-1]).astype(np.float32)


def _interp_profile(times: Any, profiles: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.empty((len(target), profiles.shape[1]), dtype=np.float32)
    for col in range(profiles.shape[1]):
        out[:, col] = interp(times, profiles[:, col], target)
    return out


def _profile_features(profile: np.ndarray, kind: str) -> np.ndarray:
    """Fixed-width, geometry-agnostic summary of a time-aligned Thomson profile."""
    x = np.asarray(profile, dtype=np.float32)
    x = np.log10(np.clip(x, 1.0e12, None)) if kind == "ne" else np.log1p(np.clip(x, 0.0, None))
    finite = np.isfinite(x)
    x = np.where(finite, x, np.nan)
    with np.errstate(all="ignore"):
        q = np.nanquantile(x, [0.05, 0.20, 0.50, 0.80, 0.95], axis=1).T
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        lo = np.nanmean(x[:, :max(1, x.shape[1] // 4)], axis=1, keepdims=True)
        hi = np.nanmean(x[:, -max(1, x.shape[1] // 4):], axis=1, keepdims=True)
        span = np.nanmax(x, axis=1, keepdims=True) - np.nanmin(x, axis=1, keepdims=True)
        present = finite.mean(axis=1, keepdims=True)
    return np.nan_to_num(np.hstack([q, mean, std, lo, hi, span, present]),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def thomson_features(row, target: np.ndarray, source: str) -> np.ndarray:
    """48 machine-agnostic kinetic features at the EFIT timestamps."""
    pieces: list[np.ndarray] = []
    for system in ("core", "edge"):
        times_key = f"thomson_{system}_times"
        for measure in ("Te", "ne"):
            key = f"thomson_{system}_{measure}"
            if times_key not in row or key not in row:
                pieces.append(np.zeros((len(target), 11), dtype=np.float32))
                continue
            try:
                profiles = stack_2d(row[key])
                if (source == "MAST" and system == "core"
                        and profiles.shape[1] == len(row["thomson_core_R"]) + 1):
                    profiles = profiles[:, 1:]
                pieces.append(_profile_features(
                    _interp_profile(row[times_key], profiles, target), measure))
            except Exception:                        # noqa: BLE001 - a missing or ragged subsystem
                pieces.append(np.zeros((len(target), 11), dtype=np.float32))
    out = np.hstack(pieces)
    # Simple pressure proxies help betaN without committing to a machine's channel layout.
    pressure = np.hstack([
        (out[:, 5] + out[:, 16]).reshape(-1, 1),
        (out[:, 27] + out[:, 38]).reshape(-1, 1),
        (out[:, 5] + out[:, 16] + out[:, 27] + out[:, 38]).reshape(-1, 1),
        (out[:, 7] - out[:, 8]).reshape(-1, 1),
    ])
    return np.hstack([out, pressure]).astype(np.float32)


def d3d_magnetic_features(row, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous, derivative and causal-lag coil features for one DIII-D shot."""
    shared_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_time = fix_d3d_ip_times(row)

    def at(offset: float) -> np.ndarray:
        return np.column_stack([
            interp(ip_time if name == "magnetics_plasma_current" else shared_time,
                   row[name], target - offset) for name in D3D_COILS])

    instant = at(0.0)
    deriv = (np.gradient(instant, target.astype(np.float32), axis=0, edge_order=1)
             if len(target) > 2 else np.zeros_like(instant))
    # The last four terms describe machine-symmetric coil energy and phase, not new labels.
    top = instant[:, [2, 4, 6, 8, 10, 12, 14, 16, 18]].sum(axis=1)
    bot = instant[:, [3, 5, 7, 9, 11, 13, 15, 17, 19]].sum(axis=1)
    derived = np.column_stack([
        top + bot, top - bot, np.sqrt(np.mean(instant[:, 2:20] ** 2, axis=1)),
        instant[:, 1] / np.maximum(np.abs(instant[:, -1]), 1.0),
    ])
    return (np.hstack([instant, deriv, instant - at(20.0), instant - at(100.0), derived])
            .astype(np.float32), instant)


def d3d_input_features(row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(features[T, 140], currents[T, 21], efit_times[T])`` from test-legal columns only."""
    target = np.asarray(row["efit_times"], dtype=np.float32)
    mag, instant = d3d_magnetic_features(row, target)
    kin = thomson_features(row, target, "DIII-D")
    duration = max(float(target[-1] - target[0]), 1.0) if len(target) else 1.0
    phase = (target - target[0]) / duration
    clock = np.column_stack([target / 1000.0, phase,
                             np.sin(np.pi * phase), np.cos(np.pi * phase)]).astype(np.float32)
    features = np.nan_to_num(np.hstack([mag, kin, clock]), nan=0.0, posinf=0.0, neginf=0.0)
    return features, instant, target


def _profile_shape(prof: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """33 shape statistics of one ``(T, channel)`` profile stack, robust to NaN channels.

    The 11 quantile summaries in ``thomson_features`` carry a profile's moments but not its
    *shape*, and the equilibrium is driven by ``p(psi)``.  This adds where the peak sits, how
    peaked it is, and 20 coarse radial bins.
    """
    p = np.where(np.isfinite(prof), prof, np.nan)
    with np.errstate(all="ignore"):
        q = np.nanpercentile(p, [5, 25, 50, 75, 90, 95, 99], axis=1).T
        mean = np.nanmean(p, axis=1, keepdims=True)
        std = np.nanstd(p, axis=1, keepdims=True)
        mx = np.nanmax(p, axis=1, keepdims=True)
        total = np.nansum(p, axis=1, keepdims=True)
        idx = np.nanargmax(np.where(np.isfinite(p), p, -np.inf), axis=1)
        r_peak = radius[np.clip(idx, 0, len(radius) - 1)].reshape(-1, 1)
        peakedness = mx / np.maximum(np.abs(mean), 1e-9)
        edges = np.linspace(0, p.shape[1], 21).astype(int)
        bins = np.stack([np.nanmean(p[:, a:b], axis=1) for a, b in zip(edges[:-1], edges[1:])], 1)
    return np.nan_to_num(np.hstack([q, mean, std, mx, total, r_peak, peakedness, bins]),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def d3d_profile_block(row, times: np.ndarray) -> np.ndarray:
    """``(T, 132)`` radial-shape summaries of the four DIII-D Thomson profiles.

    Appended after the base 140 so a member can take either width from one cache: the first 140
    columns are exactly what the base feature vector has always been.
    """
    extra = []
    for system in ("core", "edge"):
        try:
            key = "thomson_core_R" if system == "core" else "thomson_edge_spatial"
            R = np.asarray(row[key], dtype=np.float64)
            t_ts = np.asarray(row[f"thomson_{system}_times"], dtype=np.float64)
            for measure in ("Te", "ne"):
                prof = np.stack([np.asarray(p, dtype=np.float64)
                                 for p in row[f"thomson_{system}_{measure}"]])
                n = min(len(t_ts), len(prof))
                aligned = np.stack([np.interp(times, t_ts[:n], np.nan_to_num(prof[:n, j]))
                                    for j in range(prof.shape[1])], axis=1)
                rr = R if len(R) == aligned.shape[1] else np.arange(aligned.shape[1], dtype=float)
                extra.append(_profile_shape(aligned, rr))
        except Exception:                            # noqa: BLE001 - a missing or ragged subsystem
            extra.extend([np.zeros((len(times), 33), np.float32)] * 2)
    return np.nan_to_num(np.hstack(extra), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def d3d_supervised(row, profile_block: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features, currents, flux targets and ``(q95, betaN)`` for one training shot."""
    features, instant, target = d3d_input_features(row)
    if profile_block:
        features = np.hstack([features, d3d_profile_block(row, target)])
    psi = stack_psirz(row["efit_psirz"])
    q = np.asarray(row["efit_q95"], dtype=np.float32)
    beta = np.asarray(row["efit_beta_n"], dtype=np.float32)
    n = min(len(target), len(psi), len(features), len(q), len(beta))
    return features[:n], instant[:n], psi[:n], np.column_stack([q[:n], beta[:n]])


def mast_input_features(row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MAST coil currents and kinetic summaries at the EFIT timestamps."""
    target = np.asarray(row["efit_times"], dtype=np.float32)
    time = np.asarray(row["magnetics_time"], dtype=np.float64)
    instant = np.column_stack([interp(time, row[name], target) for name in MAST_COILS])
    kin = thomson_features(row, target, "MAST")
    phase = (target - target.min()) / max(float(target.max() - target.min()), 1.0)
    clock = np.column_stack([target / 1000.0, phase, np.sin(np.pi * phase), np.cos(np.pi * phase)])
    return (np.nan_to_num(np.hstack([instant, kin, clock]), nan=0.0).astype(np.float32),
            instant, target)


def geometry_arrays(row) -> dict[str, np.ndarray]:
    return {key: np.asarray(row[key]) for key in
            ("coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
             "coil_angle1", "coil_angle2", "efit_grid_R", "efit_grid_Z")}


def filament_psi(grid_r: np.ndarray, grid_z: np.ndarray,
                 coil_r: float, coil_z: float) -> np.ndarray:
    """Poloidal-flux Green function [Wb/rad/A] for a circular filament.

    The finite conductor rectangles in the dataset are represented by their centroids; the small
    finite-width error is absorbed downstream, and the physical SI form transfers to MAST.
    """
    r, z = np.meshgrid(grid_r.astype(np.float64), grid_z.astype(np.float64))
    denom = (r + coil_r) ** 2 + (z - coil_z) ** 2
    m = np.clip(4.0 * r * coil_r / np.maximum(denom, 1e-16), 1e-12, 1.0 - 1e-10)
    bracket = (1.0 - m / 2.0) * ellipk(m) - ellipe(m)
    return (4.0e-7 * np.sqrt(r * coil_r) * bracket
            / np.maximum(np.sqrt(m), 1e-8)).astype(np.float32)


def green_basis(row, current_columns: Iterable[str]) -> tuple[list[str], np.ndarray]:
    """Point-filament Green maps aggregated per current input column.

    A MAST current is an ampere-turn total shared across its conductor rectangles, so its
    response is the mean of its turn-level maps.  DIII-D's lumped rectangles have one entry each.
    """
    columns = list(current_columns)
    geom = geometry_arrays(row)
    digest = hashlib.sha256()
    for key in ("coil_input_column", "coil_R", "coil_Z", "efit_grid_R", "efit_grid_Z"):
        digest.update(np.asarray(geom[key]).astype("U").tobytes() if key == "coil_input_column"
                      else np.asarray(geom[key], dtype=np.float32).tobytes())
    digest.update("|".join(columns).encode())
    key = digest.hexdigest()
    if key in _GREEN_CACHE:
        return _GREEN_CACHE[key]
    names = np.asarray(geom["coil_input_column"]).astype(str)
    maps = []
    for column in columns:
        inds = np.flatnonzero(names == column)
        if not len(inds):
            maps.append(np.zeros((65, 65), dtype=np.float32))
            continue
        maps.append(np.mean([filament_psi(geom["efit_grid_R"], geom["efit_grid_Z"],
                                          float(geom["coil_R"][j]), float(geom["coil_Z"][j]))
                             for j in inds], axis=0, dtype=np.float32))
    _GREEN_CACHE[key] = (columns, np.stack(maps))
    return _GREEN_CACHE[key]


def vacuum_psi(row, instant: np.ndarray, current_columns: list[str]) -> np.ndarray:
    """Physical vacuum-coil flux map for one shot, in Wb/rad."""
    _, basis = green_basis(row, current_columns)
    # input currents are kA / kA-turn; the Green function wants amps / ampere-turns.
    return np.einsum("tc,crz->trz", instant[:, :len(current_columns)] * 1000.0, basis,
                     optimize=True)

SIGNED = [c for c in D3D_COILS if c != "magnetics_bcoil"]   # Ip, F-coils, ECOILA share one toroidal direction


def ip_sign(row) -> float:
    """Sign of the shot's plasma current: the median Ip over the EFIT times on the corrected Ip time axis
    (organizer notice; inputs only)."""
    from nm1.data_fixes import fix_d3d_ip_times
    t = np.asarray(row["efit_times"], dtype=np.float64)
    ip = interp(fix_d3d_ip_times(row), row["magnetics_plasma_current"], t)
    s = float(np.sign(np.nanmedian(ip)))
    return s if s != 0 else 1.0


def sign_normalized(row):
    """A copy of a DIII-D row whose Ip, F-coil and ECOILA currents are multiplied by the shot's Ip sign (the
    toroidal-field coil keeps its sign): on reversed-current shots the inputs then look like a normal shot, and
    the target (stored psi, q95, beta_N, li) does not reverse.  Rows with positive Ip are returned unchanged."""
    s = ip_sign(row)
    if s > 0:
        return row
    out = row.copy()
    for c in SIGNED:
        out[c] = np.asarray(row[c], dtype=np.float64) * s
    return out

