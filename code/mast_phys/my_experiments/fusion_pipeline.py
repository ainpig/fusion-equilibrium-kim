"""Data and physics utilities for the Fusion Equilibrium Challenge.

This module deliberately keeps the competition data out of source control.  It downloads only
the selected public parquet files into ``my_experiments/data_cache`` and turns DIII-D training
shots into a compact HDF5 training cache.  The feature construction uses only columns available
on the public test split.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import requests
from scipy.special import ellipe, ellipk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_fixes import align_mast_thomson_core, fix_d3d_ip_times


REPO_ID = "Sophelio/fusion-equilibrium-challenge"
HERE = Path(__file__).resolve().parent
# The archive location, overridable so the relocated fitting scripts can be pointed at this
# project's own data directory without editing them.
CACHE_ROOT = Path(os.environ.get("FUSION_CACHE_ROOT") or (HERE / "data_cache"))

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
    "magnetics_efps_current",
    # v14: the vessel's induced current (vessel_v14.py), a derived input column; it sits before the
    # plasma current so curr[:, -1] stays Ip and curr[:, 11] the TF current everywhere
    "vessel_current_v14",
    "magnetics_plasma_current",
]

_GREEN_CACHE: dict[str, tuple[list[str], np.ndarray]] = {}


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"authorization": f"Bearer {token}"} if token else {}


def list_config_files(config: str, retries: int = 6) -> list[dict[str, Any]]:
    """Return parquet metadata in the exact (Hub-tree) order for one split config.

    The order is load-bearing: submissions key shots by position (`shot_0007_psirz`), so a listing
    in a different order would silently mis-align every prediction with its ground truth. That is
    why this never falls back to a local directory scan -- only to a listing this function itself
    recorded from the Hub.

    The Hub answers 429 under load, which used to abort a rebuild whose parquets were all already
    cached. So: retry with backoff, persist each successful listing, and fall back to the recorded
    one when the API stays unavailable.
    """
    cache = CACHE_ROOT / "listings" / f"{config}.json"
    if os.environ.get("FUSION_OFFLINE") == "1":
        if not cache.exists():
            raise FileNotFoundError(f"Offline mode requires recorded Hub listing: {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    url0 = (
        f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/"
        f"data/{config}?recursive=false&expand=false&limit=1000"
    )
    session = requests.Session()
    last: Exception | None = None
    for attempt in range(retries):
        files: list[dict[str, Any]] = []
        url = url0
        try:
            while url:
                response = session.get(url, headers=_auth_headers(), timeout=60)
                response.raise_for_status()
                page = response.json()
                files.extend(x for x in page
                             if x.get("type") == "file" and x["path"].endswith(".parquet"))
                link = response.headers.get("link", "")
                match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                url = match.group(1) if match else ""
        except Exception as exc:                       # noqa: BLE001 - retried, then reported
            last = exc
            wait = min(60, 5 * 2 ** attempt)
            print(f"  listing {config}: {type(exc).__name__}, retrying in {wait}s "
                  f"({attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            continue
        if not files:
            raise RuntimeError(f"No parquet files found for {config}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(files), encoding="utf-8")
        return files
    if cache.exists():
        files = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  listing {config}: Hub unavailable ({type(last).__name__}); using the recorded "
              f"listing of {len(files)} files from {cache}", flush=True)
        return files
    raise RuntimeError(f"could not list {config} and no recorded listing at {cache}") from last


def _local_path(repo_path: str, root: Path = CACHE_ROOT) -> Path:
    return root / repo_path


def download_file(meta: dict[str, Any], root: Path = CACHE_ROOT) -> Path:
    """Fetch a public parquet atomically, reusing a complete cached copy when possible."""
    path = _local_path(meta["path"], root)
    expected = int(meta.get("size") or 0)
    if path.exists() and (not expected or path.stat().st_size == expected):
        return path
    if os.environ.get("FUSION_OFFLINE") == "1":
        raise FileNotFoundError(f"Offline mode: missing or incomplete cached input {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{meta['path']}?download=true"
    tmp = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            tmp.unlink(missing_ok=True)
            with requests.get(url, headers=_auth_headers(), stream=True, timeout=(20, 300)) as response:
                response.raise_for_status()
                with open(tmp, "wb") as out:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            out.write(block)
            actual = tmp.stat().st_size
            if expected and actual != expected:
                raise RuntimeError(f"partial download: expected {expected}, got {actual}")
            tmp.replace(path)
            return path
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after retries for {meta['path']}: {last_error}")


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


def _interp(times: Any, values: Any, target: np.ndarray) -> np.ndarray:
    """NaN-aware, zero-order-safe linear interpolation onto ``target``."""
    t = np.asarray(times, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    valid = np.isfinite(t) & np.isfinite(v)
    if valid.sum() == 0:
        return np.zeros(len(target), dtype=np.float32)
    if valid.sum() == 1:
        return np.full(len(target), v[valid][0], dtype=np.float32)
    return np.interp(target, t[valid], v[valid], left=v[valid][0], right=v[valid][-1]).astype(np.float32)


def _interp_profile(times: Any, profiles: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Interpolate a Thomson ``(time, channel)`` profile without leaking NaNs."""
    out = np.empty((len(target), profiles.shape[1]), dtype=np.float32)
    for col in range(profiles.shape[1]):
        out[:, col] = _interp(times, profiles[:, col], target)
    return out


def _profile_features(profile: np.ndarray, kind: str) -> np.ndarray:
    """Fixed-width, geometry-agnostic summary of a time-aligned Thomson profile."""
    x = np.asarray(profile, dtype=np.float32)
    if kind == "ne":
        x = np.log10(np.clip(x, 1.0e12, None))
    else:
        x = np.log1p(np.clip(x, 0.0, None))
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
    return np.nan_to_num(np.hstack([q, mean, std, lo, hi, span, present]), nan=0.0,
                         posinf=0.0, neginf=0.0).astype(np.float32)


def thomson_features(row: pd.Series | dict[str, Any], target: np.ndarray, source: str) -> np.ndarray:
    """Return 48 machine-agnostic kinetic features at EFIT timestamps."""
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
                if source == "MAST" and system == "core" and profiles.shape[1] == len(row["thomson_core_R"]) + 1:
                    profiles = profiles[:, 1:]
                aligned = _interp_profile(row[times_key], profiles, target)
                pieces.append(_profile_features(aligned, measure))
            except Exception:
                pieces.append(np.zeros((len(target), 11), dtype=np.float32))
    # Simple pressure proxies help beta_N without committing to a machine's channel layout.
    out = np.hstack(pieces)
    pressure = np.hstack([
        (out[:, 5] + out[:, 16]).reshape(-1, 1),
        (out[:, 27] + out[:, 38]).reshape(-1, 1),
        (out[:, 5] + out[:, 16] + out[:, 27] + out[:, 38]).reshape(-1, 1),
        (out[:, 7] - out[:, 8]).reshape(-1, 1),
    ])
    return np.hstack([out, pressure]).astype(np.float32)


def d3d_magnetic_features(row: pd.Series | dict[str, Any], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous, derivative and causal-lag coil features for a DIII-D frame."""
    shared_time = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_time = fix_d3d_ip_times(row)
    current = []
    for name in D3D_COILS:
        current.append(_interp(ip_time if name == "magnetics_plasma_current" else shared_time,
                               row[name], target))
    instant = np.column_stack(current)
    deriv = np.gradient(instant, target.astype(np.float32), axis=0, edge_order=1) if len(target) > 2 else np.zeros_like(instant)
    lag20 = np.column_stack([
        _interp(ip_time if name == "magnetics_plasma_current" else shared_time, row[name], target - 20.0)
        for name in D3D_COILS
    ])
    lag100 = np.column_stack([
        _interp(ip_time if name == "magnetics_plasma_current" else shared_time, row[name], target - 100.0)
        for name in D3D_COILS
    ])
    # The last four terms describe machine-symmetric coil energy and phase, rather than new labels.
    top = instant[:, [2, 4, 6, 8, 10, 12, 14, 16, 18]].sum(axis=1)
    bot = instant[:, [3, 5, 7, 9, 11, 13, 15, 17, 19]].sum(axis=1)
    ip = instant[:, -1]
    tf = instant[:, 1]
    derived = np.column_stack([
        top + bot, top - bot, np.sqrt(np.mean(instant[:, 2:20] ** 2, axis=1)),
        tf / np.maximum(np.abs(ip), 1.0),
    ])
    return np.hstack([instant, deriv, instant - lag20, instant - lag100, derived]).astype(np.float32), instant


def d3d_input_features(row: pd.Series | dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the DIII-D feature representation using only columns present on public test."""
    target = np.asarray(row["efit_times"], dtype=np.float32)
    mag, instant = d3d_magnetic_features(row, target)
    kin = thomson_features(row, target, "DIII-D")
    duration = max(float(target[-1] - target[0]), 1.0) if len(target) else 1.0
    phase = (target - target[0]) / duration
    clock = np.column_stack([target / 1000.0, phase, np.sin(np.pi * phase), np.cos(np.pi * phase)]).astype(np.float32)
    features = np.nan_to_num(np.hstack([mag, kin, clock]), nan=0.0, posinf=0.0, neginf=0.0)
    return features, instant, target


def d3d_features(row: pd.Series | dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build legal model features and targets from a full DIII-D training row."""
    features, instant, target = d3d_input_features(row)
    psi = stack_psirz(row["efit_psirz"])
    q = np.asarray(row.get("efit_q95", np.full(len(target), np.nan)), dtype=np.float32)
    beta = np.asarray(row.get("efit_beta_n", np.full(len(target), np.nan)), dtype=np.float32)
    n = min(len(target), len(psi), len(features), len(q), len(beta))
    return features[:n], instant[:n], psi[:n], np.column_stack([q[:n], beta[:n]])


def mast_features(row: pd.Series | dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Feature representation available for MAST physics/calibration inference."""
    target = np.asarray(row["efit_times"], dtype=np.float32)
    time = np.asarray(row["magnetics_time"], dtype=np.float64)
    instant = np.column_stack([_interp(time, row[name], target) for name in MAST_COILS])
    kin = thomson_features(row, target, "MAST")
    phase = (target - target.min()) / max(float(target.max() - target.min()), 1.0)
    extra = np.column_stack([target / 1000.0, phase, np.sin(np.pi * phase), np.cos(np.pi * phase)])
    return np.nan_to_num(np.hstack([instant, kin, extra]), nan=0.0), instant


def geometry_arrays(row: pd.Series | dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(row[key])
        for key in ("coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                    "coil_angle1", "coil_angle2", "efit_grid_R", "efit_grid_Z")
    }


def save_geometry(row: pd.Series | dict[str, Any], source: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    out = CACHE_ROOT / f"{source.lower().replace('-', '_')}_geometry.npz"
    if not out.exists():
        np.savez_compressed(out, **geometry_arrays(row))
    return out


def filament_psi(grid_r: np.ndarray, grid_z: np.ndarray, coil_r: float, coil_z: float) -> np.ndarray:
    """Poloidal-flux Green function [Wb/rad/A] for a circular filament.

    The finite conductor rectangles in the dataset are represented by their centroids here; the
    learned residual handles the small finite-width error, while this feature is transferable to
    MAST because it uses physical coordinates and SI units.
    """
    r, z = np.meshgrid(grid_r.astype(np.float64), grid_z.astype(np.float64))
    denom = (r + coil_r) ** 2 + (z - coil_z) ** 2
    m = np.clip(4.0 * r * coil_r / np.maximum(denom, 1e-16), 1e-12, 1.0 - 1e-10)
    k = np.sqrt(m)
    bracket = (1.0 - m / 2.0) * ellipk(m) - ellipe(m)
    return (4.0e-7 * np.sqrt(r * coil_r) * bracket / np.maximum(k, 1e-8)).astype(np.float32)


def green_basis(row: pd.Series | dict[str, Any], current_columns: Iterable[str]) -> tuple[list[str], np.ndarray]:
    """Aggregate point-filament Green maps per current input column.

    A MAST current is an ampere-turn total shared across its conductor rectangles, so its response
    is the mean of its turn-level Green maps.  DIII-D's lumped rectangles have one entry each.
    """
    columns = list(current_columns)
    geom = geometry_arrays(row)
    digest = hashlib.sha256()
    for key in ("coil_input_column", "coil_R", "coil_Z", "efit_grid_R", "efit_grid_Z"):
        digest.update(np.asarray(geom[key]).astype("U").tobytes() if key == "coil_input_column"
                      else np.asarray(geom[key], dtype=np.float32).tobytes())
    digest.update("|".join(columns).encode())
    cache_key = digest.hexdigest()
    cached = _GREEN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    names = np.asarray(geom["coil_input_column"]).astype(str)
    maps = []
    for column in columns:
        inds = np.flatnonzero(names == column)
        if not len(inds):
            maps.append(np.zeros((65, 65), dtype=np.float32))
            continue
        pieces = [filament_psi(geom["efit_grid_R"], geom["efit_grid_Z"],
                               float(geom["coil_R"][j]), float(geom["coil_Z"][j])) for j in inds]
        maps.append(np.mean(pieces, axis=0, dtype=np.float32))
    result = columns, np.stack(maps)
    _GREEN_CACHE[cache_key] = result
    return result


def vacuum_psi(row: pd.Series | dict[str, Any], instant: np.ndarray, current_columns: list[str]) -> np.ndarray:
    """Physical vacuum-coil flux map for one shot, in Wb/rad."""
    _, basis = green_basis(row, current_columns)
    # input currents are kA / kA-turn; the Green function uses amps / ampere-turns.
    return np.einsum("tc,crz->trz", instant[:, :len(current_columns)] * 1000.0, basis, optimize=True)


def _create_group(h5: h5py.File, name: str, feature_dim: int) -> h5py.Group:
    if name in h5:
        return h5[name]
    group = h5.create_group(name)
    options = dict(maxshape=(None,), chunks=True, compression="lzf")
    group.create_dataset("features", shape=(0, feature_dim), maxshape=(None, feature_dim),
                         chunks=(256, feature_dim), compression="lzf", dtype="f4")
    group.create_dataset("currents", shape=(0, len(D3D_COILS)), maxshape=(None, len(D3D_COILS)),
                         chunks=(256, len(D3D_COILS)), compression="lzf", dtype="f4")
    group.create_dataset("targets", shape=(0, 65, 65), maxshape=(None, 65, 65),
                         chunks=(64, 65, 65), compression="lzf", dtype="f2")
    group.create_dataset("scalars", shape=(0, 2), maxshape=(None, 2), chunks=(256, 2),
                         compression="lzf", dtype="f4")
    group.create_dataset("path_index", shape=(0,), maxshape=(None,), chunks=(1024,),
                         compression="lzf", dtype="i4")
    return group


def _append(group: h5py.Group, features: np.ndarray, currents: np.ndarray, targets: np.ndarray,
            scalars: np.ndarray, path_index: int) -> None:
    n0 = len(group["features"])
    n1 = n0 + len(features)
    for key, values in (("features", features), ("currents", currents), ("targets", targets),
                        ("scalars", scalars), ("path_index", np.full(len(features), path_index, dtype=np.int32))):
        ds = group[key]
        ds.resize((n1,) + ds.shape[1:])
        ds[n0:n1] = values


def _is_val(path: str, val_fraction: float) -> bool:
    value = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < val_fraction


def prepare_d3d_cache(n_shots: int, seed: int, workers: int = 8, val_fraction: float = 0.12,
                      cache_path: Path | None = None) -> Path:
    """Download a deterministic random DIII-D subset and append legal features to HDF5."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path or CACHE_ROOT / "d3d_supervised.h5"
    files = list_config_files("diii_d_train")
    rng = np.random.default_rng(seed)
    choice = np.sort(rng.choice(len(files), size=min(n_shots, len(files)), replace=False))
    selected = [(int(i), files[int(i)]) for i in choice]
    with h5py.File(cache_path, "a") as h5:
        existing = set()
        for grp_name in ("train", "val"):
            if grp_name in h5:
                existing.update(np.asarray(h5[grp_name]["path_index"], dtype=np.int32).tolist())
        remaining = [(i, meta) for i, meta in selected if i not in existing]
        if not remaining:
            print(f"cache already has requested selection: {cache_path}")
            return cache_path
        print(f"Preparing {len(remaining)} DIII-D shots ({len(existing)} already cached) ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[int, Future[Path]] = {i: pool.submit(download_file, meta) for i, meta in remaining}
            for done, (index, meta) in enumerate(remaining, start=1):
                local = futures[index].result()
                try:
                    row = pd.read_parquet(local).iloc[0]
                    features, currents, targets, scalars = d3d_features(row)
                    valid = np.isfinite(features).all(axis=1) & np.isfinite(targets.reshape(len(targets), -1)).all(axis=1)
                    features, currents, targets, scalars = (x[valid] for x in (features, currents, targets, scalars))
                    if not len(features):
                        raise RuntimeError("no finite training frames")
                    group = _create_group(h5, "val" if _is_val(meta["path"], val_fraction) else "train", features.shape[1])
                    _append(group, features, currents, targets, scalars, index)
                    save_geometry(row, "DIII-D")
                except Exception as exc:
                    print(f"  SKIP {meta['path']}: {type(exc).__name__}: {exc}")
                if done % 25 == 0 or done == len(remaining):
                    train_n = len(h5["train"]["features"]) if "train" in h5 else 0
                    val_n = len(h5["val"]["features"]) if "val" in h5 else 0
                    print(f"  {done}/{len(remaining)} shots; frames train={train_n:,}, val={val_n:,}")
        h5.attrs["feature_dim"] = int(h5["train"]["features"].shape[1])
        h5.attrs["feature_version"] = "v1-coils-kinetics-lags"
    return cache_path


def load_row(meta: dict[str, Any]) -> pd.Series:
    return pd.read_parquet(download_file(meta)).iloc[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare an HDF5 cache for the DIII-D model.")
    parser.add_argument("--n-shots", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    print(prepare_d3d_cache(args.n_shots, args.seed, args.workers, cache_path=args.cache))
