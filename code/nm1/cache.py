"""Turn the DIII-D training parquets into a PCA-coefficient cache, in two passes.

Pass 1 (``basis``) fits the 256-component flux basis on a shot-level sample.
Pass 2 (``build``) walks every shot once and stores, per EFIT frame, the 140 input features, the
21 raw coil currents, the flux map's coefficients in that basis, and ``(q95, betaN)``.

Only coefficients are stored.  The regressor never sees a flux map any other way, and 1.38 M
frames of coefficients are 1.4 GB instead of 23 GB of maps -- which is what lets the whole cache
sit on the GPU and an epoch take seconds.

Shots are visited in the recorded Hub listing order and the train/validation split is the hash of
the file name alone, so both are stable across rebuilds and across machines.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from nm1 import ARTIFACTS, DATA, CACHE_DATA
from nm1.features import D3D_COILS, d3d_supervised, read_row
from nm1.fetch import is_val, listing, local_path
from nm1.model import green_amps

CACHE = CACHE_DATA / "d3d_coeff.h5"
BASIS = ARTIFACTS / "flux_basis.pt"


def training_shots(split: str = "train", allow_partial: bool = False) -> list[Path]:
    """Existing parquet paths for one split, in recorded listing order."""
    files = listing("diii_d_train")
    keep = [f for f in files
            if (split == "all") or (is_val(Path(f["path"]).name) == (split == "val"))]
    paths = [local_path(f) for f in keep]
    missing = [p for p in paths if not p.exists()]
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} of {len(paths)} {split} parquets are missing "
                         f"(first: {missing[0].name}); run nm1.fetch first, or pass "
                         f"--allow-partial for a development run")
    if missing:
        print(f"WARNING: partial {split} split -- {len(paths) - len(missing)} of {len(paths)} "
              f"shots present. This is a development run, not the real one.", flush=True)
    return [p for p in paths if p.exists()]


def fit_basis(n_components: int = 256, pca_shots: int = 1500, pca_frames: int = 240_000,
              seed: int = 20260920, workers: int = 6, out: Path = BASIS,
              allow_partial: bool = False) -> Path:
    """Randomized PCA of the flux maps, fitted on training shots only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = training_shots("train", allow_partial)
    pick = paths if len(paths) <= pca_shots else [
        paths[i] for i in np.linspace(0, len(paths) - 1, pca_shots).astype(int)]
    per_shot = max(1, pca_frames // len(pick))
    rng = np.random.default_rng(seed)
    print(f"basis: sampling <={per_shot} frames from each of {len(pick)} shots "
          f"({len(paths)} training shots available)", flush=True)

    def sample(path: Path) -> np.ndarray:
        try:
            _, _, psi, _ = d3d_supervised(read_row(path))
        except Exception as exc:                    # noqa: BLE001 - one bad shot must not abort
            print(f"  skip {path.name}: {type(exc).__name__} {exc}", flush=True)
            return np.zeros((0, 65 * 65), np.float32)
        good = np.isfinite(psi.reshape(len(psi), -1)).all(axis=1)
        psi = psi[good]
        if len(psi) > per_shot:
            psi = psi[np.linspace(0, len(psi) - 1, per_shot).astype(int)]
        return psi.reshape(len(psi), -1)

    # Written straight into one preallocated block: concatenating a list of 1,500 arrays would
    # hold two copies of several GB at the moment it joins them.
    flat = np.empty((pca_frames + per_shot, 65 * 65), dtype=np.float32)
    filled, started = 0, time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ahead = {i: pool.submit(sample, pick[i]) for i in range(min(2 * workers, len(pick)))}
        for done in range(1, len(pick) + 1):
            block = ahead.pop(done - 1).result()
            nxt = done - 1 + 2 * workers
            if nxt < len(pick):
                ahead[nxt] = pool.submit(sample, pick[nxt])
            take = min(len(block), len(flat) - filled)
            flat[filled:filled + take] = block[:take]
            filled += take
            if done % 200 == 0 or done == len(pick):
                print(f"  {done}/{len(pick)} shots, {filled:,} frames, "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
            if filled == len(flat):
                break
    flat = flat[:filled]
    if len(flat) > pca_frames:
        flat = flat[rng.choice(len(flat), pca_frames, replace=False)]
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    matrix = torch.from_numpy(flat - mean).to(device)
    q = min(n_components, matrix.shape[0] - 1, matrix.shape[1])
    print(f"  randomized PCA: {matrix.shape[0]:,} frames x {matrix.shape[1]} pixels, q={q}",
          flush=True)
    with torch.no_grad():
        _, _, v = torch.pca_lowrank(matrix, q=q, center=False, niter=4)
        coeff = matrix @ v
        coeff_std = coeff.std(dim=0).clamp_min(1e-5).cpu().numpy().astype(np.float32)
        # float64 accumulation: the residual is ~1e-5 of the total over 40 M+ terms, which a
        # float32 sum cannot resolve.
        # In row blocks, so an 8 GB card does not hold a second copy of the matrix.
        residual, total = 0.0, 0.0
        for a in range(0, matrix.shape[0], 16384):
            m = matrix[a:a + 16384]
            residual += float(((m - coeff[a:a + 16384] @ v.T).double() ** 2).sum())
            total += float((m.double() ** 2).sum())
        explained = 1.0 - residual / total
    components = v.T.cpu().numpy().astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pca_mean": mean, "pca_components": components, "coeff_std": coeff_std,
                "n_components": int(q), "pca_shots": len(pick), "pca_frames": int(len(flat)),
                "seed": seed, "explained_variance": explained}, out)
    print(f"  basis explains {explained:.8f} of the sampled flux variance -> {out}", flush=True)
    return out


def _shot_arrays(path: Path, mean: torch.Tensor, comps: torch.Tensor, device: torch.device):
    """One shot's cache rows, or ``None`` if it has nothing usable.

    A shot that cannot be parsed is skipped and counted rather than aborting a 6,000-shot build.
    """
    try:
        # 272 columns: the base 140 first, then the radial profile-shape block. A member takes
        # whichever width it declares, so both feature sets come out of one pass.
        features, currents, psi, scalars = d3d_supervised(read_row(path), profile_block=True)
        good = (np.isfinite(features).all(axis=1)
                & np.isfinite(psi.reshape(len(psi), -1)).all(axis=1))
        if good.sum() < 3:
            return None
        features, currents, scalars = features[good], currents[good], scalars[good]
        flat = torch.from_numpy(psi[good].reshape(int(good.sum()), -1)).to(device)
        coeff = ((flat - mean) @ comps.T).cpu().numpy().astype(np.float32)
        return features, currents, coeff, scalars
    except Exception as exc:                        # noqa: BLE001 - counted and reported
        print(f"  skip {path.name}: {type(exc).__name__} {exc}", flush=True)
        return None


def build(basis: Path = BASIS, out: Path = CACHE, workers: int = 6,
          limit: int = 0) -> Path:
    """Project every training frame onto the basis and store the cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(basis, map_location="cpu", weights_only=False)
    mean = torch.from_numpy(state["pca_mean"]).to(device)
    comps = torch.from_numpy(state["pca_components"]).to(device)
    n_comp = comps.shape[0]

    files = listing("diii_d_train")
    items = [(i, local_path(f), is_val(Path(f["path"]).name)) for i, f in enumerate(files)]
    items = [(i, p, v) for i, p, v in items if p.exists()]
    if limit:
        items = items[:limit]
    print(f"cache: {len(items)} shots present of {len(files)} listed, {n_comp} components "
          f"-> {out}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    started, kept, skipped = time.perf_counter(), 0, 0
    with h5py.File(out, "w") as h5:
        groups: dict[str, h5py.Group] = {}
        # A bounded read-ahead window: pool.map would submit all 6,207 shots at once and keep
        # every finished block alive until the writer caught up.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ahead = {i: pool.submit(_shot_arrays, items[i][1], mean, comps, device)
                     for i in range(min(2 * workers, len(items)))}
            for done in range(1, len(items) + 1):
                index, path, val = items[done - 1]
                result = ahead.pop(done - 1).result()
                nxt = done - 1 + 2 * workers
                if nxt < len(items):
                    ahead[nxt] = pool.submit(_shot_arrays, items[nxt][1], mean, comps, device)
                if result is None:
                    skipped += 1
                    continue
                features, currents, coeff, scalars = result
                name = "val" if val else "train"
                if name not in groups:
                    g = h5.create_group(name)
                    for key, width, dtype in (("features", features.shape[1], "f4"),
                                              ("currents", currents.shape[1], "f4"),
                                              ("coeff", n_comp, "f4"),
                                              ("scalars", 2, "f4")):
                        g.create_dataset(key, (0, width), maxshape=(None, width), dtype=dtype,
                                         chunks=(4096, width))
                    g.create_dataset("shot", (0,), maxshape=(None,), dtype="i4", chunks=(8192,))
                    groups[name] = g
                g = groups[name]
                for key, arr in (("features", features), ("currents", currents),
                                 ("coeff", coeff), ("scalars", scalars),
                                 ("shot", np.full(len(features), index, np.int32))):
                    d = g[key]
                    n = d.shape[0]
                    d.resize(n + len(arr), axis=0)
                    d[n:] = arr
                kept += len(features)
                if done % 250 == 0 or done == len(items):
                    h5.flush()
                    el = time.perf_counter() - started
                    print(f"  {done}/{len(items)} shots, {kept:,} frames, {el / 60:.1f} min "
                          f"(~{el / done * (len(items) - done) / 60:.0f} min left)", flush=True)
        for name, g in groups.items():
            print(f"  {name}: {g['features'].shape[0]:,} frames, "
                  f"{len(np.unique(g['shot'][:])):,} shots")
        # The analytic coil response of this machine, so a member can be trained on the vacuum
        # block without re-deriving the geometry. DIII-D coil geometry is fixed across shots;
        # only the Thomson chord positions move, and those are not in this map.
        h5.create_dataset("green", data=green_amps(read_row(items[0][1]), D3D_COILS[:-1]))
        h5.attrs["n_components"] = n_comp
        h5.attrs["basis"] = str(basis)
        h5.attrs["skipped_shots"] = skipped
    print(f"cache: {kept:,} frames, {skipped} shots skipped, "
          f"{out.stat().st_size / 1e9:.2f} GB in {(time.perf_counter() - started) / 60:.0f} min",
          flush=True)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"frames": kept, "skipped_shots": skipped, "n_components": n_comp,
         "basis": str(basis), "shots": len(items)}, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("basis", "build", "all"))
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--pca-shots", type=int, default=1500)
    ap.add_argument("--pca-frames", type=int, default=240_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--basis", type=Path, default=BASIS)
    ap.add_argument("--out", type=Path, default=CACHE)
    ap.add_argument("--allow-partial", action="store_true",
                    help="development runs only: proceed with whatever shots are downloaded")
    a = ap.parse_args()
    if a.stage in ("basis", "all"):
        fit_basis(a.components, a.pca_shots, a.pca_frames, workers=a.workers, out=a.basis,
                  allow_partial=a.allow_partial)
    if a.stage in ("build", "all"):
        build(a.basis, a.out, a.workers, a.limit)


if __name__ == "__main__":
    main()
