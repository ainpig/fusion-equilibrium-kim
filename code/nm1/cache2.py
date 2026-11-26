"""Build the second feature block for every frame of the coefficient cache, row-aligned with it.

``cache.py`` stores the base 140(+132) features, the flux coefficients and the scalars.  This adds
``features2.extended_features`` for exactly the same frames, in exactly the same order, so a member
can take ``[base | ext]`` by concatenating two arrays.  Alignment is checked, not assumed: every
shot's frame count and every frame's EFIT time must match the coefficient cache, or the build
stops.

Also stored, for training only (they are targets, never inputs): EFIT's own ``li``, ``R_axis``,
``Z_axis``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from nm1 import DATA, CACHE_DATA
from nm1.cache import CACHE
from nm1.features import read_row, stack_psirz
from nm1.features2 import N_EXT, extended_features
from nm1.fetch import listing, local_path

CACHE2 = CACHE_DATA / "d3d_ext.h5"
AUX = ("efit_li", "efit_r_axis", "efit_z_axis")


def _one(job):
    index, path, expect = job
    try:
        row = read_row(path)
        target = np.asarray(row["efit_times"], dtype=np.float32)
        psi = stack_psirz(row["efit_psirz"])
        q = np.asarray(row["efit_q95"], dtype=np.float32)
        b = np.asarray(row["efit_beta_n"], dtype=np.float32)
        n = min(len(target), len(psi), len(q), len(b))
        good = np.isfinite(psi[:n].reshape(n, -1)).all(axis=1)
        del psi
        ext = extended_features(row, target[:n])[good]
        aux = np.column_stack([np.asarray(row[c], dtype=np.float32)[:n] if c in row.index
                               else np.full(n, np.nan, np.float32) for c in AUX])[good]
        t = (target[:n][good] / 1000.0).astype(np.float32)
        if len(ext) != expect:
            return index, None, f"frame count {len(ext)} != cache {expect}"
        return index, (ext, aux, t), ""
    except Exception as exc:                         # noqa: BLE001 - reported by the parent
        return index, None, f"{type(exc).__name__}: {exc}"


def build(out: Path = CACHE2, workers: int = 10, limit: int = 0) -> Path:
    files = listing("diii_d_train")
    started = time.perf_counter()
    with h5py.File(CACHE, "r") as h5:
        plan = {}
        for split in ("train", "val"):
            shot = h5[split]["shot"][:]
            clock = h5[split]["features"][:, 136]
            ids, first, counts = np.unique(shot, return_index=True, return_counts=True)
            order = np.argsort(first)
            plan[split] = (ids[order], first[order], counts[order], clock)
    with h5py.File(out, "w") as o:
        for split, (ids, first, counts, clock) in plan.items():
            if limit:
                ids, first, counts = ids[:limit], first[:limit], counts[:limit]
            total = int(counts.sum()) if limit else len(clock)
            g = o.create_group(split)
            d_ext = g.create_dataset("ext", (total, N_EXT), dtype="f4", chunks=(4096, N_EXT))
            d_aux = g.create_dataset("aux", (total, len(AUX)), dtype="f4", chunks=(8192, len(AUX)))
            jobs = [(int(i), local_path(files[int(i)]), int(c)) for i, c in zip(ids, counts)]
            pos = {int(i): int(f) for i, f in zip(ids, first)}
            done, bad = 0, []
            with ProcessPoolExecutor(workers) as pool:
                for index, res, err in pool.map(_one, jobs, chunksize=4):
                    done += 1
                    if res is None:
                        bad.append((index, err))
                        continue
                    ext, aux, t = res
                    a = pos[index]
                    if not np.allclose(t, clock[a:a + len(t)], atol=1e-6):
                        bad.append((index, "efit times disagree with the coefficient cache"))
                        continue
                    d_ext[a:a + len(ext)] = ext
                    d_aux[a:a + len(aux)] = aux
                    if done % 250 == 0 or done == len(jobs):
                        el = time.perf_counter() - started
                        print(f"  {split}: {done}/{len(jobs)} shots, {el / 60:.1f} min", flush=True)
            if bad:
                for index, err in bad[:20]:
                    print(f"  FAILED shot {index}: {err}", flush=True)
                raise SystemExit(f"{len(bad)} {split} shots could not be aligned")
        o.attrs["n_ext"] = N_EXT
        o.attrs["aux"] = ",".join(AUX)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) in "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    out.with_suffix(".json").write_text(json.dumps({"n_ext": N_EXT, "aux": AUX}), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=CACHE2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.workers, a.limit)


if __name__ == "__main__":
    main()
