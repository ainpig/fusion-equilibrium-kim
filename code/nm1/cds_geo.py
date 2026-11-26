"""The current-diffusion block on the flux coordinate of an equilibrium ("CDS-geo").

`diffusion.py` places every Thomson channel on a fixed ellipse. Placed instead on sqrt(psi_N) of
the frame's own equilibrium, the simulated li follows EFIT's li at within-shot correlation 0.70
against 0.14 (40 held-out shots) -- and a *predicted* equilibrium does as well as the true one,
because psi_N is blind to the flux offset and scale that carry most of a prediction's error.

So this is a second stage. At inference the equilibrium is the first-stage ensemble's own flux
map for the shot, and the held-out rows use exactly that, so their scores are honest. The
training rows use an out-of-fold first stage (stage1.oof_coefficients) when the recipe names
one: on the stored flux instead, the second stage learns from the true geometry and meets only
a predicted one at test time, which costs more than the block brings.

    python -m nm1.cds_geo --oof        # both splits; val on the first-stage blend, train out-of-fold
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from nm1 import CACHE_DATA
from nm1.diffusion import COLUMNS, N_CDS, FluxMap, diffusion_features

CDSG_CACHE = CACHE_DATA / "d3d_cdsg.h5"
_B: dict = {}


def _init(basis_path: str) -> None:
    import torch
    state = torch.load(basis_path, map_location="cpu", weights_only=False)
    _B["mean"] = state["pca_mean"].astype(np.float64)
    _B["comps"] = state["pca_components"].astype(np.float64)


def features_from_psi(row, efit_times_ms: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The CDS block for one shot with the Thomson channels on ``psi``'s flux coordinate."""
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    fmap = FluxMap(efit_times_ms, psi, R, Z)
    return diffusion_features(row, efit_times_ms, xmap=fmap)


def _job(job):
    k, path, times_s, coeff = job
    try:
        import pandas as pd
        row = pd.read_parquet(path, columns=list(COLUMNS) + ["efit_grid_R", "efit_grid_Z"]).iloc[0]
        psi = (_B["mean"] + coeff.astype(np.float64) @ _B["comps"]).reshape(-1, 65, 65)
        return k, features_from_psi(row, np.asarray(times_s) * 1000.0, psi), ""
    except Exception as exc:                        # noqa: BLE001 - reported by the parent
        return k, None, f"{type(exc).__name__}: {exc}"


STAGE1 = None          # the first stage: the manifest's stage1_weights blend (members.json)


def build(out: Path = CDSG_CACHE, workers: int = 9, splits=("val", "train"),
          stage1=STAGE1, oof: list[list[str]] | None = None) -> Path:
    """Held-out rows on the first stage's own prediction, train rows on the out-of-fold first
    stage ``oof`` (stage1.oof_members) -- or on the stored flux when it is None. Both computed
    here from checkpoints (nm1/stage1.py), never read from a file."""
    from nm1.cache import BASIS, CACHE
    from nm1.fetch import listing, local_path
    files = listing("diii_d_train")
    started = time.perf_counter()
    first = {}
    if "val" in splits:
        if stage1 is None and oof:
            from nm1.stage1 import val_oof_coefficients
            first["val"] = val_oof_coefficients(oof)
            print(f"  out-of-fold first stage on the held-out rows: {first['val'].shape}", flush=True)
        else:
            from nm1.stage1 import val_coefficients
            first["val"] = val_coefficients(stage1, "stage1_weights")
            print(f"  first stage ({stage1}) on the held-out rows: {first['val'].shape}", flush=True)
    if "train" in splits and oof:
        from nm1.stage1 import oof_coefficients
        first["train"] = oof_coefficients(oof)
        print(f"  out-of-fold first stage ({len(oof)} folds) on the training rows: "
              f"{first['train'].shape}", flush=True)
    with h5py.File(CACHE, "r") as h5, h5py.File(out, "a") as o:
        for split in splits:
            shot = h5[split]["shot"][:]
            clock = h5[split]["features"][:, 136].astype(np.float64)
            coeff_all = first[split] if split in first else h5[split]["coeff"]
            if split == "val" and split not in first:
                raise SystemExit("the held-out rows need the first stage")
            if len(coeff_all) != len(shot):
                raise SystemExit(f"{split} coefficients have {len(coeff_all)} rows, "
                                 f"the cache {len(shot)}")
            b = np.flatnonzero(np.diff(shot)) + 1
            st = np.concatenate([[0], b])
            en = np.concatenate([b, [len(shot)]])
            if split in o:
                del o[split]
            d = o.create_group(split).create_dataset("cds", (len(shot), N_CDS), dtype="f4")

            def jobs():
                for k, (s, e) in enumerate(zip(st, en)):
                    yield (k, local_path(files[int(shot[s])]), clock[s:e],
                           np.asarray(coeff_all[s:e], dtype=np.float32))
            bad = []
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(BASIS),)) as pool:
                for done, (k, res, err) in enumerate(pool.map(_job, jobs(), chunksize=2), 1):
                    if res is None:
                        bad.append((k, err))
                        continue
                    d[st[k]:en[k]] = res
                    if done % 250 == 0 or done == len(st):
                        print(f"  {split}: {done}/{len(st)} shots, "
                              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
            if bad:
                for k, err in bad[:20]:
                    print(f"  FAILED shot {k}: {err}", flush=True)
                raise SystemExit(f"{len(bad)} {split} shots failed")
        o.attrs["n_cds"] = N_CDS
        if "val" in splits:
            o.attrs["val_equilibrium"] = ("out-of-fold first stage, computed in-process" if stage1 is None else "first-stage blend, computed in-process")
        if "train" in splits:
            o.attrs["train_equilibrium"] = (
                f"out-of-fold first stage, {len(oof)} folds: "
                + "; ".join(",".join(Path(p).stem for p in f) for f in oof)
                if oof else "stored flux (coefficient cache)")
        meta = {k: str(v) for k, v in o.attrs.items()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=CDSG_CACHE)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="members.json whose stage1_oof members give every row's first-stage equilibrium")
    ap.add_argument("--blend-stage1", action="store_true",
                    help="held-out rows on the manifest's stage1_weights blend instead of the out-of-fold members")
    a = ap.parse_args()
    from nm1.stage1 import oof_members
    build(a.out, a.workers, tuple(a.splits.split(",")), a.manifest if a.blend_stage1 else None,
          oof_members(a.manifest))