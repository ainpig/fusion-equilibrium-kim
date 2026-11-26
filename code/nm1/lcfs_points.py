"""Boundary labels of training frames for the transformers' boundary-sensitivity term (``train_tfm --bnd-file``).

    python -m nm1.lcfs_points --out <dev>/lcfs_points.h5 [--K 32] [--labels <shape-head lab>/labels.h5]

Frames: the training-split frames the shape heads are labelled on (``labels.h5`` ``train/idx``: 16 per shot plus
the first and last three).  For each frame the training flux (the cache's PCA coefficients) gives the LCFS through
the official extractor; stored per frame:

  pts    (K, 2)  K boundary points equally spaced in arc length, as (row, col) grid coordinates
  xref   (2,)    the boundary point where |grad psi| is smallest (the x-point side), (row, col)
  grad   (K,)    |grad psi| at the K points
  rgeo   ()      geometric major radius of the boundary, (R_max + R_min) / 2

Labels come from the training flux only.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from nm1 import OUTPUTS, ROOT

_G: dict = {}


def _init(R, Z, mean, comps, K):
    sys.path.insert(0, str(ROOT / "fusion_scoring"))
    from nm1.guard import box_mask
    m = box_mask(R, Z)
    _G.update(R=R, Z=Z, mc=m, mf=m.astype(float), mean=mean, comps=comps, K=K)


def _bilinear(field: np.ndarray, rc: np.ndarray) -> np.ndarray:
    r, c = rc[:, 0], rc[:, 1]
    r0 = np.clip(np.floor(r).astype(int), 0, field.shape[0] - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, field.shape[1] - 2)
    fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * field[r0, c0] + (1 - fr) * fc * field[r0, c0 + 1]
            + fr * (1 - fc) * field[r0 + 1, c0] + fr * fc * field[r0 + 1, c0 + 1])


def _job(job):
    from lcfs import extract_lcfs_with_sign
    i, coeff = job
    R, Z, K = _G["R"], _G["Z"], _G["K"]
    maps = (_G["mean"] + coeff.astype(np.float64) @ _G["comps"]).reshape(-1, len(Z), len(R))
    out = []
    for psi in maps:
        contour, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", _G["mc"], _G["mf"])
        if contour is None or not np.isfinite(contour).all():
            out.append(None)
            continue
        rc = np.stack([(contour[:, 1] - Z[0]) / (Z[1] - Z[0]), (contour[:, 0] - R[0]) / (R[1] - R[0])], 1)
        gz, gr = np.gradient(psi, Z, R)
        gmag = np.sqrt(gz * gz + gr * gr)
        g_all = _bilinear(gmag, rc)
        seg = np.r_[0.0, np.cumsum(np.hypot(np.diff(contour[:, 0]), np.diff(contour[:, 1])))]
        s = np.linspace(0.0, seg[-1], K, endpoint=False)
        k = np.searchsorted(seg, s, side="right") - 1
        k = np.clip(k, 0, len(rc) - 2)
        f = ((s - seg[k]) / np.maximum(seg[k + 1] - seg[k], 1e-12))[:, None]
        pts = rc[k] * (1 - f) + rc[k + 1] * f
        out.append((pts.astype(np.float32), rc[int(np.argmin(g_all))].astype(np.float32),
                    _bilinear(gmag, pts).astype(np.float32),
                    np.float32(0.5 * (contour[:, 0].max() + contour[:, 0].min()))))
    return i, out


def build(out: Path, labels: Path, K: int = 32, workers: int = 6) -> Path:
    import torch
    from nm1.cache import BASIS, CACHE
    from nm1.shapeheads import grid
    R, Z = grid()
    p = torch.load(BASIS, map_location="cpu", weights_only=False)
    with h5py.File(labels, "r") as f:
        idx = f["train/idx"][:]
    with h5py.File(CACHE, "r") as h:
        coeff = h["train"]["coeff"][idx]
    jobs = [(a, coeff[a:a + 64]) for a in range(0, len(idx), 64)]
    res: list = [None] * len(idx)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(R, Z, p["pca_mean"].astype(float), p["pca_components"].astype(float), K)) as pool:
        for done, (a, vals) in enumerate(pool.map(_job, jobs, chunksize=2), 1):
            res[a:a + len(vals)] = vals
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} blocks, {time.perf_counter() - t0:.0f}s", flush=True)
    ok = np.asarray([r is not None for r in res])
    keep = [r for r in res if r is not None]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with h5py.File(tmp, "w") as o:
        o["idx"] = idx[ok].astype(np.int64)
        o["pts"] = np.stack([r[0] for r in keep])
        o["xref"] = np.stack([r[1] for r in keep])
        o["grad"] = np.stack([r[2] for r in keep])
        o["rgeo"] = np.asarray([r[3] for r in keep], np.float32)
        o.attrs["K"] = K
        o.attrs["source"] = "training flux (PCA coefficients of diii_d_train), official LCFS extractor"
    tmp.replace(out)
    print(f"wrote {out}: {int(ok.sum()):,} of {len(idx):,} frames with a boundary", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=OUTPUTS / "shape_heads_lab" / "labels.h5")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    build(a.out, a.labels, a.K, a.workers)


if __name__ == "__main__":
    main()
