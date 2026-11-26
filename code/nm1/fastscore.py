"""Official-functional scoring of cache-space predictions, with the ground-truth side cached.

``score.py`` re-reads every held-out parquet and re-extracts every ground-truth LCFS on each call,
which is right for the deployed chain and slow for comparing candidates.  Here the reference side
(ground-truth flux, the scalars, the contours, each shot's mean major radius) is built once per
half and kept on disk, so a candidate costs only its own contour extraction.

Predictions are given as PCA coefficients for the cache's ``val`` frames -- the same frames, in
the same order -- plus ``(q95, betaN)``, and are reconstructed and stored as float32 exactly as a
submission would be.  Nothing here changes the metric: the per-shot worker is
``local_score.score_shot`` and the reduction is ``metrics.finalize_machine``.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nm1 import ARTIFACTS, ROOT
from nm1.score import MASK, _init_worker, _reference, _W, held_out, load_truth

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
import local_score                                                    # noqa: E402
from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS              # noqa: E402
from metrics import Accum, finalize_machine                           # noqa: E402

REF_DIR = ARTIFACTS / "refcache"


def _score_job(job):
    gt, ref, pred, psi_sign, means = job
    return local_score.score_shot(gt, ref, pred, _W["R"], _W["Z"], _W["mc"], _W["mf"],
                                  psi_sign, means)


class Reference:
    """Ground truth for one held-out part, aligned to the coefficient cache's ``val`` rows."""

    def __init__(self, half: str, workers: int = 10):
        self.half = half
        REF_DIR.mkdir(parents=True, exist_ok=True)
        path = REF_DIR / f"{half}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                d = pickle.load(fh)
        else:
            d = self._build(half, workers)
            with open(path, "wb") as fh:
                pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.__dict__.update(d)

    @staticmethod
    def _build(half: str, workers: int) -> dict:
        from nm1.cache import CACHE
        from nm1.fetch import is_val, listing, local_path
        import h5py

        t0 = time.perf_counter()
        paths = held_out(half, 0)
        index = {local_path(f): i for i, f in enumerate(listing("diii_d_train"))
                 if is_val(Path(f["path"]).name)}
        with h5py.File(CACHE, "r") as h5:
            vshot = h5["val"]["shot"][:]
        starts = {}
        b = np.flatnonzero(np.diff(vshot)) + 1
        st = np.concatenate([[0], b])
        en = np.concatenate([b, [len(vshot)]])
        for s, e in zip(st, en):
            starts[int(vshot[s])] = (int(s), int(e))
        shots = load_truth(paths)
        keep, rows = [], []
        for s in shots:
            sid = index.get(s["path"])
            if sid is None or sid not in starts:
                continue
            a, e = starts[sid]
            if e - a != len(s["psi"]):
                print(f"  skip {s['path'].name}: cache has {e - a} frames, truth {len(s['psi'])}")
                continue
            keep.append(s)
            rows.append((a, e))
        print(f"reference[{half}]: {len(keep)} shots, {sum(len(s['psi']) for s in keep):,} frames;"
              f" extracting ground-truth contours ...", flush=True)
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(str(MASK),)) as pool:
            refs = list(pool.map(_reference, [s["psi"] for s in keep], chunksize=2))
        psi_sum = psi_sumsq = psi_n = 0.0
        scal_sum, scal_sumsq, scal_n = (np.zeros(N_SCALARS) for _ in range(3))
        cons_sum, cons_sumsq, cons_n = (np.zeros(N_CONS) for _ in range(3))
        for s, ref in zip(keep, refs):
            g = s["psi"].astype(np.float64)
            psi_sum += g.sum()
            psi_sumsq += (g ** 2).sum()
            psi_n += g.size
            for j, name in enumerate(SCALARS):
                v = np.asarray(s[name], dtype=np.float64)
                v = v[np.isfinite(v)]
                scal_sum[j] += v.sum()
                scal_sumsq[j] += (v ** 2).sum()
                scal_n[j] += v.size
            cg, cmk = ref[1], ref[2]
            for j in range(N_CONS):
                v = cg[cmk[:, j], j]
                cons_sum[j] += v.sum()
                cons_sumsq[j] += (v ** 2).sum()
                cons_n[j] += v.size
        stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
        means = (psi_sum / psi_n, np.where(scal_n > 0, scal_sum / np.maximum(scal_n, 1), 0.0),
                 np.where(cons_n > 0, cons_sum / np.maximum(cons_n, 1), 0.0))
        gts = [{"psi": s["psi"].astype(np.float32), "q95": s["q95"], "betaN": s["betaN"],
                "name": s["path"].name} for s in keep]
        print(f"reference[{half}] built in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        return {"gts": gts, "refs": refs, "rows": rows, "stats": stats, "means": means}

    def cons_truth(self) -> np.ndarray:
        """``(frames, 7)`` ground-truth derived scalars, NaN where the derivation fails."""
        return np.concatenate([r[1] for r in self.refs])

    def score(self, coeff: np.ndarray, q95: np.ndarray, betaN: np.ndarray,
              pca_mean: np.ndarray, comps: np.ndarray, workers: int = 10,
              flux_dtype=np.float32, psi_override=None) -> dict:
        """Score predictions given for ALL cache ``val`` rows (row-aligned with the cache)."""
        # predictions are rebuilt per shot in each pass and at most 2 x workers shots are in flight, so the
        # whole half's float64 maps are never held (or pickled) at once; same jobs, same order, same result
        comps64 = comps.astype(np.float64)
        mean64 = pca_mean.astype(np.float64)

        def pred(i, a, e):
            if psi_override is not None:
                psi = psi_override(i, a, e)
            else:
                psi = (mean64 + coeff[a:e].astype(np.float64) @ comps64).reshape(-1, 65, 65)
            return {"psirz": psi.astype(flux_dtype).astype(np.float64),
                    "q95": np.asarray(q95[a:e], np.float64),
                    "betaN": np.asarray(betaN[a:e], np.float64)}
        totals = {sg: 0.0 for sg in PSI_SIGNS}
        mean_psi = self.means[0]
        for i, (a, e) in enumerate(self.rows):
            rr, _, _ = local_score.psi_residuals(self.gts[i]["psi"], pred(i, a, e)["psirz"], mean_psi)
            for sg in PSI_SIGNS:
                totals[sg] += rr[sg]
        acc = Accum()
        acc.psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
        pending = deque()
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(str(MASK),)) as pool:
            for i, (a, e) in enumerate(self.rows):
                g = self.gts[i]
                pending.append(pool.submit(_score_job, ({"psi": g["psi"], "q95": g["q95"], "betaN": g["betaN"]},
                                                        self.refs[i], pred(i, a, e), acc.psi_sign, self.means)))
                if len(pending) >= 2 * workers:
                    acc.add(pending.popleft().result())
            while pending:
                acc.add(pending.popleft().result())
        return finalize_machine(acc, self.stats)


def fmt(res: dict) -> str:
    per = " ".join(f"{k}={v:.5f}" for k, v in res["r2_cons_each"].items())
    return (f"S={res['S']:.6f} psi={res['r2_psi']:.6f} qb={res['r2_qb']:.6f} "
            f"(q95={res['r2_each']['q95']:.5f} bN={res['r2_each']['betaN']:.5f}) "
            f"1-D={1 - res['dlcfs']:.6f} cons={res['consistency']:.6f} | {per}")


if __name__ == "__main__":
    for half in sys.argv[1:] or ["selection"]:
        Reference(half, workers=min(10, os.cpu_count() or 1))