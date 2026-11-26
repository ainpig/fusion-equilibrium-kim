"""Score the DIII-D pipeline with the official metric, on shots it never trained on.

    S = 0.55*R2_psi + 0.15*R2{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency

``fusion_scoring/`` is the competition scorer's own code, copied unmodified, so the functionals
here are the ones Codabench runs; what differs is the data, which is held-out training shots
rather than the withheld test fold.  Treat it as a faithful proxy for model selection, not as a
leaderboard prediction.

The held-out split is used in three disjoint parts that are never mixed -- see ``held_out``.
``--half evaluation`` is the number to quote; nothing has chosen anything on those shots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nm1 import ARTIFACTS, ROOT
from nm1.cache import training_shots
from nm1.features import read_row, stack_psirz

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
import local_score                                                    # noqa: E402
from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS              # noqa: E402
from metrics import Accum, finalize_machine                           # noqa: E402

MASK = ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz"
_W: dict = {}


def _init_worker(mask_path: str) -> None:
    z = np.load(mask_path)
    _W.update(R=z["grid_R"], Z=z["grid_Z"], mc=z["mask_coarse"].astype(bool))
    _W["mf"] = _W["mc"].astype(np.float64)
    local_score.MACHINE = "DIII-D"


def _reference(psi: np.ndarray):
    return local_score.build_reference(psi, _W["R"], _W["Z"], _W["mc"], _W["mf"])


def _score_one(job):
    gt, ref, pred, psi_sign, means = job
    return local_score.score_shot(gt, ref, pred, _W["R"], _W["Z"], _W["mc"], _W["mf"],
                                  psi_sign, means)


def _mapped(pool, fn, jobs, step, label):
    out = []
    for begin in range(0, len(jobs), step):
        out.extend(pool.map(fn, jobs[begin:begin + step]))
        print(f"  {min(begin + step, len(jobs))}/{len(jobs)} {label}", flush=True)
    return out


HALVES = ("calibration", "selection", "evaluation", "all")


def held_out(half: str = "evaluation", n_shots: int = 0, seed: int = 2027) -> list[Path]:
    """Held-out shot paths, in three disjoint parts that are never mixed.

    The trainer picks each member's epoch by maximising its own score on ``shots[0::2]``, so any
    number measured there is an *in-selection* number. It is fine for watching a run converge and
    misleading for choosing between members: the earlier project measured a 1536x8 EMA member
    ranking first on that half and mid-table on shots the trainer had not touched, and found EMA
    members made the ensemble worse when chosen that way.

        calibration  shots[0::2]  the trainer's own selection half -- biased by construction
        selection    shots[1::4]  never seen by the trainer; where the ensemble and the
                                  post-processing are chosen
        evaluation   shots[3::4]  never seen by anything; the only number worth quoting
    """
    paths = sorted(training_shots("val", allow_partial=True))
    if half == "calibration":
        paths = paths[0::2]
    elif half == "selection":
        paths = paths[1::4]
    elif half == "evaluation":
        paths = paths[3::4]
    elif half != "all":
        raise ValueError(f"unknown half {half}; expected one of {HALVES}")
    if n_shots and n_shots < len(paths):
        rng = np.random.default_rng(seed)
        paths = [paths[i] for i in np.sort(rng.choice(len(paths), n_shots, replace=False))]
    return paths


def held_out_ids(half: str) -> set[int]:
    """The same three parts as `held_out`, as **cache shot indices**.

    The split has to be defined once. `held_out` slices sorted *paths*; the cache labels each
    frame with the shot's position in the recorded listing. Those two agree on order but not on
    membership -- two shots in the listing failed to parse and are absent from the cache, so
    slicing `[0::2]` on each gives different sets, and a run that trains on one while scoring on
    the other leaks. Measured: 33 of the 208 evaluation shots crossed over that way.

    So this derives the ids from the paths, which is what the scorer means by a half.
    """
    from nm1.cache import is_val, listing, local_path
    index = {local_path(f): i for i, f in enumerate(listing("diii_d_train"))
             if is_val(Path(f["path"]).name)}
    return {index[p] for p in held_out(half, 0) if p in index}


def load_truth(paths: list[Path]) -> list[dict]:
    shots = []
    for path in paths:
        row = read_row(path)
        psi = stack_psirz(row["efit_psirz"])
        q = np.asarray(row["efit_q95"], dtype=np.float64)
        b = np.asarray(row["efit_beta_n"], dtype=np.float64)
        n = min(len(psi), len(q), len(b))
        ok = np.isfinite(psi[:n]).all(axis=(1, 2))
        if ok.sum() < 5:
            continue
        shots.append({"path": path, "psi": psi[:n][ok], "q95": q[:n][ok], "betaN": b[:n][ok],
                      "keep": ok, "n": n})
        del row
    return shots


def score(predictions: list[dict], shots: list[dict], workers: int = 8) -> dict:
    """Run the official functionals over aligned prediction/ground-truth shot lists."""
    local_score.MACHINE = "DIII-D"
    pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(MASK),))
    step = max(1, 2 * workers)
    try:
        print("building ground-truth references (LCFS + 7 derived scalars) ...", flush=True)
        refs = _mapped(pool, _reference, [s["psi"] for s in shots], step, "shots")

        psi_sum = psi_sumsq = psi_n = 0.0
        scal_sum, scal_sumsq, scal_n = (np.zeros(N_SCALARS) for _ in range(3))
        cons_sum, cons_sumsq, cons_n = (np.zeros(N_CONS) for _ in range(3))
        for s, ref in zip(shots, refs):
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
        mean_psi = psi_sum / psi_n
        means = (mean_psi, np.where(scal_n > 0, scal_sum / np.maximum(scal_n, 1), 0.0),
                 np.where(cons_n > 0, cons_sum / np.maximum(cons_n, 1), 0.0))

        # One global flux sign per machine, exactly as the platform scorer fixes it.
        totals = {sg: 0.0 for sg in PSI_SIGNS}
        for s, p in zip(shots, predictions):
            rr, _, _ = local_score.psi_residuals(s["psi"], p["psirz"], mean_psi)
            for sg in PSI_SIGNS:
                totals[sg] += rr[sg]
        acc = Accum()
        acc.psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
        jobs = [({"psi": s["psi"], "q95": s["q95"], "betaN": s["betaN"]}, ref, p,
                 acc.psi_sign, means) for s, ref, p in zip(shots, refs, predictions)]
        for part in _mapped(pool, _score_one, jobs, step, "shots scored"):
            acc.add(part)
        return finalize_machine(acc, stats)
    finally:
        pool.shutdown()


def predict_held_out(shots: list[dict], manifest: Path, stack: Path | None,
                     flux_dtype: str = "float32", device: str | None = None, full_chain: bool = True,
                     pred_cache: Path | None = None) -> list[dict]:
    """Run the deployed inference chain on held-out shots, with the targets dropped first.

    ``full_chain`` (default) adds the per-shot post-processing that ``nm1.predict.build`` applies to the
    submitted file (li guard and field correction); without it only the ensemble and the stack's
    in-predictor steps are scored.  `device` exists so a scoring run can be made on the CPU while the GPU
    is busy with the MAST leg. It changes where the same arithmetic happens, not what is computed.

    ``pred_cache`` (a directory, one per manifest + stack) keeps each shot's finished prediction, so a run
    that is interrupted resumes where it stopped; shots found there are not predicted again.
    """
    from nm1.predict import Predictor, post_pool
    cache = Path(pred_cache) if pred_cache else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    def cache_file(i):
        return cache / f"{i:04d}_{Path(shots[i]['path']).stem}.npz"

    def keep(key, psi_g, sc):
        done[key] = (psi_g, sc)
        if cache is not None:
            tmp = cache_file(key).with_suffix(".tmp.npz")
            np.savez(tmp, psi=np.asarray(psi_g), q=np.asarray(sc[0]), b=np.asarray(sc[1]))
            os.replace(tmp, cache_file(key))

    done = {}
    if cache is not None:
        for i in range(len(shots)):
            if cache_file(i).exists():
                z = np.load(cache_file(i))
                done[i] = (z["psi"], (z["q"], z["b"]))
        if done:
            print(f"  {len(done)}/{len(shots)} shots taken from {cache}", flush=True)
        if len(done) == len(shots):
            return _assemble(shots, done, flux_dtype)
    predictor = Predictor(manifest, stack, device=device)
    pool, extra = post_pool(predictor) if (full_chain and stack is not None) else (None, None)
    for i, s in enumerate(shots):
        if i in done:
            continue
        row = read_row(s["path"], drop_targets=True)
        pred = predictor.predict(row)
        bump = predictor.stack.get("interior_bump") if stack is not None else None
        if pool is not None and bump and bump.get("enabled"):
            from nm1.interior import apply_shot
            pred["psirz"] = apply_shot(pred["psirz"], row, pred["times"], bump)
        if pool is not None:
            grid_r = np.asarray(row["efit_grid_R"], dtype=np.float64)
            grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
            for key, psi_g, sc in pool.submit(i, pred["psirz"], grid_r, grid_z, (pred["q95"], pred["betaN"]), **extra(pred)):
                keep(key, psi_g, sc)
        else:
            keep(i, pred["psirz"], (pred["q95"], pred["betaN"]))
        if (i + 1) % 50 == 0 or i + 1 == len(shots):
            print(f"  {i + 1}/{len(shots)} shots predicted", flush=True)
    if pool is not None:
        for key, psi_g, sc in pool.drain():
            keep(key, psi_g, sc)
    return _assemble(shots, done, flux_dtype)


def _assemble(shots, done, flux_dtype):
    out = []
    for i, s in enumerate(shots):
        psi, (q, b) = done[i]
        keep = s["keep"]
        out.append({"psirz": np.asarray(psi)[:s["n"]][keep].astype(flux_dtype),
                    "q95": np.asarray(q)[:s["n"]][keep].astype(np.float32),
                    "betaN": np.asarray(b)[:s["n"]][keep].astype(np.float32)})
    return out


def report(name: str, result: dict) -> None:
    print(f"\n{name}")
    print(f"  S={result['S']:.9f}   r2_psi={result['r2_psi']:.9f}  r2_qb={result['r2_qb']:.6f}  "
          f"1-D_LCFS={1 - result['dlcfs']:.6f}  Consistency={result['consistency']:.6f}")
    print("  " + json.dumps({k: round(v, 5) for k, v in result["r2_cons_each"].items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half", default="evaluation",
                    choices=HALVES)
    ap.add_argument("--n-shots", type=int, default=0, help="0 = every shot in the half")
    ap.add_argument("--manifest", type=Path, default=ARTIFACTS / "members.json")
    ap.add_argument("--stack", type=Path, default=ARTIFACTS / "stack.json")
    ap.add_argument("--no-stack", action="store_true", help="score the bare ensemble")
    ap.add_argument("--ensemble-only", action="store_true",
                    help="skip the per-shot li guard / field correction")
    ap.add_argument("--flux-dtype", default="float32", choices=("float16", "float32", "float64"))
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "scores.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pred-cache", type=Path, help="keep finished shots here so an interrupted run resumes")
    ap.add_argument("--device", help="'cpu' to leave the GPU to another run; default: cuda if free")
    a = ap.parse_args()

    paths = held_out(a.half, a.n_shots)
    print(f"{a.half} half: {len(paths)} held-out shots", flush=True)
    shots = load_truth(paths)
    print(f"loaded {len(shots)} shots, {sum(len(s['psi']) for s in shots):,} frames", flush=True)
    predictions = predict_held_out(shots, a.manifest, None if a.no_stack else a.stack,
                                   a.flux_dtype, a.device, full_chain=not a.ensemble_only, pred_cache=a.pred_cache)
    result = score(predictions, shots, a.workers)
    name = a.tag or ("ensemble" if a.no_stack else "ensemble+stack")
    report(f"{name}  [{a.half}, {len(shots)} shots, {sum(len(s['psi']) for s in shots):,} "
           f"frames, {a.flux_dtype}]", result)
    previous = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
    previous[f"{name}:{a.half}:{a.flux_dtype}"] = {
        **result, "shots": len(shots), "frames": int(sum(len(s["psi"]) for s in shots))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(previous, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()