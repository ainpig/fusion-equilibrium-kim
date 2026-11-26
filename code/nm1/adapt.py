"""Transductive self-training on a test fold's inputs (chain A).  No label of the fold exists or is read.

    python -m nm1.adapt run --fold private --dev <dev> --base <final run> --base-release <v1 release>
                            --out <adapted artifacts> --work <work dir>

1 base       the base chain's predictions of the fold (``nm1.infer`` on the v1 release)
2 rows       per test shot, the input blocks every member reads (base + profile block, coil currents, features2,
             current-diffusion block, and the flux-coordinate block on the base chain's own flux) and the
             pseudo-labels: the predicted flux projected on the PCA basis, the predicted q95 / betaN  (CPU)
3 caches     copies of the four training caches with the fold's rows appended to the held-out group, so that
             ``--all-data`` trains on every labelled row plus these rows
4 members    every member of the base manifest except the q95 group (``dlq``, kept as trained) and the flux
             specialist, continued from its final weights for 4 epochs at lr 3e-4 (``nm1.train --init``)
5 heads      the 12 shape heads retrained on the adapted encoders (``nm1.shapeheads``)
6 release    manifest and stack of the base run with the adapted members; fp16 release in
             ``<work>/release_adapted_<fold>``

Every step skips what exists, so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from nm1 import ROOT

PSEUDO_SHOT = 1_000_000          # shot ids of the fold's rows in the augmented caches
KEEP_GROUPS = ("dlq",)           # trained with --scalar-scale, which does not combine with --init
NAMES = {"public": "diii_d_public_test", "private": "diii_d_private_test"}


def _run(cmd, env_extra: dict, log: Path, dry: bool) -> None:
    print(f"   {' '.join(map(str, cmd))}", flush=True)
    if dry:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8", **env_extra}
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        f.flush()
        for attempt in range(10):
            try:
                rc = subprocess.run([sys.executable, "-u", "-m", *map(str, cmd)], cwd=ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
                break
            except OSError as exc:
                print(f"     launch failed ({exc}); retrying in 60 s", flush=True)
                time.sleep(60)
        else:
            raise SystemExit(f"could not launch {cmd[0]}")
    if rc != 0:
        raise SystemExit(f"{cmd[0]} failed (exit {rc}), log {log}")
    print(f"     {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ 2 rows
_B: dict = {}


def _init(basis: str, pred: str) -> None:
    import torch
    state = torch.load(basis, map_location="cpu", weights_only=False)
    _B["mean"] = state["pca_mean"].astype(np.float64)
    _B["comps"] = state["pca_components"].astype(np.float64)
    _B["pred"] = pred


def _shot(job):
    k, path = job
    from nm1.cds_geo import features_from_psi
    from nm1.diffusion import diffusion_features
    from nm1.features import d3d_input_features, d3d_profile_block, read_row
    from nm1.features2 import extended_features
    try:
        row = read_row(path, True)
        base, currents, times = d3d_input_features(row)
        feats = np.hstack([base, d3d_profile_block(row, times)]).astype(np.float32)
        ext = extended_features(row, times).astype(np.float32)
        cds = diffusion_features(row, times).astype(np.float32)
        with np.load(_B["pred"]) as z:
            psi = np.asarray(z[f"shot_{k:04d}_psirz"], np.float64)
            q95 = np.asarray(z[f"shot_{k:04d}_q95"], np.float32)
            bn = np.asarray(z[f"shot_{k:04d}_betaN"], np.float32)
        cdsg = features_from_psi(row, np.asarray(times, np.float64), psi).astype(np.float32)
        coeff = ((psi.reshape(len(psi), -1) - _B["mean"]) @ _B["comps"].T).astype(np.float32)
        good = (np.isfinite(feats).all(1) & np.isfinite(coeff).all(1) & np.isfinite(q95) & np.isfinite(bn))
        return k, dict(features=feats[good], currents=np.asarray(currents, np.float32)[good], coeff=coeff[good],
                       scalars=np.stack([q95, bn], 1)[good], ext=ext[good], cds=cds[good], cdsg=cdsg[good]), ""
    except Exception as exc:                        # noqa: BLE001 - reported by the parent
        return k, None, f"{type(exc).__name__}: {exc}"


def build_rows(fold: str, pred: Path, out: Path, workers: int) -> Path:
    from nm1.cache import BASIS
    from nm1.fetch import listing, local_path
    if out.exists():
        return out
    paths = [local_path(f) for f in listing(NAMES[fold])]
    rows: dict[int, dict] = {}
    bad = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(BASIS), str(pred))) as pool:
        for n, (k, res, err) in enumerate(pool.map(_shot, list(enumerate(paths)), chunksize=1), 1):
            if res is None:
                bad.append((k, err))
            else:
                rows[k] = res
            if n % 50 == 0 or n == len(paths):
                print(f"  rows: {n}/{len(paths)} shots, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    for k, err in bad[:20]:
        print(f"  skipped shot {k}: {err}", flush=True)
    tmp = out.with_suffix(".tmp")
    with h5py.File(tmp, "w") as h:
        for key in ("features", "currents", "coeff", "scalars", "ext", "cds", "cdsg"):
            h[key] = np.concatenate([rows[k][key] for k in sorted(rows)])
        h["shot"] = np.concatenate([np.full(len(rows[k]["coeff"]), PSEUDO_SHOT + k, np.int32) for k in sorted(rows)])
        h.attrs["source"] = f"{NAMES[fold]} inputs; pseudo-labels = the base chain's predictions ({pred.name})"
    tmp.replace(out)
    print(f"  pseudo-labelled rows: {sum(len(r['coeff']) for r in rows.values()):,} frames of {len(rows)} shots", flush=True)
    return out


# ------------------------------------------------------------------ 3 caches
def _append(src: Path, dst: Path, pieces: dict[str, np.ndarray]) -> None:
    if dst.exists():
        return
    tmp = dst.with_suffix(".tmp")
    shutil.copyfile(src, tmp)
    with h5py.File(tmp, "a") as h:
        g = h["val"]
        for key, arr in pieces.items():
            d = g[key]
            data = d[:]
            del g[key]
            g.create_dataset(key, data=np.concatenate([data, arr.astype(data.dtype)]))
    tmp.replace(dst)


def build_caches(rows: Path, cache_dir: Path) -> None:
    from nm1.cache import CACHE
    from nm1.cache2 import AUX, CACHE2
    from nm1.cds_geo import CDSG_CACHE
    from nm1.diffusion import CDS_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(rows, "r") as r:
        p = {k: r[k][:] for k in r}
    n = len(p["shot"])
    _append(CACHE, cache_dir / CACHE.name, {k: p[k] for k in ("features", "currents", "coeff", "scalars", "shot")})
    _append(CACHE2, cache_dir / CACHE2.name, {"ext": p["ext"], "aux": np.full((n, len(AUX)), np.nan, np.float32)})
    _append(CDS_CACHE, cache_dir / CDS_CACHE.name, {"cds": p["cds"]})
    # the flux-coordinate cache stores its block as `cds`
    _append(CDSG_CACHE, cache_dir / CDSG_CACHE.name, {"cds": p["cdsg"]})


# ------------------------------------------------------------------ run
def cmd_run(a) -> None:
    from nm1.pipeline import MEMBERS, SPECIALIST, heads, write_stack
    fold, work, out, base, rel = a.fold, a.work.resolve(), a.out.resolve(), a.base.resolve(), a.base_release.resolve()
    logs = work / "logs"
    name = f"{NAMES[fold]}.npz"
    print("== 1 base chain prediction", flush=True)
    pred = work / "base" / name
    if not pred.exists():
        _run(["nm1.infer", "--fold", fold, "--leg", "d3d", "--out", work / "base"], {"NM1_ARTIFACTS": str(rel)},
             logs / "infer_base.log", a.dry_run)
    print("== 2 pseudo-labelled rows", flush=True)
    rows = work / f"pseudo_{fold}.h5"
    if not a.dry_run:
        os.environ["NM1_ARTIFACTS"] = str(base)
        build_rows(fold, pred, rows, a.workers)
    print("== 3 augmented caches", flush=True)
    cache_dir = work / "cache"
    if not a.dry_run:
        build_caches(rows, cache_dir)
    print("== 4 members", flush=True)
    man = json.loads((base / "members.json").read_text(encoding="utf-8"))
    keep = {m for g in KEEP_GROUPS for m in man["groups"].get(g, [])}
    args = {n: ar for n, _, ar, _ in MEMBERS}
    args[SPECIALIST[0]] = SPECIALIST[1]
    (out / "members").mkdir(parents=True, exist_ok=True)
    env = {"NM1_ARTIFACTS": str(base), "NM1_CACHE_DATA": str(cache_dir)}
    for rel_m in man["members"] + [man["specialist"]]:
        n = Path(rel_m).stem
        dst = out / rel_m
        if rel_m in keep:
            if not dst.exists() and not a.dry_run:
                shutil.copyfile(base / rel_m, dst)
                shutil.copyfile((base / rel_m).with_suffix(".json"), dst.with_suffix(".json"))
            continue
        if dst.exists() and dst.with_suffix(".json").exists():
            continue
        flags = [x for x in args[n] if x != "--base-only"]
        _run(["nm1.train", "--init", base / rel_m, "--output", dst, "--epochs", str(a.epochs), "--lr", str(a.lr),
              "--all-data", "--save-epoch", str(a.epochs), *([] if "--base-only" in args[n] else ["--ext"]), *flags],
             env, logs / f"adapt_{n}.log", a.dry_run)
    print("== 5 heads", flush=True)
    from nm1 import OUTPUTS
    head_list = heads(out, [Path(m).stem for m in man["members"]], OUTPUTS / "shape_heads_lab", logs, a.dry_run,
                      a.workers)
    print("== 6 manifest, stack, release", flush=True)
    if not a.dry_run:
        for f in ("flux_basis.pt", "mode_jacobian.npz"):
            if not (out / f).exists():
                shutil.copyfile(base / f, out / f)
        m2 = dict(man)
        m2["note"] = (man.get("note", "") + f" | self-trained on the {fold} fold's inputs (pseudo-labels of the base "
                      f"chain, {a.epochs} epochs, lr {a.lr}; q95 group kept)")
        (out / "members.json").write_text(json.dumps(m2, indent=2), encoding="utf-8")
        write_stack(out, head_list, False)
    rel_out = work / f"release_adapted_{fold}"
    if not (rel_out / "members.json").exists():
        _run(["nm1.release", "--src", out, "--out", rel_out], {"NM1_ARTIFACTS": str(out)}, logs / "release.log",
             a.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run",))
    ap.add_argument("--fold", choices=("public", "private"), default="private")
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="the final run's artifact directory")
    ap.add_argument("--base-release", type=Path, required=True, help="the v1 fp16 release (base chain)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmd_run(a)


if __name__ == "__main__":
    main()
