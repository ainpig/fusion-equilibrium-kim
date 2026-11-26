"""Preprocessing for the DIII-D model, CPU-heavy and resumable.

    python -m nm1.prepare [stages ...]

Stages, in order: basis (PCA of the training flux) -> cache (coefficients and base features of every
training frame) -> cache2 (second feature block) -> cds (current-diffusion block) -> jacobian (mode Jacobian
of the official functionals). A stage whose output exists is skipped unless ``--force``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from nm1 import ROOT

ORDER = ["basis", "cache", "cache2", "cds", "jacobian"]


def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _module(name: str, *args: str) -> None:
    # process pools inside: a subprocess, so spawned workers do not re-import a parent holding CUDA
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, "-u", "-m", name, *args], check=True, cwd=ROOT, env=env)


def stage_basis(a) -> None:
    from nm1.cache import BASIS, fit_basis
    if _done(BASIS) and not a.force:
        print(f"basis: {BASIS} exists, skipping")
        return
    fit_basis(a.components, a.pca_shots, a.pca_frames, workers=a.workers_io)


def stage_cache(a) -> None:
    from nm1.cache import BASIS, CACHE, build
    if _done(CACHE) and not a.force:
        print(f"cache: {CACHE} exists, skipping")
        return
    build(BASIS, CACHE, a.workers_io)


def stage_cache2(a) -> None:
    from nm1.cache2 import CACHE2
    if _done(CACHE2) and not a.force:
        print(f"cache2: {CACHE2} exists, skipping")
        return
    _module("nm1.cache2", "--workers", str(a.workers_cpu))


def stage_cds(a) -> None:
    from nm1.diffusion import CDS_CACHE
    if _done(CDS_CACHE) and not a.force:
        print(f"cds: {CDS_CACHE} exists, skipping")
        return
    _module("nm1.diffusion", "--splits", "val,train", "--workers", str(a.workers_cpu))


def stage_jacobian(a) -> None:
    from nm1.jacobian import OUT
    if _done(OUT) and not a.force:
        print(f"jacobian: {OUT} exists, skipping")
        return
    _module("nm1.jacobian", "--n-modes", str(a.jacobian_modes), "--n-frames", str(a.jacobian_frames),
            "--workers", str(a.workers_cpu))


STAGES = {"basis": stage_basis, "cache": stage_cache, "cache2": stage_cache2, "cds": stage_cds,
          "jacobian": stage_jacobian}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", choices=["all", *ORDER], default=["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--components", type=int, default=256)
    ap.add_argument("--pca-shots", type=int, default=1500)
    ap.add_argument("--pca-frames", type=int, default=240_000)
    ap.add_argument("--jacobian-modes", type=int, default=96)
    ap.add_argument("--jacobian-frames", type=int, default=640)
    ap.add_argument("--workers-io", type=int, default=6)
    ap.add_argument("--workers-cpu", type=int, default=6)
    a = ap.parse_args()
    chosen = ORDER if "all" in a.stages else [s for s in ORDER if s in a.stages]
    t0 = time.perf_counter()
    for name in chosen:
        print(f"\n##### {name} ({(time.perf_counter() - t0) / 60:.0f} min elapsed)", flush=True)
        STAGES[name](a)
    print(f"\nprepare done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
