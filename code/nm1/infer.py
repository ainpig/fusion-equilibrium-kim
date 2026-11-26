"""Inference entrypoint: regenerate the submitted prediction files from the shipped weights.

    python -m nm1.infer --fold public --out <fresh dir> [--leg d3d|mast|both] [--limit N]

DIII-D: the ensemble in ``NM1_ARTIFACTS`` (``members.json`` / ``stack.json``) with the shape-functional
field correction.  MAST: ``nm1.mast_phys`` (physics transfer from DIII-D priors; MAST inputs only) with
``recipes/mast_phys.json``.  Every run is audited by
``nm1.input_policy`` (official parquets only, no pre-existing .npz, no network).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from nm1 import ARTIFACTS, OUTPUTS

MAST_RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "mast_phys.json"   # 10-13: physics-transfer MAST leg


def run(out: Path, fold: str = "public", leg: str = "both", limit: int = 0, manifest: Path | None = None,
        stack: Path | None = None, mast_cfg: dict | None = None, mast_recipe: Path = MAST_RECIPE) -> dict:
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a fresh output directory; inference never reuses earlier predictions")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["FUSION_RUN_ROOT"] = str(out)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    from nm1.input_policy import install
    policy = install(out)
    import torch
    torch.set_num_threads(2)
    started, results = time.perf_counter(), {}
    if leg in ("d3d", "both"):
        from nm1.predict import build
        target = out / f"diii_d_{fold}_test.npz"
        results["d3d"] = str(build(f"diii_d_{fold}_test", target, manifest or ARTIFACTS / "members.json",
                                   stack or ARTIFACTS / "stack.json", prefetch=2, flux_dtype="float32", limit=limit))
        torch.cuda.empty_cache()
    if leg in ("mast", "both") and "mast_settings" in json.loads(Path(mast_recipe).read_text(encoding="utf-8")):
        # physics-transfer MAST leg (nm1.mast_phys): DIII-D priors + the fold's MAST inputs, its own audit
        from nm1.mast_phys import average_runs, run_recipe
        spec = json.loads(Path(mast_recipe).read_text(encoding="utf-8"))
        final = out / f"mast_{fold}_test.npz"
        # the solve is not bit-reproducible: ``runs`` independent runs of the unchanged recipe are averaged
        # frame by frame (one run when the recipe has no ``runs``)
        runs = int(spec.get("runs", 1))
        files = [run_recipe(out / ("mast_leg" if runs == 1 else f"mast_leg_r{i + 1}"), Path(mast_recipe),
                            workers=4, limit=limit, fold=fold, steps=int(spec.get("steps", 600)))
                 for i in range(runs)]
        merged = final if not spec.get("smooth_ms") else out / f"mast_{fold}_test.unsmoothed.npz"
        if runs == 1:
            Path(files[0]).replace(merged)
        else:
            average_runs([Path(f) for f in files], merged)
        if spec.get("smooth_ms"):
            # post-solve time smoothing
            from nm1.mast_phys import smooth_file
            smooth_file(merged, final, float(spec["smooth_ms"]), fold)
        results["mast"] = str(final)
    elif leg in ("mast", "both"):
        from nm1 import mast
        recipe = json.loads(Path(mast_recipe).read_text(encoding="utf-8"))
        cfg = dict(mast.CONFIG)
        cfg.update(recipe["config"])
        cfg.update(mast_cfg or {})
        tmp = out / "mast_leg"
        results["mast"] = str(mast.build(tmp, fold, limit, cfg, chunk=recipe.get("chunk", 300)))
        final = out / f"mast_{fold}_test.npz"
        Path(results["mast"]).replace(final)
        results["mast"] = str(final)
    from nm1.validate import validate
    if os.environ.get("NM1_SKIP_SHOTS"):          # a resumed part: validated after nm1.salvage stitches the parts
        results = {k: v for k, v in results.items() if k != "d3d"} | {"d3d_part": results.get("d3d")}
    for machine, path in results.items():
        if machine == "d3d_part":
            continue
        config = f"{'diii_d' if machine == 'd3d' else 'mast'}_{fold}_test"
        if validate(Path(path), config, max_shots=limit):
            raise ValueError("fresh output failed validation: " + path)
    if policy.report["denied"] or policy.report["network_attempts"]:
        raise PermissionError("data-access audit failed")
    report = dict(fold=fold, leg=leg, scope="subset" if limit else "full fold", outputs=results,
                  seconds=time.perf_counter() - started, input_policy=policy.report)
    (out / "infer_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    policy.save()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", choices=("public", "private"), default="public")
    ap.add_argument("--leg", choices=("d3d", "mast", "both"), default="both")
    ap.add_argument("--out", type=Path, default=OUTPUTS / f"infer_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--stack", type=Path)
    ap.add_argument("--mast-recipe", type=Path, default=MAST_RECIPE)
    ap.add_argument("--f-p", type=float)
    a = ap.parse_args()
    cfg = {"f_p": a.f_p} if a.f_p is not None else None
    print(json.dumps(run(a.out, a.fold, a.leg, a.limit, a.manifest, a.stack, cfg, a.mast_recipe), indent=2))


if __name__ == "__main__":
    main()
