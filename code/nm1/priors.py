"""Fit the MAST predictor's constants here, from the DIII-D parquets, rather than carrying them.

The zero-shot MAST leg has no trained weights, but it does have a handful of physical
proportionality constants -- a pressure ratio, a temperature exponent, q corrections, two
uncertainty scales. Every one of them is a least-squares or quantile fit on **DIII-D** data, and
every fitting script reads the raw ``diii_d_train`` parquets directly. So they are refitted in
this run alongside everything else, and nothing is inherited as a number.

The split is the same hash of the same file name the ensemble trains on, so a prior is never
fitted on a shot the ensemble was scored on.

    python -m nm1.priors            # fit the ones that are missing
    python -m nm1.priors --force    # refit all of them
    python -m nm1.priors --compare <dir>   # refit and diff against another set

What cannot be fitted here is the MAST vessel wall: 37 hardware limiter coordinates are a
property of the machine, not of any DIII-D data, and they are handled separately -- see
``zeroshot_mast/WALL.md``.
"""
from __future__ import annotations

import argparse
import os
import runpy
import json
import sys
import time
from pathlib import Path

from nm1 import DATA, PROJECT, ROOT

RUNTIME = ROOT / "mast_phys"
ARTIFACTS = RUNTIME / "my_experiments" / "artifacts"

_BETA_PRIOR = "__beta_uncertainty_prior"
_BETA_FEATURES = "__beta_shape_features"
_GEOMETRY_PRIOR = "__geometry_beta_priors"


def _beta_shape_features(rebuild: bool = False) -> None:
    """The shape-descriptor frames the geometric betaN priors are fitted on.

    ``build_cache`` is the original script's own function, called with ``with_thermal=False``:
    that skips the Thomson branch entirely, so this reads nothing but the split in
    ``d3d_pressure_prior_v13.json`` and the DIII-D parquets it names. Both cohorts come from
    that split, so the frames here are the same ones the ensemble treats as held out.
    """
    from my_experiments.check_synthetic_beta_d3d_v17 import build_cache
    for part in ("calibration", "validation"):
        target = ARTIFACTS / f"d3d_{part}_beta_features_v17.npz"
        if target.exists() and not rebuild:
            print(f"{target.name:<44} exists", flush=True)
            continue
        build_cache(ARTIFACTS, part)
        print(f"{target.name:<44} {target.stat().st_size / 1e6:.1f} MB", flush=True)


def _geometry_beta_priors() -> None:
    """Three Bayesian linear betaN priors over plasma shape, fitted on DIII-D calibration.

    Everything here reads DIII-D only: the three priors are fitted on the DIII-D calibration frames and scored
    on the DIII-D validation frames (``source_validation_r2`` in the output: constant -0.01, shift_li 0.86,
    geometry 0.91). The runtime uses the ``geometry`` prior, the best of the three on DIII-D validation
    (``joint_beta_confidence_v20.joint_estimate``).
    """
    import numpy as np
    from my_experiments.geometry_beta_prior_v17 import fit_prior, predict_prior
    train = np.load(ARTIFACTS / "d3d_calibration_beta_features_v17.npz")
    valid = np.load(ARTIFACTS / "d3d_validation_beta_features_v17.npz")
    priors = {k: fit_prior(train["X"], train["betaN"] / train["beta_scale"], k)
              for k in ("constant", "shift_li", "geometry")}
    source_validation = {}
    for k, prior in priors.items():
        mean, _ = predict_prior(prior, valid["X"])
        y = valid["betaN"] / valid["beta_scale"]
        source_validation[k] = float(1 - np.sum((mean - y) ** 2) / np.sum((y - y.mean()) ** 2))
    (ARTIFACTS / "d3d_geometry_beta_priors_v17.json").write_text(
        json.dumps({"priors": priors, "source_validation_r2": source_validation}, indent=2),
        encoding="utf-8")
    print(f"calibration {len(train['X'])} frames / validation {len(valid['X'])} frames", flush=True)
    print(json.dumps(source_validation, indent=2), flush=True)


def _beta_uncertainty_prior() -> None:
    """Mean and variance of the DIII-D pressure-moment population.

    The prior is a summary of ``d3d_pressure_moments_v14.npz``, which is DIII-D throughout.
    """
    import numpy as np
    source = np.load(ARTIFACTS / "d3d_pressure_moments_v14.npz")["calibration"][:, 2]
    prior = {"fitted_on": "DIII-D only", "mean": float(source.mean()),
             "variance": float(source.var()), "frames": len(source)}
    (ARTIFACTS / "d3d_beta_uncertainty_prior_v14.json").write_text(
        json.dumps(prior, indent=2), encoding="utf-8")
    print(json.dumps(prior, indent=2), flush=True)

# In dependency order: a later entry may read an earlier one's output. `takes_out` marks the two
# scripts that accept `--out`; the rest write to `ROOT/my_experiments/artifacts/` themselves,
# which relocating them already redirected here.
PRIORS = [
    ("fit_current_profile_d3d", "d3d_current_profile.json", True,
     "current-profile exponents through the Grad-Shafranov operator, 150 shots"),
    ("fit_kinetic_prior_d3d", "kinetic_prior_v11.json", True,
     "p(psi) ~ Te^n exponent and the current elongation, 128 shots"),
    ("calibrate_pressure_d3d_v13", "d3d_pressure_prior_v13.json", False,
     "total-to-electron pressure ratio, 64 shots / 860 frames"),
    ("calibrate_q_prior_d3d_v13", "d3d_q_shape_prior_v13.json", False,
     "shape-based q correction and its interval, 887 frames"),
    ("calibrate_ampere_q_v15", "d3d_ampere_q_prior_v15.json", False,
     "Ampere-law q95 against the stored one, 887 frames"),
    ("calibrate_geometric_q_v15", "d3d_geometric_q_prior_v15.json", False,
     "geometric q estimators and their conditioning"),
    ("calibrate_pressure_moments_v14", "d3d_pressure_moments_v14.json", False,
     "pressure moments per frame, the population the beta prior summarises"),
    (_BETA_PRIOR, "d3d_beta_uncertainty_prior_v14.json", False,
     "mean and variance of that population -- two numbers, computed here"),
    ("calibrate_uncertainty_d3d_v14", "d3d_uncertainty_calibration_v14.json", False,
     "posterior uncertainty scale, 452 frames"),
    ("calibrate_force_uncertainty_v16", "d3d_force_uncertainty_v16.json", False,
     "force-balance residual tolerance at the 95th percentile"),
    (_BETA_FEATURES, "d3d_validation_beta_features_v17.npz", False,
     "shape descriptors for both betaN cohorts, from the pressure-prior split"),
    (_GEOMETRY_PRIOR, "d3d_geometry_beta_priors_v17.json", False,
     "three shape-based betaN priors, fitted on DIII-D calibration"),
]

# The entries above that are driven from this module rather than by running a script.
_LOCAL = {_BETA_PRIOR: _beta_uncertainty_prior, _BETA_FEATURES: _beta_shape_features,
          _GEOMETRY_PRIOR: _geometry_beta_priors}


def _prepare() -> None:
    """Point the relocated scripts at this project's archive, before any of them is imported."""
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    sys.dont_write_bytecode = True
    # The scripts read `CACHE_ROOT/'data/diii_d_train'` and this project keeps the archive at
    # `<data>/diii_d_train`, so the root they want is its parent. `fusion_pipeline` reads this
    # variable at import time, which is why it is set here rather than patched afterwards.
    os.environ["FUSION_CACHE_ROOT"] = str(DATA.parent if DATA.name == "data" else PROJECT)
    # the runtime's input policy (my_experiments/__init__) lets NPZ outputs go only to its run directory:
    # for the fits that is the prior artifacts directory
    os.environ.setdefault("FUSION_RUN_ROOT", str(ARTIFACTS))


def fit_one(module: str, output: str, takes_out: bool) -> float:
    """Run one fitting script exactly as its own command line would."""
    started = time.perf_counter()
    if module in _LOCAL:
        if module == _BETA_FEATURES:
            _beta_shape_features(rebuild=True)       # fit_all only calls this to (re)build it
        else:
            _LOCAL[module]()
        return time.perf_counter() - started
    argv = sys.argv
    sys.argv = [module] + (["--out", str(ARTIFACTS / output)] if takes_out else [])
    try:
        # run_name="__main__" so the script's own argparse block runs: these are scripts, not
        # libraries, and several have no importable entry point.
        runpy.run_module(f"my_experiments.{module}", run_name="__main__", alter_sys=True)
    except SystemExit as exit_code:                 # argparse/scripts exit cleanly on success
        if exit_code.code not in (None, 0):
            raise
    finally:
        sys.argv = argv
    return time.perf_counter() - started


def _newest_training_parquet() -> float:
    newest = 0.0
    with os.scandir(DATA / "diii_d_train") as entries:
        for entry in entries:
            if entry.name.endswith(".parquet"):
                newest = max(newest, entry.stat().st_mtime)
    return newest


def fit_all(force: bool = False, only: list[str] | None = None) -> None:
    """Fit every constant that is missing or stale, in dependency order.

    A constant is reused only if it is newer than the training parquets and than every constant
    before it in ``PRIORS`` (a later entry may read an earlier one's output). These files live in
    git, so on a clean checkout "exists" is not enough: the parquets a fresh run fetches are newer
    than the checkout, and every constant is refitted from them. Refitting an early entry makes
    every later one stale in turn.
    """
    _prepare()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    floor = _newest_training_parquet()
    for module, output, takes_out, what in PRIORS:
        if only and module not in only:
            continue
        target = ARTIFACTS / output
        if target.exists() and target.stat().st_mtime >= floor and not force:
            print(f"{output:<38} exists (newer than its inputs), skipped", flush=True)
            floor = max(floor, target.stat().st_mtime)
            continue
        print(f"\n=== {output}  ({what})", flush=True)
        seconds = fit_one(module, output, takes_out)
        print(f"=== {output}: {seconds / 60:.1f} min", flush=True)
        floor = max(floor, target.stat().st_mtime)


def compare(reference: Path) -> int:
    """Diff the freshly fitted constants against another set, value by value."""
    mismatches = 0
    for _, output, _, _ in PRIORS:
        new, old = ARTIFACTS / output, Path(reference) / output
        if output.endswith(".npz"):                     # an intermediate, not a constant
            continue
        if not old.exists():
            print(f"{output:<38} nothing to compare with")
            continue
        a = json.loads(new.read_text(encoding="utf-8"))
        b = json.loads(old.read_text(encoding="utf-8"))
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if differing:
            mismatches += 1
            print(f"{output:<38} {len(differing)} differ: {differing[:4]}")
        else:
            print(f"{output:<38} all {len(a)} entries agree")
    return mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="refit even if the output exists")
    ap.add_argument("--only", nargs="+", help="only these modules")
    ap.add_argument("--compare", type=Path, help="after fitting, compare the values with this directory")
    a = ap.parse_args()
    fit_all(a.force, a.only)
    if a.compare:
        print()
        raise SystemExit(1 if compare(a.compare) else 0)


if __name__ == "__main__":
    main()
