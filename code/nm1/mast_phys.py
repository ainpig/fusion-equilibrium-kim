from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from nm1 import OUTPUTS, ROOT, DATA
from nm1.fetch import LISTINGS, listing, local_path

RUNTIME = Path(os.environ.get("FUSION_MAST_RUNTIME") or (ROOT / "mast_phys")).resolve()

LEGACY = ROOT                      # recorded in the run record as the source workspace

CONFIG = {
    "mast_seed_batch_shots": 64,
    "mast_seed_steps": 600,
    "mast_force_batch_shots": 96,
    "mast_temporal_batch_shots": 18,
    "mast_final_batch_shots": 144,
    "mast_final_batch_frames": 8192,
    "stable_sampler": False,
}

def _assets() -> list[str]:

    out = []
    for pattern in ("*.py", "my_experiments/*.py", "fusion_scoring/*.py"):
        out.extend(str(p.relative_to(RUNTIME)).replace("\\", "/")
                   for p in RUNTIME.glob(pattern) if p.is_file())
    from nm1.priors import PRIORS
    out.extend("my_experiments/artifacts/" + name for _, name, _, _ in PRIORS
               if name.endswith(".json"))
    return sorted(set(out))

def build(output_dir: Path, workers: int = 4, limit: int = 0, resume: bool = False,
          stop_after: str = "all", fold: str = "public", steps: int = 600,
          q95_physical: bool = False, edge_pairs: bool = False,
          sol_divisor: float = 2.0, vessel_g: float = 0.0, vessel_tau: float = 10.0,
          q95_limited_fix: bool = False, anchor_weight: float = 1e-3,
          anchor_peak: bool = False, fold_fraction: float = 1.0,
          anchor_peak_mode: str = "max", pressure_branch: bool = True,
          betan_limited_fill: bool = False, q95_limited_fill: bool = False,
          betan_diverted_blend: float = 0.0, q95_diverted_blend: float = 0.0,
          axis_from_baseline: bool = False, batches: dict | None = None, stable_sampler: bool = False,
          fill_model: str = "catboost", inner_wall_gap_m: float = 0.0) -> Path:

    if fold not in ("public", "private"):
        raise ValueError(f"unknown fold {fold!r}")
    from nm1.winspawn import install as retry_worker_spawn
    retry_worker_spawn()
    output_dir = Path(output_dir).resolve()
    if resume:
        # the same run continued after a stop: its own intermediates only (run id + SHA256 checked per file)
        if not (output_dir / "run.json").exists():
            raise FileNotFoundError("resume needs the run's run.json")
        os.environ["NM1_RESUME_DIR"] = str(output_dir / "intermediate")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh output directory (or resume this same run)")
    os.environ["FUSION_RUN_ROOT"] = str(output_dir)
    from nm1.input_policy import install as input_boundary
    boundary = input_boundary(output_dir)
    config_name = f"mast_{fold}_test"
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))

    for cached in RUNTIME.rglob("__pycache__"):
        if not cached.resolve().is_relative_to(RUNTIME.resolve()):
            raise ValueError("Bytecode cleanup escaped the runtime")
        for f in cached.glob("*"):
            f.unlink()
        cached.rmdir()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    if not sol_divisor > 0:
        raise ValueError("sol_divisor must be positive")
    os.environ["FUSION_MAST_EDGE_PAIRS"] = "1" if edge_pairs else "0"
    os.environ["FUSION_MAST_SOL_DIVISOR"] = repr(float(sol_divisor))
    # plasma support starts this far outboard of the solenoid rectangle (centre-column tube and armour; 0 = as before)
    if not 0.0 <= inner_wall_gap_m < 0.1:
        raise ValueError("inner_wall_gap_m must be in [0, 0.1) m")
    os.environ["FUSION_MAST_INNER_GAP_M"] = repr(float(inner_wall_gap_m))
    if vessel_g < 0 or vessel_tau <= 0:
        raise ValueError("vessel_g >= 0 and vessel_tau > 0 are required")
    os.environ["FUSION_MAST_VESSEL_G"] = repr(float(vessel_g))
    os.environ["FUSION_MAST_VESSEL_TAU"] = repr(float(vessel_tau))
    if not anchor_weight > 0:
        raise ValueError("anchor_weight must be positive")
    os.environ["FUSION_MAST_ANCHOR_W"] = repr(float(anchor_weight))
    if anchor_peak_mode not in ("max", "centroid"):
        raise ValueError("anchor_peak_mode is 'max' or 'centroid'")
    os.environ["FUSION_MAST_ANCHOR_PEAK"] = ("0" if not anchor_peak
                                             else "1" if anchor_peak_mode == "max" else "2")
    if not 0.0 <= fold_fraction <= 1.0:
        raise ValueError("fold_fraction must be within [0, 1]")
    os.environ["FUSION_MAST_FOLD_FRACTION"] = repr(float(fold_fraction))
    for key in ("FUSION_MAST_PFACTOR", "FUSION_MAST_PRIOR_W"):
        os.environ.pop(key, None)
    import torch
    from my_experiments import reproduce_from_inputs_v25 as runner
    from my_experiments.mast_runtime import mast
    from my_experiments.reproduction_io_v25 import install

    if not torch.cuda.is_available():
        raise SystemExit("the MAST seed optimizer needs a CUDA GPU")
    if torch.cuda.get_device_properties(0).total_memory < 12 * 2 ** 30:
        os.environ.setdefault("NM1_MAST_MAX_GPU_FRAMES", "2048")   # throughput setting for 8 GB cards
    torch.set_float32_matmul_precision("highest")

    files = listing(config_name)
    if limit:
        files = files[:limit]
    paths = [local_path(f).resolve() for f in files]
    if not paths or any(not p.is_relative_to((DATA / config_name).resolve()) for p in paths):
        raise ValueError("Only the requested official MAST input fold may be read")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} MAST parquets are missing; "
                         f"run: python -m nm1.fetch --config {config_name}")

    output_dir = output_dir.resolve()
    config = dict(CONFIG, mast_seed_steps=steps)
    # batch sizes are throughput settings (16 GB RAM machine: smaller batches); frames are optimised independently
    config.update({k: int(v) for k, v in (batches or {}).items() if k in CONFIG and k.endswith(("_shots", "_frames"))})
    # deterministic seed optimisation: grid_sample backward in float64 (no atomic float32 accumulation)
    config["stable_sampler"] = bool(stable_sampler)
    if q95_physical:
        config["q95_physical"] = True
    if q95_limited_fix:
        if not q95_physical:
            raise ValueError("the v14 limited-frame q95 rule extends the v11 q95 regularization")
        config["q95_limited_fix"] = True
    if betan_limited_fill or q95_limited_fill:
        if not q95_limited_fix:
            raise ValueError("the v17 limited-frame fill needs the v14 limited flags (--q95-limited-fix)")
        if betan_limited_fill:
            config["betaN_limited_fill"] = True
        if q95_limited_fill:
            config["q95_limited_fill"] = True
    for name, w, needs in (("betaN_diverted_blend", betan_diverted_blend, betan_limited_fill),
                           ("q95_diverted_blend", q95_diverted_blend, q95_limited_fill)):
        if w:
            if not 0.0 < w <= 1.0 or not needs:
                raise ValueError(f"{name} needs its limited-frame fill and a weight in (0, 1]")
            config[name] = float(w)
    if inner_wall_gap_m:
        config["inner_wall_gap_m"] = float(inner_wall_gap_m)
    if fill_model not in ("catboost", "transformer"):
        raise ValueError(f"unknown fill_model {fill_model!r}")
    if fill_model != "catboost":
        config["fill_model"] = fill_model
    if axis_from_baseline:
        if not q95_physical:
            raise ValueError("the axis-from-baseline step runs after the q95 physical stage (--q95-physical)")
        config["axis_from_baseline"] = True
    if edge_pairs:
        config["mast_edge_pairs"] = True

    if sol_divisor != 2.0:
        config["mast_sol_divisor"] = float(sol_divisor)
    if vessel_g > 0:
        config["mast_vessel_g_ka_per_v"] = float(vessel_g)
        config["mast_vessel_tau_ms"] = float(vessel_tau)
    if anchor_weight != 1e-3 or anchor_peak:
        config["mast_anchor_weight"] = float(anchor_weight)
        config["mast_anchor_te_peak"] = bool(anchor_peak)
    if fold_fraction != 1.0:
        config["mast_fold_fraction"] = float(fold_fraction)
    if anchor_peak and anchor_peak_mode != "max":
        config["mast_anchor_peak_mode"] = anchor_peak_mode
    if not pressure_branch:
        config["mast_no_pressure_branch"] = True
    if RUNTIME != (ROOT / "mast_phys").resolve():
        config["runtime"] = RUNTIME.name
    from my_experiments import mast_equilibrium_v12
    if mast_equilibrium_v12.SOLENOID_PARALLEL_CIRCUITS != float(sol_divisor):
        raise RuntimeError("the solenoid divisor was read before it was set")
    from my_experiments.kinetic_equilibrium import read_inputs
    from my_experiments.released_geometry import configure
    geometry = configure(read_inputs(paths[0]))
    version_name = "v3-1+q95-physical" if q95_physical else "v3-1"
    if edge_pairs:
        version_name += "+edge-pairs"
    if sol_divisor != 2.0:
        version_name += f"+sol/{sol_divisor:g}"
    if vessel_g > 0:
        version_name += f"+vessel{vessel_g:g}"
    if anchor_peak or anchor_weight != 1e-3:
        version_name += f"+anchor{anchor_weight:g}{'peak' if anchor_peak else ''}"
    if q95_limited_fix:
        version_name += "+q95-limited"
    if betan_limited_fill:
        version_name += "+betaN-fill"
    if q95_limited_fill:
        version_name += "+q95-fill"
    if betan_diverted_blend:
        version_name += f"+betaN-blend{betan_diverted_blend:g}"
    if q95_diverted_blend:
        version_name += f"+q95-blend{q95_diverted_blend:g}"
    if axis_from_baseline:
        version_name += "+axis-base"
    if fold_fraction != 1.0:
        version_name += f"+fold{fold_fraction:g}"
    if anchor_peak and anchor_peak_mode != "max":
        version_name += f"+{anchor_peak_mode}"
    if not pressure_branch:
        version_name += "+no-pressure-branch"
    identity = {"version": version_name, "machine": "mast", "shots": len(paths), "config": config,
                "inputs": [{"path": str(p), "bytes": p.stat().st_size} for p in paths]}
    if fold != "public":

        identity["fold"] = fold
    run = runner.Run(output_dir, identity, bool(resume))
    os.environ["MPLCONFIGDIR"] = str(output_dir / "matplotlib")
    specification = {
        "runtime": str(RUNTIME), "output": str(output_dir),

        "source_workspace": str(LEGACY),
        "assets": _assets(),
        "inputs": [str(p) for p in paths],
        "metadata": [str((LISTINGS / f"{config_name}.json").resolve())],
    }
    policy = install(specification)
    print(f"MAST: {len(paths)} shots, {workers} workers, seed steps "
          f"{config['mast_seed_steps']}, output {output_dir}", flush=True)
    started = time.perf_counter()
    try:
        result = mast(run, paths, config, specification, workers, stop_after)
        if (policy.denied or policy.network_attempts or boundary.report['denied']
                or boundary.report['network_attempts']):
            raise PermissionError("the MAST run read outside its asset/input manifest")
        summary = {"result": result, "seconds": time.perf_counter() - started,
                   "version": version_name, "fold": fold, "shots": len(paths), "config": config,
                   "previous_prediction_cache_used": False,
                   "external_mast_data_used": False,
                   "mast_targets_used": False,
                   "gpu": torch.cuda.get_device_name(), "numerical_support": geometry[3],
                   "input_policy": boundary.report,
                   "complete": stop_after == "all", "scope": "subset" if limit else "full fold"}
        (output_dir / "mast_run.json").write_text(json.dumps(summary, indent=2),
                                                  encoding="utf-8")
        print(json.dumps(summary["result"], indent=2), flush=True)
    finally:
        policy.save()
        boundary.save()
    return output_dir / f"{config_name}.npz"

def run_recipe(output_dir: Path, recipe: Path, workers: int = 4, limit: int = 0, fold: str = "public",
               steps: int = 600, resume: bool = False) -> Path:
    """The submitted MAST settings from ``recipes/mast_phys.json`` (q95 physical stage and the ``mast_settings``)."""
    spec = json.loads(Path(recipe).read_text(encoding="utf-8"))
    return build(output_dir, workers=workers, limit=limit, resume=resume, fold=fold, steps=steps,
                 q95_physical=bool(spec.get("mast_q95_physical", False)), batches=spec.get("batches"),
                 stable_sampler=bool(spec.get("stable_sampler", False)),
                 fill_model=str(spec.get("fill_model", "catboost")),
                 inner_wall_gap_m=float(spec.get("inner_wall_gap_m", 0.0)),
                 **spec.get("mast_settings", {}))


def average_runs(files: list[Path], out: Path) -> Path:
    """Frame-wise mean of independent runs of the same recipe (same shots, frames and keys; float64 sum, stored in
    the runs' own dtype).  Prints the run-to-run spread (mean |difference| over mean |value| per key kind)."""
    import numpy as np
    from nm1.predict import NpzStreamWriter
    runs = [np.load(f) for f in files]
    keys = sorted(runs[0].files)
    if any(sorted(r.files) != keys for r in runs):
        raise ValueError("runs differ in their keys")
    out.parent.mkdir(parents=True, exist_ok=True)
    spread, w = {}, NpzStreamWriter(out)
    try:
        for k in keys:
            xs = [r[k].astype(np.float64) for r in runs]
            if any(x.shape != xs[0].shape for x in xs):
                raise ValueError(f"runs differ in shape at {k}")
            w.write(k, np.mean(xs, axis=0).astype(runs[0][k].dtype))
            s = spread.setdefault(k.rsplit("_", 1)[-1], [0.0, 0.0])
            s[0] += float(np.nanmean(np.abs(xs[0] - xs[1])))
            s[1] += float(np.nanmean(np.abs(xs[0])))
    finally:
        w.close()
    for kind, (d, a) in spread.items():
        print(f"MAST runs: mean |run 1 - run 2| / mean |value|  {kind}: {d / a:.4%}", flush=True)
    return out


def smooth_file(src: Path, out: Path, sigma_ms: float, fold: str = "public", median: int = 0,
                scalar_ip: bool = False, q95_ms: float | None = None, beta_ms: float | None = None) -> Path:
    """Time smoothing of a finished MAST file (``nm1.mast.smooth_shot``): the plasma flux per ampere and q95 / beta_N
    filtered by a Gaussian of ``sigma_ms`` over the shot's EFIT frames; the coil flux and Ip stay exact per frame.
    Inputs of the fold only (coil currents, Ip, times).  ``median``: odd running-median window applied first
    (isolated failed frames).  ``scalar_ip``: q95 and beta_N are filtered as q95 * |Ip| and beta_N * |Ip| and divided
    back, so their exact 1 / Ip dependence follows Ip frame by frame as the flux per ampere does.  ``q95_ms`` /
    ``beta_ms``: a width of their own for q95 / beta_N (the same Gaussian over the EFIT times; 0 = unsmoothed);
    None keeps ``sigma_ms``."""
    import numpy as np
    from nm1 import mast
    from nm1.predict import NpzStreamWriter
    paths = [local_path(f) for f in listing(f"mast_{fold}_test")]
    M = mast.machine(mast.read_inputs(paths[0]), dict(mast.CONFIG), device="cpu")
    z = np.load(src)

    def own(raw, t_s, ms):
        if not ms:
            return raw
        d = (t_s[:, None] - t_s[None, :]) / (ms / 1000.0)
        g = np.where(np.abs(d) <= 3.0, np.exp(-0.5 * d ** 2), 0.0)
        return (g / g.sum(1, keepdims=True)) @ raw

    out.parent.mkdir(parents=True, exist_ok=True)
    w = NpzStreamWriter(out)
    try:
        for j, p in enumerate(paths):
            psi = z[f"shot_{j:04d}_psirz"].astype(np.float32)
            sc = np.stack([z[f"shot_{j:04d}_q95"], z[f"shot_{j:04d}_betaN"]]).astype(np.float64)
            row = mast.read_inputs(p)
            ip = np.maximum(np.abs(mast.frame_inputs(row)["ip"]), 1e4) if scalar_ip else 1.0
            ps, scs = mast.smooth_shot(M, row, psi, sc * ip, sigma_ms / 1000.0, median)
            scs = scs / ip
            t_s = np.asarray(row["efit_times"], float) * mast.TIME_UNIT_S
            if q95_ms is not None:
                scs[0] = own(sc[0], t_s, q95_ms)
            if beta_ms is not None:
                scs[1] = own(sc[1], t_s, beta_ms)
            w.write(f"shot_{j:04d}_psirz", ps.astype(np.float32))
            w.write(f"shot_{j:04d}_q95", scs[0].astype(np.float32))
            w.write(f"shot_{j:04d}_betaN", scs[1].astype(np.float32))
    finally:
        w.close()
    return out


def repair_limited(src: Path, out: Path, flags: list, fold: str = "public") -> Path:
    """Flux of the frames the solve classifies as inboard-limited, replaced by the plasma flux per ampere of the
    shot's resolved frames: linear in time between the nearest resolved frames, held at the shot ends; the coil flux
    and Ip stay exact per frame; q95 / beta_N are left as they are (the limited-frame fill already treats them).
    Grounds: the dataset documentation states every EFIT timeslice in the challenge is a diverted plasma (limited
    frames were removed), so a frame our solve finds limited is a solve failure, not a limited plasma.
    ``flags``: one boolean array per shot (q_physical_v14.limited_flags on the run's ``mast_final`` flux)."""
    import numpy as np
    from nm1 import mast
    from nm1.predict import NpzStreamWriter
    paths = [local_path(f) for f in listing(f"mast_{fold}_test")]
    M = mast.machine(mast.read_inputs(paths[0]), dict(mast.CONFIG), device="cpu")
    z = np.load(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, changed = NpzStreamWriter(out), 0
    try:
        for j, p in enumerate(paths):
            psi = z[f"shot_{j:04d}_psirz"].astype(np.float64)
            lim = np.asarray(flags[j], bool)
            if lim.any() and (~lim).any():
                row = mast.read_inputs(p)
                vac, ip = mast._vac_and_ip(M, row)
                t = np.asarray(row["efit_times"], float)
                plasma = ((psi - vac) / ip[:, None, None]).reshape(len(t), -1)
                good = np.flatnonzero(~lim)
                for k in np.flatnonzero(lim):
                    plasma[k] = _interp_rows(t[k], t[good], plasma[good])
                psi = vac + ip[:, None, None] * plasma.reshape(psi.shape)
                changed += int(lim.sum())
            w.write(f"shot_{j:04d}_psirz", psi.astype(np.float32))
            w.write(f"shot_{j:04d}_q95", z[f"shot_{j:04d}_q95"])
            w.write(f"shot_{j:04d}_betaN", z[f"shot_{j:04d}_betaN"])
    finally:
        w.close()
    print(f"limited-frame flux repair: {changed} frames", flush=True)
    return out


def _interp_rows(tq, t, rows):
    """Row-wise linear interpolation of ``rows`` (n, m) sampled at increasing ``t`` to time ``tq``; held at the ends."""
    import numpy as np
    if tq <= t[0]:
        return rows[0]
    if tq >= t[-1]:
        return rows[-1]
    i = int(np.searchsorted(t, tq))
    a = (tq - t[i - 1]) / (t[i] - t[i - 1])
    return (1 - a) * rows[i - 1] + a * rows[i]


def combine_keys(flux_file: Path, scalar_file: Path, out: Path) -> Path:
    """The submitted MAST file: the flux maps of ``flux_file`` with q95 / beta_N of ``scalar_file`` (the scorer grades
    the flux and each scalar separately).  Both files must hold the same shots and frames."""
    import numpy as np
    from nm1.predict import NpzStreamWriter
    a, b = np.load(flux_file), np.load(scalar_file)
    shots = sorted({k.rsplit("_", 1)[0] for k in a.files if k.endswith("_psirz")})
    out.parent.mkdir(parents=True, exist_ok=True)
    w = NpzStreamWriter(out)
    try:
        for s in shots:
            p, q, bn = a[f"{s}_psirz"], b[f"{s}_q95"], b[f"{s}_betaN"]
            if not len(p) == len(q) == len(bn):
                raise ValueError(f"{s}: frame counts differ between the two runs")
            w.write(f"{s}_psirz", p)
            w.write(f"{s}_q95", q)
            w.write(f"{s}_betaN", bn)
    finally:
        w.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path,
                    default=OUTPUTS / f"mast_v3_1_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="diagnostic subset; changes the fold")
    ap.add_argument("--resume", action="store_true", help="unsupported in v3-1; use a fresh directory")
    ap.add_argument("--stop-after", default="all", choices=("all", "seed", "joint"))
    ap.add_argument("--fold", default="public", choices=("public", "private"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--q95-physical", action="store_true",
                    help="v11: q95 from the exact B_t/Ip scaling with a regularized shape factor")
    ap.add_argument("--edge-pairs", action="store_true",
                    help="v14: equal-temperature pairs at the plasma edge in the seed solve")
    ap.add_argument("--sol-divisor", type=float, default=2.0,
                    help="solenoid turn current = feed current / this (v3-1 .. v13: 2)")
    ap.add_argument("--vessel-g", type=float, default=0.0,
                    help="v14: vessel conductance, kA per V of loop voltage (0 = no vessel current)")
    ap.add_argument("--vessel-tau", type=float, default=10.0, help="v14: vessel L/R time, ms")
    ap.add_argument("--q95-limited-fix", action="store_true",
                    help="v14: inboard-limited frames are unresolved geometry in the q95 rule")
    ap.add_argument("--anchor-weight", type=float, default=1e-3,
                    help="seed axis-anchor weight (v3-1 .. v13: 0.001)")
    ap.add_argument("--anchor-peak", action="store_true",
                    help="v14: anchor the seed axis on the Thomson Te maximum")
    ap.add_argument("--fold-fraction", type=float, default=1.0,
                    help="v14: share of the vertical fold's expansion applied (v3-1 .. v13: 1)")
    ap.add_argument("--anchor-peak-mode", choices=("max", "centroid"), default="max",
                    help="v14: Te maximum, or Te-weighted centroid of the top 10 percent")
    ap.add_argument("--no-pressure-branch", action="store_true",
                    help="v14: submit the baseline branch's flux (no density/pressure branch)")
    ap.add_argument("--betan-limited-fill", action="store_true",
                    help="v17: limited frames take betaN from trees fitted on the run's diverted frames")
    ap.add_argument("--q95-limited-fill", action="store_true",
                    help="v17: limited frames take the q95 shape factor from the same kind of trees")
    ap.add_argument("--betan-diverted-blend", type=float, default=0.0,
                    help="v18: weight of the trees in the diverted frames' betaN (0 = v17)")
    ap.add_argument("--q95-diverted-blend", type=float, default=0.0,
                    help="v18: weight of the trees in the diverted frames' q95 shape factor (0 = v17)")
    ap.add_argument("--axis-from-baseline", action="store_true",
                    help="submitted axis radius from the run's baseline branch, LCFS kept")
    a = ap.parse_args()
    print(build(a.output_dir, a.workers, a.limit, a.resume, a.stop_after, a.fold, a.steps,
                a.q95_physical, a.edge_pairs, a.sol_divisor, a.vessel_g, a.vessel_tau,
                a.q95_limited_fix, a.anchor_weight, a.anchor_peak, a.fold_fraction,
                a.anchor_peak_mode, not a.no_pressure_branch, a.betan_limited_fill, a.q95_limited_fill,
                a.betan_diverted_blend, a.q95_diverted_blend, a.axis_from_baseline))

if __name__ == "__main__":
    main()
