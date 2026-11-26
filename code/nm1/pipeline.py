"""v1 build: the base DIII-D ensemble from the official parquets, in two training runs.

    NM1_ARTIFACTS=<final dir> python -m nm1.pipeline build --dev <development dir> [--dry-run] [--lanes 1]

1 prepare        PCA basis, caches, current-diffusion block, mode Jacobian (``nm1.prepare``, CPU)
2 dev stage 1    development run on the training split, epochs picked on the calibration third:
                 16 first-stage members, the flux specialist, 4 out-of-fold members (half of the training shots each)
3 cdsg           current-diffusion block on the flux coordinate: training rows on the out-of-fold first stage,
                 held-out rows on the out-of-fold blend (``nm1.cds_geo``)
4 dev stage 2    the two second-stage members of the development run
5 dev heads      the 9 shape heads whose encoders the development run has; development manifest and stack
6 final members  71 members + specialist on every labelled shot (``--all-data``); a member with a development twin
                 stops at the twin's epoch and source, the rest at the end of their schedule
7 final heads    the 12 shape heads on the final encoders; final manifest (``members.json``) and ``stack.json``

Every step skips what exists, so a stopped build resumes.  Weights: each group's weight is split evenly over its
members (flux: dprod 1.5, xd 1, ctx 1, cds 4, cdsg 5; q95: prod 1, ext 2, ctx 2, dlq 5); the first stage of the
second-stage block is the flux blend without the cdsg group.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from nm1 import ARTIFACTS, OUTPUTS, ROOT, WORK

# ---------------------------------------------------------------- member definitions
_LI = ["--scalar-weights", "1,1,1,1,1,1,3"]
_HIST = ["--temporal-context=-40,-15,-5,-1"]
_WIDE = ["--width", "1536", "--depth", "8", "--ema", "0.999"]
_V1 = ["--ctx", "--cds", "--cds-columns", "34"]
_V2 = ["--ctx", "--cds", "--cds-columns", "41"]
_V3 = ["--ctx", "--cds", "--cds-columns", "52"]
_D = ["--base-only", "--cds", "--cds-columns", "41"]
_XD = ["--cds", "--cds-columns", "41"]
_G = ["--ctx", "--cdsg", "--cdsg-columns", "41"]
_CG = [*_V2, "--cdsg", "--cdsg-columns", "41"]
_DIL = ["--dilate", "0.75,1.33", "--dilate-p", "0.5", "--scalar-scale"]


def _w(kind: str, seed) -> list[str]:
    return ["--weights", "jacobian" if kind == "jac" else "gram", "--seed", str(seed)]


def _base(kind, seed, *extra):
    return ["--base-only", *_w(kind, seed), *extra]


# (name, group, train arguments, epochs) in the order of the submitted manifest
MEMBERS: list[tuple[str, str, list[str], int]] = [
    # prod: base block only
    ("jac_s1234", "prod", _base("jac", 1234), 70), ("jac_s7", "prod", _base("jac", 7), 70),
    ("jac_s21", "prod", _base("jac", 21), 70), ("jac_s99", "prod", _base("jac", 99), 70),
    ("gram_s4001", "prod", _base("gram", 4001), 70), ("gram_s4002", "prod", _base("gram", 4002), 70),
    ("gram_s4003", "prod", _base("gram", 4003), 70), ("gram_s4004", "prod", _base("gram", 4004), 70),
    ("hist_s4301", "prod", _base("gram", 4301, *_HIST), 70), ("hist_s4302", "prod", _base("gram", 4302, *_HIST), 150),
    ("vac_s4102", "prod", _base("gram", 4102, "--vac-features", "64"), 70),
    ("wide_s4701", "prod", _base("gram", 4701, *_WIDE), 70),
    ("jacli_s1234", "prod", _base("jac", 1234, *_LI), 70), ("jacli_s7", "prod", _base("jac", 7, *_LI), 70),
    ("gramli_s4001", "prod", _base("gram", 4001, *_LI), 70), ("gramli_s4002", "prod", _base("gram", 4002, *_LI), 70),
    # ext: + features2 block
    ("xgram_s4001", "ext", _w("gram", 4001), 70), ("xgram_s4002", "ext", _w("gram", 4002), 70),
    ("xjac_s1234", "ext", _w("jac", 1234), 70), ("xjac_s7", "ext", _w("jac", 7), 70),
    ("xjacli_s1234", "ext", [*_w("jac", 1234), *_LI], 70),
    # ctx: + shot context
    ("xcgram_s4001", "ctx", [*_w("gram", 4001), "--ctx"], 70), ("xcgram_s4002", "ctx", [*_w("gram", 4002), "--ctx"], 70),
    ("xcjac_s1234", "ctx", [*_w("jac", 1234), "--ctx"], 70), ("xcjac_s7", "ctx", [*_w("jac", 7), "--ctx"], 70),
    ("xcjacli_s1234", "ctx", [*_w("jac", 1234), "--ctx", *_LI], 70),
    # cds: + current-diffusion block
    ("xcdgram_s4001", "cds", [*_w("gram", 4001), *_V1], 70), ("xcdjac_s1234", "cds", [*_w("jac", 1234), *_V1], 70),
    ("xcd2gram_s4001", "cds", [*_w("gram", 4001), *_V2], 70), ("xcd2jac_s1234", "cds", [*_w("jac", 1234), *_V2], 70),
    ("xcdgram_s4002", "cds", [*_w("gram", 4002), *_V1], 70),
    ("xcd2jacli_s1234", "cds", [*_w("jac", 1234), *_V2, *_LI], 70),
    ("xcd2jac_s7", "cds", [*_w("jac", 7), *_V2], 70), ("xcd2gram_s4002", "cds", [*_w("gram", 4002), *_V2], 70),
    ("xcd2gram_s4003", "cds", [*_w("gram", 4003), *_V2], 70), ("xcd2jac_s21", "cds", [*_w("jac", 21), *_V2], 70),
    ("xcd2wide_s4701", "cds", [*_w("gram", 4701), *_WIDE, *_V2], 70),
    ("xcd3gram_s4001", "cds", [*_w("gram", 4001), *_V3], 70), ("xcd3jac_s1234", "cds", [*_w("jac", 1234), *_V3], 70),
    ("xcd2gram_s4004", "cds", [*_w("gram", 4004), *_V2], 70), ("xcd2jac_s99", "cds", [*_w("jac", 99), *_V2], 70),
    # dprod: base + current-diffusion block
    ("dgram_s4001", "dprod", [*_w("gram", 4001), *_D], 70), ("djac_s1234", "dprod", [*_w("jac", 1234), *_D], 70),
    ("dgram_s4002", "dprod", [*_w("gram", 4002), *_D], 70), ("dgram_s4003", "dprod", [*_w("gram", 4003), *_D], 70),
    ("djac_s7", "dprod", [*_w("jac", 7), *_D], 70), ("dgram_s4004", "dprod", [*_w("gram", 4004), *_D], 70),
    ("djac_s21", "dprod", [*_w("jac", 21), *_D], 70),
    ("dhist_s4301", "dprod", [*_w("gram", 4301), *_D, *_HIST], 70),
    ("dvac_s4102", "dprod", [*_w("gram", 4102), *_D, "--vac-features", "64"], 70),
    # xd: features2 + current-diffusion block, no shot context
    ("xdgram_s4001", "xd", [*_w("gram", 4001), *_XD], 70), ("xdjac_s1234", "xd", [*_w("jac", 1234), *_XD], 70),
    ("xdgram_s4002", "xd", [*_w("gram", 4002), *_XD], 70), ("xdjac_s7", "xd", [*_w("jac", 7), *_XD], 70),
    # cdsg: second stage (current-diffusion block on the flux coordinate)
    ("oxcdg2jac_s1234", "cdsg", [*_w("jac", 1234), *_G], 70), ("oxcdcg2jac_s1234", "cdsg", [*_w("jac", 1234), *_CG], 70),
    ("oxcdg2jac_s7", "cdsg", [*_w("jac", 7), *_G], 70), ("oxcdg2jac_s21", "cdsg", [*_w("jac", 21), *_G], 70),
    ("oxcdg2jac_s99", "cdsg", [*_w("jac", 99), *_G], 70), ("oxcdcg2jac_s7", "cdsg", [*_w("jac", 7), *_CG], 70),
    ("oxcdcg2jac_s21", "cdsg", [*_w("jac", 21), *_CG], 70),
    # dlq: dilation-augmented, scaled q95 head
    ("dlxcd2gram_s4001", "dlq", [*_w("gram", 4001), *_V2, *_DIL], 70),
    ("dlxcd2gram_s4002", "dlq", [*_w("gram", 4002), *_V2, *_DIL], 70),
    ("dlxcd2gram_s4003", "dlq", [*_w("gram", 4003), *_V2, *_DIL], 70),
    ("dlxcd2gram_s4004", "dlq", [*_w("gram", 4004), *_V2, *_DIL], 70),
    ("dlxcd2jac_s1234", "dlq", [*_w("jac", 1234), *_V2, *_DIL], 70),
    ("dlxcd2jac_s7", "dlq", [*_w("jac", 7), *_V2, *_DIL], 70),
    ("dlxcd2jac_s21", "dlq", [*_w("jac", 21), *_V2, *_DIL], 70),
    ("dlxcd2jac_s99", "dlq", [*_w("jac", 99), *_V2, *_DIL], 70),
    ("dlxcjac_s1234", "dlq", [*_w("jac", 1234), "--ctx", *_DIL], 70),
    ("dlxcgram_s4001", "dlq", [*_w("gram", 4001), "--ctx", *_DIL], 70),
]
SPECIALIST = ("psi_specialist", ["--base-only", "--weights", "gram", "--seed", "5101",
                                 "--loss-weights", "1.0,0.0,0.25", "--select", "r2_psi"], 70)
OOF = [("oofjac_f0", [*_w("jac", 1234), *_V2, "--folds", "2", "--fold", "0"], 70),
       ("oofjac_f1", [*_w("jac", 1234), *_V2, "--folds", "2", "--fold", "1"], 70),
       ("oofgram_f0", [*_w("gram", 4001), *_V2, "--folds", "2", "--fold", "0"], 70),
       ("oofgram_f1", [*_w("gram", 4001), *_V2, "--folds", "2", "--fold", "1"], 70)]
STAGE1_OOF = [["members/oofjac_f0.pt", "members/oofgram_f0.pt"], ["members/oofjac_f1.pt", "members/oofgram_f1.pt"]]
# the development run: one or more members of every group (the submitted development manifest)
DEV = ["gram_s4001", "jacli_s1234", "xjac_s1234", "xcgram_s4001", "xcjac_s1234", "xcd2gram_s4001", "xcd2jac_s1234",
       "xcd2jacli_s1234", "xcd2jac_s7", "dgram_s4001", "djac_s1234", "dgram_s4002", "xdgram_s4001", "xdjac_s1234",
       "oxcdg2jac_s1234", "oxcdcg2jac_s7", "dlxcd2gram_s4001", "dlxcd2jac_s1234"]
GROUP_ORDER = ["prod", "ext", "ctx", "cds", "dprod", "xd", "cdsg", "dlq"]
GROUP_WEIGHTS = {"dprod": 1.5, "xd": 1.0, "ctx": 1.0, "cds": 4.0, "cdsg": 5.0}
GROUP_WEIGHTS_Q95 = {"prod": 1.0, "ext": 2.0, "ctx": 2.0, "dlq": 5.0}
SECOND_STAGE = "cdsg"

# shape heads of the field correction: (prefix, seed, encoder, label file, streamed, frame-weighted, target weights)
_UNIFORM, _EXACTW = "1,1,1,1,1,1,1", "0.3,0.3,2,2,2,1,0.5"
HEADS = [
    ("v12", 12002, "members/oxcdg2jac_s1234.pt", "labels.h5", False, False, _UNIFORM),
    ("v12", 12003, "members/oxcdcg2jac_s7.pt", "labels.h5", False, False, _UNIFORM),
    ("v12", 12102, "members/oxcdg2jac_s1234.pt", "labels_exact.h5", True, False, _UNIFORM),
    ("v12", 12103, "members/oxcdcg2jac_s7.pt", "labels_exact.h5", True, False, _UNIFORM),
    ("v12", 12104, "members/oxcdg2jac_s99.pt", "labels_exact.h5", True, True, _EXACTW),
    ("v12", 12105, "members/oxcdcg2jac_s21.pt", "labels_exact.h5", True, True, _EXACTW),
] + [("v13", seed, enc, "labels_exact.h5", True, False, "0.3,0.3,2,2,2,0.5,2") for seed, enc in (
    (13001, "members/xcd2jac_s1234.pt"), (13002, "members/djac_s1234.pt"), (13003, "members/xcjac_s1234.pt"),
    (13004, "members/dlxcd2jac_s1234.pt"), (13005, "members/oxcdg2jac_s7.pt"), (13006, "members/xdjac_s1234.pt"))]

STACK = {
    "fitted_on": "structure and ratios inherited from the earlier v-series selection on DIII-D held-out shots; "
                 "re-checked in nm1 on the selection half",
    "selected_on": "selection half, official functionals (DIII-D only); see docs/NEW_V1_METHODS.md",
    "evaluation_half_untouched": False,
    "scalar_stack": None,
    "flux_projection": {"amplitude_weight": 0.1, "gauge_weight": 0.2, "note": "same as v3 (v3 psi specialist)"},
    "temporal": None,
    "interior_bump": {"enabled": False},
    "li_guard": {"enabled": True, "li_max": 4.9, "knee": 4.0, "slope": 0.4, "workers": 6,
                 "note": "li above the knee is relabelled to knee + slope*(li - knee), capped at li_max, inside the "
                         "LCFS only (nm1/guard.py); contours unchanged"},
    "inference": {"offload_members": True,
                  "note": "members live in host RAM and visit the GPU one at a time (an 8 GB card cannot hold 71 "
                          "members); set false on a large GPU"},
    "field_correction": {
        "enabled": True, "method": "v13", "members": [],
        "groups": ["dprod", "xd", "ctx", "cds", "cdsg", "prod", "ext", "dlq"], "workers": 10,
        "config": {
            "recipe": {"R_axis": ["base", 0.0, 0.0], "Z_axis": ["base", 0.0, 0.0], "kappa": ["groups", 0.0, 0.35],
                       "tri_top": ["groups", 0.0, 0.35], "tri_bot": ["groups", 0.0, 0.35],
                       "volume": ["base", 0.0, 0.35], "li": ["base", 0.5, 0.5]},
            "objective_weights": [0, 0, 1, 1, 1, 1, 0], "strength": 1.0, "eps": 0.5, "iters": 2, "ridge": 0.001,
            "clip": 0.4, "fractions": [1.0, 0.6, 0.3], "skip_below": 0.0005, "axis_strength": 1.0,
            "axis_max_change": 0.012, "li_strength": 1.0, "li_max_change": 0.15, "offset_group": "dlq",
            "offset_weight": 0.5}},
}


# ---------------------------------------------------------------- helpers
def _done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _env(artifacts: Path) -> dict:
    return {**os.environ, "NM1_ARTIFACTS": str(artifacts), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}


def _run(cmd: list[str], artifacts: Path, log: Path, dry: bool) -> None:
    full = [sys.executable, "-u", "-m", *cmd]
    print(f"   [{artifacts.name}] {' '.join(cmd)}", flush=True)
    if dry:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        for attempt in range(10):          # Windows refuses CreateProcess now and then under load
            try:
                rc = subprocess.run(full, cwd=ROOT, env=_env(artifacts), stdout=f, stderr=subprocess.STDOUT).returncode
                break
            except OSError as exc:
                print(f"     launch failed ({exc}); retrying in 60 s", flush=True)
                time.sleep(60)
        else:
            raise SystemExit(f"could not launch {cmd[0]}")
    if rc != 0:
        raise SystemExit(f"{cmd[0]} failed (exit {rc}), log {log}")
    print(f"     {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def _member(name: str, args: list[str], epochs: int, artifacts: Path, logs: Path, dry: bool,
            extra: list[str] = ()) -> None:
    out = artifacts / "members" / f"{name}.pt"
    if _done(out) and _done(out.with_suffix(".json")):
        return
    base_only = "--base-only" in args
    args = [a for a in args if a != "--base-only"]
    _run(["nm1.train", "--output", str(out), "--epochs", str(epochs), *([] if base_only else ["--ext"]), *args,
          *extra], artifacts, logs / f"{artifacts.name}_{name}.log", dry)


def _row(name: str):
    for n, g, a, e in MEMBERS:
        if n == name:
            return n, g, a, e
    raise KeyError(name)


def manifest(artifacts: Path, names: list[str], note: str) -> dict:
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for n in names:
        groups[_row(n)[1]].append(f"members/{n}.pt")
    groups = {g: v for g, v in groups.items() if v}
    members = [f"members/{n}.pt" for n in names]

    def split(gw: dict, exclude=()) -> list[float]:
        tot = sum(w for g, w in gw.items() if groups.get(g) and g not in exclude)
        w = [0.0] * len(members)
        for g, gwt in gw.items():
            if g in exclude or not groups.get(g):
                continue
            for m in groups[g]:
                w[members.index(m)] = gwt / len(groups[g]) / tot
        return w

    return {"members": members, "weights": split(GROUP_WEIGHTS), "weights_q95": split(GROUP_WEIGHTS_Q95),
            "stage1_weights": split(GROUP_WEIGHTS, exclude=(SECOND_STAGE,)), "stage1_oof": STAGE1_OOF,
            "specialist": f"members/{SPECIALIST[0]}.pt", "groups": groups,
            "group_weights": dict(GROUP_WEIGHTS), "group_weights_q95": dict(GROUP_WEIGHTS_Q95), "note": note}


def write_manifest(artifacts: Path, names: list[str], dry: bool) -> Path:
    path = artifacts / "members.json"
    man = manifest(artifacts, names, "New v1 (nm1): 71-member two-stage ensemble, see nm1/pipeline.py")
    if dry:
        print(f"   write {path} ({len(names)} members)")
        return path
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"   wrote {path} ({len(names)} members)", flush=True)
    return path


def heads(artifacts: Path, names: list[str], lab_shared: Path, logs: Path, dry: bool, workers: int) -> list[str]:
    base = ["nm1.shapeheads"]
    for step, made in (("labels", "labels.h5"), ("exact-labels", "labels_exact.h5")):
        if not (lab_shared / made).exists():
            _run(base + [step, "--out", str(lab_shared), "--workers", str(workers)], artifacts,
                 logs / f"heads_{step}.log", dry)
    lab = artifacts / "heads_lab"
    out = []
    for prefix, seed, encoder, labels, stream, weighted, tw in HEADS:
        if Path(encoder).stem not in names:          # the development run has 9 of the 12 encoders
            continue
        target = artifacts / "heads" / f"shape_s{seed}.pt"
        out.append(f"heads/{target.name}")
        if _done(target):
            continue
        if not dry:
            lab.mkdir(parents=True, exist_ok=True)
            for f in ("labels.h5", "labels_exact.h5"):
                if not (lab / f).exists():
                    shutil.copyfile(lab_shared / f, lab / f)
        cmd = base + ["train", "--out", str(lab), "--seed", str(seed), "--epochs", "100", "--encoder", encoder,
                      "--labels-source", labels, "--target-weights", tw]
        cmd += (["--stream-training"] if stream else []) + (["--frame-weighted"] if weighted else [])
        _run(cmd, artifacts, logs / f"{artifacts.name}_head_s{seed}.log", dry)
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lab / f"shape_s{seed}.pt", target)
    return out


def write_stack(artifacts: Path, head_list: list[str], dry: bool) -> None:
    s = json.loads(json.dumps(STACK))
    s["field_correction"]["members"] = head_list
    if dry:
        print(f"   write {artifacts / 'stack.json'} ({len(head_list)} heads)")
        return
    (artifacts / "stack.json").write_text(json.dumps(s, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- build
def build(a) -> None:
    dev, fin = a.dev.resolve(), ARTIFACTS.resolve()
    if dev == fin:
        raise SystemExit("the development and the final run need two artifact directories")
    logs = WORK / "logs_build"
    dry = a.dry_run
    t0 = time.perf_counter()

    print("== 1 prepare", flush=True)
    _run(["nm1.prepare", "all", "--workers-io", str(a.workers_io), "--workers-cpu", str(a.workers_cpu)], dev,
         logs / "prepare.log", dry)
    if not dry:
        (fin / "members").mkdir(parents=True, exist_ok=True)
        for f in ("flux_basis.pt", "mode_jacobian.npz"):
            if not (fin / f).exists():
                shutil.copyfile(dev / f, fin / f)

    print("== 2 development run, first stage", flush=True)
    for n in DEV:
        if _row(n)[1] != SECOND_STAGE:
            _member(n, _row(n)[2], _row(n)[3], dev, logs, dry)
    _member(SPECIALIST[0], SPECIALIST[1], SPECIALIST[2], dev, logs, dry)
    for n, args, ep in OOF:
        _member(n, args, ep, dev, logs, dry)
    write_manifest(dev, DEV, dry)

    print("== 3 second-stage block", flush=True)
    from nm1.cds_geo import CDSG_CACHE
    if not _done(CDSG_CACHE):
        _run(["nm1.cds_geo", "--manifest", str(dev / "members.json"), "--workers", str(a.workers_cpu)], dev,
             logs / "cds_geo.log", dry)

    print("== 4 development run, second stage", flush=True)
    for n in DEV:
        if _row(n)[1] == SECOND_STAGE:
            _member(n, _row(n)[2], _row(n)[3], dev, logs, dry)

    print("== 5 development heads, manifest, stack", flush=True)
    lab_shared = OUTPUTS / "shape_heads_lab"
    write_stack(dev, heads(dev, DEV, lab_shared, logs, dry, a.workers_io), dry)
    write_manifest(dev, DEV, dry)

    print("== 6 final run: every member on every labelled shot", flush=True)
    for n, args, ep in OOF:                       # the second-stage block's first stage, shared by both runs
        src, dst = dev / "members" / f"{n}.pt", fin / "members" / f"{n}.pt"
        if not dry and not dst.exists():
            shutil.copyfile(src, dst)
            shutil.copyfile(src.with_suffix(".json"), dst.with_suffix(".json"))
    order = [(SPECIALIST[0], SPECIALIST[1], SPECIALIST[2])] + [(n, a_, e) for n, _, a_, e in MEMBERS]
    for n, args, ep in order:
        extra = ["--all-data"]
        twin = dev / "members" / f"{n}.json"
        if twin.exists():                        # a development twin (DEV members and the specialist): its epoch
            t = json.loads(twin.read_text(encoding="utf-8"))
            extra += ["--save-epoch", str(int(t["epoch"])), "--save-source", str(t.get("source", "raw"))]
        elif n in DEV:
            if not dry:
                raise SystemExit(f"development twin of {n} missing: {twin}")
            extra += ["--save-epoch", "<twin>", "--save-source", "<twin>"]
        else:                                    # no twin: the end of its schedule
            extra += ["--save-epoch", str(int(ep)), "--save-source", "raw"]
        _member(n, args, ep, fin, logs, dry, extra)

    print("== 7 final heads, manifest, stack", flush=True)
    write_stack(fin, heads(fin, [r[0] for r in MEMBERS], lab_shared, logs, dry, a.workers_io), dry)
    write_manifest(fin, [r[0] for r in MEMBERS], dry)
    print(f"build done in {(time.perf_counter() - t0) / 3600:.1f} h", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build",))
    ap.add_argument("--dev", type=Path, required=True, help="artifact directory of the development run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers-io", type=int, default=6)
    ap.add_argument("--workers-cpu", type=int, default=6)
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
