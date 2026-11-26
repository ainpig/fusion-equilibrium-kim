"""T4 build: the T4 chain (``recipes/v4.json``), on top of the final ensemble (``nm1.pipeline build``).
Every stage skips what already exists, so a stopped build resumes.

    python -m nm1.v5build all --dev <development run> --final <final run> --release <release dir> [--dry-run]
    python -m nm1.v5build predict --dev <dev> --final <final> --release <release> --base-release <base release>
                          --fold public|private --out <dir>

Stages of ``all``:
  1 g1           the two ramp-down / li specialist members (as v2)
  2 lcfs         boundary-sensitivity labels (as v2)
  3 transformers for each recipe entry: warm start from its final member (``init``: the G1 member for the two
                 li-tail ones, with their li x3 / ramp x3 weights), boundary + LCFS-Ip fine-tune, then the moe3
                 extension (three discharge-phase experts grown from the trained transformer)
  4 release      fp16 release + G1 members + the moe3 transformers; manifest: each transformer takes the recipe's
                 share of its group's flux weight; ``seq_groups`` (if enabled in the recipe) lets the field
                 correction's group maps mix in the group's transformers
  5 chains       the T2 and T3 chains (``recipes/v2.json``, ``recipes/v3_t3.json``) through ``nm1.v2build all``

``predict``: the T4 chain, the T2 / T3 chains and the fold's self-trained chain (``nm1.adapt``) on the fold, then
the li-mean composition (li relabelled to the mean of the four chains' li, ``nm1.hybrid``) with the self-trained
chain's q95 / betaN (h1).  The MAST file comes from ``python -m nm1.infer --leg mast`` (``nm1.mast_phys``).
"""
from __future__ import annotations

import argparse
import json
import time
import shlex
from pathlib import Path

from nm1 import ROOT
from nm1.v2build import _done, _run, stage_g1, stage_lcfs

RECIPE = ROOT / "recipes" / "v5.json"
NAMES = {"public": "diii_d_public_test.npz", "private": "diii_d_private_test.npz"}


def stage_transformers(rec, dev: Path, final: Path, logs: Path, dry: bool) -> list[tuple[Path, str]]:
    t = rec["transformer"]
    out = []
    for r in t["replace"]:
        src, extra = r["source"], shlex.split(r.get("extra", ""))
        init = final / r.get("init", "members") / f"{src}.pt"
        warm = final / "members_tfm" / f"tw_{src}.pt"
        if not _done(warm):
            _run(["nm1.train_tfm", "--output", str(warm), "--all-data", "--save-epoch", str(t["final_epoch"]), "--resume",
                  *shlex.split(t["args"]), *extra, "--init-member", str(init)], final, logs / f"tfm_final_{src}.log", dry)
        bnd = final / "members_tfm" / f"tw_{src}_bnd.pt"
        if not _done(bnd):
            _run(["nm1.train_tfm", "--output", str(bnd), "--all-data", "--resume", *shlex.split(t["boundary"]), *extra,
                  "--init-seq", str(warm), "--bnd-file", str(dev / "lcfs_points.h5")], final, logs / f"tfm_bnd_{src}.log", dry)
        last = bnd
        if t.get("moe3"):
            moe = final / "members_tfm" / f"tw_{src}_bnd_moe3.pt"
            if not _done(moe):
                _run(["nm1.train_tfm", "--output", str(moe), "--all-data", "--resume", *shlex.split(t["moe3"]), *extra,
                      "--bnd-file", str(dev / "lcfs_points.h5"), "--init-seq", str(bnd)], final, logs / f"tfm_moe3_{src}.log", dry)
            last = moe
        out.append((last, r["group"]))
    return out


def v5_manifest(rec, base: dict, tfms: list[tuple[str, str]]) -> dict:
    """G1 members join their group (group weight unchanged), every transformer takes its group's share."""
    d = json.loads(json.dumps(base))
    n0 = len(d["members"])
    grp = rec["g1"]["group"]
    new = [f"members_g1/{m['name']}.pt" for m in rec["g1"]["members"]]
    d["members"] += new
    d["groups"][grp] = d["groups"][grp] + new
    gw = d["group_weights"]; tot = sum(gw[g] for g in gw if d["groups"].get(g))
    w = [0.0] * len(d["members"])
    for g, ms in d["groups"].items():
        if g in gw:
            for m in ms:
                w[d["members"].index(m)] = gw[g] / len(ms) / tot
    for k, v in list(d.items()):
        if k != "weights" and isinstance(v, list) and len(v) == n0 and all(isinstance(x, (int, float)) for x in v):
            d[k] = v + [0.0] * len(new)
    n1 = len(d["members"])
    share = rec["transformer"]["group_share"]
    by_group: dict[str, list[str]] = {}
    for rel, g in tfms:
        by_group.setdefault(g, []).append(rel)
    for g, rels in by_group.items():
        idx = [d["members"].index(m) for m in d["groups"][g]]
        gtot = sum(w[i] for i in idx)
        for i in idx:
            w[i] *= (1.0 - share[g])
        for rel in rels:
            d["members"].append(rel); w.append(share[g] * gtot / len(rels))
    d["weights"] = w
    for k, v in list(d.items()):
        if k != "weights" and isinstance(v, list) and len(v) == n1 and all(isinstance(x, (int, float)) for x in v):
            d[k] = v + [0.0] * (len(d["members"]) - n1)
    sg = rec.get("seq_groups") or {}
    if sg.get("enabled"):
        d["seq_groups"] = {g: {"share": float(sg["share"]), "members": rels} for g, rels in by_group.items()}
    d["note"] = rec["note"]
    return d


def stage_release(rec, final: Path, release: Path, tfms, logs: Path, dry: bool) -> Path:
    if not (release / "members.json").exists():
        _run(["nm1.release", "--src", str(final), "--out", str(release)], final, logs / "release_v5.log", dry)
    path = release / f"members_{rec.get('name', 'v5')}.json"
    if dry:
        print(f"   write {path} (G1 members + {len(tfms)} transformers, recipe weights)")
        return path
    import torch
    from nm1.release import _half
    for sub, names in (("members_g1", [m["name"] for m in rec["g1"]["members"]]), ("members_tfm", [p.stem for p, _ in tfms])):
        (release / sub).mkdir(exist_ok=True)
        for n in names:
            dst = release / sub / f"{n}.pt"
            if not dst.exists():
                torch.save(_half(torch.load(final / sub / f"{n}.pt", map_location="cpu", weights_only=False)), dst)
    base = json.loads((release / "members.json").read_text(encoding="utf-8"))
    man = v5_manifest(rec, base, [(f"members_tfm/{p.name}", g) for p, g in tfms])
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"   wrote {path} ({len(man['members'])} members)")
    return path


def cmd_all(a) -> None:
    rec = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    logs = Path(a.release).parent / "logs_v5"
    print("== 1 G1 members"); stage_g1(rec, a.dev, a.final, logs, a.dry_run)
    print("== 2 boundary labels"); stage_lcfs(rec, a.dev, logs, a.dry_run)
    print("== 3 transformers"); tfms = stage_transformers(rec, a.dev, a.final, logs, a.dry_run)
    print("== 4 release + manifest"); stage_release(rec, a.final, a.release, tfms, logs, a.dry_run)
    print("== 5 T2 / T3 chains")
    for tag, r in rec["composition"]["chains"].items():
        rel = Path(a.release).parent / f"release_{tag}"
        _run(["nm1.v2build", "all", "--recipe", str(ROOT / r), "--dev", str(a.dev), "--final", str(a.final),
              "--release", str(rel)], a.final, logs / f"chain_{tag}.log", a.dry_run)


def cmd_predict(a) -> None:
    rec = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    out = Path(a.out); logs = out / "logs"; name = NAMES[a.fold]
    chains = {"t5": (Path(a.release), f"members_{rec.get('name', 'v5')}.json")}
    for tag in rec["composition"]["chains"]:
        chains[tag] = (Path(a.release).parent / f"release_{tag}", "members_v2.json")
    preds = {}

    def _done(d: Path) -> bool:
        # an inference run is complete only when it wrote infer_run.json (after validation); the prediction file
        # itself is streamed under its final name, so its existence alone does not mean the run finished
        if (d / "infer_run.json").exists() and (d / name).exists():
            return True
        if d.exists() and any(d.iterdir()) and not a.dry_run:
            # a run stopped part-way (power cut, kill): set it aside so the rerun gets the fresh directory it needs
            aside = d.with_name(d.name + time.strftime(".incomplete_%m%d_%H%M%S"))
            d.replace(aside)
            print(f"   incomplete {d.name} moved to {aside.name}", flush=True)
        return False

    for tag, (rel, man) in chains.items():
        d = out / f"chain_{tag}"
        if not _done(d):
            _run(["nm1.infer", "--fold", a.fold, "--leg", "d3d", "--manifest", str(rel / man), "--out", str(d)], rel,
                 logs / f"infer_{tag}.log", a.dry_run)
        preds[tag] = d / name
    adapted = out / "self_trained"
    if not _done(adapted):
        work = out / "adapt_work"
        rel_adapted = work / f"release_adapted_{a.fold}"
        if not (rel_adapted / "members.json").exists():
            _run(["nm1.adapt", "run", "--fold", a.fold, "--dev", str(a.dev), "--base", str(a.final), "--base-release",
                  str(a.base_release), "--out", str(out / "adapted_artifacts"), "--work", str(work)],
                 a.final, logs / "adapt.log", a.dry_run)
        _run(["nm1.infer", "--fold", a.fold, "--leg", "d3d", "--out", str(adapted)], rel_adapted,
             logs / "infer_self_trained.log", a.dry_run)
    mid = out / "li_mean" / name
    order = [preds["t5"]] + [preds[t] for t in rec["composition"]["chains"]] + [adapted / name]
    config = name[:-len(".npz")]

    def _valid(f: Path) -> bool:
        # a composed file counts only when it holds every shot of the fold (a failed run can leave a stub)
        if not f.exists() or a.dry_run:
            return f.exists()
        from nm1.validate import validate
        return validate(f, config) == 0

    if not _valid(mid):
        if mid.exists():
            mid.replace(mid.with_name(mid.name + time.strftime(".invalid_%m%d_%H%M%S")))
        _run(["nm1.hybrid", str(preds["t5"]), ",".join(str(p) for p in order), str(mid), "--workers", "6",
              "--fold", a.fold], Path(a.release), logs / "li_mean.log", a.dry_run)
    final = out / name
    _run(["nm1.hybrid", str(mid), str(adapted / name), str(final), "--workers", "4", "--scalars-only",
          "--fold", a.fold], Path(a.release), logs / "h1.log", a.dry_run)
    if not a.dry_run and not _valid(final):
        raise SystemExit(f"{final} failed validation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("all", "predict"))
    ap.add_argument("--recipe", type=Path, default=RECIPE)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--base-release", type=Path, help="predict: the v1 fp16 release (the self-training starts from it)")
    ap.add_argument("--fold", choices=("public", "private"), default="public")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.dev, a.final, a.release = a.dev.resolve(), a.final.resolve(), a.release.resolve()
    (cmd_all if a.command == "all" else cmd_predict)(a)


if __name__ == "__main__":
    main()