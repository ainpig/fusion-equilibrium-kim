"""v2 build: a transformer chain (T2 ``recipes/v2.json``, T3 ``recipes/v3_t3.json``) on top of the final ensemble
(``nm1.pipeline build``).  Every stage skips what already exists, so a stopped build resumes.

    python -m nm1.v2build all --recipe <recipe> --dev <development run> --final <final run> --release <release dir>
                              [--dry-run]

Stages of ``all``:
  1 g1           the recipe's two ramp-down x3 / li x3 cdsg members: a development twin on the training split picks
                 the epoch, the final member is trained on every labelled shot and stops there
  2 lcfs         boundary-sensitivity labels of the training frames (``nm1.lcfs_points``, ``<dev>/lcfs_points.h5``)
  3 transformers for each recipe entry: warm start from its final member, then the boundary + LCFS-Ip fine-tune
                 (``nm1.train_tfm``; a transformer already trained from the same source with the same arguments, in
                 ``<final>/members_tfm``, is used as it is)
  4 release      fp16 release (``nm1.release``) + G1 members + transformers; manifest ``members_v2.json``: the G1
                 members join their group, each transformer takes the recipe's share of its group's flux weight
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from nm1 import ROOT


def _done(path: Path) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def _run(cmd: list[str], artifacts: Path, log: Path, dry: bool) -> None:
    """``python -m <cmd>`` with ``NM1_ARTIFACTS=<artifacts>``, output appended to ``log``."""
    print(f"   [{Path(artifacts).name}] {' '.join(cmd)}", flush=True)
    if dry:
        return
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NM1_ARTIFACTS": str(artifacts), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        for attempt in range(10):          # Windows refuses CreateProcess now and then under load
            try:
                rc = subprocess.run([sys.executable, "-u", "-m", *cmd], cwd=ROOT, env=env, stdout=f,
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


def stage_g1(rec, dev: Path, final: Path, logs: Path, dry: bool) -> None:
    for m in rec["g1"]["members"]:
        name, args = m["name"], shlex.split(m["args"])
        d = Path(dev) / "members_g1" / f"{name}.pt"
        if not (_done(d) and _done(d.with_suffix(".json"))):
            _run(["nm1.train", "--output", str(d), *args], dev, logs / f"g1_dev_{name}.log", dry)
        f = Path(final) / "members_g1" / f"{name}.pt"
        if not (_done(f) and _done(f.with_suffix(".json"))):
            if dry and not d.with_suffix(".json").exists():
                twin = {"epoch": "<twin>", "source": "<twin>"}
            else:
                twin = json.loads(d.with_suffix(".json").read_text(encoding="utf-8"))
            _run(["nm1.train", "--output", str(f), *args, "--all-data", "--save-epoch", str(twin["epoch"]),
                  "--save-source", str(twin.get("source", "raw"))], final, logs / f"g1_final_{name}.log", dry)


def stage_lcfs(rec, dev: Path, logs: Path, dry: bool) -> None:
    out = Path(dev) / "lcfs_points.h5"
    if not _done(out):
        K = int(((rec.get("transformer") or {}).get("lcfs_points") or {}).get("K", 32))
        _run(["nm1.lcfs_points", "--out", str(out), "--K", str(K)], dev, logs / "lcfs_points.log", dry)


def stage_transformers(rec, dev: Path, final: Path, logs: Path, dry: bool) -> list[tuple[Path, str]]:
    t = rec["transformer"]
    out = []
    for r in t["replace"]:
        src, extra = r["source"], shlex.split(r.get("extra", ""))
        init = Path(final) / r.get("init", "members") / f"{src}.pt"
        warm = Path(final) / "members_tfm" / f"tw_{src}.pt"
        if not _done(warm):
            _run(["nm1.train_tfm", "--output", str(warm), "--all-data", "--save-epoch", str(t["final_epoch"]), "--resume",
                  *shlex.split(t["args"]), *extra, "--init-member", str(init)], final, logs / f"tfm_final_{src}.log", dry)
        bnd = Path(final) / "members_tfm" / f"tw_{src}_bnd.pt"
        if not _done(bnd):
            _run(["nm1.train_tfm", "--output", str(bnd), "--all-data", "--resume", *shlex.split(t["boundary"]), *extra,
                  "--init-seq", str(warm), "--bnd-file", str(Path(dev) / "lcfs_points.h5")], final,
                 logs / f"tfm_bnd_{src}.log", dry)
        out.append((bnd, r["group"]))
    return out


def stage_release(rec, final: Path, release: Path, tfms, logs: Path, dry: bool) -> Path:
    from nm1.v5build import v5_manifest
    if not (Path(release) / "members.json").exists():
        sib = [Path(release).parent / d for d in ("release_t4",) if (Path(release).parent / d / "members.json").exists()]
        _run(["nm1.release", "--src", str(final), "--out", str(release), *(["--link-from", str(sib[0])] if sib else [])],
             final, logs / "release.log", dry)
    path = Path(release) / f"members_{rec.get('name', 'v2')}.json"
    if dry:
        print(f"   write {path} (G1 members + {len(tfms)} transformers, recipe weights)")
        return path
    import torch
    from nm1.release import _half
    for sub, names in (("members_g1", [m["name"] for m in rec["g1"]["members"]]), ("members_tfm", [p.stem for p, _ in tfms])):
        (Path(release) / sub).mkdir(exist_ok=True)
        for n in names:
            dst = Path(release) / sub / f"{n}.pt"
            if not dst.exists():
                torch.save(_half(torch.load(Path(final) / sub / f"{n}.pt", map_location="cpu", weights_only=False)), dst)
    base = json.loads((Path(release) / "members.json").read_text(encoding="utf-8"))
    man = v5_manifest(rec, base, [(f"members_tfm/{p.name}", g) for p, g in tfms])
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"   wrote {path} ({len(man['members'])} members)")
    return path


def cmd_all(a) -> None:
    rec = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    logs = Path(a.release).parent / f"logs_{rec.get('name', 'v2')}"
    print("== 1 G1 members"); stage_g1(rec, a.dev, a.final, logs, a.dry_run)
    print("== 2 boundary labels"); stage_lcfs(rec, a.dev, logs, a.dry_run)
    print("== 3 transformers"); tfms = stage_transformers(rec, a.dev, a.final, logs, a.dry_run)
    print("== 4 release + manifest"); stage_release(rec, a.final, a.release, tfms, logs, a.dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("all",))
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.dev, a.final, a.release = a.dev.resolve(), a.final.resolve(), a.release.resolve()
    cmd_all(a)


if __name__ == "__main__":
    main()
