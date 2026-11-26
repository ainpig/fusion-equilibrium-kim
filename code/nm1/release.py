"""fp16 release of a trained ensemble: the files inference reads, with SHA256 sums.

    python -m nm1.release --src <artifact dir> --out <release dir>

Member, transformer and head weights are stored in half precision (the network parameters only; feature and
target statistics, the PCA basis and every other array stay as trained).  Measured on held-out shots the effect
on the score is 1e-6.  The manifest and stack are copied; inference then runs with ``NM1_ARTIFACTS=<release>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def _half(payload: dict) -> dict:
    """The checkpoint with its ``state_dict`` floating-point tensors in float16."""
    out = dict(payload)
    sd = out.get("state_dict")
    if isinstance(sd, dict):
        out["state_dict"] = {k: (v.detach().cpu().half() if torch.is_tensor(v) and v.is_floating_point() else v)
                             for k, v in sd.items()}
    out.pop("opt", None)
    out["release"] = "fp16 state_dict"
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _referenced(src: Path) -> list[str]:
    man = json.loads((src / "members.json").read_text(encoding="utf-8"))
    stack = json.loads((src / "stack.json").read_text(encoding="utf-8"))
    rel = list(man["members"])
    if man.get("specialist"):
        rel.append(man["specialist"])
    fc = stack.get("field_correction") or {}
    rel += list(fc.get("members") or []) + list(fc.get("li_members") or [])
    return list(dict.fromkeys(rel))


def release(src: Path, out: Path, link_from: Path | None = None) -> Path:
    """``link_from``: an earlier release of the same run; files it already holds are hard-linked, not rewritten."""
    import os
    out.mkdir(parents=True, exist_ok=True)
    rels = _referenced(src)
    for rel in rels:
        dst = out / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_from is not None and (Path(link_from) / rel).exists():
            os.link(Path(link_from) / rel, dst)
            side = (Path(link_from) / rel).with_suffix(".json")
            if side.exists():
                os.link(side, dst.with_suffix(".json"))
            continue
        torch.save(_half(torch.load(src / rel, map_location="cpu", weights_only=False)), dst)
        side = (src / rel).with_suffix(".json")
        if side.exists():
            shutil.copyfile(side, dst.with_suffix(".json"))
    for f in ("flux_basis.pt", "members.json", "stack.json"):
        shutil.copyfile(src / f, out / f)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(out).as_posix()}\n" for p in files), encoding="utf-8")
    print(f"released {len(rels)} weight files -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--link-from", type=Path, help="an earlier release of the same run (hard links, no new copies)")
    a = ap.parse_args()
    release(a.src, a.out, a.link_from)


if __name__ == "__main__":
    main()
