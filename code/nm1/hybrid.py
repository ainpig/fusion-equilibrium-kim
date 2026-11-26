"""Per-key composition of two DIII-D prediction files (Challenge 1).

The scorer grades every output key on its own: the flux map (R2_psi, D_LCFS and the seven derived scalars) and
the two submitted scalars q95 and betaN.  The base chain gives the better flux and boundary; the chain
self-trained on the fold's inputs (``nm1.adapt``) gives the better q95, betaN and internal inductance li.  The
submitted file therefore takes

    psirz      from the base file, with each frame's li moved to the adapted file's li by the monotone
               relabelling inside the LCFS that the field correction already uses (``nm1.v12_shape.relabel_li``):
               the contour, hence D_LCFS, kappa, triangularity and volume, stays the base's (frames whose li
               differs by less than THRESH are left as they are)
    q95, betaN from the adapted file


    python -m nm1.hybrid <base npz> <adapted npz> <out npz> [--workers 10] [--scalars-only]
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nm1 import ROOT

THRESH = 0.002          # frames whose li differs by less are left as they are


def _li_job(args):
    sys.path.insert(0, str(ROOT / "fusion_scoring"))
    from nm1.v12_shape import relabel_li
    from nm1.v13_shape import _grid_mask, functionals
    base, srcs, R, Z = args
    mask, mf = _grid_mask(R, Z)
    out = np.array(base, np.float64, copy=True)
    moved = 0
    for k in range(len(base)):
        t = np.mean([functionals(src[k].astype(np.float64), R, Z, mask, mf)[6] for src in srcs])
        b0 = functionals(out[k], R, Z, mask, mf)[6]
        if np.isfinite(t) and np.isfinite(b0) and abs(t - b0) > THRESH:
            out[k], note = relabel_li(out[k], R, Z, float(t), 1.0, 0.5)
            moved += note.get("status") == "corrected"
    return out.astype(np.float32), moved


def _grid(fold: str):
    from nm1.features import read_row
    from nm1.fetch import listing, local_path
    row = read_row(local_path(listing(f"diii_d_{fold}_test")[0]), drop_targets=True)
    return np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)


def compose(base: Path, adapted: list[Path], out: Path, fold: str = "public", workers: int = 10,
            scalars_only: bool = False) -> Path:
    """Write the composed file.  `adapted` may hold several files: their q95 / betaN and li are averaged."""
    from nm1.predict import NpzStreamWriter
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    b = np.load(base)
    srcs = [np.load(p) for p in adapted]

    def mean_key(k):
        if len(srcs) == 1:
            return srcs[0][k]
        return np.mean([s[k] for s in srcs], 0).astype(srcs[0][k].dtype)

    keys = list(b.keys())
    shots = sorted({k.rsplit("_", 1)[0] for k in keys if k.endswith("_psirz")})
    wr = NpzStreamWriter(out)
    moved = 0
    try:
        if scalars_only:
            for k in keys:
                wr.write(k, b[k] if k.endswith("_psirz") else mean_key(k))
        else:
            R, Z = _grid(fold)
            # resumable: every finished shot is kept in <out>.parts/ (atomic write), so a run killed by an
            # out-of-memory stop restarts from the shots it had not finished; a shot's part is reused only if
            # the base and source files are the ones it was computed from (their sizes and mtimes in parts.json)
            parts = out.with_name(out.name + ".parts")
            parts.mkdir(exist_ok=True)
            stamp = {"base": [str(base), Path(base).stat().st_size, Path(base).stat().st_mtime],
                     "adapted": [[str(p), Path(p).stat().st_size, Path(p).stat().st_mtime] for p in adapted]}
            sj = parts / "parts.json"
            if sj.exists() and json.loads(sj.read_text(encoding="utf-8")) != json.loads(json.dumps(stamp)):
                raise RuntimeError(f"{parts} belongs to other input files; move it away first")
            sj.write_text(json.dumps(stamp), encoding="utf-8")

            def done(sh):
                return (parts / f"{sh}.npz").exists()

            # a bounded window of shots in flight: submitting every shot at once queues gigabytes of pickled maps
            with ProcessPoolExecutor(workers) as pool:
                pending, nxt = deque(), 0
                for k in keys:
                    s, kind = k.rsplit("_", 1)
                    while nxt < len(shots) and len(pending) < 2 * workers:
                        sh = shots[nxt]
                        pending.append((sh, None if done(sh) else
                                        pool.submit(_li_job, (b[f"{sh}_psirz"], [x[f"{sh}_psirz"] for x in srcs], R, Z))))
                        nxt += 1
                    if kind == "psirz":
                        s0, fut = pending.popleft()
                        assert s0 == s, (s0, s)
                        if fut is None:
                            with np.load(parts / f"{s0}.npz") as f:
                                v, m = f["v"], int(f["m"])
                        else:
                            v, m = fut.result()
                            tmp = parts / f"{s0}.tmp.npz"
                            np.savez(tmp, v=v, m=np.int64(m))
                            os.replace(tmp, parts / f"{s0}.npz")
                        moved += m
                    else:
                        v = mean_key(k)
                    wr.write(k, v)
    finally:
        wr.close()
    print(f"wrote {out} ({len(shots)} shots; li moved on {moved} frames)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path)
    ap.add_argument("adapted", help="one adapted prediction file, or several joined by commas (averaged)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--fold", choices=("public", "private"), default="public")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--scalars-only", action="store_true", help="take q95 / betaN only; keep the base flux unchanged")
    a = ap.parse_args()
    compose(a.base, [Path(p) for p in a.adapted.split(",")], a.out, a.fold, a.workers, a.scalars_only)


if __name__ == "__main__":
    main()