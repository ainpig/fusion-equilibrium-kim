"""Agreement score: the official metric with a reference submission standing in for the ground truth.

    python tools/agree.py --machine MAST --ref ref/mast_private_test.npz --cand out/mast.npz [--shots 0-49] [--workers 6]

The scorer modules are the starter kit's own (the official starter kit @4b6af06, path in $STARTER). Identical files give
S = 1. Only shots present in the candidate are scored, so a partial run can be checked early.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import os
STARTER = Path(os.environ.get("STARTER") or (Path(__file__).resolve().parents[1] / "starter"))
sys.path.insert(0, str(STARTER / "fusion_scoring"))

from common import AXIS_SIGN, CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

MASK = {"DIII-D": "d3d_envelope.npz", "MAST": "mast_envelope.npz"}
N_POINTS, N_ITER = 512, 22
LI_IDX = CONS_SCALARS.index("li")
_G = {}


def _grid(machine):
    m = np.load(STARTER / "fusion_scoring" / "masks" / MASK[machine])
    mc = m["mask_coarse"].astype(bool)
    return m["grid_R"], m["grid_Z"], mc, mc.astype(np.float64)


def shot_ids(z):
    return sorted({int(k[5:9]) for k in z.files if k.endswith("_psirz")})


def load(z, i):
    return {"psi": z[f"shot_{i:04d}_psirz"].astype(np.float64),
            "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
            "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)}


def ref_orientation(machine, path, ids):
    """Factor that puts the reference file in the machine's stored convention (as real ground truth is)."""
    R, Z, mc, mf = _grid(machine)
    z = np.load(path)
    ok = {1.0: 0, -1.0: 0}
    for i in ids[:: max(1, len(ids) // 8)][:8]:
        psi = z[f"shot_{i:04d}_psirz"].astype(np.float64)
        for k in range(0, psi.shape[0], max(1, psi.shape[0] // 4)):
            for s in ok:
                ok[s] += extract_lcfs_with_sign(psi[k], R, Z, machine, mc, mf, axis_sign=s)[0] is not None
    s = max(ok, key=ok.get)
    return s * AXIS_SIGN[machine]


def reference(args):
    machine, path, i, fac = args
    R, Z, mc, mf = _grid(machine)
    g = load(np.load(path), i)
    g["psi"] *= fac
    T = g["psi"].shape[0]
    cont, cons = [], np.full((T, N_CONS), np.nan)
    for k in range(T):
        c = extract_lcfs(g["psi"][k], R, Z, machine, mc, mf, n_points=N_POINTS)
        cont.append(c)
        v = derive_frame(g["psi"][k], R, Z, machine, mc, mf, contour=c)
        cons[k] = [v[n] for n in CONS_SCALARS]
    rg = [major_radius(c) for c in cont if c is not None]
    return i, cont, cons, np.isfinite(cons), (float(np.mean(rg)) if rg else float("nan")), g


def score_one(args):
    machine, i, ref, gt, pred, psi_sign, means = args
    R, Z, mc, mf = _grid(machine)
    mean_psi, mean_scal, mean_cons = means
    contours, cons_gt, cmask, rgeo = ref
    T = gt["psi"].shape[0]
    axis_sign = AXIS_SIGN[machine] * psi_sign
    p = pred["psi"]
    ss_psi = {s: float(np.sum((gt["psi"] - s * p) ** 2)) for s in PSI_SIGNS}
    ss_scal = np.zeros(N_SCALARS)
    for j, n in enumerate(SCALARS):
        g, q = gt[n], pred[n]
        d = np.isfinite(g)
        q = np.where(np.isfinite(q[d]), q[d], mean_scal[j])
        ss_scal[j] = float(np.sum((g[d] - q) ** 2))
    ss_cons = np.zeros(N_CONS)
    cf, cfail = np.zeros(N_CONS, np.int64), np.zeros(N_CONS, np.int64)
    ds, nf, nfr = [], 0, 0
    for k in range(T):
        rc, need = contours[k], cmask[k]
        if rc is None and not need.any():
            continue
        ex, _ = extract_lcfs_with_sign(p[k], R, Z, machine, mc, mf, n_iter=N_ITER, axis_sign=axis_sign)
        if rc is not None:
            nfr += 1
            if ex is None:
                nf += 1
                ds.append(1.0)
            else:
                ds.append(min(1.0, symmetric_hausdorff(ex, np.asarray(rc)) / rgeo))
        if need.any():
            v = derive_frame(p[k], R, Z, machine, mc, mf, contour=ex, with_li=bool(need[LI_IDX]), axis_sign=axis_sign)
            for j in np.nonzero(need)[0]:
                x = v[CONS_SCALARS[j]]
                cf[j] += 1
                if not np.isfinite(x):
                    x = mean_cons[j]
                    cfail[j] += 1
                ss_cons[j] += (cons_gt[k, j] - x) ** 2
    tot = float(np.sum((gt["psi"] - mean_psi) ** 2))
    return {"ss_res_psi": ss_psi, "psi_pixels": int(gt["psi"].size), "psi_nonfinite": 0,
            "ss_res_scal": ss_scal, "ss_res_cons": ss_cons, "cons_frames": cf, "cons_fails": cfail,
            "shot_r2_psi": {s: 1 - ss_psi[s] / tot for s in PSI_SIGNS}, "shot_dlcfs": float(np.mean(ds)) if ds else None,
            "lcfs_frames": nfr, "lcfs_fails": nf, "ref_lcfs_dropped": sum(c is None for c in contours), "missing": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["DIII-D", "MAST"], required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--shots", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    zc = np.load(a.cand)
    ids = shot_ids(zc)
    if a.shots:
        lo, hi = map(int, a.shots.split("-"))
        ids = [i for i in ids if lo <= i <= hi]
    fac = ref_orientation(a.machine, a.ref, ids)
    with ProcessPoolExecutor(a.workers) as ex:
        refs = {r[0]: r[1:] for r in ex.map(reference, [(a.machine, a.ref, i, fac) for i in ids])}
    ps = pss = pn = 0.0
    ssum, ssq, sn = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    csum, csq, cn = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for i in ids:
        cont, cons, cm, rg, g = refs[i]
        ps += g["psi"].sum(); pss += (g["psi"] ** 2).sum(); pn += g["psi"].size
        for j, n in enumerate(SCALARS):
            v = g[n][np.isfinite(g[n])]
            ssum[j] += v.sum(); ssq[j] += (v ** 2).sum(); sn[j] += v.size
        for j in range(N_CONS):
            v = cons[cm[:, j], j]
            csum[j] += v.sum(); csq[j] += (v ** 2).sum(); cn[j] += v.size
    stats = dict(psi_sum=ps, psi_sumsq=pss, psi_n=pn, scal_sum=ssum, scal_sumsq=ssq, scal_n=sn,
                 cons_sum=csum, cons_sumsq=csq, cons_n=cn)
    means = (ps / pn, ssum / np.maximum(sn, 1), csum / np.maximum(cn, 1))
    preds = {i: load(zc, i) for i in ids}
    tot = {s: sum(float(np.sum((refs[i][4]["psi"] - s * preds[i]["psi"]) ** 2)) for i in ids) for s in PSI_SIGNS}
    sign = min(PSI_SIGNS, key=tot.get)
    print(f"reference orientation factor {fac:+.0f}, candidate sign {sign:+.0f}")
    acc = Accum()
    acc.psi_sign = sign
    jobs = [(a.machine, i, refs[i][:4], refs[i][4], preds[i], sign, means) for i in ids]
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(score_one, jobs):
            acc.add(part)
    r = finalize_machine(acc, stats)
    print(f"{a.machine} shots={len(ids)}  agreement S={r['S']:.6f}  R2psi={r['r2_psi']:.6f}  "
          f"R2qb={r['r2_qb']:.6f} {r['r2_each']}  1-D={1 - r['dlcfs']:.6f}  cons={r['consistency']:.6f}")
    print("cons:", r["r2_cons_each"], " lcfs_fail", r["lcfs_fail_frac"], " sign", r["psi_sign"])


if __name__ == "__main__":
    main()
