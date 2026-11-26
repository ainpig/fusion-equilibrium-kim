"""Sensitivity of the seven scored Consistency scalars to each flux mode.

This is what turns the Consistency term into something a regressor can be trained against.  For
frame f and mode m the coefficient is perturbed by +-h*coeff_std[m], the seven scalars are
re-derived with the official scoring code, and

    J[f, j, m] = d f_j / d c_m .

Because ``SS_res_j = sum_f (sum_m J[f,j,m] dc_m)^2`` and ``R2_j = 1 - SS_res_j / (N Var(s_j))``,
a whitened-coefficient loss is proportional to the metric's pooled Consistency residual under

    w_m = coeff_std_m^2 * sum_j mean_f(J[f,j,m]^2) / Var(s_j) .

The full quadratic form is stored too.  Collapsing it to its diagonal assumes the model's
coefficient errors are uncorrelated across modes; they are not, since every mode comes off one
shared trunk.

Step size: ``h`` is deliberately of the order of the model's *actual* coefficient error rather
than infinitesimal.  The LCFS extraction is a bisection over a discrete contour, so a tiny step
measures quantization noise instead of a derivative.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from nm1 import ARTIFACTS, ROOT
from nm1.cache import BASIS, CACHE

sys.path.insert(0, str(ROOT / "fusion_scoring"))
from common import AXIS_SIGN, CONS_SCALARS                    # noqa: E402
from derive import derive_frame                               # noqa: E402
from lcfs import extract_lcfs_with_sign                       # noqa: E402

OUT = ARTIFACTS / "mode_jacobian.npz"
_G: dict = {}


def _init(mask_path: str, pca_mean, comps, coeff_std, n_modes: int, h: float,
          machine: str) -> None:
    m = np.load(mask_path)
    _G.update(R=m["grid_R"], Z=m["grid_Z"], mc=m["mask_coarse"].astype(bool),
              sign=AXIS_SIGN[machine], machine=machine, pca_mean=pca_mean, comps=comps,
              coeff_std=coeff_std, n_modes=n_modes, h=h)
    _G["mf"] = _G["mc"].astype(np.float64)


def _derive(psi_flat: np.ndarray) -> np.ndarray:
    R, Z, mc, mf, s = _G["R"], _G["Z"], _G["mc"], _G["mf"], _G["sign"]
    machine = _G["machine"]
    frame = psi_flat.reshape(65, 65).astype(np.float64)
    contour, _ = extract_lcfs_with_sign(frame, R, Z, machine, mc, mf, axis_sign=s)
    v = derive_frame(frame, R, Z, machine, mc, mf, contour=contour, with_li=True, axis_sign=s)
    return np.array([v[k] for k in CONS_SCALARS])


def _derive_base(coeff: np.ndarray) -> np.ndarray:
    return _derive(_G["pca_mean"] + coeff @ _G["comps"])


def _jacobian_one(coeff: np.ndarray) -> np.ndarray:
    pca_mean, comps, cstd = _G["pca_mean"], _G["comps"], _G["coeff_std"]
    n_modes, h = _G["n_modes"], _G["h"]
    base = pca_mean + coeff @ comps
    J = np.zeros((7, n_modes))
    for m in range(n_modes):
        step = h * cstd[m]
        J[:, m] = (_derive(base + step * comps[m]) - _derive(base - step * comps[m])) / (2 * step)
    return J


class _SerialPool:
    """Drop-in for ProcessPoolExecutor where subprocesses cannot be spawned. Same result, slower."""

    def __init__(self, initializer, initargs):
        initializer(*initargs)

    def map(self, fn, items, chunksize=1):
        return [fn(x) for x in items]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pool(workers: int, initargs):
    if workers and workers > 1:
        try:
            p = ProcessPoolExecutor(workers, initializer=_init, initargs=initargs)
            p.submit(int, 0).result(timeout=60)         # prove a worker can actually start
            return p
        except Exception as exc:                        # noqa: BLE001 - reported, then serial
            print(f"  process pool unavailable ({type(exc).__name__}); running serially")
    return _SerialPool(_init, initargs)


def build(cache: Path = CACHE, basis: Path = BASIS, out: Path = OUT, n_modes: int = 96,
          n_frames: int = 640, h: float = 0.25, workers: int = 12, seed: int = 17,
          var_frames: int = 3000, machine: str = "DIII-D") -> Path:
    state = torch.load(basis, map_location="cpu", weights_only=False)
    pca_mean, comps, cstd = state["pca_mean"], state["pca_components"], state["coeff_std"]
    n_modes = min(n_modes, len(cstd))

    rng = np.random.default_rng(seed)
    with h5py.File(cache, "r") as h5:
        g = h5["train"]
        n = len(g["coeff"])
        coeff = g["coeff"][np.sort(rng.choice(n, min(n_frames, n), replace=False))]
        var_coeff = g["coeff"][np.sort(rng.choice(n, min(var_frames, n), replace=False))]

    if machine != "DIII-D":
        raise ValueError("the mode Jacobian is a DIII-D quantity (it is built from DIII-D flux maps)")
    mask = str(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    initargs = (mask, pca_mean, comps, cstd, n_modes, h, machine)

    started = time.perf_counter()
    print(f"deriving {len(var_coeff)} frames for the pooled-R2 denominators ...", flush=True)
    with _pool(workers, initargs) as pool:
        gv = np.stack(list(pool.map(_derive_base, list(var_coeff), chunksize=8)))
    var_s = np.nanvar(gv, axis=0)
    print("  " + "  ".join(f"{k}={v:.4g}" for k, v in zip(CONS_SCALARS, var_s)), flush=True)

    print(f"computing {len(coeff)} x (7 x {n_modes}) Jacobians, h={h} std ...", flush=True)
    with _pool(workers, initargs) as pool:
        J = np.stack(list(pool.map(_jacobian_one, list(coeff), chunksize=2)))
    finite = np.isfinite(J).all(axis=(1, 2))
    J = J[finite]
    print(f"  usable frames: {finite.sum()}/{len(finite)}  "
          f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)

    J2 = np.nanmean(J ** 2, axis=0)
    contrib = J2 * (cstd[:n_modes] ** 2)[None, :] / var_s[:, None]
    w = contrib.sum(axis=0)
    Jw = J * cstd[:n_modes][None, None, :] / np.sqrt(var_s)[None, :, None]
    # Per scalar as well as summed. The summed form is what the metric weights by, but keeping the
    # seven apart lets the loss rebalance them -- `li` carries the least sensitivity of the seven
    # and sits furthest from its own ceiling, and whether that trade pays is measurable, not
    # derivable. Costs 7x96x96 float64 = 516 KB.
    gram_j = np.einsum("fjm,fjn->jmn", Jw, Jw) / len(Jw)
    gram_j = 0.5 * (gram_j + np.swapaxes(gram_j, 1, 2))
    gram = gram_j.sum(axis=0)
    gram = 0.5 * (gram + gram.T)                        # symmetric by construction
    dg = np.diag(gram)
    off = np.abs(gram - np.diag(dg))
    ev = np.linalg.eigvalsh(gram)
    print(f"gram: diag matches the per-mode weight to {np.abs(dg - w).max() / w.max():.2e} "
          f"relative; off-diagonal mass {off.sum() / np.abs(gram).sum():.3f}; "
          f"eigenvalues in [{ev.min():.3e}, {ev.max():.3e}]", flush=True)

    order = np.argsort(-w)
    cum = np.cumsum(w[order]) / w.sum()
    print(f"top-20 modes by metric weight: {order[:20].tolist()}")
    for frac in (0.5, 0.9, 0.99):
        print(f"  {frac:.0%} of the sensitivity is in the top "
              f"{int(np.searchsorted(cum, frac)) + 1} modes")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, J2=J2, contrib=contrib, weight=w, gram=gram, gram_per_scalar=gram_j,
             var_s=var_s,
             coeff_std=cstd[:n_modes], n_modes=n_modes, h=h, scalars=np.array(CONS_SCALARS))
    out.with_suffix(".json").write_text(json.dumps(
        {"n_modes": n_modes, "h": h, "n_frames": int(finite.sum()),
         "weight": w.tolist(),
         "per_scalar_total": {k: float(contrib[j].sum()) for j, k in enumerate(CONS_SCALARS)}},
        indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--basis", type=Path, default=BASIS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--n-modes", type=int, default=96)
    ap.add_argument("--n-frames", type=int, default=640)
    ap.add_argument("--h", type=float, default=0.25, help="step in units of coeff_std")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--var-frames", type=int, default=3000)
    a = ap.parse_args()
    build(a.cache, a.basis, a.out, a.n_modes, a.n_frames, a.h, a.workers, a.seed, a.var_frames)


if __name__ == "__main__":
    main()