"""The flux *gradient* as a quadratic form in the same coefficients the model predicts.

The loss so far constrains `psi` and, through the measured Jacobian, the seven scalars derived
from its contour. Nothing constrains the field *inside* the boundary, and two of the scalars are
built from exactly that:

    li ~ integral |grad psi|^2 / R  dR dZ        (poloidal field energy)
    Delta* psi = -mu0 R J_phi                     (the Grad-Shafranov operator: the current)

A model can match `psi` closely, get the contour right, and still carry the wrong internal
gradient -- which is the shape the physics-informed operator literature reports for purely
data-driven models, and it is where `li` and `kappa` sit against their ceilings here.

The saving grace is linearity. The reconstruction is `psi = mean + sum_m c_m phi_m`, so a
coefficient error `dc` produces a gradient error `sum_m dc_m grad(phi_m)`, and

    integral |grad d_psi|^2 w  =  dc^T G dc ,   G[m,n] = sum_pixels w (dR phi_m dR phi_n
                                                                      + dZ phi_m dZ phi_n)

`G` depends only on the basis, so it is computed once here and the training term costs one more
matrix product against a 256x256 matrix -- the same shape and the same cost as the Gram term the
Consistency surrogate already uses.

`w = 1/R` is not arbitrary: `dV = 2 pi R dR dZ` and `B_pol = |grad psi| / R`, so the poloidal
field energy `integral B_pol^2 dV` is `2 pi integral |grad psi|^2 / R dR dZ`. Weighting by `1/R`
makes this term the field energy of the error, which is what `li` normalises.

    python -m nm1.gradient            # write artifacts/grad_gram.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nm1 import ARTIFACTS
from nm1.cache import BASIS, training_shots

OUT = ARTIFACTS / "grad_gram.npz"


def grid() -> tuple[np.ndarray, np.ndarray]:
    """The DIII-D flux grid, read from a training shot rather than hard-coded."""
    import pandas as pd
    row = pd.read_parquet(training_shots("train", allow_partial=True)[0],
                          columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    return np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)


def build(basis: Path = BASIS, out: Path = OUT) -> Path:
    st = torch.load(basis, map_location="cpu", weights_only=False)
    comps = st["pca_components"].astype(np.float64)          # (n_comp, 4225)
    cstd = st["coeff_std"].astype(np.float64)
    n = len(comps)
    r, z = grid()
    side = int(round(len(comps[0]) ** 0.5))
    if side * side != comps.shape[1] or side != len(r):
        raise SystemExit(f"basis is {comps.shape[1]} long, grid is {len(r)}x{len(z)}")

    maps = comps.reshape(n, len(z), len(r))
    # Central differences with the real spacing; `np.gradient` handles the edges one-sided.
    dz_, dr_ = np.gradient(maps, z, r, axis=(1, 2))
    w = (1.0 / r)[None, :] * np.ones((len(z), 1))            # the 1/R of the field energy
    w = w / w.sum()                                          # keep the matrix O(1)

    flat_r = (dr_ * np.sqrt(w)[None]).reshape(n, -1)
    flat_z = (dz_ * np.sqrt(w)[None]).reshape(n, -1)
    G = flat_r @ flat_r.T + flat_z @ flat_z.T
    G = 0.5 * (G + G.T)

    # Whiten into the units the trainer works in (`dz = dc / coeff_std`) and normalise by the
    # gradient energy the basis itself carries, so the term reads like the other `1 - R2` terms.
    Gw = G * cstd[:, None] * cstd[None, :]
    total = float(np.trace(Gw))
    Gw /= total

    ev = np.linalg.eigvalsh(Gw)
    off = np.abs(Gw - np.diag(np.diag(Gw)))
    print(f"grad gram: {n} modes, off-diagonal mass {off.sum() / np.abs(Gw).sum():.3f}, "
          f"eigenvalues in [{ev.min():.3e}, {ev.max():.3e}]", flush=True)
    order = np.argsort(-np.diag(Gw))
    cum = np.cumsum(np.diag(Gw)[order])
    for frac in (0.5, 0.9, 0.99):
        print(f"  {frac:.0%} of the gradient energy is in the top "
              f"{int(np.searchsorted(cum, frac)) + 1} modes")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, gram=Gw, raw=G, coeff_std=cstd, grid_R=r, grid_Z=z)
    out.with_suffix(".json").write_text(json.dumps(
        {"n_modes": n, "weight": "1/R", "normalised_by": "trace", "total_before_norm": total,
         "off_diagonal_mass": float(off.sum() / np.abs(Gw).sum())}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", type=Path, default=BASIS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    build(a.basis, a.out)


if __name__ == "__main__":
    main()
