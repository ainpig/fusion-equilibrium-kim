"""The scorer's magnetic-axis functional, per frame and differentiable, on PCA coefficients.

``R_axis`` and ``Z_axis`` are two of the seven Consistency scalars, and ``Z_axis`` is the weakest
of them (R2 0.964 against 0.99+ for the triangularities).  The scorer computes both as

    strongest interior local maximum of phi = AXIS_SIGN * psi   (o_point.find_o_point)
    + a three-point parabolic refinement along R and along Z     (o_point._subpixel)

The Gram surrogate in ``train.py`` sees this only through a Jacobian averaged over 640 frames,
while the refinement's sensitivity depends on where each frame's axis sits on the 5 cm Z grid.
But the refinement is a closed-form function of five flux values, so it can be evaluated exactly
for every frame: find the peak cell once on the *true* flux (``targets``), then evaluate the same
formula on the *predicted* flux at those five points (``predict``) -- which is differentiable.

For a quadratic peak the refinement returns the same vertex from any cell of the stencil, so
evaluating it at the true cell is a faithful surrogate even when the prediction's own peak lands in
a neighbouring cell.
"""
from __future__ import annotations

import numpy as np
import torch

from nm1 import ROOT

MASK = ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz"
AXIS_SIGN = -1.0                                   # DIII-D: the axis is a psi MINIMUM


def _grid():
    z = np.load(MASK)
    return (np.asarray(z["grid_R"], np.float64), np.asarray(z["grid_Z"], np.float64),
            np.asarray(z["mask_coarse"], bool))


def _erode(mask: np.ndarray, cells: int = 2) -> np.ndarray:
    m = mask.copy()
    for _ in range(cells):
        e = m.copy()
        e[1:, :] &= m[:-1, :]
        e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]
        e[:, :-1] &= m[:, 1:]
        m = e
    return m


@torch.no_grad()
def targets(coeff: np.ndarray, pca_mean: np.ndarray, comps: np.ndarray,
            device: str = "cuda", chunk: int = 32768) -> dict[str, np.ndarray]:
    """Peak cell and refined axis of ``pca_mean + coeff @ comps`` for every row.

    Mirrors ``o_point.find_o_point`` exactly: 8-neighbour local maxima inside the envelope mask
    eroded by two cells, the strongest one wins, global argmax inside the mask as the fallback.
    """
    R, Z, mask = _grid()
    inner = torch.from_numpy(_erode(mask, 2)).to(device)
    inner[0, :] = inner[-1, :] = False
    inner[:, 0] = inner[:, -1] = False
    maskt = torch.from_numpy(mask).to(device)
    mean_t = torch.from_numpy(pca_mean.astype(np.float64)).to(device)
    comps_t = torch.from_numpy(comps.astype(np.float64)).to(device)
    n = len(coeff)
    iz_all = np.empty(n, np.int16)
    ir_all = np.empty(n, np.int16)
    ra = np.empty(n, np.float64)
    za = np.empty(n, np.float64)
    dR, dZ = R[1] - R[0], Z[1] - Z[0]
    for a in range(0, n, chunk):
        c = torch.from_numpy(coeff[a:a + chunk].astype(np.float64)).to(device)
        phi = (AXIS_SIGN * (mean_t + c @ comps_t)).reshape(-1, 65, 65)
        B = phi.shape[0]
        ctr = phi[:, 1:-1, 1:-1]
        ge = torch.ones_like(ctr, dtype=torch.bool)
        for dz in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dz == 0 and dr == 0:
                    continue
                ge &= ctr >= phi[:, 1 + dz:64 + dz, 1 + dr:64 + dr]
        lm = torch.zeros_like(phi, dtype=torch.bool)
        lm[:, 1:-1, 1:-1] = ge
        lm &= inner
        cand = torch.where(lm, phi, torch.full_like(phi, -torch.inf)).reshape(B, -1)
        best = cand.argmax(1)
        has = torch.isfinite(cand.gather(1, best[:, None]))[:, 0]
        fb = torch.where(maskt, phi, torch.full_like(phi, -torch.inf)).reshape(B, -1).argmax(1)
        idx = torch.where(has, best, fb)
        iz = idx // 65
        ir = idx % 65
        b = torch.arange(B, device=device)

        def d1(p, q, r):
            den = p - 2 * q + r
            out = torch.clamp(0.5 * (p - r) / torch.where(den == 0, torch.ones_like(den), den),
                              -1.0, 1.0)
            return torch.where(den == 0, torch.zeros_like(out), out)

        izc = iz.clamp(1, 63)
        irc = ir.clamp(1, 63)
        dzv = d1(phi[b, izc - 1, ir], phi[b, izc, ir], phi[b, izc + 1, ir])
        drv = d1(phi[b, iz, irc - 1], phi[b, iz, irc], phi[b, iz, irc + 1])
        dzv = torch.where((iz > 0) & (iz < 64), dzv, torch.zeros_like(dzv))
        drv = torch.where((ir > 0) & (ir < 64), drv, torch.zeros_like(drv))
        s = slice(a, a + B)
        iz_all[s] = iz.cpu().numpy()
        ir_all[s] = ir.cpu().numpy()
        ra[s] = R[ir.cpu().numpy()] + drv.cpu().numpy() * dR
        za[s] = Z[iz.cpu().numpy()] + dzv.cpu().numpy() * dZ
    return {"iz": iz_all, "ir": ir_all, "R_axis": ra, "Z_axis": za}


def cached_targets(path=None) -> tuple[dict, dict]:
    """Axis targets for every ``train`` and ``val`` row of the coefficient cache, built once."""
    import h5py

    from nm1 import ARTIFACTS
    from nm1.cache import BASIS, CACHE

    path = path or ARTIFACTS / "axis_targets.npz"
    keys = ("iz", "ir", "R_axis", "Z_axis")
    if path.exists():
        z = np.load(path)
        return ({k: z["tr_" + k] for k in keys}, {k: z["va_" + k] for k in keys})
    state = torch.load(BASIS, map_location="cpu", weights_only=False)
    with h5py.File(CACHE, "r") as h5:
        tr = targets(h5["train"]["coeff"][:], state["pca_mean"], state["pca_components"])
        va = targets(h5["val"]["coeff"][:], state["pca_mean"], state["pca_components"])
    np.savez(path, **{"tr_" + k: v for k, v in tr.items()}, **{"va_" + k: v for k, v in va.items()})
    return tr, va


class AxisHead:
    """Differentiable refined axis of a predicted flux map, evaluated at given peak cells."""

    def __init__(self, pca_mean: np.ndarray, comps: np.ndarray, coeff_std: np.ndarray,
                 device: str | torch.device = "cuda"):
        R, Z, _ = _grid()
        self.dR, self.dZ = float(R[1] - R[0]), float(Z[1] - Z[0])
        self.R = torch.from_numpy(R).float().to(device)
        self.Z = torch.from_numpy(Z).float().to(device)
        # whitened coefficients -> flux: psi = mean + (z * std) @ comps
        self.basis = torch.from_numpy((comps * coeff_std[:, None]).T.astype(np.float32)
                                      ).to(device)                      # (4225, n_comp)
        self.mean = torch.from_numpy(pca_mean.astype(np.float32)).to(device)

    def predict(self, z: torch.Tensor, iz: torch.Tensor, ir: torch.Tensor):
        """``(R_axis, Z_axis)`` of the flux implied by whitened coefficients ``z``."""
        iz = iz.long().clamp(1, 63)
        ir = ir.long().clamp(1, 63)
        pts = torch.stack([iz * 65 + ir, iz * 65 + ir - 1, iz * 65 + ir + 1,
                           (iz - 1) * 65 + ir, (iz + 1) * 65 + ir], 1)        # (B, 5)
        rows = self.basis[pts]                                               # (B, 5, n_comp)
        psi = self.mean[pts] + torch.einsum("bpk,bk->bp", rows, z.float())
        phi = AXIS_SIGN * psi
        c, rl, rr, zl, zr = phi.unbind(1)

        def d1(p, q, r):
            den = p - 2 * q + r
            den = torch.minimum(den, torch.full_like(den, -1e-9))          # a peak is concave
            return torch.clamp(0.5 * (p - r) / den, -1.0, 1.0)

        return (self.R[ir] + d1(rl, c, rr) * self.dR, self.Z[iz] + d1(zl, c, zr) * self.dZ)
