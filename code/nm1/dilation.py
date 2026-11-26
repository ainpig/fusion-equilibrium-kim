"""The exact dilation symmetry of the equilibrium problem, as training-data augmentation.

Scale every poloidal current (F-coils, the Ohmic coil, the plasma current) by lambda and the pressure by
lambda^2 and the Grad-Shafranov solution is the same shape with the flux scaled by lambda. Everything the
network reads changes in a known way:

    currents, their derivatives, 20 ms / 100 ms changes, the coil sums   x lambda   (bcoil, the toroidal field, stays)
    bcoil / Ip                                                           / lambda
    log10 of the electron density (pressure x lambda^2 at fixed Te)       + 2 log10 lambda
    flux map                    psi -> lambda psi   =>  c' = lambda c + (lambda - 1) V m
    q95 -> q95 / lambda,  betaN -> lambda betaN      (toroidal field fixed)

A trained member is far from equivariant under it (a 20% scale change moves its prediction by more than its own
error), so the symmetry is information the data never showed it. Implemented for the base 140 block, in the
whitened space the network reads, on the fly.
"""
from __future__ import annotations

import math

import numpy as np
import torch

N_BASE = 140


def base_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(covariant mask, ratio column, log10-density shift multiplicity)`` for the 140 base columns."""
    cov = np.zeros(N_BASE)
    for b in range(4):
        off = 21 * b
        cov[off] = 1.0                                   # ECOILA
        cov[off + 2:off + 21] = 1.0                      # F-coils and Ip; bcoil (off + 1) stays
    cov[[84, 85, 86]] = 1.0                              # top + bottom, top - bottom, rms of the coils
    ratio = np.zeros(N_BASE)
    ratio[87] = 1.0                                      # bcoil / Ip
    shift = np.zeros(N_BASE)
    for block in (88 + 11, 88 + 33):                     # core and edge electron-density summaries
        shift[[block + k for k in (0, 1, 2, 3, 4, 5, 7, 8)]] = 1.0   # quantiles, mean, low/high ends (not std, span, presence)
    shift[132] = shift[133] = 1.0                        # pressure proxies that contain one density mean
    shift[134] = 2.0                                     # ... and the sum that contains two
    return cov, ratio, shift


class BasePlan:
    def __init__(self, fmean: np.ndarray, fstd: np.ndarray, pca_mean: np.ndarray, comps: np.ndarray,
                 coeff_std: np.ndarray, device) -> None:
        cov, ratio, shift = base_columns()
        mean, std = fmean.astype(np.float64), fstd.astype(np.float64)
        t = lambda v: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
        self.cov, self.ratio = t(cov), t(ratio)
        self.a = t(mean / std)                            # mu / sigma
        self.shift = t(shift / std)                       # per-column log10 shift in whitened units
        vm = comps.astype(np.float64) @ pca_mean.astype(np.float64)
        self.vm_z = t(vm / coeff_std.astype(np.float64))

    def apply(self, xb, zb, sb, lam, smean, sstd):
        """Dilate a batch by ``lam`` (B,): inputs, whitened coefficient targets, whitened (q95, betaN)."""
        l = lam[:, None]
        xb = (xb + (l - 1.0) * self.cov * (xb + self.a) + (1.0 / l - 1.0) * self.ratio * (xb + self.a)
              + 2.0 * torch.log10(l) * self.shift)
        zb = l * zb + (l - 1.0) * self.vm_z
        raw = sb * sstd + smean
        raw = torch.stack([raw[:, 0] / lam, raw[:, 1] * lam], dim=1)
        return xb, zb, (raw - smean) / sstd


UNSEEN = -0.5                                            # features2: "this chord does not cover the grid point"


def base_rules(n: int = N_BASE):
    """``(pw, shift, unseen, mask, vloop)`` for the 140 base columns.

    The toroidal field scales with the rest, so q95, betaN and everything the diffusion simulation derives from
    q stay put (checked against a rebuilt block: the current-diffusion columns agree to 4e-8); bcoil / Ip is
    then invariant, and the density summaries shift by 2 log10 lambda.
    """
    pw = np.zeros(n)
    pw[:84] = 1.0                                        # 21 currents (bcoil too), their derivative, 20/100 ms changes
    pw[[84, 85, 86]] = 1.0                               # top + bottom, top - bottom, rms of the coils
    _, _, shift = base_columns()
    z = np.zeros(n, bool)
    return pw, shift, z, z, z


def ext_rules():
    """The 298 features2 columns: coil windows scale (bcoil's do not); the electron-density grids scale with
    lambda^2 (they are linear in ne, and a missing point stays missing); the four density-level crossings of the
    vertical chord cannot be transformed without the profile and are marked missing; log1p(Te) and the
    temperature crossings do not change."""
    from nm1.features2 import CORE_Z, N_CORE, N_EXT, TAN_R
    pw = np.zeros(N_EXT)
    shift = np.zeros(N_EXT)
    unseen = np.zeros(N_EXT, bool)
    mask = np.zeros(N_EXT, bool)
    pw[:6 * 21] = 1.0                                    # every current's window statistics, bcoil's included
    k, core0 = len(CORE_Z), 6 * 21
    pw[core0 + k:core0 + 2 * k] = 2.0
    unseen[core0 + k:core0 + 2 * k] = True
    mask[core0 + 2 * k + 6:core0 + 2 * k + 10] = True
    kt, tan0 = len(TAN_R), core0 + N_CORE
    pw[tan0 + kt:tan0 + 2 * kt] = 2.0
    unseen[tan0 + kt:tan0 + 2 * kt] = True
    return pw, shift, unseen, mask, np.zeros(N_EXT, bool)


def ctx_rules(n: int = 16):
    pw = np.zeros(n)
    pw[1] = 1.0                                          # the shot's peak current; the rest are ratios or times
    z = np.zeros(n, bool)
    return pw, np.zeros(n), z, z, z


def cds_rules(n: int = 41):
    """Current-diffusion block: li, q0 (B_t scales with Ip), the enclosed-current fractions, the resistive-time
    memories and Te are invariant; the flux columns and the Ohmic-coil current scale with lambda; the loop
    voltage (stored as sign * log1p|V|) scales inside the log."""
    pw = np.zeros(n)
    pw[[34, 35, 36, 37, 38, 39]] = 1.0
    vloop = np.zeros(n, bool)
    vloop[[5, 14]] = True
    z = np.zeros(n, bool)
    return pw, np.zeros(n), z, z, vloop


class Plan:
    """Dilation rules for the whole whitened input matrix, registered block by block (as trainer appends them)."""

    def __init__(self) -> None:
        self.parts = {k: [] for k in ("mu", "sigma", "pw", "shift", "unseen", "mask", "vloop")}
        self.width = 0

    def add(self, mean, std, rules) -> None:
        pw, shift, unseen, mask, vloop = rules
        for k, v in (("mu", mean), ("sigma", std), ("pw", pw), ("shift", shift), ("unseen", unseen),
                     ("mask", mask), ("vloop", vloop)):
            self.parts[k].append(np.asarray(v))
        self.width += len(mean)

    def finalize(self, device, pca_mean, comps, coeff_std) -> None:
        cat = {k: np.concatenate(v) for k, v in self.parts.items()}
        f = lambda v: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device)
        self.mu, self.sigma = f(cat["mu"]), f(cat["sigma"])
        self.cols_pw = torch.from_numpy(np.flatnonzero(cat["pw"] != 0)).to(device)
        self.pw = f(cat["pw"][cat["pw"] != 0])
        self.unseen_pw = torch.from_numpy(cat["unseen"][cat["pw"] != 0]).to(device)
        self.cols_sh = torch.from_numpy(np.flatnonzero(cat["shift"] != 0)).to(device)
        self.sh = f(cat["shift"][cat["shift"] != 0])
        self.cols_mask = torch.from_numpy(np.flatnonzero(cat["mask"])).to(device)
        self.cols_v = torch.from_numpy(np.flatnonzero(cat["vloop"])).to(device)
        vm = comps.astype(np.float64) @ pca_mean.astype(np.float64)
        self.vm_z = f(vm / coeff_std.astype(np.float64))

    def apply(self, xb, zb, sb, lam, smean, sstd):
        l = lam[:, None]
        x = xb.float()
        delta = torch.zeros_like(x)
        if len(self.cols_pw):
            raw = x[:, self.cols_pw] * self.sigma[self.cols_pw] + self.mu[self.cols_pw]
            new = raw * torch.pow(l, self.pw[None, :])
            new = torch.where(self.unseen_pw[None, :] & ((raw - UNSEEN).abs() < 2e-3), raw, new)
            delta[:, self.cols_pw] = (new - raw) / self.sigma[self.cols_pw]
        if len(self.cols_sh):
            delta[:, self.cols_sh] += 2.0 * torch.log10(l) * self.sh[None, :] / self.sigma[self.cols_sh]
        if len(self.cols_v):
            y = x[:, self.cols_v] * self.sigma[self.cols_v] + self.mu[self.cols_v]
            y2 = torch.sign(y) * torch.log1p(l * torch.expm1(y.abs()))
            delta[:, self.cols_v] = (y2 - y) / self.sigma[self.cols_v]
        if len(self.cols_mask):
            raw = x[:, self.cols_mask] * self.sigma[self.cols_mask] + self.mu[self.cols_mask]
            delta[:, self.cols_mask] = (UNSEEN - raw) * (lam != 1.0)[:, None].float() / self.sigma[self.cols_mask]
        zb = l * zb + (l - 1.0) * self.vm_z
        return (x + delta).to(xb.dtype), zb, sb            # q95 and betaN do not change


def sample_lambda(n: int, lo: float, hi: float, p: float, device) -> torch.Tensor:
    """Log-uniform scale factors in [lo, hi] for a fraction ``p`` of the rows, 1 for the rest."""
    lam = torch.exp(torch.empty(n, device=device).uniform_(math.log(lo), math.log(hi)))
    return torch.where(torch.rand(n, device=device) < p, lam, torch.ones_like(lam))
