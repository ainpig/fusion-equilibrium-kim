"""Test fixed-coil boundary conditions after a v21 vertical displacement.

This module reads inputs and predictions only. The full variant translates the
inferred plasma field while retaining the measured vacuum field. The balanced
variant additionally removes its uniform-field increment at the predicted axis;
R**2 and Z are homogeneous Grad-Shafranov solutions, so this adjustment adds no
toroidal plasma current. The remaining higher multipoles can change asymmetry.
No coefficient, shot grouping, or offset is fitted to MAST equilibrium labels.
"""
import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.interpolate import RectBivariateSpline
from my_experiments.vertical_null_v21 import VerticalNull
from my_experiments.vertical_transfer_v21 import extracted_axis


class CoilPreserving:
    def __init__(self, row, r, z, mask):
        self.null = VerticalNull(row, r, z)
        self.r, self.z, self.mask = np.asarray(r, float), np.asarray(z, float), mask
        self.rr, self.zz = np.meshgrid(self.r, self.z)

    def candidates(self, row, times, shifted, shift_m):
        shifted = np.asarray(shifted, np.float64)
        if abs(shift_m) < 1e-9:
            return {k: shifted.astype(np.float32) for k in ('fixed_coils', 'balanced_coils')}
        vacuum = np.einsum('tc,czr->tzr', self.null.currents(row, times), self.null.basis, optimize=True)
        displaced = np.stack([ndshift(f, (shift_m/(self.z[1]-self.z[0]), 0.), order=3, mode='nearest') for f in vacuum])
        correction = vacuum-displaced
        full = correction-correction.mean((1, 2))[:, None, None]
        balanced = correction.copy()
        own_axis = extracted_axis(shifted, self.r, self.z, self.mask)
        for i, (ra, za) in enumerate(own_axis):
            spl = RectBivariateSpline(self.z, self.r, correction[i])
            gr = float(spl.ev(za, ra, dx=0, dy=1))
            gz = float(spl.ev(za, ra, dx=1, dy=0))
            balanced[i] -= gr*(self.rr**2-ra**2)/(2*ra)+gz*(self.zz-za)
        balanced -= balanced.mean((1, 2))[:, None, None]
        result = dict(fixed_coils=(shifted+full).astype(np.float32),
                      balanced_coils=(shifted+balanced).astype(np.float32))
        assert all(np.isfinite(p).all() for p in result.values())
        return result
