"""Deployment wrapper for the fixed-coil correction with a saturation fallback.

An expansion factor clipped at v21's existing gain limit is an unresolved
displacement estimate. In that case retain v21 rather than propagate the
truncated estimate into another physical boundary-condition update. The gate
uses only that pre-existing inference limit, never an identity or label.
"""
import numpy as np
from my_experiments.coil_preserving_v22 import CoilPreserving
from my_experiments.vertical_transfer_v21 import GAIN_LIMIT


class MastCorrection:
    def __init__(self, row, r, z, mask, fold):
        self.enabled = bool(float(fold['gain']) < GAIN_LIMIT)
        self.coil = CoilPreserving(row, r, z, mask) if self.enabled else None

    def predict(self, row, times, shifted, shift_m):
        if not self.enabled:
            return np.asarray(shifted, np.float32).copy()
        return self.coil.candidates(row, times, shifted, shift_m)['balanced_coils']
