"""Vacuum vertical-field null Z from coil currents alone.

Physics, no labels: a plasma ring sits where the externally applied radial field
vanishes, because F_Z = -2*pi*R*Ip*B_R. B_R = -(1/R) dpsi/dZ is computed from the
released coil geometry and the measured circuit currents, so this quantity is
defined identically on both machines and reads no equilibrium target whatsoever.
"""
import numpy as np
from my_experiments.mast_equilibrium_v12 import circuit_basis
from my_experiments.fusion_pipeline import MAST_COILS, _interp

# Circuits that shape the poloidal field. The plasma's own current is excluded (it
# is what we are locating), and the TF rod carries no poloidal flux.
PF_COLUMNS = [c for c in MAST_COILS if c not in
              ('magnetics_plasma_current', 'magnetics_tf_current')]


class VerticalNull:
    def __init__(self, row, r, z):
        self.r = np.asarray(r, float)
        self.z = np.asarray(z, float)
        basis = circuit_basis(row, self.r, self.z)          # (n_circuit, nz, nr) Wb/rad/A
        self.keep = [MAST_COILS.index(c) for c in PF_COLUMNS]
        self.basis = np.asarray(basis, np.float64)[self.keep]
        # dpsi/dZ of each circuit's Green map: B_R = -(1/R) dpsi/dZ.
        self.dbasis = np.gradient(self.basis, self.z, axis=1)

    def currents(self, row, times):
        return np.column_stack([_interp(row['magnetics_time'], row[k], times)
                                for k in PF_COLUMNS]) * 1000.0

    def null(self, row, times, r_query, window=0.6):
        """Z of the vacuum B_R = 0 surface at each frame's plasma major radius.

        Returns (Z_null, strength) where strength is |d B_R / dZ| at the null --
        the decay index proxy that says how firmly the field pins the plasma.
        """
        amps = self.currents(row, times)
        r_query = np.clip(np.asarray(r_query, float), self.r[0], self.r[-1])
        ir = np.clip(np.searchsorted(self.r, r_query)-1, 0, len(self.r)-2)
        w = (r_query-self.r[ir])/(self.r[ir+1]-self.r[ir])
        # dpsi/dZ profile along Z at each frame's own major radius
        prof = np.einsum('tc,czr->tzr', amps, self.dbasis, optimize=True)
        t = np.arange(len(times))
        col = prof[t, :, ir]*(1-w)[:, None]+prof[t, :, ir+1]*w[:, None]   # (T, nz)
        br = -col/r_query[:, None]
        inside = np.abs(self.z) <= window
        zz = self.z[inside]
        b = br[:, inside]
        out = np.full(len(times), np.nan)
        slope = np.full(len(times), np.nan)
        sign = np.sign(b)
        cross = np.nonzero(sign[:, :-1]*sign[:, 1:] < 0)
        # Keep, per frame, the crossing closest to the midplane.
        best = {}
        for t, k in zip(*cross):
            if t not in best or abs(zz[k]) < abs(zz[best[t]]):
                best[t] = k
        for t, k in best.items():
            b0, b1 = b[t, k], b[t, k+1]
            frac = b0/(b0-b1)
            out[t] = zz[k]+frac*(zz[k+1]-zz[k])
            slope[t] = (b1-b0)/(zz[k+1]-zz[k])
        return out, slope
