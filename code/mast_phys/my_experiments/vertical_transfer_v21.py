"""Label-free vertical re-placement of a frozen MAST flux fold.

The axis heights our reconstruction extracts are less spread across the fold than the
vacuum vertical-field null: the height at which the measured coil currents put B_R = 0,
which is where force balance F_Z = -2*pi*R*Ip*B_R places a current ring (on the public
fold the two spreads are 3.4 cm and 6.7 cm). The expansion factor is set by matching the
spread of our own extracted axis heights to the spread of the null -- a distribution
alignment between two quantities computed from the submitted inputs and our own
predictions. No MAST target of any kind enters, and no limit on where an axis may sit
is taken from MAST: the only bound on a shot's re-placement is that of the null itself.
The fold only expands; an axis is never carried past the height that shot's own coil
currents call for, and one that already sits at or beyond it is left where it is.

Only shot-level constants are applied; within-shot motion is left untouched.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import shift as ndshift

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'fusion_scoring'))
from o_point import find_o_point                                       # noqa: E402
from my_experiments.vertical_null_v21 import VerticalNull              # noqa: E402

# Saturation guard on the fold-level expansion factor, and the flag the later stages read to see whether it
# saturated (`gain < GAIN_LIMIT`). A generic bound, not a fitted number: the raw gain would have to reach it
# for it to act.
GAIN_LIMIT = 6.0
AXIS_SIGN_MAST = 1.0


def extracted_axis(psi, r, z, mask):
    """Scorer-exact O-point per frame, using the published finder unmodified."""
    out = np.empty((len(psi), 2))
    for k, frame in enumerate(psi):
        phi = AXIS_SIGN_MAST*np.asarray(frame, np.float64)
        if not np.isfinite(phi).all():
            lo = np.nanmin(phi) if np.isfinite(phi).any() else 0.0
            phi = np.nan_to_num(phi, nan=lo-1.0, posinf=np.nanmax(phi), neginf=lo-1.0)
        ra, za, _, _, _ = find_o_point(phi, r, z, mask)
        out[k] = (ra, za)
    return out


class VerticalTransfer:
    """Two passes: measure the fold, then re-place each shot."""

    def __init__(self, geometry_row, r, z, mask):
        self.r = np.asarray(r, float)
        self.z = np.asarray(z, float)
        self.mask = np.asarray(mask, bool)
        self.dZ = float(self.z[1]-self.z[0])
        self.null = VerticalNull(geometry_row, self.r, self.z)
        self.shots = []

    def observe(self, psi, row, times):
        """Record one shot's axis height and its vacuum null. Returns both."""
        axis = extracted_axis(psi, self.r, self.z, self.mask)
        zn, _ = self.null.null(row, times, axis[:, 0])
        self.shots.append((float(np.nanmean(axis[:, 1])), float(np.nanmean(zn)), len(psi)))
        return axis, zn

    def solve(self):
        """Fold-level gain and per-shot shift, from observations alone.

        Both populations are centred on their MEDIAN, not their mean. The extracted
        axis heights are strongly one-sided, so a mean reference would drag the whole
        majority upward to pay for that tail. Centring on the median leaves the
        majority where it is and expands only the shots our own reconstruction
        already places off the fold centre.
        """
        zp = np.asarray([s[0] for s in self.shots])
        zn = np.asarray([s[1] for s in self.shots])
        finite = np.isfinite(zp) & np.isfinite(zn)
        centre_pred = float(np.median(zp[finite]))
        centre_null = float(np.median(zn[finite]))
        spread_pred = float(np.sqrt(np.mean((zp[finite]-centre_pred)**2)))
        spread_null = float(np.sqrt(np.mean((zn[finite]-centre_null)**2)))
        raw = spread_null/spread_pred if spread_pred > 1e-9 else 1.0
        gain = float(np.clip(raw, 1.0, GAIN_LIMIT))
        # v14 run setting (fusion_eq.mast --fold-fraction; unset = 1, v3-1 .. v13): the share of the expansion
        # that is applied. `gain` itself, which later stages read as the saturation flag, is unchanged.
        import os as _os
        fraction = float(_os.environ.get('FUSION_MAST_FOLD_FRACTION', '1'))
        applied = 1.0+fraction*(gain-1.0)
        target = np.where(finite, centre_pred+applied*(zp-centre_pred), zp)
        # The expansion moves an axis away from the fold centre towards the height its own coil currents call
        # for (the vacuum null). It stops at that height, and an axis that already sits at or beyond it is left
        # where it is: the fold only expands, and never carries a shot past its own null. This replaces any fixed
        # limit on where an axis may sit or how far a shot may move.
        reach = np.abs(zn-centre_pred)
        same_side = np.sign(zn-centre_pred) == np.sign(zp-centre_pred)
        overshoot = finite & same_side & (reach < np.abs(target-centre_pred))
        already_beyond = overshoot & (np.abs(zp-centre_pred) >= reach)
        target = np.where(already_beyond, zp, np.where(overshoot, zn, target))
        shift = np.where(finite, target-zp, 0.)
        self.gain, self.fold_mean, self.shift = gain, centre_pred, shift
        return dict(gain=round(gain, 6), raw_gain=round(float(raw), 6), applied_gain=round(applied, 6),
                    centre_pred_Z=round(centre_pred, 6), centre_null_Z=round(centre_null, 6),
                    spread_pred=round(spread_pred, 6), spread_null=round(spread_null, 6),
                    shots=int(len(zp)),
                    moved_shots=int(np.sum(np.abs(shift) > 1e-4)),
                    null_limited=int(overshoot.sum()),
                    already_beyond_null=int(already_beyond.sum()),
                    mean_abs_shift=round(float(np.abs(shift).mean()), 6),
                    max_abs_shift=round(float(np.abs(shift).max()), 6))

    def apply(self, index, psi):
        """Rigidly move one shot; edges replicate, interior is cubic."""
        d = float(self.shift[index])
        if abs(d) < 1e-9:
            return np.asarray(psi, np.float32), 0.0
        moved = np.stack([ndshift(f, (d/self.dZ, 0.0), order=3, mode='nearest')
                          for f in np.asarray(psi, np.float64)])
        return moved.astype(np.float32), d
