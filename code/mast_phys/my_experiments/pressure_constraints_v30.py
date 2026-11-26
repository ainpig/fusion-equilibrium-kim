"""Add observed electron-pressure isosurface pairs to the v29 constraints.

v29 gained by appending density equal-value pairs to the temperature pairs the
inverse equilibrium already used. The quantity that actually enters
Grad-Shafranov is the electron pressure p = ne*Te, and its iso-value radii are
not recoverable from either factor's radii because the product is more peaked
than both.

Measured on the 70-shot holdout inputs (check_pressure_pairs_v30):
  * pressure yields 43,712 valid pairs, more than temperature (42,986) and far
    more than density (34,920), so it also constrains frames where ne alone fails;
  * its radii sit a median 3.9 cm from the temperature pairs and 12.2 cm from the
    density pairs, against the 10.3 cm separation between Te and ne that produced
    the v29 gain.

Construction is identical to v29's so the change is one extra observation, not a
new method. No MAST label is read here.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import isotonic_regression
from my_experiments.density_constraints_v29 import prepare_kinetic as density_constraints
from my_experiments.mast_inputs_v15 import profiles

FRACTIONS = np.linspace(.25, .85, 9)


def equal_value_pairs(r, values, centres, counts, shape):
    """The v29 equal-value construction applied to an arbitrary midplane profile."""
    pairs = np.zeros(shape, np.float32)
    valid = np.zeros(shape[:2], np.float32)
    for i, v in enumerate(values):
        if counts[i] < 12 or (v > 0).sum() < 12:
            continue
        split = int(np.argmin(np.abs(r-centres[i])))
        if split < 3 or split > len(r)-4:
            continue
        smooth = gaussian_filter1d(np.asarray(v, float), 2)
        scale = max(float(np.max(smooth)), 1.)
        smooth = smooth/scale
        left = isotonic_regression(smooth[:split+1], increasing=True).x
        right = isotonic_regression(smooth[split:], increasing=False).x
        top = min(left[-1], right[0])
        for j, fraction in enumerate(FRACTIONS):
            value = fraction*top
            if value <= max(left[0], right[-1]) or top <= 0:
                continue
            pairs[i, j] = [np.interp(value, left, r[:split+1]),
                           np.interp(value, right[::-1], r[split:][::-1])]
            valid[i, j] = 1.
    return pairs, valid


def prepare_kinetic(row, times, exponent):
    kin, (pairs, valid, pressure) = density_constraints(row, times, exponent)
    r, te, ne, counts = profiles(row, times)
    centres = np.asarray(kin, float)[:, 0]
    # p_e in the same arbitrary scale the other profiles use; only equal-value
    # radii matter, and the construction normalises by the profile peak.
    extra, extra_valid = equal_value_pairs(
        r, te*ne, centres, counts, (len(times), len(FRACTIONS), 2))
    return kin, (np.concatenate([pairs, extra], axis=1),
                 np.concatenate([valid, extra_valid], axis=1), pressure)
