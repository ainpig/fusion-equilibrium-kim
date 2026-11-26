"""Source-uncertainty-gated beta upgrade of the released v17 ensemble.

Three standard errors is a fixed signal-strength rule, not a target-fitted
threshold or a claimed confidence bound on MAST reconstruction error.
"""
import numpy as np


def refine_beta(beta_v17, beta_v16, readout, prior):
    previous, original = np.asarray(beta_v17, float), np.asarray(beta_v16, float)
    if previous.shape != original.shape or previous.ndim != 1:
        raise ValueError('Mismatched beta inputs')
    n = len(previous)
    precision = np.asarray(readout.get('physics_weight', np.zeros(n)), float)
    physical = np.asarray(readout.get('beta_physics', np.zeros(n)), float)
    if precision.shape != previous.shape or physical.shape != previous.shape:
        raise ValueError('Mismatched pressure diagnostics')
    if prior['mean'] <= 0 or prior['variance'] <= 0:
        raise ValueError('Invalid source pressure prior')
    good = np.isfinite(physical)&np.isfinite(precision)&(precision > 0)&(physical > 0)
    precision = np.where(good, np.clip(precision, 1e-15, 1.), 1e-15)
    physical = np.where(good, physical, 0.)
    variance = prior['variance']*np.maximum(1., (physical/prior['mean'])**2)
    sigma2 = variance*(1/precision-1)
    relative_noise = np.where(good, 9*sigma2/np.maximum(physical**2, 1e-20), np.inf)
    weight = 1-.5*np.clip(relative_noise, 0., 1.)
    output = (previous+2*(weight-.5)*(previous-original)).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError('Non-finite beta output')
    return output, {'changed_frames': int(np.count_nonzero(output != np.asarray(beta_v17, np.float32))),
                    'posterior_weight_mean': float(weight.mean()) if n else .5,
                    'posterior_weight_max': float(weight.max()) if n else .5}
