"""Propagate the v22 geometry change through the existing CUDA pressure readout.

The v20 kinetic increment is retained as an increment. The inherited v17/v18
pressure estimator is reevaluated on the improved flux; no MAST target enters.
"""
import numpy as np
from my_experiments.beta_readout_v18 import refine_beta


def candidates(base22, old18, anchor16, posterior, readout, prior, resolved):
    base = np.asarray(base22, np.float64)
    new17 = ((np.asarray(anchor16, float)+np.asarray(posterior, float))*.5).astype(np.float32)
    new18, _ = refine_beta(new17, anchor16, readout, prior)
    delta = np.asarray(new18, float)-np.asarray(old18, float)
    delta[~resolved] = 0.
    raw = np.asarray(readout.get('beta_physics', np.zeros(len(base))), float)
    precision = np.clip(np.asarray(readout.get('physics_weight', np.zeros(len(base))), float), 1e-15, 1.)
    variance = prior['variance']*np.maximum(1., (raw/prior['mean'])**2)
    noise = variance*(1/precision-1)
    gate = .5*np.clip(1-9*noise/np.maximum(raw**2, 1e-20), 0., 1.)
    gate[(raw <= 0) | ~resolved] = 0.
    result = {name: np.maximum(base+w*delta, 0.).astype(np.float32) for name, w in
              [('beta_delta', 1.), ('beta_half_delta', .5), ('beta_confident_delta', gate)]}
    assert all(np.isfinite(v).all() for v in result.values())
    return result, dict(new_beta18=new18, delta=delta, gate=gate)
