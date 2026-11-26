"""Input-only joint kinetic pressure readout with a fixed confidence limit."""
import json
import numpy as np
from my_experiments.pipeline_v17 import ART
from my_experiments.synthetic_beta_model_v17 import uncertainty
from my_experiments.geometry_beta_prior_v17 import predict_prior
from my_experiments.gs_uncertainty_v14 import smooth_readout


def smooth_signal(t, e, valid, linear=False):
    """Gaussian-weighted (optionally local-linear) denoising of an input kinetic amplitude over time."""
    t, e = np.asarray(t, float), np.asarray(e, float)
    width = max(12., 2*np.median(np.diff(t))) if len(t)>1 else 12.
    result = e.copy()
    for i in np.flatnonzero(valid):
        selected = valid&(abs(t-t[i])<=3*width)
        dt = (t[selected]-t[i])/width
        w = np.exp(-.5*dt**2)
        if linear and selected.sum()>=3:
            x = np.column_stack([np.ones(len(dt)), dt])
            c = np.linalg.lstsq(x*np.sqrt(w[:, None]), e[selected]*np.sqrt(w), rcond=1e-12)[0]
            result[i] = max(c[0], 0.)
        else:
            result[i] = float(w@e[selected]/w.sum())
    return result


def joint_estimate(x, scale, floor, resolved, times, beta_v16):
    """Reproduce the existing source-only joint MAP + electron trajectory.

    x and scale are computed from predicted flux and permitted diagnostics by
    extract_kinetic_profile_pressure_v17 with --joint-time. No labels enter.
    """
    prior_record = json.loads((ART/'d3d_geometry_beta_priors_v17.json').read_text(encoding='utf-8'))
    prior = prior_record['priors']['geometry']
    calibration = json.loads((ART/'d3d_uncertainty_calibration_v14.json').read_text(encoding='utf-8'))['prior']
    raw = x[:, 0]*scale
    sigma = uncertainty(x)*scale
    pm, ps = predict_prior(prior, x)
    pm, ps = pm*scale, ps*scale
    weight = ps**2/(ps**2+sigma**2)
    mean = weight*raw+(1-weight)*pm
    post_sigma = ps*np.sqrt(1-weight)
    mode = np.maximum(mean, floor)
    mode[~resolved] = beta_v16[~resolved]
    post_sigma[~resolved] = np.sqrt(calibration['variance'])
    anchor = np.maximum(smooth_readout(times, mode, post_sigma), floor).astype(np.float32)
    anchor[~resolved] = beta_v16[~resolved]
    anchor = anchor.astype(float)
    trajectory = anchor.copy()
    valid = resolved & (floor > 0)
    if valid.sum() >= 3:
        electron = smooth_signal(times, floor, valid, linear=True)
        ratio = max(1., float(anchor[valid].sum()/electron[valid].sum()))
        trajectory[valid] = np.maximum(ratio*electron[valid], floor[valid])
    result = (.5*(anchor+trajectory)).astype(np.float32)
    result[~resolved] = beta_v16[~resolved]
    return result


def refine(beta_v18, joint_beta, x, scale, resolved, readout_v17, prior):
    base, joint = np.asarray(beta_v18, float), np.asarray(joint_beta, float)
    n = len(base)
    if len(joint) != n or x.shape != (n, 23):
        raise ValueError('Inconsistent pressure inputs')
    raw = x[:, 0]*scale
    sigma = uncertainty(x)*scale
    precision = np.clip(np.asarray(readout_v17.get('physics_weight', np.zeros(n)), float), 1e-15, 1.)
    old_raw = np.asarray(readout_v17.get('beta_physics', np.zeros(n)), float)
    old_variance = prior['variance']*np.maximum(1., (old_raw/prior['mean'])**2)
    old_sigma = np.sqrt(old_variance*(1/precision-1))
    gate = .5*np.clip(1-9*(sigma/np.maximum(raw, 1e-20))**2, 0, 1)
    disagreement = abs(joint-base)/np.maximum(np.hypot(sigma, old_sigma), 1e-8)
    gate *= np.minimum(1., 9/np.maximum(disagreement**2, 1e-20))
    gate[~np.asarray(resolved, bool) | (raw <= 0)] = 0.
    result = (base+gate*(joint-base)).astype(np.float32)
    if not np.isfinite(result).all() or np.min(result) < 0:
        raise ValueError('Invalid joint pressure result')
    return result, gate
