"""Source/synthetic-only geometric priors in poloidal pressure units."""
import numpy as np


def descriptors(x, kind):
    shift = x[:, 8]/np.maximum(x[:, 6], .05)
    if kind == 'constant':
        return np.zeros((len(x), 0))
    if kind == 'shift_li':
        return shift[:, None]
    if kind == 'geometry':
        return np.column_stack([shift, x[:, 14], x[:, 7]-1, x[:, 6]**2])
    if kind == 'finite_aspect':
        eps2 = x[:, 6]**2
        return np.column_stack([shift, x[:, 14], x[:, 7]-1, eps2,
                                eps2*shift, eps2*x[:, 14]])
    if kind == 'geometry_kinetic':
        return np.column_stack([shift, x[:, 14], x[:, 7]-1, x[:, 6]**2, x[:, -1]])
    raise ValueError(kind)


def fit_prior(x, target, kind, sample_weight=None,
              fitted_on='DIII-D calibration only'):
    raw = descriptors(x, kind)
    weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, float)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Expected positive finite per-frame training weights')
    weights = weights*(len(x)/weights.sum())
    centre = np.average(raw, axis=0, weights=weights)
    spread = np.maximum(np.sqrt(np.average((raw-centre)**2, axis=0, weights=weights)), 1e-6)
    z = (raw-centre)/spread
    design = np.column_stack([np.ones(len(x)), z])
    response = target+.5*x[:, 14] if kind == 'shift_li' else target
    regular = np.eye(design.shape[1])*.01*len(x)
    regular[0, 0] = 0
    covariance = np.linalg.inv(design.T@(weights[:, None]*design)+regular)
    coef = covariance@design.T@(weights*response)
    residual = response-design@coef
    return {'fitted_on': fitted_on, 'kind': kind, 'centre': centre.tolist(),
            'spread': spread.tolist(), 'coefficients': coef.tolist(), 'covariance': covariance.tolist(),
            'residual_variance': float(np.average(residual**2, weights=weights)), 'calibration_frames': len(x)}


def predict_prior(prior, x):
    if prior['fitted_on'] not in ('DIII-D calibration only',
                                'DIII-D calibration and synthetic free-boundary training only'):
        raise ValueError('Prior must exclude measured target-machine training data')
    raw = descriptors(x, prior['kind'])
    z = (raw-np.asarray(prior['centre']))/np.asarray(prior['spread'])
    design = np.column_stack([np.ones(len(x)), z])
    mean = design@prior['coefficients']
    if prior['kind'] == 'shift_li':
        mean -= .5*x[:, 14]
    residual_variance = max(prior['residual_variance'], prior.get('inverse_variance_floor', 0.))
    variance = residual_variance*(1+np.einsum('ni,ij,nj->n', design, prior['covariance'], design))
    if z.shape[1]:
        variance *= np.maximum(1., np.mean(z*z, axis=1)/9.)
    return np.maximum(mean, 0.), np.sqrt(np.maximum(variance, 1e-8))
