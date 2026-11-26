"""Denoise normalized source current, retaining actual time-varying actuators.

The response correction conserves observed Ip at every original frame. It
does not blur the supplied vacuum field or reuse a neighbouring frame's Ip.
The 12 ms scale is the pre-existing input/readout timescale, not MAST-fitted.
"""
import numpy as np
import torch
from my_experiments.virial_pressure_v14 import discrete_jr


def smoother(times, mode='local_linear'):
    t = np.asarray(times, float)
    if mode not in ('gaussian', 'local_linear'):
        raise ValueError(mode)
    if len(t)<3:
        return np.eye(len(t))
    width = max(12., 2*np.median(np.diff(t)))
    dt = (t[None]-t[:, None])/width
    kernel = np.exp(-.5*dt**2)*(abs(dt)<=3)
    if mode=='gaussian':
        return kernel/kernel.sum(1)[:, None]
    s0, s1, s2 = kernel.sum(1), (kernel*dt).sum(1), (kernel*dt**2).sum(1)
    denominator = s0*s2-s1*s1
    matrix = kernel*(s2[:, None]-dt*s1[:, None])/np.maximum(denominator[:, None], 1e-20)
    isolated = denominator<=1e-12*s0*s0
    matrix[isolated] = np.eye(len(t))[isolated]
    return matrix


def refine(eq, cases, initial_fields, mode='local_linear'):
    r, z = eq.r, eq.z
    rr, zz = np.meshgrid(r, z)
    area = float((r[1]-r[0])*(z[1]-z[0]))
    wall = eq.fine_mask.cpu().numpy().reshape(rr.shape)>0
    outputs, records = [], []
    for case, seed in zip(cases, initial_fields):
        curr = eq.currents(case['row'], case['times'])
        vacuum = np.einsum('tc,czr->tzr', curr*1000, eq.basis, optimize=True)
        current = np.array([np.maximum(discrete_jr(p.astype(float)-v, r, z)/rr*area, 0.)*wall for p, v in zip(seed, vacuum)])
        total = current.sum((1, 2))
        if np.any(total<=0):
            raise ValueError('Unresolved source current')
        weights = current.reshape(len(seed), -1)/total[:, None]
        kernel = smoother(case['times'], mode)
        smoothed = np.maximum(kernel@weights, 0.)
        smoothed /= smoothed.sum(1)[:, None]
        difference = smoothed-weights
        assert np.max(abs(difference.sum(1)))<1e-12
        with torch.no_grad():
            correction = eq.grid_response(eq.t(difference)*eq.t(curr[:, -1]*1000)[:, None]).cpu().numpy()
        result = (np.asarray(seed)+correction).astype(np.float32)
        assert np.isfinite(result).all()
        outputs.append(result)
        records.append({'shot': case['shot'], 'frames': len(seed),
                        'max_current_fraction_change': float(np.abs(difference).sum(1).max()/2),
                        'max_absolute_flux_correction': float(np.max(abs(correction))),
                        'normalized_current_error': float(np.max(abs(difference.sum(1))))})
    return outputs, {'method': 'normalized-current temporal filtering', 'mode': mode,
                     'MAST_training': False, 'actual_actuator_inputs_preserved': True, 'records': records}
