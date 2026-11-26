"""CUDA Green-response correction for negative numerical plasma-current lobes.

Subtract the known vacuum field before differentiating. Project the inferred
source onto nonnegative current weights, with its normalized net current held
fixed, and apply only the difference through the existing GPU Green operator.
This is not a new learned inverse-current model or a replacement of the flux.
"""
import numpy as np
import torch
from my_experiments.virial_pressure_v14 import discrete_jr


def refine(eq, row, times, phi, enabled=True):
    phi = np.asarray(phi, np.float32)
    if not enabled:
        return phi.copy(), dict(enabled=False, negative_fraction_mean=0., negative_fraction_max=0., peak_change=0.)
    curr = eq.currents(row, times)
    vacuum = np.einsum('tc,czr->tzr', curr*1000., eq.basis, optimize=True)
    r, z = eq.r, eq.z
    rr, _ = np.meshgrid(r, z)
    area = float((r[1]-r[0])*(z[1]-z[0]))
    wall = eq.fine_mask.cpu().numpy().reshape(rr.shape)>0
    current = np.stack([discrete_jr(p.astype(float)-v, r, z)/rr*area*wall for p, v in zip(phi, vacuum)])
    signed = current.sum((1, 2))
    pos = np.maximum(current, 0.)
    positive = pos.sum((1, 2))
    valid = (signed>0)&(positive>0)&np.isfinite(current).all((1, 2))
    neg_fraction = (positive-signed)/np.maximum(positive, 1e-20)
    before = current/np.maximum(signed, 1e-20)[:, None, None]
    after = pos/np.maximum(positive, 1e-20)[:, None, None]
    delta = (after-before).reshape(len(phi), -1)
    delta[~valid] = 0.
    assert np.max(abs(delta.sum(1))) < 1e-10
    with torch.inference_mode():
        change = eq.grid_response(eq.t(delta)*eq.t(curr[:, -1]*1000.)[:, None]).cpu().numpy()
    # A field's gauge is arbitrary; retain the existing calibrated mean.
    change -= change.mean((1, 2))[:, None, None]
    result = (phi.astype(float)+change).astype(np.float32)
    assert np.isfinite(result).all()
    return result, dict(enabled=True, negative_fraction_mean=float(neg_fraction[valid].mean()),
                        negative_fraction_max=float(neg_fraction[valid].max()),
                        peak_change=float(np.max(abs(change))), corrected_frames=int(valid.sum()))
