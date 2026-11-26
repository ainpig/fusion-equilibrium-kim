"""Small bounded residual pressure estimator trained without MAST labels."""
import numpy as np
import torch
from torch import nn


class BetaResidual(nn.Module):
    def __init__(self, width, bound=2.):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.bound*torch.tanh(self.net(x).squeeze(-1)/self.bound)


def uncertainty(x):
    return np.maximum(np.hypot(1.8198859634382136*x[:, 2], x[:, 3]), 1e-6)


def feature_transform(x, scale, mode='invariant'):
    if mode == 'invariant':
        return np.asarray(x)
    noise = uncertainty(x)
    significance = np.column_stack([np.clip(x[:, 0]/noise, -100, 100), np.clip(x[:, 1]/noise, -100, 100)])
    if mode == 'pressure_only':
        # The physical estimators already use actual R, Z and field/current
        # scales. Avoid asking a residual model to extrapolate raw aspect ratio
        # or grid-size fingerprints beyond its source-machine training range.
        columns = [*range(6), 7, 14, *range(15, 23)]
        if x.shape[1]>23:
            columns += list(range(23, x.shape[1]))
        return np.column_stack([x[:, columns], x[:, 8]/np.maximum(x[:, 6], .05), significance])
    if mode == 'invariant_snr':
        return np.column_stack([x, significance])
    if mode != 'field_scale':
        raise ValueError(mode)
    return np.column_stack([x, np.log1p(scale), x[:, 0]*scale, x[:, 2]*scale, x[:, 3]*scale,
                            significance])


@torch.inference_mode()
def predict(checkpoint, x, scale, device='cuda'):
    if checkpoint['fitted_on'] not in ('synthetic GS teachers only', 'synthetic GS teachers and DIII-D calibration only', 'DIII-D equilibria with input-like perturbations',
                                      'synthetic free-boundary GS teachers only', 'synthetic free-boundary GS teachers and DIII-D calibration only'):
        raise ValueError('Invalid training provenance')
    if 'ensemble_members' in checkpoint:
        values = [predict(member, x, scale, device) for member in checkpoint['ensemble_members']]
        means = np.stack([v[0] for v in values])
        sigma = np.sqrt(np.mean(np.stack([v[1]**2 for v in values]), 0)+np.var(means, 0))
        return means.mean(0), sigma, np.stack([v[2] for v in values]).mean(0)
    transformed = feature_transform(x, scale, checkpoint.get('feature_mode', 'invariant'))
    if checkpoint.get('feature_mode') == 'field_scale' and len(checkpoint['centre']) == x.shape[1]+4:
        transformed = transformed[:, :-2]  # Preserve the first experimental schema.
    model = BetaResidual(transformed.shape[1], checkpoint.get('correction_bound', 2.)).to(device)
    model.load_state_dict(checkpoint['state'])
    standardized = np.clip((transformed-checkpoint['centre'])/checkpoint['spread'], -12, 12)
    adjustment = model(torch.as_tensor(standardized, dtype=torch.float32, device=device)).cpu().numpy()
    corrected = np.maximum(x[:, 0]+uncertainty(x)*adjustment, 0.)*scale
    sigma = uncertainty(x)*scale*checkpoint['residual_sigma_scale']
    return corrected, sigma, adjustment
