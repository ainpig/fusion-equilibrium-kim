"""Scale-invariant pressure-reconstruction diagnostics for synthetic transfer.

Inputs are an estimated equilibrium, observed Ip/TF, and its geometry. No
target beta or source/target shot identity enters this feature extractor.
"""
import numpy as np
from my_experiments.gpu_profiles_v15 import pressure_batch, MU0

FEATURES = ['beta_fit', 'beta_virial', 'sigma_jackknife', 'sigma_model', 'gs_residual',
            'current_ratio', 'epsilon', 'elongation', 'axis_shift', 'normalized_flux_span',
            'mean_s', 'var_s', 'grid_dr', 'grid_dz', 'li_proxy']+[f'loo_{i}' for i in range(8)]


def features(prepared, batch_frames=1024):
    results = []
    for first in range(0, len(prepared['phi']), batch_frames):
        part = {k: v if k in ('r', 'z') else v[first:first+batch_frames] for k, v in prepared.items()}
        result = pressure_batch(part)
        phi = np.asarray(part['phi'], float)
        r, z = np.asarray(part['r'], float), np.asarray(part['z'], float)
        rr = r[None, None, :]
        inside = np.asarray(part['inside'], bool)
        ip, fv = abs(np.asarray(part['ip'])), abs(np.asarray(part['fvac']))
        a, rg = np.asarray(part['minor']), np.asarray(part['rgeo'])
        dr, dz = float(r[1]-r[0]), float(z[1]-z[0])
        jr = np.zeros_like(phi)
        jr[:, 1:-1, 1:-1] = -((phi[:, 1:-1, 2:]-2*phi[:, 1:-1, 1:-1]+phi[:, 1:-1, :-2])/dr**2
            -(phi[:, 1:-1, 2:]-phi[:, 1:-1, :-2])/(2*dr*r[None, None, 1:-1])
            +(phi[:, 2:, 1:-1]-2*phi[:, 1:-1, 1:-1]+phi[:, :-2, 1:-1])/dz**2)/MU0
        pr = np.gradient(phi, r, axis=2, edge_order=2)
        pz = np.gradient(phi, z, axis=1, edge_order=2)
        volume = np.sum(rr*inside, axis=(1, 2))
        pvir = -.5*np.sum(jr*pr*inside, axis=(1, 2))/volume
        norm = 2*MU0*100*a*rg/(fv*ip/1e6)
        # Poloidal magnetic pressure expressed in beta_N units. Dividing by
        # this scale removes current/field/device-size amplitudes.
        beta_scale = 4e-6*ip/fv*rg/a
        axis = np.asarray(part['axis'])
        span = axis[:, 2]-np.asarray(part['pb'])
        s = np.clip((axis[:, 2, None, None]-phi)/span[:, None, None], 0, 1)
        mean_s = np.sum(s*rr*inside, axis=(1, 2))/volume
        var_s = np.sum((s-mean_s[:, None, None])**2*rr*inside, axis=(1, 2))/volume
        current = np.sum(jr/rr*inside, axis=(1, 2))*dr*dz
        li = 4*np.pi/(MU0**2*ip**2*rg)*np.sum((pr**2+pz**2)/rr*inside, axis=(1, 2))*dr*dz
        model_sigma = result['functional_sensitivity_norm']*result['residual_norm']/np.sqrt(3)
        columns = [result['beta_full_fit']/beta_scale, pvir*norm/beta_scale,
                   result['sigma_jackknife']/beta_scale, model_sigma/beta_scale,
                   result['gs_residual'], current/ip, a/rg, np.asarray(part['height'])/a,
                   (axis[:, 0]-rg)/a, span/np.maximum(2e-7*ip*rg, 1e-8), mean_s, var_s,
                   dr/a, dz/np.asarray(part['height']), li]
        x = np.column_stack(columns+[result['leave_sector_out'][:, i]/beta_scale for i in range(8)])
        results.append((np.nan_to_num(x, nan=0., posinf=100., neginf=-100.), beta_scale))
    return np.concatenate([x for x, s in results]), np.concatenate([s for x, s in results])
