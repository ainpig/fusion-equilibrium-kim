"""Regenerate the joint kinetic pressure evidence from legal shot inputs.

This is the input-to-feature path behind the cached v17 experiments reused by
the v20 readout. It reads no target values and does not fit shared parameters.
"""
import numpy as np
from my_experiments.staging_v17 import prepare_shot
from my_experiments.mast_inputs_v15 import profiles
from my_experiments.fusion_pipeline import _interp
from my_experiments.mast_constants import MAST_TF_TURNS
from my_experiments.beta_features_v17 import features
from my_experiments.gpu_profiles_v15 import prepare_geometry
from my_experiments.slice_pressure_v17 import cached_geometry
from my_experiments.closed_plasma_geometry_v17 import close_domain
from my_experiments.thermal_floor_v14 import electron_beta
from my_experiments.kinetic_profile_pressure_v17 import prepare_system, joint_systems, solve_prepared


def prepare_joint(row, times, phi):
    n = len(times)
    stage = prepare_shot(row, times, phi)
    r, z = np.asarray(row['efit_grid_R']), np.asarray(row['efit_grid_Z'])
    ip = abs(_interp(row['magnetics_time'], row['magnetics_plasma_current'], times)*1000)
    fv = abs(_interp(row['magnetics_time'], row['magnetics_tf_current'], times)*1000*MAST_TF_TURNS*2e-7)
    tsr, te, ne, _ = profiles(row, times)
    x, scale = np.zeros((n, 23)), np.ones(n)
    good, floor = np.zeros(n, bool), np.zeros(n)
    if stage['prepared'] is None:
        return x, scale, floor, good
    old_x, _ = features(stage['prepared'])
    closed, ids = [], []
    for i, old in zip(stage['indices'], old_x):
        try:
            g = close_domain(cached_geometry(phi[i], r, z, old, ip[i]), r, z)
            p = prepare_geometry(phi[i:i+1], r, z, [g], ip[i:i+1], fv[i:i+1])
            pn = (g['axis'][2]-phi[i])/(g['axis'][2]-g['xpoint'][2])
            if np.sum(p['inside'][0]&(pn > .015)&(pn < .985)) < 60:
                continue
            floor[i] = electron_beta(g, tsr, te[i], ne[i], fv[i], ip[i])
            closed.append(p)
            ids.append(i)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not ids:
        return x, scale, floor, good
    prepared = {k: closed[0][k] if k in ('r', 'z') else np.concatenate([p[k] for p in closed]) for k in closed[0]}
    result, normalizer = features(prepared)
    ids = np.asarray(ids, int)
    x[ids], scale[ids], good[ids] = result, normalizer, True
    systems, valid_ids = [], []
    for i in ids:
        try:
            g = close_domain(cached_geometry(phi[i], r, z, x[i], ip[i]), r, z)
            systems.append(prepare_system(phi[i], r, z, g, ip[i], fv[i], tsr, te[i], ne[i]))
            valid_ids.append(i)
        except (ValueError, np.linalg.LinAlgError):
            good[i] = False
    if systems:
        hs, gs, counts = joint_systems(np.asarray(times)[valid_ids], systems)
        for j, i in enumerate(valid_ids):
            try:
                result = solve_prepared(systems[j], hs[j], gs[j], counts[j])
                for column, key in [(0, 'beta'), (2, 'sigma_jackknife'), (3, 'sigma_model')]:
                    x[i, column] = result[key]/scale[i]
                x[i, 4] = result['gs_residual']
                x[i, 15:23] = result['leave_sector_out']/scale[i]
            except (ValueError, np.linalg.LinAlgError):
                good[i] = False
    return x, scale, floor, good
