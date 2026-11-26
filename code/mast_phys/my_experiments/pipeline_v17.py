"""Production v17 MAST inference: temporal current refinement and the closed-domain posterior.

MAST beta uses a fixed equal ensemble of v16 and that posterior. Neither learned parameters nor
ensemble weights are fitted to MAST targets.

The DIII-D half of the original module is gone: this project trains its own DIII-D leg (see
``fusion_eq/train.py``), so the class that loaded the old stacked checkpoints, and the two
constants it read, had no caller and no checkpoints to load.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

from my_experiments.pipeline_v16 import project
from my_experiments.temporal_current_v17 import refine
from my_experiments.fast_geometry_v17 import physical_geometry
from my_experiments.mast_equilibrium_v13 import ThomsonEquilibriumV13
from my_experiments.mast_inputs_v15 import profiles
from my_experiments.thermal_floor_v14 import electron_beta
from my_experiments.gpu_profiles_v15 import prepare_geometry
from my_experiments.gpu_readout_v15 import readout_staged
from my_experiments.calibrate_ampere_q_v15 import ampere_q
from my_experiments.q_shape_prior_v13 import shape_q, ellipse_q
from my_experiments.fusion_pipeline import MAST_COILS, _interp
from my_experiments.mast_constants import MAST_TF_TURNS

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'my_experiments/artifacts'




def mast_mask():
    from my_experiments.released_geometry import grid
    return grid()[2]


def prepare_mast(row, times, phi):
    """Same closed-domain posterior staging as the validated experiment.

    The unused virial readout is omitted; contour and current-fit predicates
    remain identical to staging_v17.prepare_shot_closed.
    """
    r, z = np.asarray(row['efit_grid_R']), np.asarray(row['efit_grid_Z'])
    rr, zz = np.meshgrid(r, z)
    points = np.column_stack([rr.ravel(), zz.ravel()])
    curr = np.column_stack([_interp(row['magnetics_time'], row[k], times) for k in MAST_COILS])
    ip = abs(curr[:, -1] * 1000)
    # F = B_t R = 2e-7 * N * I, with `curr[:, 11]` = magnetics_tf_current in kA (the challenge
    # ships MAST's TF in kA, not kA*turn, and gives it no poloidal-plane rectangle -- so N is an
    # external machine constant, like DIII-D's 144). N = 24 is inherited and its source is not
    # recorded; inverting MAST's published B_t = 0.52 T at R0 = 0.85 m against the shipped TF
    # current (86 kA peak) gives N = 25.7, so this may be ~7% low. Disclosed in METHODS.md 1.2.
    fv = abs(curr[:, 11] * 1000 * MAST_TF_TURNS * 2e-7)
    tsr, te, ne, _ = profiles(row, times)
    geometries, ids, floors, qa, qh, qe, qc = [], [], [], [], [], [], []
    for i, frame in enumerate(phi):
        try:
            g = physical_geometry(frame, r, z, mast_mask())
            pn = (g['axis'][2] - frame) / (g['axis'][2] - g['xpoint'][2])
            inside = Polygon(g['boundary']).contains_points(points).reshape(rr.shape)
            if np.count_nonzero((pn > .015) & (pn < .985) & inside) < 60:
                raise ValueError('Too few current cells')
            # v14: the batched GPU fit also drops the grid edges; require its exact count here
            cells = (pn > .015) & (pn < .985) & inside
            cells[[0, -1], :] = False
            cells[:, [0, -1]] = False
            if np.count_nonzero(cells) < 60:
                raise ValueError('Too few current cells')
            av = ampere_q(g['spline'], g['qcurve'], fv[i], ip[i])
            hv = shape_q(g['qcurve'], *g['axis'][:2], fv[i], ip[i])
            ev = ellipse_q(g['qcurve'], *g['axis'][:2], fv[i], ip[i])
            centre = (g['qcurve'].max(0) + g['qcurve'].min(0)) / 2
            cv = shape_q(g['qcurve'], *centre, fv[i], ip[i])
            if not np.isfinite(hv):
                hv = ev
            if not np.isfinite(cv):
                cv = ev
            if not np.isfinite(av) or not np.isfinite(hv):
                raise ValueError('Invalid q')
            floor = electron_beta(g, tsr, te[i], ne[i], fv[i], ip[i])
            ids.append(i)
            geometries.append(g)
            floors.append(floor)
            qa.append(av)
            qh.append(hv)
            qe.append(ev)
            qc.append(cv)
        except (ValueError, np.linalg.LinAlgError):
            continue
    ids = np.asarray(ids, int)
    result = {'indices': ids, 'times': np.asarray(times), 'n': len(times), 'prepared': None}
    if len(ids):
        result.update(prepared=prepare_geometry(np.asarray(phi)[ids], r, z, geometries, ip[ids], fv[ids]),
                      thermal_floor=np.asarray(floors), q_ampere=np.asarray(qa),
                      q_shape=np.asarray(qh), q_ellipse=np.asarray(qe), q_centre=np.asarray(qc))
    return result


class MastPredictor:
    def __init__(self, geometry):
        prior = json.loads((ART / 'kinetic_prior_v11.json').read_text())
        factor = json.loads((ART / 'd3d_pressure_prior_v13.json').read_text())['total_to_electron_pressure']
        self.eq = ThomsonEquilibriumV13(geometry, prior, factor)
        self.geometry = geometry

    def fields(self, cases, baselines):
        for case in cases:
            for key in ('coil_R', 'coil_Z', 'coil_width', 'coil_height', 'coil_input_column',
                        'efit_grid_R', 'efit_grid_Z'):
                if not np.array_equal(case['row'][key], self.geometry[key]):
                    raise ValueError('Changed geometry: ' + key)
        updated, temporal = refine(self.eq, cases, [b['psirz'] for b in baselines], 'local_linear')
        outputs, details = [], []
        for case, phi, record in zip(cases, updated, temporal['records']):
            final, changed, force = project(self.eq, case['row'], case['times'], phi)
            outputs.append(final)
            details.append({'temporal': record, 'force': force, 'force_changed_frames': int(changed.sum())})
        return outputs, details

    def finish(self, staged, baselines):
        read = readout_staged(staged, [b['q95'] for b in baselines], [b['betaN'] for b in baselines],
                              batch_frames=1024, q_method='conditioned')
        results = []
        for stage, baseline, (q, beta, detail) in zip(staged, baselines, read):
            beta = (.5 * (np.asarray(baseline['betaN'], float) + np.asarray(beta, float))).astype(np.float32)
            unresolved = np.ones(len(beta), bool)
            unresolved[stage['indices']] = False
            # An unresolved geometry supplies no evidence for modifying a scalar.
            beta[unresolved] = baseline['betaN'][unresolved]
            q[unresolved] = baseline['q95'][unresolved]
            if not np.isfinite(q).all() or not np.isfinite(beta).all():
                raise ValueError('Non-finite MAST scalar')
            detail['beta_ensemble_weights'] = [.5, .5]
            results.append((q.astype(np.float32), beta, detail))
        return results
