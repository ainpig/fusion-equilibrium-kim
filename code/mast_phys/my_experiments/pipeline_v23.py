"""Source-calibrated q and CUDA pressure updates on the frozen v22 equilibrium.

CPU contour staging is shared between q and pressure. Large independent
pressure systems are solved on CUDA in batches that fit the 16 GiB device.
All existing calibration records are DIII-D-only. No MAST target is opened.
"""
import json
import numpy as np
from my_experiments.pipeline_v17 import ART, prepare_mast
from my_experiments.geometry_q_v23 import GeometryQ
from my_experiments.robust_q_v16 import robust_q
from my_experiments.gpu_readout_v15 import readout_staged
from my_experiments.pressure_delta_v23 import candidates
from my_experiments.vertical_transfer_v21 import GAIN_LIMIT
from my_experiments.fusion_pipeline import MAST_COILS, _interp


def prepare(row, times, phi, previous_q, gain):
    if float(gain) >= GAIN_LIMIT:
        return dict(enabled=False, q95=np.asarray(previous_q, np.float32).copy(), prepared=None)
    stage = prepare_mast(row, times, phi)
    q = np.asarray(previous_q, np.float32).copy()
    ids = stage['indices']
    if len(ids):
        q[ids] = robust_q(stage['q_ampere'], stage['q_shape'], stage['q_ellipse'])
    unresolved = np.ones(len(q), bool)
    unresolved[ids] = False
    if unresolved.any():
        estimates, _ = GeometryQ().candidates(row, np.asarray(times)[unresolved], np.asarray(phi)[unresolved], q[unresolved], gain)
        q[unresolved] = estimates['q_robust']
    assert q.dtype == np.float32 and np.isfinite(q).all()
    currents = np.column_stack([_interp(row['magnetics_time'], row[k], times) for k in MAST_COILS])
    return dict(enabled=True, q95=q, prepared=stage, times=np.asarray(times, float), currents=currents)


def finish(staged, baselines22, beta18, beta16, batch_frames=8192):
    prior = json.loads((ART/'d3d_uncertainty_calibration_v14.json').read_text(encoding='utf-8'))
    assert prior['fitted_on'] == 'DIII-D only'
    active = [i for i, s in enumerate(staged) if s['enabled']]
    values = readout_staged([staged[i]['prepared'] for i in active],
                            [baselines22[i]['q95'] for i in active], [beta16[i] for i in active],
                            batch_frames=batch_frames, q_method='conditioned') if active else []
    read = dict(zip(active, values))
    predictions, details = [], []
    for i, (stage, base) in enumerate(zip(staged, baselines22)):
        beta = np.asarray(base['betaN'], np.float32).copy()
        detail = dict(enabled=stage['enabled'], resolved_pressure_frames=0)
        if stage['enabled']:
            _, posterior, record = read[i]
            resolved = np.zeros(len(beta), bool)
            resolved[stage['prepared']['indices']] = True
            choices, _ = candidates(beta, beta18[i], beta16[i], posterior, record, prior['prior'], resolved)
            beta = choices['beta_delta']
            detail['resolved_pressure_frames'] = int(resolved.sum())
        predictions.append(dict(psirz=np.asarray(base['psirz'], np.float32), q95=stage['q95'], betaN=beta))
        details.append(detail)
    return predictions, details


class CurrentCorrection:
    def __init__(self, geometry_row):
        from my_experiments.pipeline_v17 import MastPredictor
        self.eq = MastPredictor(geometry_row).eq

    def apply(self, staged, predictions):
        from my_experiments.positive_current_v23 import refine
        result, details = [], []
        for stage, pred in zip(staged, predictions):
            if stage['enabled']:
                # A cache of legal circuit inputs already interpolated at the
                # original EFIT times. Interpolation at these same knots is exact.
                row = dict(magnetics_time=stage['times'], **dict(zip(MAST_COILS, stage['currents'].T)))
                phi, detail = refine(self.eq, row, stage['times'], pred['psirz'])
            else:
                phi, detail = pred['psirz'], dict(enabled=False)
            result.append(dict(pred, psirz=phi))
            details.append(detail)
        return result, details
