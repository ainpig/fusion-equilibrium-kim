"""Read q95 from the improved v22 flux using existing DIII-D calibrations.

No MAST target or fitted target coefficient is accessed. The two physical
readouts are the inherited conditioned shape rule and the inherited median of
Ampere/axis-shape/elliptical estimates. Keep the existing saturation fallback.
"""
import json
import numpy as np
from my_experiments.pipeline_v17 import ART, ROOT, MAST_TF_TURNS
from my_experiments.fast_geometry_v17 import physical_geometry
from my_experiments.fusion_pipeline import _interp
from my_experiments.q_shape_prior_v13 import shape_q, ellipse_q
from my_experiments.calibrate_ampere_q_v15 import ampere_q
from my_experiments.robust_q_v16 import robust_q
from my_experiments.gpu_readout_v15 import conditioned_q
from my_experiments.vertical_transfer_v21 import GAIN_LIMIT


class GeometryQ:
    def __init__(self):
        self.prior = json.loads((ART/'d3d_geometric_q_prior_v15.json').read_text(encoding='utf-8'))
        assert self.prior['fitted_on'] == 'DIII-D only'
        from my_experiments.released_geometry import grid
        self.r, self.z, self.mask = grid()

    def candidates(self, row, times, phi, previous, fold_gain):
        previous = np.asarray(previous, np.float32)
        conditioned, robust = previous.copy(), previous.copy()
        evidence = np.full((len(phi), 3), np.nan)
        resolved = np.zeros(len(phi), bool)
        if float(fold_gain) < GAIN_LIMIT:
            ip = abs(_interp(row['magnetics_time'], row['magnetics_plasma_current'], times)*1000.)
            # Same external turn count as the main path; see METHODS.md 1.2.
            fv = abs(_interp(row['magnetics_time'], row['magnetics_tf_current'], times)*1000.*MAST_TF_TURNS*2e-7)
            for i, frame in enumerate(phi):
                try:
                    g = physical_geometry(frame, self.r, self.z, self.mask)
                    h = shape_q(g['qcurve'], *g['axis'][:2], fv[i], ip[i])
                    e = ellipse_q(g['qcurve'], *g['axis'][:2], fv[i], ip[i])
                    a = ampere_q(g['spline'], g['qcurve'], fv[i], ip[i])
                    if not np.isfinite(h):
                        h = e
                    q, _ = conditioned_q(np.array([h]), np.array([e]), self.prior)
                    if not np.isfinite(q[0]) or q[0] <= 0:
                        continue
                    conditioned[i] = q[0]
                    robust[i] = robust_q(a, h, e) if np.isfinite(a) and a > 0 else q[0]
                    evidence[i] = a, h, e
                    resolved[i] = True
                except (ValueError, np.linalg.LinAlgError):
                    continue
        half = ((previous.astype(float)+conditioned.astype(float))*.5).astype(np.float32)
        result = dict(q_conditioned=conditioned, q_robust=robust, q_half_conditioned=half)
        assert all(np.isfinite(v).all() for v in result.values())
        return result, dict(evidence=evidence, resolved=resolved)
