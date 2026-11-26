"""Median of three physical q estimators; all scale calibration is DIII-D-only."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def calibration():
    art=ROOT/'my_experiments/artifacts'
    shape=json.loads((art/'d3d_geometric_q_prior_v15.json').read_text())
    ampere=json.loads((art/'d3d_ampere_q_prior_v15.json').read_text())
    assert shape['fitted_on']==ampere['fitted_on']=='DIII-D only'
    scale=ampere['ratio'] if ampere['validation_r2_calibrated']>ampere['validation_r2_raw'] else 1.
    return scale,shape['methods']['axis']['ratio'],shape['methods']['ellipse']['ratio']


def robust_q(ampere,axis,ellipse):
    scales=calibration();values=np.stack([np.asarray(x,float)*s for x,s in zip((ampere,axis,ellipse),scales)])
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Resolved positive q estimates required')
    return np.median(values,axis=0)
