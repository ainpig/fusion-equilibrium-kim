"""Release candidate: source-calibrated minimum-motion vertical force repair."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from my_experiments.force_projection_v16 import infer_translation
from my_experiments.gpu_readout_v15 import readout_staged
ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def tolerance():
    data=json.loads((ROOT/'my_experiments/artifacts/d3d_force_uncertainty_v16.json').read_text())
    if data['fitted_on']!='DIII-D only':raise ValueError('Force tolerance must be source-only')
    return float(data['source_residual_tolerance'])


def project(eq,row,times,baseline_phi):
    phi,detail=infer_translation(eq,times,eq.currents(row,times),baseline_phi,force_tolerance=tolerance())
    changed=np.asarray([abs(d['shift_m'])>=1e-7 for d in detail['frames']])
    assert np.array_equal(phi[~changed],np.asarray(baseline_phi)[~changed])
    return phi,changed,detail


def finish(times,changed,staged,q0,b0):
    """Batch only changed shots; keep local, unaffected outputs byte-identical."""
    active=[i for i,x in enumerate(changed) if x.any()]
    output=readout_staged([staged[i] for i in active],[q0[i] for i in active],[b0[i] for i in active],batch_frames=4096,q_method='conditioned') if active else []
    results={i:v for i,v in zip(active,output)};final=[]
    for i,(t,mask,q,b) in enumerate(zip(times,changed,q0,b0)):
        if i not in results:
            final.append((np.asarray(q).copy(),np.asarray(b).copy(),{'unchanged_shot':True,'beta_affected_frames':0}));continue
        nq,nb,detail=results[i];nq[~mask]=np.asarray(q)[~mask]
        width=3*max(12.,2*np.median(np.diff(t))) if len(t)>1 else 36.
        affected=np.min(abs(np.asarray(t)[:,None]-np.asarray(t)[mask][None,:]),axis=1)<=width
        nb[~affected]=np.asarray(b)[~affected];detail['beta_affected_frames']=int(affected.sum())
        if not np.isfinite(nq).all() or not np.isfinite(nb).all():raise ValueError('Non-finite v16 scalar')
        final.append((nq.astype(np.float32),nb.astype(np.float32),detail))
    return final
