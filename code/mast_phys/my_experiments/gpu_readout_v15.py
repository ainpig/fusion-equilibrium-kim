"""CUDA-batched scalar readout with source-only priors and CPU contour staging."""
import json
from pathlib import Path
import numpy as np
from my_experiments.physical_scalars_v13 import physical_geometry
from my_experiments.mast_inputs_v15 import profiles
from my_experiments.thermal_floor_v14 import electron_beta,truncate_normal
from my_experiments.adaptive_prior_v14 import relative_prior
from my_experiments.gs_uncertainty_v14 import smooth_readout
from my_experiments.gpu_profiles_v15 import prepare_geometry,pressure_batch
from my_experiments.calibrate_ampere_q_v15 import ampere_q
from my_experiments.q_shape_prior_v13 import shape_q,ellipse_q
from my_experiments.fusion_pipeline import MAST_COILS,_interp
from my_experiments.mast_constants import MAST_TF_TURNS
ROOT=Path(__file__).resolve().parents[1]


def conditioned_q(shape,ellipse,prior):
    """Keep the detailed contour only inside its DIII-D conditioning envelope."""
    if prior['fitted_on']!='DIII-D only' or prior['conditioning']['fitted_on']!='DIII-D only':
        raise ValueError('Non-source q calibration')
    a=np.asarray(shape)*prior['methods']['axis']['ratio'];e=np.asarray(ellipse)*prior['methods']['ellipse']['ratio']
    c=prior['conditioning'];outside=(a<e*c['ratio_low'])|(a>e*c['ratio_high'])|~np.isfinite(a)
    return np.where(outside,e,a),outside


def prepare_shot(row,times,phi):
    r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z'])
    from my_experiments.released_geometry import grid
    _, _, mask = grid(row)
    curr=np.column_stack([_interp(row['magnetics_time'],row[k],times) for k in MAST_COILS])
    ip=abs(curr[:,-1]*1000);fv=abs(curr[:,11]*1000*MAST_TF_TURNS*2e-7);tsr,te,ne,_=profiles(row,times)
    geometries=[];indices=[];floors=[];qa=[];qh=[];qe=[];qc=[]
    from matplotlib.path import Path as Polygon
    rr,zz=np.meshgrid(r,z);points=np.column_stack([rr.ravel(),zz.ravel()])
    for i,frame in enumerate(phi):
        try:
            g=physical_geometry(frame,r,z,mask)
            # GPU and CPU use the same positive-pressure fit cell requirement.
            pn=(g['axis'][2]-frame)/(g['axis'][2]-g['xpoint'][2])
            if np.count_nonzero((pn>.015)&(pn<.985)&mask)<60:raise ValueError('Too few current cells')
            # v14: also the exact count the batched GPU fit requires (inside the boundary polygon, grid
            # edges excluded), so one small-boundary frame cannot stop its whole batch; frames that pass
            # the mask count but would fail this one never occurred before v14 (they stopped the run).
            cells=Polygon(g['boundary']).contains_points(points).reshape(rr.shape)&(pn>.015)&(pn<.985)
            cells[[0,-1],:]=False;cells[:,[0,-1]]=False
            if np.count_nonzero(cells)<60:raise ValueError('Too few current cells')
            a=ampere_q(g['spline'],g['qcurve'],fv[i],ip[i])
            h=shape_q(g['qcurve'],*g['axis'][:2],fv[i],ip[i]);e=ellipse_q(g['qcurve'],*g['axis'][:2],fv[i],ip[i])
            centre=(g['qcurve'].max(0)+g['qcurve'].min(0))/2
            cq=shape_q(g['qcurve'],*centre,fv[i],ip[i])
            if not np.isfinite(cq):cq=e
            if not np.isfinite(h):h=e
            if not np.isfinite(a) or not np.isfinite(h):raise ValueError('Invalid q geometry')
            floor=electron_beta(g,tsr,te[i],ne[i],fv[i],ip[i])
            indices.append(i);geometries.append(g);floors.append(floor);qa.append(a);qh.append(h);qe.append(e);qc.append(cq)
        except (ValueError,np.linalg.LinAlgError):continue
    if not indices:return {'indices':np.array([],int),'times':np.asarray(times),'n':len(times),'prepared':None}
    p=prepare_geometry(np.asarray(phi)[indices],r,z,geometries,ip[indices],fv[indices])
    return {'indices':np.asarray(indices),'times':np.asarray(times),'n':len(times),'prepared':p,
            'thermal_floor':np.asarray(floors),'q_ampere':np.asarray(qa),'q_shape':np.asarray(qh),'q_ellipse':np.asarray(qe),'q_centre':np.asarray(qc)}


def finish_shot(staged,gpu_result,q_fallback,beta_fallback,q_method='ampere'):
    art=ROOT/'my_experiments/artifacts';bcal=json.loads((art/'d3d_uncertainty_calibration_v14.json').read_text());prior=bcal['prior']
    qprior=json.loads((art/'d3d_q_shape_prior_v13.json').read_text());acal=json.loads((art/'d3d_ampere_q_prior_v15.json').read_text())
    geometric=json.loads((art/'d3d_geometric_q_prior_v15.json').read_text())
    if geometric['fitted_on']!='DIII-D only':raise ValueError('Non-source q calibration')
    ratio=acal['ratio'] if acal['validation_r2_calibrated']>acal['validation_r2_raw'] else 1.
    n=staged['n'];ids=staged['indices'];q=np.asarray(q_fallback,float).copy();beta=np.asarray(beta_fallback,float).copy()
    if len(ids)==0:return q.astype(np.float32),beta.astype(np.float32),{'unresolved_frames':n}
    if q_method=='ampere':values=staged['q_ampere']*ratio
    elif q_method=='shape':values=staged['q_shape']*qprior['q_ratio_center']
    elif q_method in ('ellipse','centre'):values=staged['q_'+q_method]*geometric['methods'][q_method]['ratio']
    elif q_method=='conditioned':values,_=conditioned_q(staged['q_shape'],staged['q_ellipse'],geometric)
    else:raise ValueError(q_method)
    q[ids]=values
    stat=gpu_result['sigma_jackknife']*bcal['scale']
    model=gpu_result['functional_sensitivity_norm']*gpu_result['residual_norm']/np.sqrt(3)
    sigma=np.full(n,np.inf);sigma[ids]=np.hypot(stat,model)
    physical=np.asarray(beta_fallback,float).copy();physical[ids]=gpu_result['beta_full_fit']
    floor=np.zeros(n);floor[ids]=staged['thermal_floor']
    m,s,w=relative_prior(physical,sigma,prior['mean'],prior['variance'],return_weight=True)
    m,s=truncate_normal(m,s,floor);beta=np.maximum(smooth_readout(staged['times'],m,s),floor)
    if not np.isfinite(q).all() or not np.isfinite(beta).all():raise RuntimeError('Non-finite GPU readout')
    return q.astype(np.float32),beta.astype(np.float32),{'unresolved_frames':n-len(ids),'q_method':q_method,
        'physics_weight':w.tolist(),'sigma_model':model.tolist(),'beta_physics':physical.tolist(),
        'q_ampere':staged['q_ampere'].tolist(),'q_shape':staged['q_shape'].tolist(),'q_ellipse':staged['q_ellipse'].tolist(),
        'q_centre':staged['q_centre'].tolist(),'resolved_indices':ids.tolist()}


def readout_staged(staged_list,q_baseline,beta_baseline,batch_frames=4096,q_method='ampere'):
    import os as _os
    # GPU memory cap (8 GB cards): frames are read out independently, so a smaller batch gives the same result
    batch_frames=min(int(batch_frames),int(_os.environ.get('NM1_MAST_MAX_GPU_FRAMES',batch_frames)))
    valid=[p for p in staged_list if p['prepared'] is not None]
    if not valid:return [finish_shot(p,{},q,b,q_method) for p,q,b in zip(staged_list,q_baseline,beta_baseline)]
    keys=valid[0]['prepared'].keys();merged={}
    for key in keys:
        merged[key]=valid[0]['prepared'][key] if key in ('r','z') else np.concatenate([p['prepared'][key] for p in valid])
    results={};total=len(merged['phi'])
    for start in range(0,total,batch_frames):
        batch={k:(v if k in ('r','z') else v[start:start+batch_frames]) for k,v in merged.items()}
        result=pressure_batch(batch)
        for k,v in result.items():results.setdefault(k,[]).append(v)
    results={k:np.concatenate(v) for k,v in results.items()};offset=0;output=[]
    for staged,q,b in zip(staged_list,q_baseline,beta_baseline):
        n=len(staged['indices']);part={k:v[offset:offset+n] for k,v in results.items()};offset+=n
        output.append(finish_shot(staged,part,q,b,q_method))
    return output
