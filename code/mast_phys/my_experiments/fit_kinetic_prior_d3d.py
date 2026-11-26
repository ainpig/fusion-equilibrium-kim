"""Dimensionless kinetic/current priors, fitted only to existing DIII-D TRAIN shots."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,stack_2d,_interp_profile
from my_experiments.build_d3d_rich_cache import is_val


def gs_current(psi,R,Z):
    dR=float(R[1]-R[0]);dZ=float(Z[1]-Z[0])
    d2R=np.zeros_like(psi);d2Z=np.zeros_like(psi);d1R=np.zeros_like(psi)
    d2R[:,:,1:-1]=(psi[:,:,2:]-2*psi[:,:,1:-1]+psi[:,:,:-2])/(dR*dR)
    d2Z[:,1:-1,:]=(psi[:,2:,:]-2*psi[:,1:-1,:]+psi[:,:-2,:])/(dZ*dZ)
    d1R[:,:,1:-1]=(psi[:,:,2:]-psi[:,:,:-2])/(2*dR)
    return -(d2R-d1R/R[None,None,:]+d2Z)/(4e-7*np.pi*R[None,None,:])


def fit(nshots,out):
    paths=sorted(p for p in (CACHE_ROOT/'data/diii_d_train').glob('*.parquet') if not is_val(p.name))
    rng=np.random.default_rng(11017)
    paths=[paths[i] for i in np.sort(rng.choice(len(paths),nshots,replace=False))]
    cols=['efit_psirz','efit_times','efit_grid_R','efit_grid_Z','efit_r_axis','efit_z_axis',
          'efit_lcfs_n','efit_lcfs_r','efit_lcfs_z','thomson_core_times','thomson_core_Te',
          'thomson_chord_name','thomson_chord_R','thomson_chord_Z']
    slopes=[];elong=[]
    for num,path in enumerate(paths):
        row=pd.read_parquet(path,columns=cols).iloc[0]
        psi=stack_psirz(row['efit_psirz']).astype(float)
        R,Z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z'])
        RR,ZZ=np.meshgrid(R,Z)
        ts=stack_2d(row['thomson_core_Te'])
        te=_interp_profile(row['thomson_core_times'],ts,np.asarray(row['efit_times']))
        keep=np.char.startswith(np.asarray(row['thomson_chord_name']).astype(str),'TS_core')
        cr,cz=np.asarray(row['thomson_chord_R'])[keep],np.asarray(row['thomson_chord_Z'])[keep]
        if len(cr)!=te.shape[1]:continue
        js=-gs_current(psi,R,Z)
        for i in range(0,len(psi),9):
            if not np.isfinite(psi[i]).all():continue
            nr=int(row['efit_lcfs_n'][i])
            lr,lz=np.asarray(row['efit_lcfs_r'][i])[:nr],np.asarray(row['efit_lcfs_z'][i])[:nr]
            good=np.isfinite(lr)&np.isfinite(lz)
            if good.sum()<12:continue
            interp=RegularGridInterpolator((Z,R),psi[i],bounds_error=False,fill_value=np.nan)
            axis=float(interp([[row['efit_z_axis'][i],row['efit_r_axis'][i]]])[0])
            boundary=float(np.nanmedian(interp(np.c_[lz[good],lr[good]])))
            if abs(boundary-axis)<1e-5:continue
            pn=(interp(np.c_[cz,cr])-axis)/(boundary-axis)
            ok=(pn>.05)&(pn<.90)&np.isfinite(te[i])&(te[i]>20)
            if ok.sum()>8:
                x=np.log(1-pn[ok]);y=np.log(te[i,ok])
                if x.std()>.10:slopes.append(float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2)))
            inside=Polygon(np.c_[lr[good],lz[good]]).contains_points(np.c_[RR.ravel(),ZZ.ravel()]).reshape(RR.shape)
            j=np.maximum(js[i],0)*inside
            if j.sum()<=0:continue
            j/=j.sum();rc=float((j*RR).sum());zc=float((j*ZZ).sum())
            sr=np.sqrt((j*(RR-rc)**2).sum());sz=np.sqrt((j*(ZZ-zc)**2).sum())
            if sr>.05 and sz>.05:elong.append(float(sz/sr))
        if (num+1)%16==0:print(f'{num+1}/{len(paths)} shots, {len(slopes)} kinetic profiles',flush=True)
    old=json.loads((ROOT/'my_experiments/artifacts/d3d_current_profile.json').read_text())
    result={k:old[k] for k in ('alpha','beta','gamma')}
    result.update(te_exponent=float(np.clip(np.median(slopes),.5,4)),
                  current_elongation=float(np.median(elong)),n_shots=len(paths),
                  n_kinetic_profiles=len(slopes),n_current_profiles=len(elong),
                  te_exponent_quantiles=np.quantile(slopes,[.1,.5,.9]).tolist(),
                  fitted_on='DIII-D train only; existing cached official challenge data',
                  training_files=[p.name for p in paths])
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='training_files'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--n-shots',type=int,default=128)
    ap.add_argument('--out',type=Path,default=ROOT/'my_experiments/artifacts/kinetic_prior_v11.json')
    a=ap.parse_args();fit(a.n_shots,a.out)
