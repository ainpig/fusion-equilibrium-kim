"""Source-machine-only total/electron pressure ratio with held-out DIII-D check.

No MAST input, label, prediction or coefficient is opened by this calibration.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as Polygon

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,stack_2d,_interp_profile,_interp
from my_experiments.build_d3d_rich_cache import is_val
from data_fixes import fix_d3d_ip_times

MU0=4e-7*np.pi
E=1.602176634e-19


def samples(path):
    import pyarrow.parquet as pq
    cols=[k for k in pq.ParquetFile(path).schema_arrow.names if k.startswith('efit_') or k.startswith('thomson_')
          or k in ('magnetics_time','magnetics_plasma_current','magnetics_plasma_current_times','magnetics_bcoil')]
    row=pd.read_parquet(path,columns=cols).iloc[0]
    t=np.asarray(row['efit_times'],float)
    psi=stack_psirz(row['efit_psirz']).astype(float)
    R,Z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z'])
    rr,zz=np.meshgrid(R,Z)
    ip=np.abs(_interp(fix_d3d_ip_times(row),row['magnetics_plasma_current'],t))/1000
    tf=np.abs(_interp(row['magnetics_time'],row['magnetics_bcoil'],t))*1000
    names=np.asarray(row['thomson_chord_name']).astype(str)
    profiles=[];positions=[]
    for system,prefix in [('core','TS_core'),('edge','TS_tangential')]:
        if f'thomson_{system}_Te' not in row:
            continue
        te=stack_2d(row[f'thomson_{system}_Te']);ne=stack_2d(row[f'thomson_{system}_ne'])
        keep=np.char.startswith(names,prefix)
        if keep.sum()!=te.shape[1] or te.shape!=ne.shape:
            continue
        profiles.append(_interp_profile(row[f'thomson_{system}_times'],te*ne*E,t))
        positions.append(np.column_stack([np.asarray(row['thomson_chord_Z'])[keep],np.asarray(row['thomson_chord_R'])[keep]]))
    if not profiles:
        return []
    pe=np.concatenate(profiles,axis=1);pos=np.concatenate(positions)
    result=[]
    for i in range(0,len(t),17):
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        good=np.isfinite(lr)&np.isfinite(lz)
        if good.sum()<12 or ip[i]<.3 or not np.isfinite(psi[i]).all():
            continue
        lr,lz=lr[good],lz[good]
        a=np.ptp(lr)/2;rg=(lr.max()+lr.min())/2
        bt=2e-7*144*tf[i]/rg
        target=float(row['efit_beta_n'][i])
        if not .5<bt<3.5 or not .05<target<8 or a<.2:
            continue
        fn=RegularGridInterpolator((Z,R),psi[i],bounds_error=False,fill_value=np.nan)
        axis=float(fn([[row['efit_z_axis'][i],row['efit_r_axis'][i]]])[0])
        bound=float(np.nanmedian(fn(np.column_stack([lz,lr]))))
        if abs(bound-axis)<1e-5:
            continue
        pn=(fn(pos)-axis)/(bound-axis)
        xp=[];yp=[]
        for lo in np.linspace(0,.95,20):
            valid=(pn>=lo)&(pn<lo+.05)&np.isfinite(pe[i])&(pe[i]>0)
            if valid.any():
                xp.append(float(np.median(pn[valid])));yp.append(float(np.median(pe[i,valid])))
        if len(xp)<5 or min(xp)>.35:
            continue
        gridpn=(psi[i]-axis)/(bound-axis)
        pressure=np.interp(gridpn,xp,yp)
        inside=Polygon(np.column_stack([lr,lz])).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(rr.shape)
        pav=float((pressure*rr*inside).sum()/(rr*inside).sum())
        if pav<50:
            continue
        beta_e=2*MU0*pav*100*a/(bt*ip[i])
        ratio=target/beta_e
        if .25<ratio<20:
            result.append([beta_e,target,ratio])
    return result


def main():
    paths=sorted(p for p in (CACHE_ROOT/'data/diii_d_train').glob('*.parquet') if not is_val(p.name))
    rng=np.random.default_rng(13017)
    chosen=[paths[i] for i in rng.choice(len(paths),96,replace=False)]
    train=[];valid=[]
    for i,p in enumerate(chosen):
        (train if i<64 else valid).extend(samples(p))
        if (i+1)%16==0:
            print(f'{i+1}/96 DIII-D shots; {len(train)} calibration / {len(valid)} validation frames',flush=True)
    tr,va=np.asarray(train),np.asarray(valid)
    # Robust, dimensionless pressure moment; no machine-specific output offset.
    factor=float(np.exp(np.median(np.log(tr[:,2]))))
    factor=float(np.clip(factor,1.,10.))
    def r2(p):return float(1-np.sum((va[:,1]-p)**2)/np.sum((va[:,1]-va[:,1].mean())**2))
    out={'fitted_on':'DIII-D training split only; no MAST data','total_to_electron_pressure':factor,
         'train_frames':len(tr),'validation_frames':len(va),'train_ratio_quantiles':np.quantile(tr[:,2],[.1,.5,.9]).tolist(),
         'd3d_validation_r2_with_thermal_factor_2':r2(2*va[:,0]),'d3d_validation_r2_with_prior':r2(factor*va[:,0]),
         'calibration_files':[p.name for p in chosen[:64]],'validation_files':[p.name for p in chosen[64:]],
         'tf_turns_source':'https://fusion.gat.com/pubs-ext/SOFT02/A24059.pdf','tf_turns':144}
    (ROOT/'my_experiments/artifacts/d3d_pressure_prior_v13.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if not k.endswith('files')},indent=2))


if __name__=='__main__':main()
