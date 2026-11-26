"""Source-only pressure-moment model: electrons plus a broader ion pressure.

Calibration and selection read DIII-D only, using the pre-existing split.
No MAST file or MAST coefficient is used here.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import lsq_linear
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,stack_2d,_interp,_interp_profile
from data_fixes import fix_d3d_ip_times
MU0=4e-7*np.pi;E=1.602176634e-19


def moments(pn,te,ne,gridpn,volume_weights):
    xs=[];press=[];dens=[];temps=[]
    for lo in np.linspace(0,.95,20):
        good=(pn>=lo)&(pn<lo+.05)&np.isfinite(te)&np.isfinite(ne)&(te>0)&(ne>0)
        if good.any():
            xs.append(float(np.median(pn[good])));press.append(float(np.median(te[good]*ne[good]*E)))
            dens.append(float(np.median(ne[good])));temps.append(float(np.median(te[good])))
    if len(xs)<5 or min(xs)>.35:raise ValueError('Incomplete thermal profile')
    xp=np.asarray(xs);tp=np.asarray(temps)
    near=xp<.7
    # Source temperature-shape prior used only to extrapolate an unobserved axis.
    t0=float(np.median(tp[near]/np.maximum(1-xp[near],.1)**.736504945183694))
    denom=float(volume_weights.sum())
    peavg=float(np.sum(np.interp(gridpn,xp,press)*volume_weights)/denom)
    navg=float(np.sum(np.interp(gridpn,xp,dens)*volume_weights)/denom)
    return np.array([peavg,E*t0*navg])


def extract(path):
    import pyarrow.parquet as pq
    cols=[k for k in pq.ParquetFile(path).schema_arrow.names if k.startswith('efit_') or k.startswith('thomson_')
          or k in ('magnetics_time','magnetics_plasma_current','magnetics_plasma_current_times','magnetics_bcoil')]
    row=pd.read_parquet(path,columns=cols).iloc[0];t=np.asarray(row['efit_times'],float)
    psi=stack_psirz(row['efit_psirz']).astype(float);r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z'])
    rr,zz=np.meshgrid(r,z);ip=np.abs(_interp(fix_d3d_ip_times(row),row['magnetics_plasma_current'],t))/1000
    fv=np.abs(_interp(row['magnetics_time'],row['magnetics_bcoil'],t))*1000*144*2e-7
    names=np.asarray(row['thomson_chord_name']).astype(str);te=[];ne=[];position=[]
    for system,prefix in [('core','TS_core'),('edge','TS_tangential')]:
        if f'thomson_{system}_Te' not in row:continue
        et=stack_2d(row[f'thomson_{system}_Te']);en=stack_2d(row[f'thomson_{system}_ne']);keep=np.char.startswith(names,prefix)
        if keep.sum()!=et.shape[1] or et.shape!=en.shape:continue
        te.append(_interp_profile(row[f'thomson_{system}_times'],et,t));ne.append(_interp_profile(row[f'thomson_{system}_times'],en,t))
        position.append(np.column_stack([np.asarray(row['thomson_chord_Z'])[keep],np.asarray(row['thomson_chord_R'])[keep]]))
    if not te:return []
    te=np.concatenate(te,axis=1);ne=np.concatenate(ne,axis=1);position=np.concatenate(position);out=[]
    for i in range(0,len(t),17):
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        good=np.isfinite(lr)&np.isfinite(lz);target=float(row['efit_beta_n'][i])
        if good.sum()<12 or ip[i]<.3 or not .05<target<8 or not np.isfinite(psi[i]).all():continue
        lr,lz=lr[good],lz[good];a=np.ptp(lr)/2;rg=(lr.max()+lr.min())/2;bt=fv[i]/rg
        if not .5<bt<3.5 or a<.2:continue
        fn=RegularGridInterpolator((z,r),psi[i],bounds_error=False,fill_value=np.nan)
        pa=float(fn([[row['efit_z_axis'][i],row['efit_r_axis'][i]]])[0]);pb=float(np.median(fn(np.column_stack([lz,lr]))))
        if abs(pb-pa)<1e-5:continue
        inside=Polygon(np.column_stack([lr,lz])).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(rr.shape)
        try:p=moments((fn(position)-pa)/(pb-pa),te[i],ne[i],(psi[i]-pa)/(pb-pa),rr*inside)
        except ValueError:continue
        beta=p*2*MU0*100*a/(bt*ip[i])
        if p[0]>50 and .25<target/beta[0]<20:out.append([*beta,target])
    return out


def main():
    art=ROOT/'my_experiments/artifacts';source=json.loads((art/'d3d_pressure_prior_v13.json').read_text())
    buffers={}
    for label in ('calibration_files','validation_files'):
        records=[]
        for i,name in enumerate(source[label]):
            records.extend(extract(CACHE_ROOT/'data/diii_d_train'/name))
            if (i+1)%16==0:print(label,i+1,'shots',len(records),'frames',flush=True)
        buffers[label]=np.asarray(records)
    tr,va=buffers['calibration_files'],buffers['validation_files']
    fit=lsq_linear(tr[:,:2],tr[:,2],bounds=([1,0],[6,6]))
    single=float(np.dot(tr[:,0],tr[:,2])/np.dot(tr[:,0],tr[:,0]))
    ion=float(np.maximum(np.dot(tr[:,1],tr[:,2]-tr[:,0])/np.dot(tr[:,1],tr[:,1]),0))
    candidates={'electron_only':[single,0.],'electron_plus_ion':[1.,ion],'two_moments':fit.x.tolist()}
    def r2(p):return float(1-np.sum((p-va[:,2])**2)/np.sum((va[:,2]-va[:,2].mean())**2))
    scores={name:r2(va[:,:2]@coef) for name,coef in candidates.items()}
    chosen=max(scores,key=scores.get)
    result={'fitted_on':'DIII-D only','selected_on':'DIII-D held-out shots only','coefficients':candidates[chosen],
            'selected':chosen,'candidates':candidates,'validation_r2':scores,
            'calibration_frames':len(tr),'validation_frames':len(va),
            'calibration_files':source['calibration_files'],'validation_files':source['validation_files']}
    (art/'d3d_pressure_moments_v14.json').write_text(json.dumps(result,indent=2))
    np.savez(art/'d3d_pressure_moments_v14.npz',calibration=tr,validation=va)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_files')},indent=2))


if __name__=='__main__':main()
