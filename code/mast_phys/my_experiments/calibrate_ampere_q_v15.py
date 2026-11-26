"""DIII-D-only calibration of scale-free, Ampere-normalized contour q.

q_A = F/(2*pi*mu0*Ip) (integral dl/(R|grad psi|))
                              (integral |grad psi| dl/R).
The product removes global derivative-amplitude errors. The small remaining
ratio reflects current outside the 95% surface and the toroidal-field profile.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from skimage.measure import find_contours
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,_interp
from data_fixes import fix_d3d_ip_times
MU0=4e-7*np.pi


def ampere_q(spline,curve,fvac,ip):
    c=np.asarray(curve,float)
    if np.linalg.norm(c[0]-c[-1])<1e-9:c=c[:-1]
    nxt=np.roll(c,-1,axis=0);mid=(c+nxt)/2;ds=np.linalg.norm(nxt-c,axis=1)
    grad=np.hypot(spline.ev(mid[:,1],mid[:,0],dy=1),spline.ev(mid[:,1],mid[:,0],dx=1))
    if np.any(grad<1e-10) or abs(ip)<1:return np.nan
    return float(abs(fvac)/(2*np.pi*MU0*abs(ip))*np.sum(ds/(mid[:,0]*grad))*np.sum(ds*grad/mid[:,0]))


def samples(path):
    cols=['efit_times','efit_psirz','efit_grid_R','efit_grid_Z','efit_lcfs_n','efit_lcfs_r','efit_lcfs_z',
          'efit_r_axis','efit_z_axis','efit_q95','magnetics_time','magnetics_plasma_current_times','magnetics_plasma_current','magnetics_bcoil']
    row=pd.read_parquet(path,columns=cols).iloc[0];t=np.asarray(row['efit_times'],float)
    r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z']);phi=-stack_psirz(row['efit_psirz']).astype(float)
    ip=np.abs(_interp(fix_d3d_ip_times(row),row['magnetics_plasma_current'],t))*1000
    fv=np.abs(_interp(row['magnetics_time'],row['magnetics_bcoil'],t))*1000*144*2e-7
    fr=np.linspace(r[0],r[-1],257);fz=np.linspace(z[0],z[-1],257);result=[]
    for i in range(0,len(t),17):
        q=float(row['efit_q95'][i]);ra,za=float(row['efit_r_axis'][i]),float(row['efit_z_axis'][i])
        if ip[i]<3e5 or not 1<q<30 or not np.isfinite(phi[i]).all():continue
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        good=np.isfinite(lr)&np.isfinite(lz)
        if good.sum()<12:continue
        sp=RectBivariateSpline(z,r,phi[i]);pa=float(sp.ev(za,ra));pb=float(np.median(sp.ev(lz[good],lr[good])))
        if pa-pb<1e-5:continue
        found=None
        for c in find_contours(sp(fz,fr),pa-.95*(pa-pb)):
            if len(c)<8 or np.linalg.norm(c[0]-c[-1])>1e-5:continue
            curve=np.column_stack([np.interp(c[:,1],np.arange(len(fr)),fr),np.interp(c[:,0],np.arange(len(fz)),fz)])
            if Polygon(curve).contains_point((ra,za)):found=curve;break
        if found is None:continue
        value=ampere_q(sp,found,fv[i],ip[i])
        if np.isfinite(value) and value>0:result.append([value,q,q/value])
    return result


def main():
    art=ROOT/'my_experiments/artifacts';split=json.loads((art/'d3d_pressure_prior_v13.json').read_text());buf={}
    for key in ('calibration_files','validation_files'):
        values=[]
        for i,name in enumerate(split[key]):
            values.extend(samples(CACHE_ROOT/'data/diii_d_train'/name))
            if (i+1)%16==0:print(key,i+1,len(values),'frames',flush=True)
        buf[key]=np.asarray(values)
    train,valid=buf['calibration_files'],buf['validation_files'];ratio=float(np.median(train[:,2]))
    def r2(p):return float(1-np.sum((p-valid[:,1])**2)/np.sum((valid[:,1]-valid[:,1].mean())**2))
    report={'fitted_on':'DIII-D only','ratio':ratio,'ratio_quantiles':np.quantile(train[:,2],[.01,.1,.5,.9,.99]).tolist(),
        'calibration_frames':len(train),'validation_frames':len(valid),'validation_r2_raw':r2(valid[:,0]),
        'validation_r2_calibrated':r2(valid[:,0]*ratio),'calibration_files':split['calibration_files'],'validation_files':split['validation_files']}
    (art/'d3d_ampere_q_prior_v15.json').write_text(json.dumps(report,indent=2));np.savez(art/'d3d_ampere_q_prior_v15.npz',calibration=train,validation=valid)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_files')},indent=2))


if __name__=='__main__':main()
