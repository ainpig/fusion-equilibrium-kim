"""Calibrate pressure uncertainty on DIII-D only; test on separate DIII-D shots."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,_interp
from my_experiments.gs_profiles_v13 import recover_profiles,pressure_and_f2,MU0
from my_experiments.gs_uncertainty_v14 import pressure_uncertainty,posterior
from data_fixes import fix_d3d_ip_times


def samples(path):
    cols=['efit_times','efit_psirz','efit_grid_R','efit_grid_Z','efit_lcfs_n','efit_lcfs_r','efit_lcfs_z',
          'efit_r_axis','efit_z_axis','efit_beta_n','magnetics_time','magnetics_plasma_current_times',
          'magnetics_plasma_current','magnetics_bcoil']
    row=pd.read_parquet(path,columns=cols).iloc[0];t=np.asarray(row['efit_times'],float)
    r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z']);phi=-stack_psirz(row['efit_psirz']).astype(float)
    ip=np.abs(_interp(fix_d3d_ip_times(row),row['magnetics_plasma_current'],t))*1000
    fv=np.abs(_interp(row['magnetics_time'],row['magnetics_bcoil'],t))*1000*144*2e-7
    fr=np.linspace(r[0],r[-1],129);fz=np.linspace(z[0],z[-1],129);rr,zz=np.meshgrid(fr,fz)
    out=[]
    for i in range(0,len(t),34):
        target=float(row['efit_beta_n'][i]);ra,za=float(row['efit_r_axis'][i]),float(row['efit_z_axis'][i])
        if ip[i]<3e5 or not .05<target<8 or not np.isfinite(phi[i]).all():continue
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        valid=np.isfinite(lr)&np.isfinite(lz)
        if valid.sum()<12:continue
        lr,lz=lr[valid],lz[valid];boundary=np.column_stack([lr,lz]);a=np.ptp(lr)/2;rg=(lr.max()+lr.min())/2
        sp=RectBivariateSpline(z,r,phi[i]);pa=float(sp.ev(za,ra));pb=float(np.median(sp.ev(lz,lr)))
        if pa-pb<1e-5:continue
        inside=Polygon(boundary).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(rr.shape)
        g={'axis':(ra,za,pa),'xpoint':(0,0,pb),'boundary':boundary,'R':fr,'Z':fz,'flux':sp(fz,fr),'inside':inside}
        try:
            fit=recover_profiles(phi[i],r,z,g,ip[i]);p,_=pressure_and_f2(fit,(pa-g['flux'])/(pa-pb),fv[i])
            pav=float((p*rr*inside).sum()/(rr*inside).sum());beta=2*MU0*100*a*rg*pav/(fv[i]*ip[i]/1e6)
            sigma,_=pressure_uncertainty(phi[i],r,z,g,ip[i],fv[i])
        except (ValueError,np.linalg.LinAlgError):continue
        if np.isfinite(beta) and np.isfinite(sigma) and sigma>1e-6:out.append([beta,sigma,target,fit['relative_residual']])
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--max-shots',type=int,default=0);args=ap.parse_args()
    art=ROOT/'my_experiments/artifacts';split=json.loads((art/'d3d_pressure_prior_v13.json').read_text());buf={}
    for key in ('calibration_files','validation_files'):
        files=split[key][:args.max_shots or None];rows=[]
        for i,name in enumerate(files):
            rows.extend(samples(CACHE_ROOT/'data/diii_d_train'/name))
            if (i+1)%8==0 or i+1==len(files):print(key,i+1,'shots',len(rows),'frames',flush=True)
        buf[key]=np.asarray(rows)
    tr,va=buf['calibration_files'],buf['validation_files']
    # Robust normal-error calibration, fixed 50% quantile; no MAST-based scale selection.
    scale=max(1.,float(np.median(abs(tr[:,0]-tr[:,2])/tr[:,1])/.6744897501960817))
    prior=json.loads((art/'d3d_beta_uncertainty_prior_v14.json').read_text())
    pred,_=posterior(va[:,0],va[:,1]*scale,prior['mean'],prior['variance'])
    def r2(x):return float(1-np.sum((x-va[:,2])**2)/np.sum((va[:,2]-va[:,2].mean())**2))
    result={'fitted_on':'DIII-D only','scale':scale,'calibration_frames':len(tr),'validation_frames':len(va),
            'validation_physics_r2':r2(va[:,0]),'validation_posterior_r2':r2(pred),
            'validation_68pct_coverage':float(np.mean(abs(va[:,0]-va[:,2])<=scale*va[:,1])),
            'max_shots':args.max_shots,'source_split':split,'prior':prior}
    tag='d3d_uncertainty_calibration_v14'+('_smoke' if args.max_shots else '')
    (art/f'{tag}.json').write_text(json.dumps(result,indent=2));np.savez(art/f'{tag}.npz',calibration=tr,validation=va)
    print(json.dumps({k:v for k,v in result.items() if k!='source_split'},indent=2))


if __name__=='__main__':main()
