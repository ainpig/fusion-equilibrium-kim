"""Calibrate contour-centre and elliptical Ampere priors on DIII-D only."""
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
from my_experiments.q_shape_prior_v13 import shape_q,ellipse_q
from data_fixes import fix_d3d_ip_times


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
        centre=(found.max(0)+found.min(0))/2
        values=[shape_q(found,ra,za,fv[i],ip[i]),shape_q(found,*centre,fv[i],ip[i]),ellipse_q(found,ra,za,fv[i],ip[i])]
        if np.isfinite(values).all() and min(values)>0:result.append([q,*values])
    return result


def main():
    art=ROOT/'my_experiments/artifacts';split=json.loads((art/'d3d_q_shape_prior_v13.json').read_text());buf={}
    for key in ('calibration_files','validation_files'):
        values=[]
        for i,name in enumerate(split[key]):
            values.extend(samples(CACHE_ROOT/'data/diii_d_train'/name))
            if (i+1)%16==0:print(key,i+1,len(values),'frames',flush=True)
        buf[key]=np.asarray(values)
    train,valid=buf['calibration_files'],buf['validation_files'];report={'fitted_on':'DIII-D only','methods':{}}
    for j,name in enumerate(('axis','centre','ellipse'),1):
        ratios=train[:,0]/train[:,j];ratio=float(np.median(ratios));pred=valid[:,j]*ratio
        report['methods'][name]={'ratio':ratio,'quantiles':np.quantile(ratios,[.01,.1,.5,.9,.99]).tolist(),
            'validation_r2':float(1-np.sum((pred-valid[:,0])**2)/np.sum((valid[:,0]-valid[:,0].mean())**2))}
    ax=report['methods']['axis']['ratio'];el=report['methods']['ellipse']['ratio']
    bounds=np.quantile(train[:,1]*ax/(train[:,3]*el),[.01,.99])
    va=valid[:,1]*ax;ve=valid[:,3]*el;outside=(va<ve*bounds[0])|(va>ve*bounds[1])
    guarded=np.where(outside,ve,va)
    report['conditioning']={'fitted_on':'DIII-D only','quantiles':[.01,.99],'ratio_low':float(bounds[0]),'ratio_high':float(bounds[1]),
        'action':'use elliptical prior outside the source-supported contour/ellipse ratio interval',
        'validation_fallback_frames':int(outside.sum()),
        'validation_r2':float(1-np.sum((guarded-valid[:,0])**2)/np.sum((valid[:,0]-valid[:,0].mean())**2))}
    report.update(calibration_frames=len(train),validation_frames=len(valid),calibration_files=split['calibration_files'],validation_files=split['validation_files'])
    (art/'d3d_geometric_q_prior_v15.json').write_text(json.dumps(report,indent=2));np.savez(art/'d3d_geometric_q_prior_v15.npz',calibration=train,validation=valid)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_files')},indent=2))


if __name__=='__main__':main()
