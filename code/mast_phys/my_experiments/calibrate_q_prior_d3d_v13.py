"""Calibrate a geometric q uncertainty band on DIII-D; never reads MAST data."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from skimage.measure import find_contours
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,stack_psirz,_interp
from my_experiments.q_shape_prior_v13 import shape_q,ellipse_q
from data_fixes import fix_d3d_ip_times


def extract(path):
    cols=['efit_times','efit_psirz','efit_grid_R','efit_grid_Z','efit_lcfs_n','efit_lcfs_r','efit_lcfs_z',
          'efit_r_axis','efit_z_axis','efit_q95','magnetics_time','magnetics_plasma_current_times',
          'magnetics_plasma_current','magnetics_bcoil']
    row=pd.read_parquet(path,columns=cols).iloc[0]
    t=np.asarray(row['efit_times'],float);r=np.asarray(row['efit_grid_R']);z=np.asarray(row['efit_grid_Z'])
    ip=np.abs(_interp(fix_d3d_ip_times(row),row['magnetics_plasma_current'],t))*1000
    fvac=np.abs(_interp(row['magnetics_time'],row['magnetics_bcoil'],t))*1000*144*2e-7
    phi=-stack_psirz(row['efit_psirz']).astype(float)
    result=[];fr=np.linspace(r[0],r[-1],129);fz=np.linspace(z[0],z[-1],129)
    for i in range(0,len(t),17):
        q=float(row['efit_q95'][i]);ra=float(row['efit_r_axis'][i]);za=float(row['efit_z_axis'][i])
        if ip[i]<3e5 or not 1<q<30 or not np.isfinite(phi[i]).all():continue
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        ok=np.isfinite(lr)&np.isfinite(lz)
        if ok.sum()<12:continue
        sp=RectBivariateSpline(z,r,phi[i]);pa=float(sp.ev(za,ra));pb=float(np.median(sp.ev(lz[ok],lr[ok])))
        if pa-pb<1e-5:continue
        level=pa-.95*(pa-pb);fine=sp(fz,fr);found=None
        for c in find_contours(fine,level):
            if len(c)<8 or np.linalg.norm(c[0]-c[-1])>1e-5:continue
            xy=np.column_stack([np.interp(c[:,1],np.arange(len(fr)),fr),np.interp(c[:,0],np.arange(len(fz)),fz)])
            if Polygon(xy).contains_point([ra,za]):found=xy;break
        if found is None:continue
        x=shape_q(found,ra,za,float(fvac[i]),float(ip[i]))
        if not np.isfinite(x):x=ellipse_q(found,ra,za,float(fvac[i]),float(ip[i]))
        if x>0 and np.isfinite(x):result.append([x,q,q/x])
    return result


def main():
    source=json.loads((ROOT/'my_experiments/artifacts/d3d_pressure_prior_v13.json').read_text())
    train=[];valid=[]
    for key,buf in [('calibration_files',train),('validation_files',valid)]:
        for i,name in enumerate(source[key]):
            buf.extend(extract(CACHE_ROOT/'data/diii_d_train'/name))
            if (i+1)%16==0:print(key,i+1,'shots',len(buf),'frames',flush=True)
    tr,va=np.asarray(train),np.asarray(valid)
    lo,center,hi=np.quantile(tr[:,2],[.01,.5,.99])
    pred=va[:,0]*center
    r2=float(1-np.sum((va[:,1]-pred)**2)/np.sum((va[:,1]-va[:,1].mean())**2))
    coverage=float(np.mean((va[:,1]>=lo*va[:,0])&(va[:,1]<=hi*va[:,0])))
    out={'fitted_on':'DIII-D only','q_ratio_low':float(lo),'q_ratio_center':float(center),'q_ratio_high':float(hi),
         'calibration_quantiles':[.01,.5,.99],'validation_q_r2':r2,'validation_interval_coverage':coverage,
         'calibration_frames':len(tr),'validation_frames':len(va),'q_fallback':float(np.median(tr[:,1])),
         'calibration_files':source['calibration_files'],'validation_files':source['validation_files']}
    (ROOT/'my_experiments/artifacts/d3d_q_shape_prior_v13.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if not k.endswith('_files')},indent=2))


if __name__=='__main__':main()
