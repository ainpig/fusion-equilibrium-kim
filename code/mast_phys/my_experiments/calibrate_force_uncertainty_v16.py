"""Calibrate force-balance discrepancy using DIII-D training labels only."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT,D3D_COILS,green_basis,stack_psirz,_interp
from my_experiments.virial_pressure_v14 import discrete_jr
from data_fixes import fix_d3d_ip_times


def extract(path):
    import pyarrow.parquet as pq
    names=pq.ParquetFile(path).schema_arrow.names
    keep=[k for k in names if k.startswith('coil_') or k in D3D_COILS or k in
        ('efit_times','efit_psirz','efit_grid_R','efit_grid_Z','efit_lcfs_n','efit_lcfs_r','efit_lcfs_z','magnetics_time','magnetics_plasma_current_times')]
    row=pd.read_parquet(path,columns=keep).iloc[0];t=np.asarray(row['efit_times'],float)
    r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z']);rr,zz=np.meshgrid(r,z)
    _,basis=green_basis(row,D3D_COILS)
    # The ambiguous source-machine central-solenoid convention is documented;
    # do not fit its multiplier on target-machine equilibria.
    cur=np.column_stack([_interp(fix_d3d_ip_times(row) if k=='magnetics_plasma_current' else row['magnetics_time'],row[k],t) for k in D3D_COILS])
    vac=np.einsum('tc,czr->tzr',cur*1000,basis,optimize=True);vz=np.gradient(vac,z.astype(float),axis=1,edge_order=2)
    phi=-stack_psirz(row['efit_psirz']).astype(float);result=[]
    for i in range(0,len(t),17):
        ip=abs(cur[i,-1]*1000)
        if ip<3e5 or not np.isfinite(phi[i]).all():continue
        n=int(row['efit_lcfs_n'][i]);lr=np.asarray(row['efit_lcfs_r'][i])[:n];lz=np.asarray(row['efit_lcfs_z'][i])[:n]
        good=np.isfinite(lr)&np.isfinite(lz)
        if good.sum()<12:continue
        inside=Polygon(np.column_stack([lr[good],lz[good]])).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(rr.shape)
        j=np.maximum(discrete_jr(phi[i],r,z)/rr,0)*inside
        if j.sum()<=0:continue
        j/=j.sum();result.append(float((j*vz[i]).sum()/(2e-7*ip)))
    return result


def main():
    art=ROOT/'my_experiments/artifacts';split=json.loads((art/'d3d_geometric_q_prior_v15.json').read_text());values={}
    for key in ('calibration_files','validation_files'):
        values[key]=np.concatenate([extract(CACHE_ROOT/'data/diii_d_train'/n) for n in split[key]])
        print(key,len(values[key]),flush=True)
    train,valid=values['calibration_files'],values['validation_files'];tolerance=float(np.quantile(abs(train),.95))
    report={'fitted_on':'DIII-D only','source_residual_tolerance':tolerance,'quantile':.95,
        'calibration_frames':len(train),'validation_frames':len(valid),'source_validation_coverage':float(np.mean(abs(valid)<=tolerance)),
        'force_quantiles':np.quantile(train,[.01,.05,.5,.95,.99]).tolist(),
        'calibration_files':split['calibration_files'],'validation_files':split['validation_files'],
        'central_solenoid_assumption':'released DIII-D lumped current convention, unit multiplier; no target-domain fitting'}
    (art/'d3d_force_uncertainty_v16.json').write_text(json.dumps(report,indent=2));np.savez(art/'d3d_force_uncertainty_v16.npz',calibration=train,validation=valid)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_files')},indent=2))


if __name__=='__main__':main()
