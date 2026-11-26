"""v16-compatible scalar staging with an independent virial pressure moment."""
import numpy as np
from pathlib import Path
from matplotlib.path import Path as Polygon
from my_experiments.physical_scalars_v13 import physical_geometry
from my_experiments.mast_inputs_v15 import profiles
from my_experiments.thermal_floor_v14 import electron_beta
from my_experiments.gpu_profiles_v15 import prepare_geometry
from my_experiments.calibrate_ampere_q_v15 import ampere_q
from my_experiments.q_shape_prior_v13 import shape_q,ellipse_q
from my_experiments.fusion_pipeline import MAST_COILS,_interp
from my_experiments.mast_constants import MAST_TF_TURNS
from my_experiments.virial_pressure_v14 import pressure_moment,MU0
from my_experiments.slice_pressure_v17 import pressure_slices
ROOT=Path(__file__).resolve().parents[1]


def prepare_shot(row,times,phi,closed=False):
    geometry_function = physical_geometry
    if closed:
        from my_experiments.fast_geometry_v17 import physical_geometry as geometry_function
    r,z=np.asarray(row['efit_grid_R']),np.asarray(row['efit_grid_Z']);rr,zz=np.meshgrid(r,z)
    from my_experiments.released_geometry import grid
    points=np.column_stack([rr.ravel(),zz.ravel()]);mask=grid(row)[2]
    curr=np.column_stack([_interp(row['magnetics_time'],row[k],times) for k in MAST_COILS]);ip=abs(curr[:,-1]*1000);fv=abs(curr[:,11]*1000*MAST_TF_TURNS*2e-7)
    tsr,te,ne,_=profiles(row,times);geometries=[];ids=[];floors=[];qa=[];qh=[];qe=[];qc=[];virial=np.full(len(times),np.nan);bounded=np.full(len(times),np.nan)
    for i,frame in enumerate(phi):
        try:
            g=geometry_function(frame,r,z,mask);pn=(g['axis'][2]-frame)/(g['axis'][2]-g['xpoint'][2])
            inside=Polygon(g['boundary']).contains_points(points).reshape(rr.shape)
            if np.count_nonzero((pn>.015)&(pn<.985)&inside)<60:raise ValueError('Too few current cells')
            av=ampere_q(g['spline'],g['qcurve'],fv[i],ip[i]);hv=shape_q(g['qcurve'],*g['axis'][:2],fv[i],ip[i]);ev=ellipse_q(g['qcurve'],*g['axis'][:2],fv[i],ip[i])
            centre=(g['qcurve'].max(0)+g['qcurve'].min(0))/2;cv=shape_q(g['qcurve'],*centre,fv[i],ip[i])
            if not np.isfinite(hv):hv=ev
            if not np.isfinite(cv):cv=ev
            if not np.isfinite(av) or not np.isfinite(hv):raise ValueError('Invalid q')
            floor=electron_beta(g,tsr,te[i],ne[i],fv[i],ip[i]);c=g['boundary'];minor=np.ptp(c[:,0])/2;rg=(c[:,0].min()+c[:,0].max())/2
            moments=pressure_slices(frame,r,z,g,ip[i],fv[i],tsr,te[i],ne[i])
            virial[i]=moments['raw'];bounded[i]=moments['electron_bounded']
            ids.append(i);geometries.append(g);floors.append(floor);qa.append(av);qh.append(hv);qe.append(ev);qc.append(cv)
        except (ValueError,np.linalg.LinAlgError):continue
    result={'indices':np.asarray(ids,int),'times':np.asarray(times),'n':len(times),'prepared':None,'virial':virial,'bounded_virial':bounded}
    if ids:
        result.update(prepared=prepare_geometry(np.asarray(phi)[ids],r,z,geometries,ip[ids],fv[ids]),
            thermal_floor=np.asarray(floors),q_ampere=np.asarray(qa),q_shape=np.asarray(qh),q_ellipse=np.asarray(qe),q_centre=np.asarray(qc))
    return result


def prepare_shot_closed(row,times,phi):
    return prepare_shot(row,times,phi,closed=True)
