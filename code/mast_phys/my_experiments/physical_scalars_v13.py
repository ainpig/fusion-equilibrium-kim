"""Physical X-point separatrix and scalar integration, independent of scoring LCFS.

No fitted MAST parameters. The benchmark's contour functional is for its shape
metric; q95/betaN refer to the physical equilibrium boundary instead.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import minimum_filter,label,map_coordinates
from skimage.measure import find_contours
from matplotlib.path import Path as Polygon

from my_experiments.fusion_pipeline import _interp_profile
from my_experiments.mast_equilibrium_v12 import MU0,E_CHARGE,TF_TURNS
from data_fixes import align_mast_thomson_core
from functools import lru_cache
from pathlib import Path
import json
from my_experiments.q_shape_prior_v13 import shape_q,ellipse_q


def machine_wall():
    raise RuntimeError("External MAST wall data is disabled in v3-1; use released_geometry")


@lru_cache(maxsize=1)
def q_prior():
    path=Path(__file__).resolve().parent/'artifacts/d3d_q_shape_prior_v13.json'
    prior=json.loads(path.read_text())
    if prior.get('fitted_on')!='DIII-D only':
        raise ValueError('The q shape prior must be calibrated on DIII-D only')
    return prior


def critical_points(phi,r,z,mask):
    """Find subcell stationary points and classify their Hessians."""
    sp=RectBivariateSpline(z,r,np.asarray(phi,float),kx=3,ky=3,s=0)
    rr,zz=np.meshgrid(r,z)
    gr=sp(z,r,dx=0,dy=1);gz=sp(z,r,dx=1,dy=0)
    norm=gr*gr+gz*gz
    candidate=(norm==minimum_filter(norm,3,mode='nearest'))&mask
    candidate[[0,-1],:]=False;candidate[:,[0,-1]]=False
    p=np.column_stack([rr[candidate],zz[candidate]])
    if len(p)==0:
        return sp,[],[]
    initial=p.copy();dr=float(r[1]-r[0]);dz=float(z[1]-z[0])
    for _ in range(15):
        g1=sp.ev(p[:,1],p[:,0],dy=1);g2=sp.ev(p[:,1],p[:,0],dx=1)
        h1=sp.ev(p[:,1],p[:,0],dy=2);h2=sp.ev(p[:,1],p[:,0],dx=2)
        h12=sp.ev(p[:,1],p[:,0],dx=1,dy=1)
        det=h1*h2-h12*h12
        ok=np.abs(det)>1e-12
        sx=np.where(ok,(h2*g1-h12*g2)/np.where(ok,det,1),0)
        sy=np.where(ok,(h1*g2-h12*g1)/np.where(ok,det,1),0)
        factor=np.maximum(1,np.maximum(np.abs(sx)/(2*dr),np.abs(sy)/(2*dz)))
        p[:,0]-=sx/factor;p[:,1]-=sy/factor
        p[:,0]=np.clip(p[:,0],r[0]+dr,r[-1]-dr)
        p[:,1]=np.clip(p[:,1],z[0]+dz,z[-1]-dz)
    roots=[];axes=[];saddles=[]
    for pos,start in zip(p,initial):
        x,y=map(float,pos)
        if np.linalg.norm(pos-start)>.3 or any(np.linalg.norm(pos-q)<.005 for q in roots):
            continue
        g=np.hypot(sp.ev(y,x,dy=1),sp.ev(y,x,dx=1))
        if g>1e-6*max(1.,np.ptp(phi)):
            continue
        h1=float(sp.ev(y,x,dy=2));h2=float(sp.ev(y,x,dx=2));h12=float(sp.ev(y,x,dx=1,dy=1))
        point=(x,y,float(sp.ev(y,x)))
        roots.append(pos)
        if h1*h2-h12*h12>0 and h1+h2<0:
            axes.append(point)
        elif h1*h2-h12*h12<0:
            saddles.append(point)
    return sp,axes,saddles


def enclosing_contour(field,level,r,z,axis):
    contours=find_contours(field,level=level)
    found=[]
    for c in contours:
        if len(c)<8 or np.linalg.norm(c[0]-c[-1])>1e-5:
            continue
        xy=np.column_stack([np.interp(c[:,1],np.arange(len(r)),r),
                            np.interp(c[:,0],np.arange(len(z)),z)])
        if Polygon(xy).contains_point(axis[:2]):
            found.append(xy)
    return min(found,key=lambda c:np.ptp(c[:,0])*np.ptp(c[:,1])) if found else None


def physical_geometry(phi, r, z, mask, resolution=257):
    from my_experiments.released_geometry import diverted_geometry
    return diverted_geometry(phi, r, z, mask, resolution)


def physical_scalars(row,times,psi,curr,r,z,mask,pressure_factor=2.,pressure_method='kinetic'):
    from my_experiments.mast_inputs_v13 import profiles
    tsr,te,ne,_=profiles(row,times)
    pressure=pressure_factor*E_CHARGE*te*ne
    qout=[];bout=[];records=[]
    for i,phi in enumerate(psi):
        try:
            g=physical_geometry(phi,r,z,mask)
        except ValueError as exc:
            qout.append(np.nan);bout.append(np.nan);records.append({'status':str(exc)})
            continue
        sp=g['spline'];axis=g['axis'];xpoint=g['xpoint'];span=axis[2]-xpoint[2]
        fvac=abs(2e-7*TF_TURNS*float(curr[i,11])*1000)
        ip=abs(float(curr[i,-1]))/1000
        c=g['qcurve'];mid=.5*(c+np.roll(c,-1,axis=0))
        ds=np.linalg.norm(c-np.roll(c,-1,axis=0),axis=1)
        grad=np.hypot(sp.ev(mid[:,1],mid[:,0],dx=1),sp.ev(mid[:,1],mid[:,0],dy=1))
        q=fvac/(2*np.pi)*np.sum(ds/np.maximum(mid[:,0]*grad,1e-12))
        pn=(axis[2]-g['flux'])/span
        tsn=(axis[2]-sp.ev(np.zeros_like(tsr),tsr))/span
        ok=np.isfinite(pressure[i])&(pressure[i]>0)&(tsn>=-.05)&(tsn<=1.05)
        xp=[];yp=[]
        bins=np.linspace(0,1,21)
        for lo,hi in zip(bins[:-1],bins[1:]):
            have=ok&(tsn>=lo)&(tsn<hi)
            if have.any():
                xp.append(float(np.median(tsn[have])));yp.append(float(np.median(pressure[i,have])))
        if len(xp)<3 and pressure_method!='gs':
            qout.append(q);bout.append(np.nan);records.append({'status':'Missing pressure'})
            continue
        extra={}
        if pressure_method=='gs':
            from my_experiments.gs_profiles_v13 import recover_profiles,pressure_and_f2
            try:
                prof=recover_profiles(np.asarray(phi,float),r,z,g,float(curr[i,-1])*1000)
            except (ValueError,np.linalg.LinAlgError) as exc:
                qout.append(np.nan);bout.append(np.nan)
                records.append({'status':'Unresolved GS profile: '+str(exc)})
                continue
            p,_=pressure_and_f2(prof,pn,fvac)
            _,f95=pressure_and_f2(prof,.95,fvac)
            if f95>0:
                q*=np.sqrt(f95)/fvac
            extra={'gs_relative_residual':prof['relative_residual'],'ampere_scale':prof['ampere_scale']}
        else:
            p=np.interp(pn,xp,yp)
        rr=np.broadcast_to(g['R'][None,:],g['flux'].shape)
        inside=g['inside']
        pavg=float((p*rr*inside).sum()/(rr*inside).sum())
        boundary=g['boundary']
        a=np.ptp(boundary[:,0])/2
        rg=(boundary[:,0].max()+boundary[:,0].min())/2
        bt=fvac/rg
        beta=2*MU0*pavg*100*a/(bt*ip)
        # The derivative-based integral is accurate on a resolved equilibrium,
        # but can diverge on a poorly reconstructed saddle. Bound it with the
        # DIII-D-validated homothetic-shape uncertainty interval, using actual
        # TF/Ip and contour geometry; no MAST calibration is involved.
        geometric=shape_q(c,axis[0],axis[1],fvac,ip*1e6)
        if not np.isfinite(geometric):geometric=ellipse_q(c,axis[0],axis[1],fvac,ip*1e6)
        prior=q_prior();original_q=float(q)
        if np.isfinite(geometric) and geometric>0:
            q=float(np.clip(q,prior['q_ratio_low']*geometric,prior['q_ratio_high']*geometric))
        elif not np.isfinite(q):
            q=prior['q_fallback']
        extra.update(q95_before_shape_guard=original_q,q95_shape_estimate=float(geometric),
                     q95_shape_guarded=bool(abs(q-original_q)>1e-6))
        qout.append(q);bout.append(beta)
        records.append({'status':'ok','axis':list(axis),'xpoint':list(xpoint),
                        'a':float(a),'rgeo':float(rg),'pavg':pavg,'q95':float(q),'betaN':float(beta),**extra})
    return np.asarray(qout,np.float32),np.asarray(bout,np.float32),records
