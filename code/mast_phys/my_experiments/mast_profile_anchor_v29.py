"""Monotone flux-profile adjustment toward the baseline's own MAST inductance."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fusion_scoring'))
from derive import _bilinear
from lcfs import extract_lcfs_with_sign
from o_point import find_o_point
from my_experiments.fast_geometry_exact_v29 import grid_inside,internal_inductance
from my_experiments.li_profile_v24 import solve,apply


def monotone_limit():
    polynomial=np.polynomial.Polynomial([0,0,1,-4,6,-4,1])
    derivative=polynomial.deriv()
    roots=derivative.deriv().roots()
    x=np.r_[0.,1.,[float(v.real) for v in roots if abs(v.imag)<1e-12 and 0<v.real<1]]
    return .95/float(np.max(np.abs(derivative(x))))


def prepare_profile(psi,r,z,mask):
    p=np.asarray(psi,float)
    contour,_=extract_lcfs_with_sign(p,r,z,'MAST',mask,mask.astype(float),n_iter=22,axis_sign=1.)
    if contour is None:
        return None
    phi=p
    _,_,iz,ir,_=find_o_point(phi,r,z,mask)
    pa=float(phi[iz,ir])
    pb=float(np.median(_bilinear(phi,r,z,contour[:,0],contour[:,1])))
    span=pa-pb
    if not np.isfinite(span) or span<=1e-8:
        return None
    rr,zz=np.meshgrid(r,z)
    inside=grid_inside(contour,r,z)
    if inside.sum()<4:
        return None
    s=np.clip((phi-pb)/span,0,1)
    delta=span*s**2*(1-s)**4*inside
    gz,gr=np.gradient(phi,z,r)
    dz,dr=np.gradient(delta,z,r)
    c=(gr*gr+gz*gz)/rr**2
    b=2*(gr*dr+gz*dz)/rr**2
    a=(dr*dr+dz*dz)/rr**2
    vw=2*np.pi*rr*float(np.mean(np.diff(r)))*float(np.mean(np.diff(z)))*inside
    numerator=np.array([np.sum(v*vw)/vw.sum() for v in (c,b,a)])
    fr=np.interp(contour[:,0],r,np.arange(len(r)))
    fz=np.interp(contour[:,1],z,np.arange(len(z)))
    r0=np.clip(np.floor(fr).astype(int),0,len(r)-2)
    z0=np.clip(np.floor(fz).astype(int),0,len(z)-2)
    tr,tz=fr-r0,fz-z0
    seg=np.linalg.norm(np.roll(contour,-1,axis=0)-contour,axis=1)
    vertex=.5*(seg+np.roll(seg,1))/seg.sum()
    weights=np.zeros(phi.size)
    for rx,zx,w in ((0,0,(1-tr)*(1-tz)),(1,0,tr*(1-tz)),(0,1,(1-tr)*tz),(1,1,tr*tz)):
        np.add.at(weights,(z0+zx)*len(r)+r0+rx,vertex*w)
    keep=weights!=0
    coefficients=np.stack([v.ravel()[keep] for v in (c,b,a)])
    weights=weights[keep]
    li=float(numerator[0]/np.sum(weights*np.sqrt(np.maximum(coefficients[0],0)))**2)
    if not np.isfinite(li) or not .1<=li<=5:
        return None
    return dict(delta=delta,coefficients=coefficients,weights=weights,numerator=numerator,li=li)


def stage_shot(job):
    base,candidate,r,z,mask=job
    staged=[]
    for b,p in zip(base,candidate):
        contour,_=extract_lcfs_with_sign(np.asarray(b,float),r,z,'MAST',mask,mask.astype(float),n_iter=22,axis_sign=1.)
        target=internal_inductance(np.asarray(b,float),r,z,'MAST',contour)
        profile=prepare_profile(p,r,z,mask)
        if not np.isfinite(target) or not .1<=target<=5:profile=None
        staged.append((profile,target))
    return staged


def anchor(base,candidate,staged):
    result=np.asarray(candidate,np.float32).copy()
    good=[i for i,(p,t) in enumerate(staged) if p is not None]
    bad=[i for i,(p,t) in enumerate(staged) if p is None]
    failures=0
    amounts=np.zeros(len(result))
    if good:
        profiles=[staged[i][0] for i in good];targets=np.asarray([staged[i][1] for i in good])
        value,li=solve(profiles,targets,max_amount=monotone_limit(),iterations=40)
        fixed=apply(result[good],profiles,value)
        safe=np.abs(li-targets)<1e-5
        fixed[~safe]=np.asarray(base)[np.asarray(good)[~safe]]
        result[good]=fixed;amounts[good]=value;failures=int((~safe).sum())
    result[bad]=np.asarray(base)[bad]
    return result,dict(unresolved=len(bad),profile_fallback=failures,max_amount=float(np.max(np.abs(amounts))),monotone_limit=monotone_limit())
