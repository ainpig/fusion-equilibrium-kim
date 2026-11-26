"""Structured-grid implementation of the scorer's identical even-odd predicate."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fusion_scoring'))
sys.path.insert(0,str(ROOT))
import local_score                          # the vendored competition scorer (zeroshot_mast/local_score.py)
from derive import _poloidal_field,_bilinear


def grid_inside(contour,r,z):
    p=np.asarray(contour,float);q=np.roll(p,-1,axis=0)
    r,z=np.asarray(r,float),np.asarray(z,float)
    crossing=(p[None,:,1]>z[:,None])!=(q[None,:,1]>z[:,None])
    dy=q[:,1]-p[:,1]
    x=p[None,:,0]+(z[:,None]-p[None,:,1])*(q[:,0]-p[:,0])[None,:]/np.where(dy!=0,dy,1.)[None,:]
    sorted_x=np.sort(np.where(crossing,x,np.inf),axis=1)
    count=crossing.sum(1)
    # The original predicate is r < x_cross. side='right' keeps equality identical.
    return np.stack([(count[i]-np.searchsorted(row,r,side='right'))%2==1 for i,row in enumerate(sorted_x)])


def internal_inductance(psi,R,Z,machine,C):
    if C is None or len(C)<8:return np.nan
    Rv=np.asarray(R,float);Zv=np.asarray(Z,float)
    Bp=_poloidal_field(psi,R,Z,machine)
    RR,ZZ=np.meshgrid(Rv,Zv)
    inside=grid_inside(C,Rv,Zv)
    if inside.sum()<4:return np.nan
    dR=float(np.mean(np.diff(Rv)));dZ=float(np.mean(np.diff(Zv)))
    dV=2.*np.pi*RR*dR*dZ
    Bp2=np.where(np.isfinite(Bp),Bp,0.)**2
    num=float(np.sum(Bp2[inside]*dV[inside]));V=float(np.sum(dV[inside]))
    if V<=0:return np.nan
    Bp2_V=num/V
    Bp_on_C=_bilinear(Bp,Rv,Zv,C[:,0],C[:,1])
    Cc=np.vstack([C,C[0]]);seg=np.sqrt(np.sum(np.diff(Cc,axis=0)**2,axis=1))
    Bp_mid=.5*(Bp_on_C+np.roll(Bp_on_C,-1))
    G=float(np.sum(Bp_mid*seg));lp=float(np.sum(seg))
    if G<=0 or lp<=0:return np.nan
    return float(Bp2_V/(G/lp)**2)


def geometry(phi,r,z,mask):
    scorer=local_score;result=[]
    for frame in phi:
        field=np.asarray(frame,float)
        contour,_=scorer.extract_lcfs_with_sign(field,r,z,'DIII-D',mask,mask.astype(float),n_iter=22,axis_sign=-1.)
        values=scorer.derive_frame(field,r,z,'DIII-D',mask,mask.astype(float),contour=contour,with_li=False,axis_sign=-1.)
        li=internal_inductance(field,r,z,'DIII-D',contour)
        values['li']=li if np.isfinite(li) and .1<=li<=5 else np.nan
        result.append([values[k] for k in scorer.CONS_SCALARS])
    return np.asarray(result)
