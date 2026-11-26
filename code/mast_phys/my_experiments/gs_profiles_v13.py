"""Recover flux functions by the local Grad-Shafranov identity, not target fitting.

With known vacuum TF, J_phi R = R^2 p'(psi) + FF'(psi)/mu0 separates the
pressure and field functions across a flux surface. Coefficients describe one
observed/predicted equilibrium and are never trained across MAST examples.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import comb,betainc
from matplotlib.path import Path as Polygon

MU0=4e-7*np.pi


def bernstein(s,degree=4):
    s=np.clip(np.asarray(s),0,1)
    return np.stack([comb(degree,k)*s**k*(1-s)**(degree-k) for k in range(degree+1)],axis=-1)


def integrated_basis(s,degree=4):
    s=np.clip(np.asarray(s),0,1)
    return np.stack([(1-betainc(k+1,degree-k+1,s))/(degree+1) for k in range(degree+1)],axis=-1)


def recover_profiles(phi,r,z,geometry,ip_amp):
    rr,zz=np.meshgrid(r,z)
    dr=float(r[1]-r[0]);dz=float(z[1]-z[0])
    drr=np.zeros_like(phi,dtype=float);dzz=np.zeros_like(phi,dtype=float);d1r=np.zeros_like(phi,dtype=float)
    drr[:,1:-1]=(phi[:,2:]-2*phi[:,1:-1]+phi[:,:-2])/dr**2
    dzz[1:-1]=(phi[2:]-2*phi[1:-1]+phi[:-2])/dz**2
    d1r[:,1:-1]=(phi[:,2:]-phi[:,:-2])/(2*dr)
    jr=-(drr-d1r/rr+dzz)/MU0
    pa=geometry['axis'][2];pb=geometry['xpoint'][2];span=pa-pb
    pn=(pa-phi)/span
    inside=Polygon(geometry['boundary']).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(phi.shape)
    valid=inside&(pn>.015)&(pn<.985)&np.isfinite(jr)
    valid[[0,-1],:]=False;valid[:,[0,-1]]=False
    if valid.sum()<60:
        raise ValueError('Too few plasma cells to resolve GS profiles')
    # Current is an available input constraint, used only to correct discretization.
    current=float(np.sum(jr[inside]/rr[inside])*dr*dz)
    ampere_scale=abs(ip_amp)/current if current>0 else 1.
    if .75<ampere_scale<1.25:
        jr=jr*ampere_scale
    scale=max(float(np.sqrt(np.mean(jr[valid]**2))),1.)
    radius=float(geometry['axis'][0])
    base=bernstein(pn[valid])
    design=np.hstack([base*(rr[valid,None]/radius)**2,base])
    target=jr[valid]/scale
    # Smooth flux functions; non-negative pressure gradient, unrestricted FF'.
    second=np.diff(np.eye(5),n=2,axis=0)
    regular=np.zeros((6,10));regular[:3,:5]=second;regular[3:,5:]=second
    mat=np.vstack([design,.01*np.sqrt(len(target))*regular])
    values=np.r_[target,np.zeros(6)]
    fitted=lsq_linear(mat,values,bounds=(np.r_[np.zeros(5),np.full(5,-np.inf)],np.full(10,np.inf)),
                      tol=1e-8,max_iter=100)
    error=float(np.linalg.norm(design@fitted.x-target)/max(np.linalg.norm(target),1e-10))
    return {'pprime':fitted.x[:5]*scale/radius**2,'ffprime_over_mu0':fitted.x[5:]*scale,
            'flux_span':span,'relative_residual':error,'ampere_scale':ampere_scale,
            'n_cells':int(valid.sum())}


def pressure_and_f2(profiles,pn,fvac):
    integral=integrated_basis(pn)*profiles['flux_span']
    pressure=integral@profiles['pprime']
    f2=fvac*fvac+2*MU0*(integral@profiles['ffprime_over_mu0'])
    return pressure,f2
