"""Pressure readout with spatial block-jackknife uncertainty.

The posterior prior is measured on DIII-D only. Eight poloidal sectors expose
unresolved pressure/FF separation in an approximate equilibrium. No MAST target
or scalar coefficient is fitted by this module.
"""
import numpy as np
from scipy.optimize import lsq_linear
from matplotlib.path import Path as Polygon
from my_experiments.gs_profiles_v13 import bernstein,integrated_basis,MU0
from my_experiments.virial_pressure_v14 import discrete_jr


def pressure_uncertainty(phi,r,z,geometry,ip_amp,fvac,return_detail=False):
    rr,zz=np.meshgrid(r,z);pa=geometry['axis'][2];span=pa-geometry['xpoint'][2];ra,za=geometry['axis'][:2]
    pn=(pa-phi)/span;inside=Polygon(geometry['boundary']).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(rr.shape)
    valid=inside&(pn>.015)&(pn<.985);valid[[0,-1],:]=False;valid[:,[0,-1]]=False
    jr=discrete_jr(phi,r,z);current=np.sum(jr[inside]/rr[inside])*float((r[1]-r[0])*(z[1]-z[0]))
    ampere=abs(ip_amp)/current if current>0 else 1.
    if .75<ampere<1.25:jr*=ampere
    scale=max(float(np.sqrt(np.mean(jr[valid]**2))),1.)
    base=bernstein(pn[valid]);x=np.hstack([base*(rr[valid,None]/ra)**2,base]);y=jr[valid]/scale
    regular=np.zeros((6,10));regular[:3,:5]=np.diff(np.eye(5),n=2,axis=0);regular[3:,5:]=regular[:3,:5]
    vw=geometry['R'][None,:]*geometry['inside'];pf=(pa-geometry['flux'])/span
    pav=(integrated_basis(pf)*vw[:,:,None]).sum((0,1))/vw.sum()*span*scale/ra**2
    c=geometry['boundary'];a=np.ptp(c[:,0])/2;rg=(c[:,0].max()+c[:,0].min())/2
    norm=2*MU0*100*a*rg/(fvac*abs(ip_amp)/1e6)
    angle=np.arctan2((zz[valid]-za)/max(np.ptp(c[:,1])/2,.1),(rr[valid]-ra)/max(a,.1))
    sectors=np.minimum(((angle+np.pi)/(2*np.pi)*8).astype(int),7)
    values=[]
    for k in range(8):
        keep=sectors!=k
        if keep.sum()<60:continue
        fitted=lsq_linear(np.vstack([x[keep],.01*np.sqrt(keep.sum())*regular]),np.r_[y[keep],np.zeros(6)],
                          bounds=(np.r_[np.zeros(5),np.full(5,-np.inf)],np.full(10,np.inf)),tol=1e-8,max_iter=100)
        values.append(float(pav@fitted.x[:5]*norm))
    sigma=float(np.sqrt(max(len(values)-1,0)*np.var(values))) if len(values)>2 else np.inf
    if return_detail:
        matrix=np.vstack([x,.01*np.sqrt(len(y))*regular]);target=np.r_[y,np.zeros(6)]
        fitted=lsq_linear(matrix,target,bounds=(np.r_[np.zeros(5),np.full(5,-np.inf)],np.full(10,np.inf)),tol=1e-8,max_iter=100)
        residual=float(np.linalg.norm(x@fitted.x-y));ynorm=float(np.linalg.norm(y))
        functional=np.r_[pav*norm,np.zeros(5)]
        sensitivity=x@np.linalg.solve(matrix.T@matrix+np.eye(10)*1e-12,functional)
        detail={'residual_norm':residual,'target_norm':ynorm,'gs_residual':residual/max(ynorm,1e-12),
                'functional_sensitivity_norm':float(np.linalg.norm(sensitivity)),
                'beta_full_fit':float(functional@fitted.x)}
        return sigma,values,detail
    return sigma,values


def posterior(beta,sigma,mean,variance):
    weight=variance/(variance+np.asarray(sigma)**2)
    return weight*np.asarray(beta)+(1-weight)*mean,weight


def smooth_readout(times,values,sigma):
    t=np.asarray(times,float);x=np.asarray(values,float);s=np.asarray(sigma,float)
    width=max(12.,2*np.median(np.diff(t))) if len(t)>1 else 12.
    kernel=np.exp(-.5*((t[:,None]-t[None,:])/width)**2)
    kernel[abs(t[:,None]-t[None,:])>3*width]=0
    weights=kernel/np.maximum(s[None,:]**2,.05**2)
    return (weights@x)/weights.sum(1)
