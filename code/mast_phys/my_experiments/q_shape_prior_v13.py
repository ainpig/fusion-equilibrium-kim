"""Derivative-free safety-factor shape prior from Ampere's law.

For locally homothetic surfaces, |grad psi| is proportional to 1/h, with
h=(position-axis).normal. Eliminating the field amplitude using Ip gives the
shape integral below. Its calibration/uncertainty are obtained on DIII-D only.
"""
import numpy as np

MU0=4e-7*np.pi


def shape_q(contour,axis_r,axis_z,fvac,ip_amp):
    c=np.asarray(contour,float)
    if len(c)>1 and np.linalg.norm(c[0]-c[-1])<1e-9:c=c[:-1]
    if len(c)<8 or abs(ip_amp)<1 or abs(fvac)<1e-9:return np.nan
    nxt=np.roll(c,-1,axis=0);delta=nxt-c;mid=(c+nxt)/2
    ds=np.linalg.norm(delta,axis=1)
    area=np.sum(c[:,0]*nxt[:,1]-c[:,1]*nxt[:,0])
    orientation=1 if area>=0 else -1
    h=orientation*((mid[:,0]-axis_r)*delta[:,1]-(mid[:,1]-axis_z)*delta[:,0])/np.maximum(ds,1e-12)
    if np.any(h<1e-5) or np.any(mid[:,0]<=0):return np.nan
    a=np.sum(ds/(mid[:,0]*h));b=np.sum(h*ds/mid[:,0])
    return float(abs(fvac)/(2*np.pi*MU0*abs(ip_amp))*a*b)


def ellipse_q(contour,axis_r,axis_z,fvac,ip_amp):
    """Fallback shape prior when a contour is not star-shaped about its axis."""
    c=np.asarray(contour,float);a=np.ptp(c[:,0])/2;b=np.ptp(c[:,1])/2
    rc=(c[:,0].max()+c[:,0].min())/2;zc=(c[:,1].max()+c[:,1].min())/2
    theta=np.linspace(0,2*np.pi,512,endpoint=False)
    circle=np.column_stack([rc+a*np.cos(theta),zc+b*np.sin(theta)])
    return shape_q(circle,rc,zc,fvac,ip_amp)
