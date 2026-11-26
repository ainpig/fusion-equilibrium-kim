"""Global radial-force pressure moment for an axisymmetric equilibrium.

Let W=p+(F^2-Fvac^2)/(2 mu0 R^2). The radial force equation gives
    d_R(R^2 W) = R J_phi d_R(psi) + 2 R p.
On a closed boundary p=0 and F=Fvac, so W=0. Hence
    <p>_V = - integral(R J_phi psi_R dR dZ)/(2 integral(R dR dZ)).
This is an integral readout, not a MAST-fitted pressure calibration.
"""
import numpy as np
from matplotlib.path import Path as Polygon
from scipy.interpolate import RectBivariateSpline
MU0=4e-7*np.pi


def discrete_jr(phi,r,z):
    f=np.asarray(phi,float);dr=float(r[1]-r[0]);dz=float(z[1]-z[0]);out=np.zeros_like(f)
    out[1:-1,1:-1]=-((f[1:-1,2:]-2*f[1:-1,1:-1]+f[1:-1,:-2])/dr**2
       -(f[1:-1,2:]-f[1:-1,:-2])/(2*dr*r[None,1:-1])
       +(f[2:,1:-1]-2*f[1:-1,1:-1]+f[:-2,1:-1])/dz**2)/MU0
    return out


def pressure_moment(phi,r,z,geometry,vacuum=None):
    rr,zz=np.meshgrid(r,z);dr=float(r[1]-r[0]);dz=float(z[1]-z[0]);poly=Polygon(geometry['boundary'])
    weight=np.zeros_like(rr,float)
    for a in (-.375,-.125,.125,.375):
        for b in (-.375,-.125,.125,.375):
            weight+=poly.contains_points(np.column_stack([(rr+a*dr).ravel(),(zz+b*dz).ravel()])).reshape(rr.shape)/16
    jr=discrete_jr(np.asarray(phi,float)-(0 if vacuum is None else vacuum),r,z)
    sp=RectBivariateSpline(z,r,np.asarray(phi,float))
    derivative=sp(z,r,dy=1)
    moment=-.5*np.sum(jr*derivative*weight)
    volume=np.sum(rr*weight)
    return float(moment/volume)
