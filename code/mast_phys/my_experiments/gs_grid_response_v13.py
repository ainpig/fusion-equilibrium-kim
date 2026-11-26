"""Differentiable free-space plasma response via a discrete GS interior solve.

Exact Green boundary data plus sine separation in Z reduce a dense 4225-square
response to small radial solves. The emitted grid satisfies the same five-point
GS identity used for scalar reconstruction.
"""
from __future__ import annotations
import numpy as np
import torch

MU0=4e-7*np.pi


class GSGridResponse:
    def __init__(self,r,z,green,device):
        self.nr=len(r);self.nz=len(z);self.device=device
        self.dr=float(r[1]-r[0]);self.dz=float(z[1]-z[0])
        n=self.nr-2;m=self.nz-2
        ri=np.asarray(r[1:-1],float)
        lower=1/self.dr**2+1/(2*self.dr*ri)
        upper=1/self.dr**2-1/(2*self.dr*ri)
        radial=np.diag(np.full(n,-2/self.dr**2))+np.diag(lower[1:],-1)+np.diag(upper[:-1],1)
        eigen=-4*np.sin(np.arange(1,m+1)*np.pi/(2*(m+1)))**2/self.dz**2
        inverse=np.stack([np.linalg.inv(radial+lam*np.eye(n)) for lam in eigen])
        sine=np.sqrt(2/(m+1))*np.sin(np.arange(1,m+1)[:,None]*np.arange(1,m+1)[None,:]*np.pi/(m+1))
        self.inverse=self.t(inverse);self.sine=self.t(sine);self.r=self.t(ri)
        self.left=float(lower[0]);self.right=float(upper[-1])
        border=np.zeros((self.nz,self.nr),bool);border[[0,-1],:]=True;border[:,[0,-1]]=True
        self.indices=torch.as_tensor(np.flatnonzero(border),device=device)
        self.boundary=green[:,self.indices] if green is not None else None

    def t(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def __call__(self,amps,boundary_values=None):
        b=amps.shape[0]
        values=amps@self.boundary if boundary_values is None else boundary_values
        full=torch.zeros((b,self.nz*self.nr),dtype=amps.dtype,device=amps.device)
        full[:,self.indices]=values
        full=full.reshape(b,self.nz,self.nr)
        density=amps.reshape(b,self.nz,self.nr)[:,1:-1,1:-1]/(self.dr*self.dz)
        rhs=-MU0*self.r[None,None,:]*density
        rhs=rhs.clone()
        rhs[:,0,:]-=full[:,0,1:-1]/self.dz**2
        rhs[:,-1,:]-=full[:,-1,1:-1]/self.dz**2
        rhs[:,:,0]-=self.left*full[:,1:-1,0]
        rhs[:,:,-1]-=self.right*full[:,1:-1,-1]
        transformed=torch.einsum('zk,bzr->bkr',self.sine,rhs)
        solved=torch.einsum('kij,bkj->bki',self.inverse,transformed)
        interior=torch.einsum('zk,bkr->bzr',self.sine,solved)
        full=full.clone();full[:,1:-1,1:-1]=interior
        return full
