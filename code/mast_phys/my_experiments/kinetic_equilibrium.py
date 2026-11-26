"""Input-only, low-dimensional free-boundary equilibrium reconstruction.

The unknown is a compact toroidal current distribution, never an EFIT label. Its integral is
the observed Ip. Coil Green functions determine the vacuum field. The solver minimizes flux
variation along nested D-shaped surfaces, with Thomson temperature locating the hot plasma.
All learned priors come from DIII-D. MAST target columns are excluded by the IO path.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_fixes import align_mast_thomson_core
from my_experiments.fusion_pipeline import (CACHE_ROOT, MAST_COILS, _interp, _interp_profile,
                                           filament_psi, green_basis)


def read_inputs(path):
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    keep = [k for k in names if k.startswith(('coil_', 'thomson_')) or
            (k.startswith('magnetics_') and 'dsep' not in k) or
            k in ('efit_times', 'efit_grid_R', 'efit_grid_Z')]
    row = pd.read_parquet(path, columns=keep).iloc[0].copy()
    # v14: the vessel's induced current, derived from this row's own coil and plasma currents
    from my_experiments.vessel_v14 import add_column
    return add_column(row)


def kinetic_geometry(row, times, exponent):
    """Fit Te=A[1-((R-Raxis)/a)^2]^nu to horizontal Thomson INPUTS.

    nu is a DIII-D temperature/normalized-flux fit. Soft-L1 suppresses bad channels. Missing
    channels are interpolated spatially only for initialization; the fit uses valid observations.
    """
    r, te, _ = align_mast_thomson_core(row)
    te = _interp_profile(row['thomson_core_times'], te, times)
    result = []
    for v in te:
        ok = np.isfinite(v) & (v > 0) & np.isfinite(r)
        if ok.sum() < 12:
            result.append([float(np.mean(r)), float(np.ptp(r)/2), 0.0])
            continue
        rr, vv = r[ok], v[ok].astype(float)
        smooth = gaussian_filter1d(np.interp(r, rr, vv), 2)
        peak = max(float(np.quantile(smooth, .97)), 1.0)
        hot = smooth > .75*peak
        center = float(np.average(r[hot], weights=smooth[hot])) if hot.any() else float(np.mean(r))
        support = r[smooth > .1*peak]
        width = max(float(np.ptp(support)/2), .15) if len(support) else float(np.ptp(r)/2)
        def residual(p):
            rho = (rr-p[0])/p[1]
            model = p[2]*np.maximum(1-rho*rho, 0)**exponent
            return (model-vv)/peak
        lower=np.array([r.min()+.05,.1,.3*peak],float)
        upper=np.array([r.max()-.05,np.ptp(r),3*peak],float)
        initial=np.clip([center,width,peak],lower+1e-7*(upper-lower),upper-1e-7*(upper-lower))
        fit = least_squares(residual,initial,bounds=(lower,upper),
                            loss='soft_l1',f_scale=.1,max_nfev=60)
        error = float(np.mean(np.minimum(residual(fit.x)**2, 1)))
        result.append([fit.x[0],fit.x[1],error])
    return np.asarray(result,dtype=np.float32)


class KineticEquilibrium:
    def __init__(self,row,prior,device='cuda'):
        self.device=torch.device(device)
        self.prior=prior
        self.r=np.asarray(row['efit_grid_R'],np.float32)
        self.z=np.asarray(row['efit_grid_Z'],np.float32)
        self.sr,self.sz=self.r[::2],self.z[::2]
        rr,zz=np.meshgrid(self.sr,self.sz)
        self.rr,self.zz=self.t(rr.ravel()),self.t(zz.ravel())
        maps=np.stack([filament_psi(self.r,self.z,float(a),float(b))
                       for a,b in zip(rr.ravel(),zz.ravel())])
        gs=maps[:,::2,::2].reshape(len(rr.ravel()),-1)
        rad=np.sqrt((self.sr[1]-self.sr[0])*(self.sz[1]-self.sz[0])/np.pi)
        np.fill_diagonal(gs,2e-7*rr.ravel()*(np.log(8*rr.ravel()/rad)-1.75))
        self.gs=self.t(gs)
        fine_r,fine_z=np.meshgrid(self.r,self.z)
        self.fine_rr,self.fine_zz=self.t(fine_r.ravel()),self.t(fine_z.ravel())
        fine=np.stack([filament_psi(self.r,self.z,float(a),float(b)).ravel()
                       for a,b in zip(fine_r.ravel(),fine_z.ravel())])
        fine_rad=np.sqrt((self.r[1]-self.r[0])*(self.z[1]-self.z[0])/np.pi)
        np.fill_diagonal(fine,2e-7*fine_r.ravel()*(np.log(8*fine_r.ravel()/fine_rad)-1.75))
        self.go=self.t(fine)
        _,basis=green_basis(row,MAST_COILS)
        basis=basis.copy()
        # P-coils are total kA-turn; the solenoid is a kA feed through individual turns.
        # Official signal dictionary: README.md, MAST columns and conductor geometry.
        nsol=np.count_nonzero(np.asarray(row['coil_input_column'])=='magnetics_sol_current')
        basis[MAST_COILS.index('magnetics_sol_current')]*=nsol
        self.basis=basis
        from my_experiments.released_geometry import grid
        _, _, env = grid(row)
        self.mask=self.t(env[::2,::2].ravel().astype(np.float32))
        self.fine_mask=self.t(env.ravel().astype(np.float32))
        self.theta=self.t(np.arange(40)*2*np.pi/40)[None,None,:]
        self.rho=self.t([.3,.55,.8,.98])[None,:,None]

    def t(self,x):
        return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def currents(self,row,times):
        return np.column_stack([_interp(row['magnetics_time'],row[k],times) for k in MAST_COILS])

    def source(self,p,fine=False):
        rr,zz,mask=(self.fine_rr,self.fine_zz,self.fine_mask) if fine else (self.rr,self.zz,self.mask)
        rc,zc,a,b,delta,shift=[p[:,i,None] for i in range(6)]
        zq=(zz[None]-zc)/b
        rq=(rr[None]-rc+delta*a*zq*zq)/a
        for _ in range(3):
            rho2=rq*rq+zq*zq
            rq=(rr[None]-rc+delta*a*zq*zq-shift*(1-rho2))/a
        rho2=rq*rq+zq*zq
        alpha,beta,gamma=(self.prior[k] for k in ('alpha','beta','gamma'))
        q=(1-rho2.clamp_min(1e-8)**alpha).clamp_min(1e-8)**beta
        q=torch.where(rho2<1,q,torch.zeros_like(q))
        q=q*((1-gamma)*rr[None]/rc+gamma*rc/rr[None])*mask
        return q/q.sum(1,keepdim=True).clamp_min(1e-20)

    def surfaces(self,p):
        rc,zc,a,b,delta,shift=[p[:,i,None,None] for i in range(6)]
        r=rc+shift*(1-self.rho**2)+a*self.rho*torch.cos(self.theta)-delta*a*(self.rho*torch.sin(self.theta))**2
        z=zc+b*self.rho*torch.sin(self.theta)
        return torch.stack([2*(r-self.r[0])/np.ptp(self.r)-1,
                            2*(z-self.z[0])/np.ptp(self.z)-1],dim=-1)

    def optimize(self,row,times,steps=180):
        kin=kinetic_geometry(row,times,self.prior['te_exponent'])
        curr=self.currents(row,times)
        params,hist,loss=self.optimize_prepared(kin,curr,steps)
        return params,kin,hist,loss

    def optimize_prepared(self,kin,curr,steps=180):
        ip=self.t(curr[:,-1]*1000)
        vac=np.einsum('tc,czr->tzr',curr*1000,self.basis,optimize=True)
        vs=self.t(vac[:,::2,::2].reshape(len(curr),-1))
        kt=self.t(kin)
        ar=kt[:,1].clamp(.22,.8)
        elong=float(self.prior['current_elongation'])
        initial=torch.stack([kt[:,0]-.10*ar,torch.zeros_like(ar),ar,
                             (elong*ar).clamp(.35,1.25),torch.full_like(ar,.25),.10*ar],1)
        lower=self.t([.45,-.30,.20,.35,-.15,-.10])
        upper=self.t([1.3,.30,.85,1.5,.75,.40])
        span=upper-lower
        initial=torch.maximum(lower+.001,torch.minimum(upper-.001,initial))
        u=torch.logit((initial-lower)/span).detach().requires_grad_(True)
        opt=torch.optim.Adam([u],lr=.04)
        best=torch.full((len(curr),),float('inf'),device=self.device)
        saved=initial.clone()
        history=[]
        for step in range(steps):
            opt.zero_grad()
            p=lower+span*torch.sigmoid(u)
            w=self.source(p)
            psi=vs+ip[:,None]*(w@self.gs)
            surf=torch.nn.functional.grid_sample(psi.reshape(-1,1,len(self.sz),len(self.sr)),
                    self.surfaces(p),align_corners=True,mode='bilinear')[:,0]
            scale=(2e-7*ip.abs()*p[:,0]).clamp_min(.01)
            equal_flux=((surf-surf.mean(2,keepdim=True))/scale[:,None,None]).square().mean((1,2))
            anchor=((p[:,0]+p[:,5]-kt[:,0])/ar).square()
            size=((p[:,2]-ar)/ar).square()
            axis_support=torch.relu((p[:,1].abs()-.10)/ar).square()
            shape=((p[:,3]/p[:,2]-elong)/elong).square()
            loss=equal_flux+.03*anchor+.015*size+.005*shape+.01*axis_support
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_([u],1)
            opt.step()
            with torch.no_grad():
                improved=loss<best
                best=torch.minimum(best,loss)
                saved[improved]=p[improved]
            if step in (0,steps//2,steps-1):
                history.append({'step':step,'loss':float(loss.mean()),'flux_residual':float(equal_flux.mean())})
        if not torch.isfinite(saved).all() or not torch.isfinite(best).all():
            raise RuntimeError('Non-finite equilibrium optimization')
        return saved.cpu().numpy(),history,best.cpu().numpy()

    def reconstruct(self,params,curr):
        with torch.no_grad():
            w=self.source(self.t(params),fine=True)
            flux=(self.t(curr*1000)@self.t(self.basis.reshape(len(MAST_COILS),-1))
                  +self.t(curr[:,-1]*1000)[:,None]*(w@self.go))
        return flux.cpu().numpy().reshape(-1,65,65)

    def predict(self,row,steps=180,stride=5):
        times=np.asarray(row['efit_times'],np.float64)
        ids=np.unique(np.r_[np.arange(0,len(times),stride),len(times)-1])
        params,kin,hist,loss=self.optimize(row,times[ids],steps)
        pp=np.column_stack([np.interp(times,times[ids],params[:,j]) for j in range(6)])
        curr=self.currents(row,times)
        flux=self.reconstruct(pp,curr)
        return flux,{'parameters':params.tolist(),'kinetic':kin.tolist(),
                     'history':hist,'residual':loss.tolist()}
