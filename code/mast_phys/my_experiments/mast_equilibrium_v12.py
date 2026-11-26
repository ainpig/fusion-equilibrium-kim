"""MAST input-only reconstruction with circuit-aware SI magnetic fields.

Shared physical constants come from machine engineering descriptions, not MAST
equilibrium labels. Each solve estimates only the supplied discharge's state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import ellipk, ellipe
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import isotonic_regression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'fusion_scoring'))

from my_experiments.fusion_pipeline import MAST_COILS, _interp_profile
from my_experiments.kinetic_equilibrium import KineticEquilibrium,kinetic_geometry
from data_fixes import align_mast_thomson_core

MU0 = 4e-7*np.pi
E_CHARGE = 1.602176634e-19
# Battaglia et al., UKAEA-CCFE-PR(19)34, section 3: MAST has two
# parallel solenoid circuits. The measured feed current is twice turn current.
# The challenge's data page leaves the solenoid scale open ("magnetics_sol_current reads exactly 2x
# the FAIR-MAST level-2 pf_active solenoid current, and we have not resolved which convention is
# correct"), so v14 can set the divisor per run (fusion_eq.mast --sol-divisor); 2 is v3-1 .. v13.
import os as _os
SOLENOID_PARALLEL_CIRCUITS = float(_os.environ.get('FUSION_MAST_SOL_DIVISOR', '2'))
# MAST's 24 centre-rod conductors, each carrying the TF circuit current (published hardware, kept in one place).
from my_experiments.mast_constants import MAST_TF_TURNS as TF_TURNS


def green_points(ro, zo, rs, zs):
    """Circular-loop psi [Wb/rad/A], broadcast over physical coordinates."""
    m = np.clip(4*ro*rs/((ro+rs)**2+(zo-zs)**2), 1e-12, 1-1e-14)
    return 4e-7*np.sqrt(ro*rs/m)*((1-m/2)*ellipk(m)-ellipe(m))


def correct_cell_green(matrix, r, z):
    """Integrate nearby rectangular sources, including their finite self field.

    The old diagonal-only circular self term leaves grid-dependent derivatives.
    Eight-point Gauss integration per dimension fixes all source/observer pairs
    within two cells. Distant pairs retain the centroid approximation.
    """
    nr, nz = len(r), len(z)
    rr, zz = np.meshgrid(r.astype(float), z.astype(float))
    indices = np.arange(nr*nz).reshape(nz,nr)
    dr, dz = float(r[1]-r[0]), float(z[1]-z[0])
    nodes, weights = leggauss(8)
    out = np.array(matrix, dtype=np.float32, copy=True)
    for iz in range(-2,3):
        for ir in range(-2,3):
            src = indices[max(0,-iz):min(nz,nz-iz),max(0,-ir):min(nr,nr-ir)].ravel()
            obs = indices[max(0,iz):min(nz,nz+iz),max(0,ir):min(nr,nr+ir)].ravel()
            v = np.zeros(len(src))
            for a,wa in zip(nodes,weights):
                for b,wb in zip(nodes,weights):
                    v += wa*wb/4*green_points(rr.ravel()[obs],zz.ravel()[obs],
                                             rr.ravel()[src]+a*dr/2,zz.ravel()[src]+b*dz/2)
            out[src,obs] = v
    return out


def circuit_basis(row,r,z):
    """Finite conductor rectangles, PF ampere-turn totals and solenoid feed A."""
    rr,zz=np.meshgrid(r.astype(float),z.astype(float))
    names=np.asarray(row['coil_input_column'])
    nodes,weights=leggauss(3)
    result=[]
    for name in MAST_COILS:
        if name=='vessel_current_v14':
            # v14: the vessel shell (vessel_v14.py), flux per ampere of total shell current
            from my_experiments.vessel_v14 import basis as vessel_basis
            result.append(vessel_basis(row,r,z))
            continue
        ids=np.flatnonzero(names==name)
        field=np.zeros_like(rr)
        for j in ids:
            if row['coil_angle1'][j]!=0 or row['coil_angle2'][j]!=0:
                raise ValueError('MAST circuit integrator requires rectangular conductors')
            for a,wa in zip(nodes,weights):
                for b,wb in zip(nodes,weights):
                    field+=wa*wb/4*green_points(rr,zz,
                            float(row['coil_R'][j])+a*float(row['coil_width'][j])/2,
                            float(row['coil_Z'][j])+b*float(row['coil_height'][j])/2)
        if name=='magnetics_sol_current':
            field/=SOLENOID_PARALLEL_CIRCUITS
        elif len(ids):
            field/=len(ids)
        result.append(field)
    return np.asarray(result,np.float32)


class MastEquilibriumV12(KineticEquilibrium):
    def __init__(self, row, prior, device='cuda'):
        super().__init__(row, prior, device=device)
        self.basis=circuit_basis(row,self.r,self.z)
        self.go = self.t(correct_cell_green(self.go.cpu().numpy(),self.r,self.z))
        # Optimize and reconstruct with identical source and output grids.
        self.sr,self.sz = self.r,self.z
        self.rr,self.zz,self.mask = self.fine_rr,self.fine_zz,self.fine_mask
        self.gs = self.go
        self.theta = self.t(np.arange(64)*2*np.pi/64)[None,None,:]

    def optimize_prepared(self, kin, curr, steps=360):
        ip=self.t(curr[:,-1]*1000)
        vac=np.einsum('tc,czr->tzr',curr*1000,self.basis,optimize=True)
        vs=self.t(vac.reshape(len(curr),-1))
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
                    self.surfaces(p),align_corners=True,mode='bicubic')[:,0]
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


def thomson_constraints(row,times):
    """Pair equal-temperature points on the two sides of the observed hot core.

    Isothermal surfaces are magnetic surfaces; no prescribed Te(psi) exponent is
    needed for these constraints. Monotone denoising is local to each observation.
    """
    radius,te,ne=align_mast_thomson_core(row)
    temperature=_interp_profile(row['thomson_core_times'],te,times)
    pressure=_interp_profile(row['thomson_core_times'],2*E_CHARGE*te*ne,times)
    levels=np.linspace(.25,.85,9)
    pairs=np.empty((len(times),len(levels),2),np.float32)
    valid=np.zeros((len(times),len(levels)),np.float32)
    paxis=np.full(len(times),np.nan,np.float32)
    for i,t in enumerate(temperature):
        ok=np.isfinite(t)&(t>0)&np.isfinite(radius)
        pairs[i]=np.column_stack([np.full(len(levels),.5),np.full(len(levels),1.2)])
        if ok.sum()<12:
            continue
        smooth=gaussian_filter1d(np.interp(radius,radius[ok],t[ok]),2)
        peak=int(np.argmax(smooth))
        pgood=pressure[i][np.isfinite(pressure[i])&(pressure[i]>0)]
        if len(pgood):
            paxis[i]=np.quantile(pgood,.95)
        if peak<3 or peak>len(radius)-4:
            continue
        left=isotonic_regression(smooth[:peak+1],increasing=True).x
        right=isotonic_regression(smooth[peak:],increasing=False).x
        for j,fraction in enumerate(levels):
            value=fraction*smooth[peak]
            if value<=max(left[0],right[-1]) or value>=min(left[-1],right[0]):
                continue
            pairs[i,j,0]=np.interp(value,left,radius[:peak+1])
            pairs[i,j,1]=np.interp(value,right[::-1],radius[peak:][::-1])
            valid[i,j]=1
    return pairs,valid,paxis


def shaped_radius_squared(rr,zz,p):
    """Exact, stable inverse of the nested shifted-D coordinate map.

    A fixed-point inverse can overflow outside the source (large |Z|), even
    though that region carries zero current. The rationalized quadratic root
    remains finite and reduces continuously to the unshifted coordinate.
    """
    rc,zc,a,b,delta,shift=[p[:,i,None] for i in range(6)]
    zq=(zz[None]-zc)/b
    x=(rr[None]-rc-shift+(delta*a+shift)*zq*zq)/a
    discriminant=1-4*(shift/a)*x
    rq=2*x/(1+torch.sqrt(discriminant.clamp_min(1e-8)))
    return rq*rq+zq*zq,discriminant>=0


class ThomsonEquilibriumV12(MastEquilibriumV12):
    """Infer current peaking/pressure balance from each discharge's Thomson data."""
    def __init__(self,row,prior,device='cuda'):
        super().__init__(row,prior,device)
        # Interior flux surfaces approximate the current-bearing plasma. Near a
        # separatrix an analytic D contour cannot represent the X-point cusp.
        self.rho=self.t(np.linspace(.1,.98,10))[None,:,None]

    def unnormalized_source(self,p,fine=False):
        rr,zz,mask=(self.fine_rr,self.fine_zz,self.fine_mask) if fine else (self.rr,self.zz,self.mask)
        rc,zc,a,b,delta,shift,beta,gamma=[p[:,i,None] for i in range(8)]
        rho2,resolved=shaped_radius_squared(rr,zz,p)
        q=(1-rho2.clamp_min(1e-8)**self.prior['alpha']).clamp_min(1e-8)**beta
        q=torch.where((rho2<1)&resolved,q,torch.zeros_like(q))
        return q*((1-gamma)*rr[None]/rc+gamma*rc/rr[None])*mask

    def source(self,p,fine=False):
        q=self.unnormalized_source(p,fine)
        return q/q.sum(1,keepdim=True).clamp_min(1e-20)

    def surfaces(self,p):
        return super().surfaces(p[:,:6])

    def optimize(self,row,times,steps=600):
        kin=kinetic_geometry(row,times,self.prior['te_exponent'])
        params,hist,loss=self.optimize_prepared(kin,self.currents(row,times),steps,
                                               thomson_constraints(row,times))
        return params,kin,hist,loss

    def predict(self,row,steps=600,stride=5):
        times=np.asarray(row['efit_times'],float)
        if not len(times):
            raise ValueError('Empty EFIT time axis')
        ids=np.unique(np.r_[np.arange(0,len(times),stride),len(times)-1])
        params,kin,hist,loss=self.optimize(row,times[ids],steps)
        pp=np.column_stack([np.interp(times,times[ids],params[:,j]) for j in range(params.shape[1])])
        return self.reconstruct(pp,self.currents(row,times)),{
            'parameters':params.tolist(),'kinetic':kin.tolist(),'history':hist,'residual':loss.tolist()}

    def optimize_prepared(self,kin,curr,steps=360,constraints=None):
        if constraints is None:
            raise ValueError('Thomson pairs and pressure required')
        pairs,valid,pressure=constraints
        ip=self.t(curr[:,-1]*1000)
        vac=np.einsum('tc,czr->tzr',curr*1000,self.basis,optimize=True)
        vs=self.t(vac.reshape(len(curr),-1))
        kt=self.t(kin);ar=kt[:,1].clamp(.22,.8)
        elong=float(self.prior['current_elongation'])
        initial=torch.stack([kt[:,0]-.1*ar,torch.zeros_like(ar),ar,
            (elong*ar).clamp(.35,1.25),torch.full_like(ar,.25),torch.full_like(ar,.1),
            torch.full_like(ar,self.prior['beta']),torch.full_like(ar,.75)],1)
        # The optimized shift is a fraction of a. |shift/a|<1/2 prevents
        # nested surfaces from folding or crossing on the outboard side.
        lower=self.t([.45,-.30,.20,.35,-.15,-.45,.15,.01])
        upper=self.t([1.3,.30,.85,1.5,.75,.45,3.,.99])
        span=upper-lower
        initial=torch.maximum(lower+.001,torch.minimum(upper-.001,initial))
        u=torch.logit((initial-lower)/span).detach().requires_grad_(True)
        opt=torch.optim.Adam([u],lr=.035)
        best=torch.full((len(curr),),float('inf'),device=self.device)
        saved=initial.clone();saved[:,5]*=saved[:,2];history=[]
        pair_r=self.t(pairs)
        pair_grid=torch.stack([2*(pair_r-self.r[0])/np.ptp(self.r)-1,
                              torch.zeros_like(pair_r)],dim=-1)
        valid_t=self.t(valid)
        has_pressure=np.isfinite(pressure)&(pressure>0)
        pressure_weight=self.t(has_pressure.astype(float))
        observed_p=self.t(np.where(has_pressure,pressure,1.)).clamp_min(1.)
        darea=float((self.r[1]-self.r[0])*(self.z[1]-self.z[0]))
        rho=self.rho.reshape(1,-1)
        for step in range(steps):
            opt.zero_grad()
            raw=lower+span*torch.sigmoid(u)
            p=torch.cat([raw[:,:5],raw[:,5:6]*raw[:,2:3],raw[:,6:]],dim=1)
            unnormalized=self.unnormalized_source(p)
            total=unnormalized.sum(1).clamp_min(1e-20)
            w=unnormalized/total[:,None]
            psi=(vs+ip[:,None]*(w@self.gs)).reshape(-1,1,65,65)
            surf=torch.nn.functional.grid_sample(psi,self.surfaces(p),align_corners=True,mode='bicubic')[:,0]
            scale=(2e-7*ip.abs()*p[:,0]).clamp_min(.01)
            shape=(1-rho**(2*self.prior['alpha'])).clamp_min(1e-8)**p[:,6,None]
            # Current-weighted angular residual: impose GS flux-function
            # consistency where current flows, without imposing a smooth LCFS.
            sw=shape*rho
            variance=((surf-surf.mean(2,keepdim=True))/scale[:,None,None]).square().mean(2)
            eq=(variance*sw).sum(1)/sw.sum(1)
            pair_flux=torch.nn.functional.grid_sample(psi,pair_grid,align_corners=True,mode='bicubic')[:,0]
            paired=((pair_flux[:,:,0]-pair_flux[:,:,1])/scale[:,None]).square()
            thermal=(paired*valid_t).sum(1)/valid_t.sum(1).clamp_min(1)
            mean_flux=surf.mean(2)
            integral=(.5*(shape[:,:-1]+shape[:,1:])*(mean_flux[:,:-1]-mean_flux[:,1:])).sum(1)
            predicted_p=(ip/(total*darea)*(1-p[:,7])/p[:,0]*integral).clamp_min(1.)
            pressure_loss=torch.log(predicted_p/observed_p).square()*pressure_weight
            anchor=((p[:,0]+p[:,5]-kt[:,0])/ar).square()
            size=((p[:,2]-ar)/ar).square()
            weak_shape=((p[:,3]/p[:,2]-elong)/elong).square()
            missing_pressure_prior=(p[:,7]-self.prior['gamma']).square()*(1-pressure_weight)
            loss=(eq+.16*thermal+.0025*pressure_loss+.001*anchor+.0005*size
                  +.0002*weak_shape+.0002*missing_pressure_prior)
            if not torch.isfinite(loss).all():
                bad=torch.nonzero(~torch.isfinite(loss)).flatten().cpu().tolist()
                raise RuntimeError(f'Non-finite objective at step {step}, frames {bad}')
            loss.mean().backward()
            if not torch.isfinite(u.grad).all():
                bad=torch.nonzero(~torch.isfinite(u.grad).all(1)).flatten().cpu().tolist()
                raise RuntimeError(f'Non-finite gradient at step {step}, frames {bad}')
            torch.nn.utils.clip_grad_norm_([u],1)
            opt.step()
            with torch.no_grad():
                improved=loss<best;best=torch.minimum(best,loss);saved[improved]=p[improved]
            if step in (0,steps//2,steps-1):
                history.append({'step':step,'loss':float(loss.mean()),'flux_residual':float(eq.mean()),
                                'isotherm_residual':float(thermal.mean()),
                                'pressure_log_error':float(pressure_loss.mean())})
        if not torch.isfinite(saved).all() or not torch.isfinite(best).all():
            raise RuntimeError('Non-finite Thomson equilibrium optimization')
        return saved.cpu().numpy(),history,best.cpu().numpy()


def contour_q(psi, r, z, boundary, axis_flux, fvac, axis_index=None):
    """Safety factor from the 95% poloidal-flux contour and physical TF current."""
    from scipy.interpolate import RectBivariateSpline
    from contour import find_contours, is_closed, encloses_point, grid_to_physical
    spline = RectBivariateSpline(z,r,psi,kx=3,ky=3,s=0)
    bflux = np.median(spline.ev(boundary[:,1],boundary[:,0]))
    level = axis_flux + .95*(bflux-axis_flux)
    ia = np.unravel_index(np.argmax(psi),psi.shape) if axis_index is None else axis_index
    candidates=[c for c in find_contours(psi,level) if is_closed(c) and encloses_point(c,*ia)]
    if not candidates:
        return np.nan
    c = grid_to_physical(max(candidates,key=len),r,z)
    points = .5*(c+np.roll(c,-1,axis=0))
    dl = np.linalg.norm(c-np.roll(c,-1,axis=0),axis=1)
    grad = np.hypot(spline.ev(points[:,1],points[:,0],dx=1),
                    spline.ev(points[:,1],points[:,0],dy=1))
    return abs(fvac)/(2*np.pi)*np.sum(dl/np.maximum(points[:,0]*grad,1e-10))


def kinetic_scalars(row, times, psi, curr, r, z, mask):
    """Derive q95 and betaN from flux geometry, TF winding and measured n_e T_e.

    p_total=2 p_e assumes T_i=T_e and singly charged quasi-neutral plasma.
    No MAST scalar labels, regression coefficients or target-derived fallback.
    """
    from scipy.interpolate import RectBivariateSpline
    from matplotlib.path import Path as PolygonPath
    from lcfs import extract_lcfs
    from o_point import find_o_point
    tsr,te,ne=align_mast_thomson_core(row)
    pressure=_interp_profile(row['thomson_core_times'],2*E_CHARGE*te*ne,times)
    rr,zz=np.meshgrid(r,z)
    pts=np.column_stack([rr.ravel(),zz.ravel()])
    outq,outb,flags=[],[],[]
    for i,frame in enumerate(psi):
        fvac=MU0/(2*np.pi)*TF_TURNS*curr[i,MAST_COILS.index('magnetics_tf_current')]*1000
        ip=abs(float(curr[i,-1]))/1000
        ra,za,iz,ir,_=find_o_point(frame,r,z,mask.astype(bool))
        boundary=extract_lcfs(frame,r,z,'MAST',mask.astype(bool),mask.astype(float),
                              n_points=256,axis_sign=1)
        if boundary is None or abs(fvac)<1e-8 or ip<1e-5:
            outq.append(6.5);outb.append(1.);flags.append('missing_closed_surface')
            continue
        spline=RectBivariateSpline(z,r,frame)
        aflux=float(spline.ev(za,ra))
        bflux=float(np.median(spline.ev(boundary[:,1],boundary[:,0])))
        span=max(aflux-bflux,1e-8)
        q=contour_q(frame,r,z,boundary,aflux,fvac,(iz,ir))
        pn=(aflux-frame)/span
        tsn=(aflux-spline.ev(np.zeros_like(tsr),tsr))/span
        ok=np.isfinite(pressure[i]) & (pressure[i]>0) & (tsn>=-.05) & (tsn<=1.1)
        bins=np.linspace(0,1,17)
        xp,yp=[],[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            have=ok & (tsn>=lo) & (tsn<hi)
            if have.any():
                xp.append(float(np.median(tsn[have])))
                yp.append(float(np.median(pressure[i,have])))
        inside=PolygonPath(boundary).contains_points(pts).reshape(frame.shape)
        a=(boundary[:,0].max()-boundary[:,0].min())/2
        rg=(boundary[:,0].max()+boundary[:,0].min())/2
        bt=abs(fvac)/rg
        if len(xp)>=3 and inside.any():
            p=np.interp(pn,np.asarray(xp),np.asarray(yp))
            pavg=float((p*rr*inside).sum()/(rr*inside).sum())
            beta=2*MU0*pavg/(bt*bt)*100*a*bt/ip
            flag='ok'
        else:
            beta=1.;flag='missing_pressure'
        outq.append(float(q) if np.isfinite(q) else 6.5)
        outb.append(float(beta))
        flags.append(flag if np.isfinite(q) else 'missing_q95_contour')
    return np.asarray(outq,np.float32),np.asarray(outb,np.float32),flags
