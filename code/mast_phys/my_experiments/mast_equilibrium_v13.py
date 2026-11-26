"""Hardware-constrained equilibrium with vertical force balance and source priors.

MAST targets are not inputs. The pressure ratio is calibrated on DIII-D alone.
v3-1 source support is derived from released conductors, not an external wall.
"""
from __future__ import annotations
import numpy as np
import torch
from matplotlib.path import Path as Polygon

from my_experiments.mast_equilibrium_v12 import ThomsonEquilibriumV12,thomson_constraints,shaped_radius_squared
from my_experiments.released_geometry import source_fraction
from my_experiments.gs_grid_response_v13 import GSGridResponse
from my_experiments.mast_inputs_v13 import prepare_kinetic


class ThomsonEquilibriumV13(ThomsonEquilibriumV12):
    def __init__(self,row,prior,pressure_factor,device='cuda'):
        super().__init__(row,prior,device)
        # Keep cell-fraction integration, using only released conductor bounds.
        fraction = source_fraction(self.r, self.z, row)
        self.fine_mask=self.fine_mask*self.t(fraction.ravel())
        self.mask=self.fine_mask
        self.pressure_factor=float(pressure_factor)
        self.grid_response=GSGridResponse(self.r,self.z,self.go,self.device)
        self.plasma_plane=float(np.median(np.asarray(row['thomson_chord_Z'],float)))

    def optimize(self,row,times,steps=600):
        kin,constraints=prepare_kinetic(row,times,self.prior['te_exponent'])
        params,hist,loss=self.optimize_prepared(kin,self.currents(row,times),steps,constraints)
        return params,kin,hist,loss

    def reconstruct(self,params,curr):
        with torch.no_grad():
            w=self.source(self.t(params),fine=True)
            plasma=self.grid_response(w*self.t(curr[:,-1]*1000)[:,None])
            vac=np.einsum('tc,czr->tzr',curr*1000,self.basis,optimize=True)
            result=plasma+self.t(vac)
        return result.cpu().numpy()

    def unnormalized_source(self,p,fine=False):
        rr,zz,mask=(self.fine_rr,self.fine_zz,self.fine_mask) if fine else (self.rr,self.zz,self.mask)
        rho2,resolved=shaped_radius_squared(rr,zz,p)
        beta_p,beta_f,alpha,gamma=[p[:,i,None] for i in range(6,10)]
        core=(1-rho2.clamp_min(1e-8)**alpha).clamp_min(1e-8)
        pp=core**beta_p;ff=core**beta_f
        src=(1-gamma)*rr[None]/p[:,0,None]*pp+gamma*p[:,0,None]/rr[None]*ff
        return torch.where((rho2<1)&resolved,src,torch.zeros_like(src))*mask

    def vertical_projection(self,p,vac_z,initial_z,iterations=3):
        """Use a midplane current prior; the measured coil field stays asymmetric.

        A horizontal Thomson chord does not observe vertical displacement.
        Letting a free vertical coordinate fit it produced unstable branches.
        This is a geometry prior at the supplied diagnostic plane, not a fitted
        MAST axis label. The actual magnetic axis is still derived from total psi.
        """
        z=torch.full_like(initial_z,self.plasma_plane)
        return torch.cat([p[:,:1],z[:,None],p[:,2:]],1)

    def optimize_prepared(self,kin,curr,steps=600,constraints=None,initial_params=None):
        if constraints is None:
            raise ValueError('Thomson observations required')
        pairs,valid,pressure=constraints
        # v14 run settings (fusion_eq.mast sets them; unset = v3-1 .. v13): the total-to-electron
        # pressure ratio and weight of the pressure prior (a label-free scan found a stronger prior adds
        # limited frames, so v14 leaves both as they were) and the weight of the axis anchor (v14 0.3 on
        # the Thomson Te maximum: the frames the seed left inboard-limited had their axis 4.4 cm inboard
        # of it, the diverted frames sit on it; see mast_inputs_v15.prepare_kinetic)
        import os as _os
        factor=float(_os.environ.get('FUSION_MAST_PFACTOR') or self.pressure_factor)
        prior_weight=float(_os.environ.get('FUSION_MAST_PRIOR_W') or 1e-5)
        anchor_weight=float(_os.environ.get('FUSION_MAST_ANCHOR_W') or 1e-3)
        # The existing extractor returns 2*p_e; rescale by the source-only ratio.
        pressure=np.asarray(pressure)*factor/2
        ip=self.t(curr[:,-1]*1000)
        vac=np.einsum('tc,czr->tzr',curr*1000,self.basis,optimize=True)
        vs=self.t(vac.reshape(len(curr),-1))
        vac_z=self.t(np.gradient(vac,self.z.astype(float),axis=1,edge_order=2).reshape(len(curr),-1))
        kt=self.t(kin);ar=kt[:,1].clamp(.22,.8)
        elong=float(self.prior['current_elongation'])
        initial=torch.stack([kt[:,0]-.1*ar,torch.zeros_like(ar),ar,
            (elong*ar).clamp(.35,1.25),torch.full_like(ar,.25),torch.full_like(ar,.1),
            torch.full_like(ar,self.prior['beta']),torch.full_like(ar,self.prior['beta']),
            torch.full_like(ar,self.prior['alpha']),torch.full_like(ar,.75)],1)
        lower=self.t([.45,-.30,.20,.35,-.15,-.45,.15,.15,.5,.01])
        upper=self.t([1.3,.30,.85,1.5,.75,.45,4.,4.,4.,.99])
        span=upper-lower
        if initial_params is not None:
            initial=self.t(initial_params).clone()
            initial[:,5]=initial[:,5]/initial[:,2]
        initial=torch.maximum(lower+.001,torch.minimum(upper-.001,initial))
        u=torch.logit((initial-lower)/span).detach().requires_grad_(True)
        opt=torch.optim.Adam([u],lr=.035)
        best=torch.full((len(curr),),float('inf'),device=self.device)
        saved=initial.clone();saved[:,5]*=saved[:,2];history=[]
        pr=self.t(pairs)
        grid=torch.stack([2*(pr-self.r[0])/np.ptp(self.r)-1,torch.zeros_like(pr)],dim=-1)
        valid_t=self.t(valid)
        have=np.isfinite(pressure)&(pressure>0)
        pw=self.t(have.astype(float));observed=self.t(np.where(have,pressure,1.)).clamp_min(1.)
        darea=float((self.r[1]-self.r[0])*(self.z[1]-self.z[0]));rho=self.rho.reshape(1,-1)
        previous_z=torch.zeros_like(ar)
        for step in range(steps):
            rate=.035*(.01+.99*.5*(1+np.cos(np.pi*step/max(steps-1,1))))
            opt.param_groups[0]['lr']=rate
            opt.zero_grad()
            raw=lower+span*torch.sigmoid(u)
            p=torch.cat([raw[:,:5],raw[:,5:6]*raw[:,2:3],raw[:,6:]],1)
            p=self.vertical_projection(p,vac_z,previous_z,6 if step==0 else 3)
            previous_z=p[:,1].detach()
            source=self.unnormalized_source(p)
            total=source.sum(1).clamp_min(1e-20);w=source/total[:,None]
            psi=vs.reshape(-1,1,65,65)+self.grid_response(ip[:,None]*w)[:,None]
            surf=torch.nn.functional.grid_sample(psi,self.surfaces(p),align_corners=True,mode='bicubic')[:,0]
            scale=(2e-7*ip.abs()*p[:,0]).clamp_min(.01)
            profile_base=(1-rho**(2*p[:,8,None])).clamp_min(1e-8)
            pressure_profile=profile_base**p[:,6,None]
            field_profile=profile_base**p[:,7,None]
            sw=((1-p[:,9,None])*pressure_profile+p[:,9,None]*field_profile)*rho
            variance=((surf-surf.mean(2,keepdim=True))/scale[:,None,None]).square().mean(2)
            eq=(variance*sw).sum(1)/sw.sum(1)
            paired=torch.nn.functional.grid_sample(psi,grid,align_corners=True,mode='bicubic')[:,0]
            thermal=(((paired[:,:,0]-paired[:,:,1])/scale[:,None]).square()*valid_t).sum(1)/valid_t.sum(1).clamp_min(1)
            mean=surf.mean(2)
            integral=(.5*(pressure_profile[:,:-1]+pressure_profile[:,1:])*(mean[:,:-1]-mean[:,1:])).sum(1)
            predp=(ip/(total*darea)*(1-p[:,9])/p[:,0]*integral).clamp_min(1.)
            # Thomson measures electron pressure, not the unknown ion/fast-ion
            # contribution. It supplies a lower bound, not a total-pressure label.
            pe=observed/factor
            ploss=torch.relu(torch.log(pe/predp)).square()*pw
            weak_pressure_prior=torch.log(predp/observed).square()*pw
            # Net vertical force on a toroidal current plasma comes from the
            # external field: Fz/(2*pi*Ip)=<d psi_vac/dZ>_J. Self force cancels.
            vertical=((w*vac_z).sum(1)/(2e-7*ip.abs()).clamp_min(.01)).square()
            anchor=((p[:,0]+p[:,5]-kt[:,0])/ar).square()
            size=((p[:,2]-ar)/ar).square()
            shape=((p[:,3]/p[:,2]-elong)/elong).square()
            loss=(eq+.16*thermal+.0025*ploss+prior_weight*weak_pressure_prior+vertical
                  +anchor_weight*anchor+.0005*size+.0002*shape
                  +.0002*(p[:,9]-self.prior['gamma']).square()*(1-pw))
            if not torch.isfinite(loss).all():
                raise RuntimeError(f'Non-finite v13 objective at iteration {step}')
            # Each row is an independent inverse problem. A batch mean makes
            # Adam's epsilon (and a global gradient clip) change the solution
            # when unrelated shots are added to the batch.
            loss.sum().backward()
            if not torch.isfinite(u.grad).all():
                raise RuntimeError(f'Non-finite v13 gradient at iteration {step}')
            with torch.no_grad():
                norms=u.grad.norm(dim=1,keepdim=True).clamp_min(1.)
                u.grad.div_(norms)
            opt.step()
            with torch.no_grad():
                improved=loss<best;best=torch.minimum(best,loss);saved[improved]=p[improved]
            if step in (0,steps//2,steps-1):
                history.append({'step':step,'loss':float(loss.mean()),'flux_residual':float(eq.mean()),
                                'isotherm_residual':float(thermal.mean()),'pressure_log_error':float(ploss.mean()),
                                'vertical_force_residual':float(vertical.mean())})
        if not torch.isfinite(saved).all() or not torch.isfinite(best).all():
            raise RuntimeError('Non-finite v13 equilibrium')
        return saved.cpu().numpy(),history,best.cpu().numpy()
