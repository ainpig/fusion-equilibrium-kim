"""Batched CUDA pressure/FF reconstruction and uncertainty; no MAST fitting.

The five pressure coefficients are nonnegative. Enumerating their 32 active
sets solves the same small convex least-squares problem as the CPU estimator.
Nine systems (full + eight leave-sector-out) are evaluated together per frame.
"""
from __future__ import annotations
import numpy as np
import torch
from matplotlib.path import Path as Polygon

MU0=4e-7*np.pi


def prepare_geometry(phi,r,z,geometries,ip,fvac,compact=True):
    rr,zz=np.meshgrid(r,z);points=np.column_stack([rr.ravel(),zz.ravel()])
    inside=np.stack([Polygon(g['boundary']).contains_points(points).reshape(rr.shape) for g in geometries])
    result={'phi':np.asarray(phi),'inside':inside,'ip':np.asarray(ip),'fvac':np.asarray(fvac),
            'axis':np.array([g['axis'] for g in geometries]),'pb':np.array([g['xpoint'][2] for g in geometries]),
            'minor':np.array([np.ptp(g['boundary'][:,0])/2 for g in geometries]),
            'height':np.array([np.ptp(g['boundary'][:,1])/2 for g in geometries]),
            'rgeo':np.array([(g['boundary'][:,0].max()+g['boundary'][:,0].min())/2 for g in geometries]),
            'r':np.asarray(r),'z':np.asarray(z)}
    if compact:
        # Five sufficient statistics replace a large fine-grid IPC transfer.
        # The exact Bernstein antiderivative is polynomial (no approximation).
        integrals=[]
        for g in geometries:
            s=np.clip((g['axis'][2]-g['flux'])/(g['axis'][2]-g['xpoint'][2]),0,1)
            w=g['R'][None,:]*g['inside'];denom=w.sum()
            sums=[np.sum(c*s**k*(1-s)**(5-k)*w) for k,c in enumerate((1,5,10,10,5))]
            integrals.append(np.cumsum(sums)/(5*denom))
        result['integral_mean']=np.asarray(integrals)
    else:
        result.update(fine=np.stack([g['flux'] for g in geometries]),fine_inside=np.stack([g['inside'] for g in geometries]),
                      fine_r=np.stack([g['R'] for g in geometries]))
    return result


def active_set_solve(h,g):
    """Exact active-set enumeration for five constrained and five free variables."""
    shape=g.shape;h=h.reshape(-1,10,10);g=g.reshape(-1,10)
    best=torch.full((len(g),),float('inf'),dtype=g.dtype,device=g.device);answer=torch.zeros_like(g)
    diagonal=torch.arange(10,device=g.device)
    for bits in range(32):
        free=torch.tensor([bool(bits&(1<<k)) for k in range(5)]+[True]*5,device=g.device,dtype=g.dtype)
        matrix=h*free[None,:,None]*free[None,None,:]
        matrix[:,diagonal,diagonal]+=1-free
        solution=torch.linalg.solve(matrix,(g*free).unsqueeze(-1)).squeeze(-1)
        feasible=(solution[:,:5]>=-1e-8).all(1)&torch.isfinite(solution).all(1)
        cost=torch.einsum('bi,bij,bj->b',solution,h,solution)-2*(solution*g).sum(1)
        use=feasible&(cost<best)
        best=torch.where(use,cost,best);answer=torch.where(use[:,None],solution,answer)
    if not torch.isfinite(best).all():raise RuntimeError('No feasible GPU pressure active set')
    answer[:,:5].clamp_min_(0)
    return answer.reshape(shape)


@torch.inference_mode()
def pressure_batch(prepared,device='cuda'):
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for v15 batched pressure')
    def t(value):
        if isinstance(value,np.ndarray) and not value.flags.writeable:value=value.copy()
        return torch.as_tensor(value,dtype=torch.float64,device=device)
    r=t(prepared['r']);z=t(prepared['z']);rr=r[None,None,:];zz=z[None,:,None]
    phi=t(prepared['phi']);inside=torch.as_tensor(prepared['inside'],device=device,dtype=torch.bool)
    axis=t(prepared['axis']);pb=t(prepared['pb']);span=axis[:,2]-pb;ra=axis[:,0];za=axis[:,1]
    ip=t(prepared['ip']).abs();fvac=t(prepared['fvac']).abs();dr=r[1]-r[0];dz=z[1]-z[0]
    pn=(axis[:,2,None,None]-phi)/span[:,None,None]
    valid=inside&(pn>.015)&(pn<.985);valid[:,[0,-1],:]=False;valid[:,:,[0,-1]]=False
    count=valid.sum((1,2)).to(torch.float64)
    if (count<60).any():raise ValueError('Too few current cells in batch')
    jr=torch.zeros_like(phi)
    jr[:,1:-1,1:-1]=-((phi[:,1:-1,2:]-2*phi[:,1:-1,1:-1]+phi[:,1:-1,:-2])/dr**2
        -(phi[:,1:-1,2:]-phi[:,1:-1,:-2])/(2*dr*r[None,None,1:-1])
        +(phi[:,2:,1:-1]-2*phi[:,1:-1,1:-1]+phi[:,:-2,1:-1])/dz**2)/MU0
    current=(jr/rr*inside).sum((1,2))*dr*dz
    ampere=torch.where(current>0,ip/current,torch.ones_like(ip))
    correction=torch.where((ampere>.75)&(ampere<1.25),ampere,torch.ones_like(ip))
    jr*=correction[:,None,None]
    scale=torch.sqrt((jr.square()*valid).sum((1,2))/count).clamp_min(1.)
    s=pn.clamp(0,1);base=torch.stack([(1-s)**4,4*s*(1-s)**3,6*s*s*(1-s)**2,4*s**3*(1-s),s**4],-1)
    base*=valid[:,:,:,None]
    x=torch.cat([base*(rr/ra[:,None,None])[:,:,:,None]**2,base],-1).flatten(1,2)
    y=(jr/scale[:,None,None]*valid).flatten(1)
    xt=x.transpose(1,2);h0=xt@x;g0=(xt@y[:,:,None]).squeeze(-1)
    regular=torch.zeros((6,10),dtype=torch.float64,device=device)
    second=torch.diff(torch.eye(5,dtype=torch.float64,device=device),n=2,dim=0)
    regular[:3,:5]=second;regular[3:,5:]=second;reg=regular.T@regular
    angle=torch.atan2((zz-za[:,None,None])/t(prepared['height'])[:,None,None],
                      (rr-ra[:,None,None])/t(prepared['minor'])[:,None,None])
    sectors=((angle+np.pi)/(2*np.pi)*8).to(torch.int64).clamp_max(7).flatten(1)
    h=[h0+.0001*count[:,None,None]*reg];g=[g0];counts=[count]
    for sector in range(8):
        excluded=((sectors==sector)&valid.flatten(1)).to(torch.float64)
        weighted=x*excluded[:,:,None]
        n=count-excluded.sum(1);counts.append(n)
        h.append(h0-xt@weighted+.0001*n[:,None,None]*reg)
        g.append(g0-(xt@(y*excluded)[:,:,None]).squeeze(-1))
    h=torch.stack(h,1);g=torch.stack(g,1);counts=torch.stack(counts,1)
    coefficients=active_set_solve(h,g)
    full=coefficients[:,0];residual=x@full[:,:,None]-y[:,:,None]
    residual_norm=torch.linalg.vector_norm(residual,dim=(1,2));target_norm=torch.linalg.vector_norm(y,dim=1)
    # Integrate Bernstein antiderivatives on the fine physical plasma region.
    if 'integral_mean' in prepared:integrated=t(prepared['integral_mean'])
    else:
        fine=t(prepared['fine']);sf=((axis[:,2,None,None]-fine)/span[:,None,None]).clamp(0,1)
        vw=t(prepared['fine_inside'])*t(prepared['fine_r'])[:,None,:];denom=vw.sum((1,2))
        totals=[]
        for k,c in enumerate((1,5,10,10,5)):
            totals.append((c*sf**k*(1-sf)**(5-k)*vw).sum((1,2)))
        integrated=torch.cumsum(torch.stack(totals,1),1)/(5*denom[:,None])
    norm=2*MU0*100*t(prepared['minor'])*t(prepared['rgeo'])/(fvac*ip/1e6)
    functional=integrated*(span*scale/ra**2*norm)[:,None]
    beta=(coefficients[:,:,:5]*functional[:,None,:]).sum(-1)
    keep=counts[:,1:]>=60;nkeep=keep.sum(1)
    loo_mean=(beta[:,1:]*keep).sum(1)/nkeep.clamp_min(1)
    sigma=torch.sqrt((nkeep-1).clamp_min(0)*(torch.square(beta[:,1:]-loo_mean[:,None])*keep).sum(1)/nkeep.clamp_min(1))
    sigma=torch.where(nkeep>=3,sigma,torch.full_like(sigma,float('inf')))
    functional10=torch.cat([functional,torch.zeros_like(functional)],1)
    response=torch.linalg.solve(h[:,0]+torch.eye(10,device=device,dtype=torch.float64)*1e-12,functional10[:,:,None])
    sensitivity=torch.linalg.vector_norm(x@response,dim=(1,2))
    result={'sigma_jackknife':sigma,'residual_norm':residual_norm,'target_norm':target_norm,
            'functional_sensitivity_norm':sensitivity,'beta_full_fit':beta[:,0],
            'leave_sector_out':beta[:,1:],'gs_residual':residual_norm/target_norm.clamp_min(1e-12)}
    return {name:value.cpu().numpy() for name,value in result.items()}
