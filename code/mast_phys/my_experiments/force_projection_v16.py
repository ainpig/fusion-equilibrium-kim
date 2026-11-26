"""Rigid current translation constrained by input-only vertical force balance.

No MAST labels, learned displacement, or shot-ID lookup is used. A root is
accepted only within a bounded search and with at least 98% current support
retained inside the engineering wall. Otherwise the original state is kept.
"""
import numpy as np
import torch
from my_experiments.virial_pressure_v14 import discrete_jr
from my_experiments.temporal_guard_v13 import guard_current


def circuit_projection(curr):
    """Optional series/anti-series engineering idealisation; audit separately."""
    result=np.asarray(curr).copy()
    for lo in (0,2,4,6):result[:,lo:lo+2]=result[:,lo:lo+2].mean(1)[:,None]
    differential=(result[:,8]-result[:,9])/2
    result[:,8]=differential;result[:,9]=-differential
    return result


def infer_translation(eq,times,curr,phi,project_circuits=False,max_shift=.4,force_tolerance=0.):
    current=circuit_projection(curr) if project_circuits else np.asarray(curr)
    vacuum=np.einsum('tc,czr->tzr',current*1000,eq.basis,optimize=True)
    original_vacuum=np.einsum('tc,czr->tzr',np.asarray(curr)*1000,eq.basis,optimize=True)
    r,z=eq.r,eq.z;rr=np.broadcast_to(r[None,:],phi.shape[1:]);area=float((r[1]-r[0])*(z[1]-z[0]))
    mask=eq.fine_mask.reshape(65,65);weights=[]
    for f,v in zip(phi,original_vacuum):
        w=np.maximum(discrete_jr(np.asarray(f,float)-v,r,z)/rr*area,0.)
        w*=mask.cpu().numpy()>0;w/=max(w.sum(),1e-20);weights.append(w)
    weights=np.asarray(weights,np.float32)
    derivative=np.gradient(vacuum,z.astype(float),axis=1,edge_order=2)
    if force_tolerance>0 and not project_circuits:
        initial_force=np.sum(weights*derivative,axis=(1,2))
        limit=float(force_tolerance)*2e-7*np.abs(np.asarray(current[:,-1],float)*1000)
        # A margin avoids changing the classification of a threshold-adjacent
        # sample due to CPU/CUDA reduction rounding. This skips work only for
        # a shot whose output was already exactly the original prediction.
        if np.all(abs(initial_force)<limit*.999):
            records=[{'shift_m':0.,'roots':[],'force_at_zero':float(f),'root_found':False,'force_tolerance':float(t)} for f,t in zip(initial_force,limit)]
            return np.asarray(phi,np.float32).copy(),{'frames':records,'guards':[],'project_circuits':False,
                'source_force_tolerance':float(force_tolerance),'unchanged_shot_fastpath':True}
    shifts=np.linspace(-max_shift,max_shift,81);records=[];out=[]
    gx,gz=torch.meshgrid(torch.linspace(-1,1,65,device=eq.device),torch.linspace(-1,1,65,device=eq.device),indexing='xy')
    grid=torch.stack([gx,gz],-1)
    with torch.no_grad():
        for first in range(0,len(weights),64):
            n=min(64,len(weights)-first);ns=len(shifts)
            base=eq.t(weights[first:first+n])/mask.clamp_min(1e-6)
            sample=grid[None,None].repeat(n,ns,1,1,1)
            sample[:,:,:,:,1]-=eq.t(shifts)[None,:,None,None]*2/np.ptp(z)
            shifted=torch.nn.functional.grid_sample(base[:,None,None].expand(-1,ns,-1,-1,-1).reshape(n*ns,1,65,65),
                sample.reshape(n*ns,65,65,2),align_corners=True,mode='bilinear',padding_mode='zeros').reshape(n,ns,65,65)
            shifted*=mask;mass=shifted.sum((2,3));shifted/=mass[:,:,None,None].clamp_min(1e-20)
            force=(shifted*eq.t(derivative[first:first+n])[:,None]).sum((2,3)).cpu().numpy()
            mass=mass.cpu().numpy();selected=[]
            for j in range(n):
                valid=mass[j]>.98;f=force[j];roots=[]
                tolerance=float(float(force_tolerance)*2e-7*abs(float(current[first+j,-1])*1000))
                target=np.sign(f[ns//2])*tolerance;excess=f-target
                if abs(f[ns//2])>tolerance:
                    for k in range(ns-1):
                        if valid[k] and valid[k+1] and excess[k]*excess[k+1]<=0:
                            if excess[k]==excess[k+1]:continue
                            roots.append(float(shifts[k]-excess[k]*(shifts[k+1]-shifts[k])/(excess[k+1]-excess[k])))
                d=min(roots,key=abs) if roots else 0.
                selected.append(d);records.append({'shift_m':d,'roots':roots,'force_at_zero':float(f[ns//2]),'root_found':bool(roots),'force_tolerance':tolerance})
            sample=grid[None].repeat(n,1,1,1);sample[:,:,:,1]-=eq.t(selected)[:,None,None]*2/np.ptp(z)
            translated=torch.nn.functional.grid_sample(base[:,None],sample,align_corners=True,mode='bilinear',padding_mode='zeros')[:,0]*mask
            translated/=translated.sum((1,2))[:,None,None].clamp_min(1e-20);out.append(translated.cpu().numpy())
        result=np.concatenate(out).reshape(len(weights),-1)
        response=eq.grid_response(eq.t(result)).cpu().numpy()
        fixed,guards=guard_current(times,result,response,current)
        plasma=eq.grid_response(eq.t(fixed)*eq.t(current[:,-1]*1000)[:,None]).cpu().numpy()
    answer=(plasma+vacuum).astype(np.float32)
    if not project_circuits:
        unchanged=np.array([abs(d['shift_m'])<1e-7 for d in records])
        answer[unchanged]=np.asarray(phi)[unchanged]
    return answer,{'frames':records,'guards':guards,'project_circuits':project_circuits,'source_force_tolerance':float(force_tolerance)}
