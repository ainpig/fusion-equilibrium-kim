"""Input-aware robust rejection of isolated equilibrium-solver branches.

Detect jumps in unit-current plasma response while the actuator/Ip inputs stay
locally regular. Interpolate CURRENT distributions, then reconstruct every
original time with its actual currents. No targets or fitted thresholds.
"""
import numpy as np


def guard_current(times,weights,unit_flux,currents):
    t=np.asarray(times,float);w=np.asarray(weights,float);h=np.asarray(unit_flux,float).reshape(len(t),-1)
    c=np.asarray(currents,float)
    if len(t)<5:return w.astype(np.float32),[]
    dt=float(np.median(np.diff(t)));window=4*dt
    controls=c/np.maximum(abs(c[:,-1:]),1.)
    bad=[];records=[]
    for i in range(len(t)):
        near=np.flatnonzero((abs(t-t[i])<=window)&(np.arange(len(t))!=i))
        if len(near)<4:continue
        center=np.median(h[near],axis=0)
        scale=np.median(np.linalg.norm(h[near]-center,axis=1))
        floor=1e-6*max(np.linalg.norm(center),1e-15)
        response_score=float(np.linalg.norm(h[i]-center)/max(scale,floor))
        cc=np.median(controls[near],axis=0)
        cs=np.median(np.linalg.norm(controls[near]-cc,axis=1))
        control_score=float(np.linalg.norm(controls[i]-cc)/max(cs,1e-6))
        if response_score>6 and control_score<3:
            bad.append(i);records.append({'frame':i,'time':float(t[i]),'response_mad_score':response_score,
                                         'control_mad_score':control_score})
    good=np.setdiff1d(np.arange(len(t)),bad)
    if len(good)<2:return w.astype(np.float32),[]
    result=w.copy()
    for i in bad:
        left=good[good<i];right=good[good>i]
        if len(left) and len(right):
            a,b=int(left[-1]),int(right[0]);fraction=(t[i]-t[a])/(t[b]-t[a])
            result[i]=(1-fraction)*w[a]+fraction*w[b]
        elif len(left):result[i]=w[left[-1]]
        else:result[i]=w[right[0]]
    for i in bad:
        result[i]=np.maximum(result[i],0)
        result[i]/=result[i].sum()
    return result.astype(np.float32),records
