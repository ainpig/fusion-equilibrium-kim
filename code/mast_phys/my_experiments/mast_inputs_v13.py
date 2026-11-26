"""Pulse-local, missing-aware Thomson preprocessing with spatial outlier checks.

Never fills a missing channel from a distant future measurement. No targets or
shared learned weights are used. Spatial interpolation only denoises one pulse.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.optimize import least_squares,isotonic_regression
from data_fixes import align_mast_thomson_core

E=1.602176634e-19


def pulse_interpolate(times,values,target):
    t=np.asarray(times,float);v=np.asarray(values,float)
    good=np.isfinite(t);t=t[good];v=v[good]
    order=np.argsort(t);t=t[order];v=v[order]
    t,indices=np.unique(t,return_index=True);v=v[indices]
    if not len(t):return np.full((len(target),v.shape[1]),np.nan)
    hi=np.clip(np.searchsorted(t,target),0,len(t)-1);lo=np.maximum(hi-1,0)
    lo=np.where(np.asarray(target)>=t[-1],len(t)-1,lo)
    gap=t[hi]-t[lo]
    weight=np.divide(np.asarray(target)-t[lo],gap,out=np.zeros(len(target)),where=gap>0).clip(0,1)
    left,right=v[lo],v[hi]
    gl=np.isfinite(left)&(left>0);gr=np.isfinite(right)&(right>0)
    nearest=np.where((weight<=.5)[:,None],left,right)
    both=gl&gr
    return np.where(both,(1-weight[:,None])*left+weight[:,None]*right,
                    np.where(np.isfinite(nearest)&(nearest>0),nearest,np.nan))


def profiles(row,times):
    r,te,ne=align_mast_thomson_core(row)
    te=pulse_interpolate(row['thomson_core_times'],te,times)
    ne=pulse_interpolate(row['thomson_core_times'],ne,times)
    clean_t=np.zeros_like(te);clean_n=np.zeros_like(ne);counts=[]
    for i,(temp,density) in enumerate(zip(te,ne)):
        good=np.isfinite(temp)&np.isfinite(density)&(temp>0)&(density>0)
        if good.sum()<12:
            counts.append(0);continue
        good &= density>=.02*np.quantile(density[good],.9)
        for values in (temp,density):
            if good.sum()<12:break
            filled=np.interp(r,r[good],np.log(values[good]))
            med=median_filter(filled,size=5,mode='nearest')
            deviation=median_filter(abs(filled-med),size=5,mode='nearest')
            good &= abs(filled-med)<=4*np.maximum(1.4826*deviation,.15)
        counts.append(int(good.sum()))
        if good.sum()<12:continue
        clean_t[i]=np.interp(r,r[good],temp[good],left=0,right=0)
        clean_n[i]=np.interp(r,r[good],density[good],left=0,right=0)
    return r,clean_t.astype(np.float32),clean_n.astype(np.float32),np.asarray(counts)


def prepare_kinetic(row,times,exponent):
    r,te,ne,counts=profiles(row,times)
    kin=[];pairs=np.zeros((len(times),9,2),np.float32);valid=np.zeros((len(times),9),np.float32)
    pressure=np.full(len(times),np.nan,np.float32)
    for i,v in enumerate(te):
        ok=v>0
        if counts[i]<12 or ok.sum()<12:
            kin.append([float(np.mean(r)),float(np.ptp(r)/2),1.]);continue
        smooth=gaussian_filter1d(v,2)
        peak=max(float(np.quantile(smooth,.97)),1.)
        hot=smooth>.75*peak
        center=float(np.average(r[hot],weights=smooth[hot]))
        support=r[smooth>.1*peak]
        width=max(float(np.ptp(support)/2),.15)
        def residual(p):
            model=p[2]*np.maximum(1-((r[ok]-p[0])/p[1])**2,0)**exponent
            return (model-v[ok])/peak
        lower=np.array([r.min()+.05,.1,.3*peak]);upper=np.array([r.max()-.05,np.ptp(r),3*peak])
        initial=np.clip([center,width,peak],lower+1e-7*(upper-lower),upper-1e-7*(upper-lower))
        fitted=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.1,max_nfev=60)
        kin.append([fitted.x[0],fitted.x[1],float(np.mean(np.minimum(residual(fitted.x)**2,1)))])
        # The broad fitted core locates the branch split even for hollow or
        # double-peaked temperatures; one exceptional channel cannot set it.
        split=int(np.argmin(abs(r-fitted.x[0])))
        pressure[i]=np.quantile(2*E*te[i]*ne[i],.95)
        if split<3 or split>len(r)-4:continue
        left=isotonic_regression(smooth[:split+1],increasing=True).x
        right=isotonic_regression(smooth[split:],increasing=False).x
        top=min(left[-1],right[0])
        for j,fraction in enumerate(np.linspace(.25,.85,9)):
            value=fraction*top
            if value<=max(left[0],right[-1]):continue
            pairs[i,j]=[np.interp(value,left,r[:split+1]),np.interp(value,right[::-1],r[split:][::-1])]
            valid[i,j]=1
    return np.asarray(kin,np.float32),(pairs,valid,pressure)
