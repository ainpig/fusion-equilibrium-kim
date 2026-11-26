"""Reject empty laser pulses before bounded, missing-aware interpolation.

No per-channel interpolation across distant observations is introduced. The
existing spatial cleanup is retained; only invalid whole pulses are skipped.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares,isotonic_regression
from data_fixes import align_mast_thomson_core
from my_experiments.mast_inputs_v13 import profiles as old_profiles
E=1.602176634e-19


def profiles(row,times):
    radius,te,ne=align_mast_thomson_core(row);t=np.asarray(row['thomson_core_times'],float)
    joint=np.isfinite(te)&np.isfinite(ne)&(te>0)&(ne>0)
    valid=np.isfinite(t)&(joint.sum(1)>=12)
    if not valid.any():return radius,np.zeros((len(times),len(radius))),np.zeros((len(times),len(radius))),np.zeros(len(times),int)
    prepared=dict(row);prepared['thomson_core_R']=radius;prepared['thomson_core_Te']=te[valid];prepared['thomson_core_ne']=ne[valid]
    prepared['thomson_core_times']=t[valid]
    r,clean_t,clean_n,counts=old_profiles(prepared,times)
    goodtime=np.sort(np.unique(t[valid]));alltime=np.sort(np.unique(t[np.isfinite(t)]))
    cadence=float(np.median(np.diff(alltime))) if len(alltime)>1 else 5.
    distance=np.min(abs(np.asarray(times)[:,None]-goodtime[None,:]),axis=1)
    distant=distance>max(20.,4*cadence)
    clean_t[distant]=0.;clean_n[distant]=0.;counts[distant]=0
    return r,clean_t,clean_n,counts


def prepare_kinetic(row,times,exponent):
    r,te,ne,counts=profiles(row,times);kin=[]
    pairs=np.zeros((len(times),9,2),np.float32);valid=np.zeros((len(times),9),np.float32);pressure=np.full(len(times),np.nan,np.float32)
    for i,v in enumerate(te):
        ok=v>0
        if counts[i]<12 or ok.sum()<12:
            kin.append([float(np.mean(r)),float(np.ptp(r)/2),1.]);continue
        smooth=gaussian_filter1d(v,2);peak=max(float(np.quantile(smooth,.97)),1.);hot=smooth>.75*peak
        centre=float(np.average(r[hot],weights=smooth[hot]));support=r[smooth>.1*peak];width=max(float(np.ptp(support)/2),.15)
        def residual(p):return (p[2]*np.maximum(1-((r[ok]-p[0])/p[1])**2,0)**exponent-v[ok])/peak
        lower=np.array([r.min()+.05,.1,.3*peak]);upper=np.array([r.max()-.05,np.ptp(r),3*peak])
        initial=np.clip([centre,width,peak],lower+1e-7*(upper-lower),upper-1e-7*(upper-lower))
        fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.1,max_nfev=60)
        import os as _os
        if _os.environ.get('FUSION_MAST_ANCHOR_PEAK')=='1':
            # v14: anchor the magnetic axis on the Te maximum itself (Te is a flux function; its
            # midplane maximum is the axis) rather than on the fitted parabola's centre. The fitted
            # centre still splits the profile for the isotherm pairs below, so those are unchanged.
            kin.append([float(r[int(np.argmax(smooth))]),fit.x[1],float(np.mean(np.minimum(residual(fit.x)**2,1)))])
        elif _os.environ.get('FUSION_MAST_ANCHOR_PEAK')=='2':
            # v14 option (not used by the v14 run): the Te-weighted centroid of the channels within 10%
            # of the maximum, so a flat or noisy top does not throw the anchor to one side
            top=smooth>=.9*float(np.max(smooth))
            kin.append([float(np.average(r[top],weights=smooth[top])),fit.x[1],float(np.mean(np.minimum(residual(fit.x)**2,1)))])
        else:
            kin.append([fit.x[0],fit.x[1],float(np.mean(np.minimum(residual(fit.x)**2,1)))])
        split=int(np.argmin(abs(r-fit.x[0])));pressure[i]=np.quantile(2*E*te[i]*ne[i],.95)
        if split<3 or split>len(r)-4:continue
        left=isotonic_regression(smooth[:split+1],increasing=True).x;right=isotonic_regression(smooth[split:],increasing=False).x
        top=min(left[-1],right[0])
        for j,fraction in enumerate(np.linspace(.25,.85,9)):
            value=fraction*top
            if value<=max(left[0],right[-1]):continue
            pairs[i,j]=[np.interp(value,left,r[:split+1]),np.interp(value,right[::-1],r[split:][::-1])];valid[i,j]=1
    return np.asarray(kin,np.float32),(pairs,valid,pressure)
