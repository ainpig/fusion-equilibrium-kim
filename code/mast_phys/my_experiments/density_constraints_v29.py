"""Add observed density isosurface pairs to the existing temperature constraints."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import isotonic_regression
from my_experiments.mast_inputs_v15 import prepare_kinetic as temperature_constraints,profiles


def prepare_kinetic(row,times,exponent):
    kin,(temperature_pairs,temperature_valid,pressure)=temperature_constraints(row,times,exponent)
    r,te,ne,counts=profiles(row,times)
    pairs=np.zeros_like(temperature_pairs)
    valid=np.zeros_like(temperature_valid)
    for i,v in enumerate(ne):
        if counts[i]<12 or (v>0).sum()<12:continue
        split=int(np.argmin(np.abs(r-kin[i,0])))
        if split<3 or split>len(r)-4:continue
        smooth=gaussian_filter1d(v.astype(float),2)
        scale=max(float(np.max(smooth)),1.)
        smooth=smooth/scale
        left=isotonic_regression(smooth[:split+1],increasing=True).x
        right=isotonic_regression(smooth[split:],increasing=False).x
        top=min(left[-1],right[0])
        for j,fraction in enumerate(np.linspace(.25,.85,9)):
            value=fraction*top
            if value<=max(left[0],right[-1]) or top<=0:continue
            pairs[i,j]=[np.interp(value,left,r[:split+1]),np.interp(value,right[::-1],r[split:][::-1])]
            valid[i,j]=1.
    return kin,(np.concatenate([temperature_pairs,pairs],axis=1),
                np.concatenate([temperature_valid,valid],axis=1),pressure)
