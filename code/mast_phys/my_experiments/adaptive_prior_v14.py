"""Scale-aware source priors for beta: do not impose a fixed high-beta ceiling."""
import numpy as np


def relative_prior(y,sigma,mean,variance,return_weight=False):
    y=np.asarray(y,float);s2=np.asarray(sigma,float)**2
    v=variance*np.maximum(1.,(y/mean)**2)
    w=v/(v+s2);post=w*y+(1-w)*mean
    result=(post,np.sqrt(v*(1-w)))
    return (*result,w) if return_weight else result


def student_prior(y,sigma,mean,variance):
    """Student-t(3) prior with the same source mean and variance, positive MAP.

Its heavy tails reduce extrapolation shrinkage. The scalar MAP stationary
equation is cubic and all real roots/endpoints are checked, not one local root.
"""
    out=[];errors=[]
    for value,error in zip(np.asarray(y,float),np.asarray(sigma,float)):
        if error<1e-10:out.append(value);errors.append(error);continue
        if not np.isfinite(error):out.append(mean);errors.append(np.sqrt(variance));continue
        noise=error**2;delta=value-mean;aa=variance
        roots=np.roots([1.,-delta,aa+4*noise,-delta*aa])
        candidates=[delta,0.]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and r.real+mean>=0]
        def loss(x):return .5*(x-delta)**2/noise+2*np.log1p(x*x/aa)
        chosen=min(candidates,key=loss)
        precision=1/noise+4*(aa-chosen**2)/(aa+chosen**2)**2
        out.append(max(chosen+mean,0.));errors.append(np.sqrt(1/max(precision,1e-8)))
    return np.asarray(out),np.asarray(errors)
