"""Condition the scalar posterior on the measured electron-pressure lower bound."""
import numpy as np
from scipy.special import log_ndtr
from my_experiments.calibrate_pressure_moments_v14 import moments,MU0


def electron_beta(geometry,tsr,te,ne,fvac,ip_amp):
    pa=geometry['axis'][2];pb=geometry['xpoint'][2]
    pn=(geometry['spline'].ev(np.zeros_like(tsr),tsr)-pa)/(pb-pa)
    try:p=moments(pn,te,ne,(geometry['flux']-pa)/(pb-pa),geometry['R'][None,:]*geometry['inside'])[0]
    except ValueError:return 0.
    c=geometry['boundary'];a=np.ptp(c[:,0])/2;rg=(c[:,0].min()+c[:,0].max())/2
    return float(max(0.,2*MU0*100*a*rg*p/(abs(fvac)*abs(ip_amp)/1e6)))


def truncate_normal(mean,sigma,lower):
    m=np.asarray(mean,float);s=np.asarray(sigma,float);lo=np.maximum(np.asarray(lower,float),0.)
    safe=np.maximum(s,1e-10);alpha=(lo-m)/safe
    ratio=np.exp(-.5*alpha**2-.5*np.log(2*np.pi)-log_ndtr(-alpha))
    result=np.maximum(m+safe*ratio,lo)
    v=np.clip(1+alpha*ratio-ratio**2,0,1)*safe**2
    result=np.where(s>1e-10,result,np.maximum(m,lo));v=np.where(s>1e-10,v,0.)
    return result,np.sqrt(v)
