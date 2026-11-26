"""Per-equilibrium pressure readouts; no target labels or shared MAST fit.

The cubic pressure / free-end FF' option follows the magnetics-only MAST
parameterisation in Berkery et al., PPCF 63 (2021) 055014, section 2.1.
"""
import numpy as np
from scipy.optimize import lsq_linear
from matplotlib.path import Path as Polygon
from my_experiments.gs_profiles_v13 import bernstein,integrated_basis,MU0
from my_experiments.virial_pressure_v14 import discrete_jr,pressure_moment


def prepare(phi,r,z,geometry,ip,fvac,vacuum=None):
    phi=np.asarray(phi,float);rr,zz=np.meshgrid(r,z);a=geometry['axis'];span=a[2]-geometry['xpoint'][2]
    inside=Polygon(geometry['boundary']).contains_points(np.column_stack([rr.ravel(),zz.ravel()])).reshape(phi.shape)
    pn=(a[2]-phi)/span;good=inside&(pn>.015)&(pn<.985);good[[0,-1]]=False;good[:,[0,-1]]=False
    if good.sum()<60:raise ValueError('Insufficient plasma cells')
    jr=discrete_jr(phi if vacuum is None else phi-vacuum,r,z)
    current=float(np.sum(jr[inside]/rr[inside])*(r[1]-r[0])*(z[1]-z[0]))
    sf=np.clip((a[2]-geometry['flux'])/span,0,1);vw=geometry['R'][None,:]*geometry['inside'];volume=float(vw.sum())
    fine_moments=(integrated_basis(sf)*vw[:,:,None]).sum((0,1))/volume
    contour=geometry['boundary'];minor=float(np.ptp(contour[:,0])/2);rg=float((contour[:,0].max()+contour[:,0].min())/2)
    return {'jr':jr[good],'s':pn[good],'r2':(rr[good]/a[0])**2,'span':span,'axis_r':float(a[0]),'rgeo':rg,
        'fine_moments':fine_moments,'current':current,'ip':abs(float(ip)),'fvac':abs(float(fvac)),
        'beta_norm':2*MU0*100*minor*rg/(abs(float(fvac))*abs(float(ip))/1e6),'minor':minor,
        'z':zz[good],'r':rr[good],'axis_z':float(a[1])}


def solve(prepared,kind='cubic_neumann',regularization=0.,ampere='gated',selection=None):
    p=prepared;s=p['s'];rr=p['r2'];jr=p['jr'].copy();scale=max(float(np.sqrt(np.mean(jr**2))),1.)
    ratio=p['ip']/p['current'] if p['current']>0 else 1.
    correction=ratio if ampere=='always' or (ampere=='gated' and .75<ratio<1.25) else 1.
    jr*=correction;scale*=correction
    if kind in ('cubic_neumann','cubic_free','thermal_neumann','thermal_free'):
        pc=np.asarray(p['thermal_gradient']) if kind.startswith('thermal') else np.array([0,.25,1/3,.25,0.])
        pb=(bernstein(s)@pc)[:,None]
        ff=np.column_stack([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s)+s**3]) if kind.endswith('neumann') else bernstein(s)
        npres=1;basis=np.column_stack([pb[:,0]*rr,ff]);rough=np.zeros((0,basis.shape[1]))
        functional=np.r_[p['fine_moments']@pc*p['span']*scale/p['axis_r']**2*p['beta_norm'],np.zeros(ff.shape[1])]
    elif kind in ('bernstein','bernstein_neumann','bernstein_neumann3'):
        b=bernstein(s);npres=5;second=np.diff(np.eye(5),n=2,axis=0)
        if kind=='bernstein':ff=b;field_rough=second
        elif kind=='bernstein_neumann':
            transform=np.zeros((5,4));transform[:4,:]=np.eye(4);transform[4,3]=1
            ff=b@transform;field_rough=second@transform
        else:
            ff=np.column_stack([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s)+s**3])
            field_rough=np.array([[1.,-2.,1.],[0.,1.,-1.]])
        basis=np.column_stack([b*rr[:,None],ff]);rough=np.zeros((3+len(field_rough),basis.shape[1]));rough[:3,:5]=second;rough[3:,5:]=field_rough
        functional=np.r_[p['fine_moments']*p['span']*scale/p['axis_r']**2*p['beta_norm'],np.zeros(ff.shape[1])]
    else:raise ValueError(kind)
    y=jr/scale;selected=np.ones(len(y),bool) if selection is None else np.asarray(selection,bool)
    x=basis[selected];y=y[selected]
    if len(y)<max(20,basis.shape[1]*3):raise ValueError('Too few selected cells')
    matrix=np.vstack([x,float(regularization)*np.sqrt(len(y))*rough]);target=np.r_[y,np.zeros(len(rough))]
    if npres==1:
        coef=np.linalg.lstsq(matrix,target,rcond=None)[0]
        if coef[0]<0:coef=np.r_[0.,np.linalg.lstsq(matrix[:,1:],target,rcond=None)[0]]
    else:
        fit=lsq_linear(matrix,target,bounds=(np.r_[np.zeros(npres),np.full(basis.shape[1]-npres,-np.inf)],np.full(basis.shape[1],np.inf)),tol=1e-9,max_iter=200)
        coef=fit.x
    residual=x@coef-y
    return {'beta':float(functional@coef),'relative_residual':float(np.linalg.norm(residual)/max(np.linalg.norm(y),1e-12)),
        'ampere_scale':ratio,'applied_ampere_scale':correction,'coefficients':coef,'functional':functional,
        'condition_number':float(np.linalg.cond(matrix)),'cells':len(y)}


def add_thermal_shape(prepared,geometry,radii,te,ne):
    """Fit a monotone electron-pressure shape to one frame's Thomson inputs.

The pressure/FF separation below estimates its unknown total-pressure
amplitude from the GS identity, not from any target-domain beta label.
"""
    pa=geometry['axis'][2];span=pa-geometry['xpoint'][2]
    s=(pa-geometry['spline'].ev(np.zeros_like(radii),radii))/span
    pressure=np.asarray(te,float)*np.asarray(ne,float)*1.602176634e-19
    ok=np.isfinite(pressure)&(pressure>0)&(s>=0)&(s<=1.05);x=[];y=[]
    for lo in np.arange(0,1,.05):
        have=ok&(s>=lo)&(s<lo+.05)
        if have.any():x.append(float(np.median(s[have])));y.append(float(np.median(pressure[have])))
    if len(x)<6 or min(x)>.35:raise ValueError('Insufficient thermal shape')
    peak=max(y);x=np.asarray(x);y=np.asarray(y)/peak
    design=np.column_stack([integrated_basis(x),np.ones(len(x))]);rough=np.zeros((3,6));rough[:,:5]=np.diff(np.eye(5),n=2,axis=0)
    matrix=np.vstack([design,.005*np.sqrt(len(x))*rough])
    fit=lsq_linear(matrix,np.r_[y,np.zeros(3)],bounds=(0,np.inf),tol=1e-9,max_iter=200)
    gradient=fit.x[:5];average=float(prepared['fine_moments']@gradient)
    if average<1e-6:raise ValueError('Flat or unresolved thermal pressure')
    prepared['thermal_gradient']=gradient/average
    prepared['thermal_beta']=(average+fit.x[5])*peak*prepared['beta_norm']
    prepared['thermal_shape_residual']=float(np.sqrt(np.mean((design@fit.x-y)**2)))
    return prepared
