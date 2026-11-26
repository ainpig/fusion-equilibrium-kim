"""GS pressure fit with conservative, pointwise electron-pressure lower bounds.

Only supplied Thomson values enter the additional inequalities. A half-cell
flux-coordinate uncertainty shifts each lower bound towards the axis. No
shared coefficient or MAST equilibrium target is fitted.
"""
import numpy as np
from scipy.optimize import minimize, nnls
from my_experiments.gs_profiles_v13 import bernstein, integrated_basis
from my_experiments.pressure_models_v17 import prepare


def pressure_constraints(geometry, r, z, radii, te, ne, pressure_scale):
    pa, pb = geometry['axis'][2], geometry['xpoint'][2]
    span = pa-pb;sp=geometry['spline']
    s = (pa-sp.ev(np.zeros_like(radii), radii))/span
    uncertainty = .5*(abs(sp.ev(np.zeros_like(radii), radii, dy=1))*(r[1]-r[0])
                     +abs(sp.ev(np.zeros_like(radii), radii, dx=1))*(z[1]-z[0]))/span
    pe = np.asarray(te)*np.asarray(ne)*1.602176634e-19
    have = np.isfinite(pe)&(pe>0)&np.isfinite(s)&(s>=0)&(s<.95)
    if have.sum()<12:
        return np.empty((0,5)), np.empty(0)
    locations, lower = [], []
    for left in np.arange(0,.95,.05):
        good=have&(s>=left)&(s<left+.05)
        if good.any():
            locations.append(max(0.,float(np.median(s[good])-np.max(uncertainty[good]))))
            lower.append(float(np.quantile(pe[good],.25))/pressure_scale)
    return integrated_basis(np.asarray(locations)), np.asarray(lower)


def fit_system(h, g, constraints, lower):
    # Eliminate the five unconstrained FF' coefficients before solving the
    # remaining small convex pressure problem.
    coupling=np.linalg.solve(h[5:,5:],h[5:,:5]);offset=np.linalg.solve(h[5:,5:],g[5:])
    reduced=h[:5,:5]-h[:5,5:]@coupling;linear=g[:5]-h[:5,5:]@offset
    reduced=.5*(reduced+reduced.T)
    chol=np.linalg.cholesky(reduced+np.eye(5)*1e-12)
    p=nnls(chol.T,np.linalg.solve(chol,linear),maxiter=1000)[0]
    violated=len(lower)>0 and np.min(constraints@p-lower)<-1e-8
    if violated:
        magnitude=max(float(np.trace(reduced)),1e-12)
        initial=np.maximum(p,1.001*np.max(lower/np.maximum(constraints.sum(1),1e-12)))
        fit=minimize(lambda v:float((.5*v@reduced@v-linear@v)/magnitude),initial,
                     jac=lambda v:(reduced@v-linear)/magnitude,method='SLSQP',bounds=[(0,None)]*5,
                     constraints=[{'type':'ineq','fun':lambda v:constraints@v-lower,'jac':lambda v:constraints}],
                     options={'maxiter':200,'ftol':1e-11})
        p=fit.x
        if not np.isfinite(p).all() or p.min()<-1e-7 or np.min(constraints@p-lower)<-1e-7:
            raise ValueError('Electron-profile pressure constraint unresolved')
    return np.r_[p,offset-coupling@p],bool(violated)


def prepare_system(phi,r,z,geometry,ip,fvac,radii,te,ne):
    data=prepare(phi,r,z,geometry,ip,fvac)
    correction=data['ip']/data['current'] if data['current']>0 else 1.
    jr=data['jr']*(correction if .75<correction<1.25 else 1.)
    scale=max(float(np.sqrt(np.mean(jr*jr))),1.)
    b=bernstein(data['s']);x=np.column_stack([b*data['r2'][:,None],b]);y=jr/scale
    rough=np.zeros((6,10));second=np.diff(np.eye(5),n=2,axis=0);rough[:3,:5]=second;rough[3:,5:]=second
    regular=rough.T@rough
    pressure_scale=data['span']*scale/data['axis_r']**2
    constraints,lower=pressure_constraints(geometry,r,z,radii,te,ne,pressure_scale)
    functional=np.r_[data['fine_moments']*pressure_scale*data['beta_norm'],np.zeros(5)]
    angle=np.arctan2((data['z']-data['axis_z'])/max(np.ptp(geometry['boundary'][:,1])/2,.1),
                    (data['r']-data['axis_r'])/max(data['minor'],.1))
    sectors=np.minimum(((angle+np.pi)/(2*np.pi)*8).astype(int),7)
    systems=[];vectors=[];counts=[];full_data=None
    for sector in (-1,*range(8)):
        keep=np.ones(len(y),bool) if sector==-1 else sectors!=sector
        hdata=x[keep].T@x[keep];h=hdata+.0001*keep.sum()*regular+np.eye(10)*1e-12;g=x[keep].T@y[keep]
        systems.append(h);vectors.append(g);counts.append(int(keep.sum()))
        if sector==-1:full_data=hdata
    return {'h':np.stack(systems),'g':np.stack(vectors),'count':np.asarray(counts),
            'x':x,'y':y,'functional':functional,'constraints':constraints,'lower':lower,'hdata':full_data}


def solve_prepared(data,h=None,g=None,counts=None):
    h=data['h'] if h is None else h;g=data['g'] if g is None else g
    counts=data['count'] if counts is None else counts
    values=[];active=[];functional=data['functional']
    for sector in range(9):
        if counts[sector]<60:
            values.append(np.nan);active.append(False);continue
        coef,changed=fit_system(h[sector],g[sector],data['constraints'],data['lower'])
        values.append(float(functional@coef));active.append(changed)
        if sector==0:residual=data['x']@coef-data['y']
    values=np.asarray(values);loo=values[1:];good=np.isfinite(loo)
    sigma=np.sqrt((good.sum()-1)*np.var(loo[good])) if good.sum()>=3 else np.inf
    dual=np.linalg.solve(h[0],functional)
    model=np.sqrt(max(float(dual@data['hdata']@dual),0))*np.linalg.norm(residual)/np.sqrt(3)
    return {'beta':values[0],'sigma_jackknife':float(sigma),'sigma_model':float(model),
            'gs_residual':float(np.linalg.norm(residual)/max(np.linalg.norm(data['y']),1e-12)),
            'leave_sector_out':np.where(good,loo,values[0]),'constraints':len(data['lower']),'active':active[0]}


def readout(phi,r,z,geometry,ip,fvac,radii,te,ne):
    return solve_prepared(prepare_system(phi,r,z,geometry,ip,fvac,radii,te,ne))


def joint_systems(times, prepared):
    """Pool dimensionless magnetic likelihoods within one discharge only.

    The pressure bounds remain frame-specific: no neighboring Thomson sample
    is silently used to impose an instantaneous pressure lower bound.
    """
    times=np.asarray(times,float)
    if len(times)!=len(prepared) or not len(times):raise ValueError('Inconsistent discharge systems')
    if len(times)>1 and np.any(np.diff(times)<=0):raise ValueError('Expected increasing times')
    width=max(12.,2*np.median(np.diff(times))) if len(times)>1 else 12.
    dt=(times[:,None]-times[None,:])/width
    kernel=np.exp(-.5*dt**2)*(abs(dt)<=3);kernel/=kernel.sum(1)[:,None]
    count=np.asarray([p['count'][0] for p in prepared],float)
    h=np.stack([p['h'] for p in prepared]);g=np.stack([p['g'] for p in prepared])
    counts=np.stack([p['count'] for p in prepared])
    pooled_h=np.einsum('ij,jskl->iskl',kernel,h/count[:,None,None,None])*count[:,None,None,None]
    pooled_g=np.einsum('ij,jsl->isl',kernel,g/count[:,None,None])*count[:,None,None]
    pooled_count=(kernel@(counts/count[:,None]))*count[:,None]
    return pooled_h,pooled_g,pooled_count
