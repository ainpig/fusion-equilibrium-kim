"""Functional-guided interpolation between existing trained flux groups."""
from __future__ import annotations

import numpy as np


GROUPS = ("dprod", "xd", "ctx", "cds", "cdsg")
WEIGHTS = np.array([1.5,1.,1.,4.,5.])/12.5
# Training-only functional scales (the source label comparison is recorded).
SCALES = np.array([.0434,.0461,.1193,.2487,.2398,1.5243])


def mix_frame(psi, R, Z, targets, strength, groups, ridge=1e-5):
    """Change flux-group weights towards independently learned shape targets.

    Every candidate stays in the nearby span of actual trained flux maps.
    A local response and a line search use only prediction-derived functionals.
    """
    from nm1.guard import box_mask
    from derive import _shape_from_contour, magnetic_axis
    from lcfs import extract_lcfs
    from nm1.predict import projected_flux
    p = np.asarray(psi,float)
    mask = box_mask(R,Z)
    def measured(field):
        C = extract_lcfs(field,R,Z,"DIII-D",mask,mask.astype(float),n_points=512)
        if C is None:return None
        shape = _shape_from_contour(C)
        axis = magnetic_axis(field,R,Z,"DIII-D",mask)
        v=np.array([axis[0],axis[1],shape["kappa"],shape["tri_top"],shape["tri_bot"],shape["volume"]])
        return v if np.isfinite(v).all() else None
    base = measured(p)
    if base is None:return psi
    # The separate interior pass handles the requested axis blend. Here it is
    # held fixed so boundary improvement cannot spend axis consistency.
    desired = np.r_[base[:2],base[2:]+np.asarray(strength[2:6])*(np.asarray(targets)[2:6]-base[2:])]
    directions = np.stack([projected_flux(g[None],p[None],1.,1.)[0]-p for g in groups])
    eps=.2
    jac=np.empty((6,len(groups)))
    for j,d in enumerate(directions):
        got=measured(p+eps*d)
        if got is None:return psi
        jac[:,j]=(got-base)/(eps*SCALES)
    rhs=(desired-base)/SCALES
    if np.linalg.norm(rhs)<1e-5:return psi
    try:
        delta=np.linalg.solve(jac.T@jac+ridge*np.eye(len(groups)),jac.T@rhs)
    except np.linalg.LinAlgError:
        return psi
    proposed=np.maximum(WEIGHTS+np.clip(delta,-.25,.25),0)
    proposed/=proposed.sum()
    delta=proposed-WEIGHTS
    change=np.einsum('g,gzr->zr',delta,directions)
    best,best_error=psi,float(np.dot(rhs,rhs))
    for fraction in (1.,.5,.25):
        field=p+fraction*change
        got=measured(field)
        if got is None:continue
        error=float(np.sum(((got-desired)/SCALES)**2))
        if error<best_error*.999:
            best,best_error=field,error
    return best
