"""A bounded interior flux-profile change, solved for its derived inductance.

The perturbation is zero at the boundary and magnetic axis, with zero first
derivative there. Its support is the prediction's own closed contour. No target
field is consulted. CUDA solves the small independent profile updates in batch.
"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'fusion_scoring'))
sys.path.insert(0, str(ROOT))
from derive import _bilinear, _points_in_polygon, internal_inductance
from lcfs import extract_lcfs_with_sign
from o_point import find_o_point
import local_score                          # the vendored competition scorer (zeroshot_mast/local_score.py)


def prepare_field(psi, r, z, mask):
    p = np.asarray(psi, float)
    contour, _ = extract_lcfs_with_sign(p, r, z, 'DIII-D', mask, mask.astype(float),
                                       n_iter=local_score.N_ITER, axis_sign=-1.)
    if contour is None or len(contour) < 8:
        return None
    phi = -p
    _, _, iz, ir, _ = find_o_point(phi, r, z, mask)
    pa = float(phi[iz, ir])
    pb = float(np.median(_bilinear(phi, r, z, contour[:, 0], contour[:, 1])))
    span = pa-pb
    if not np.isfinite(span) or span <= 1e-8:
        return None
    rr, zz = np.meshgrid(r, z)
    inside = _points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(rr.shape)
    if inside.sum() < 4:
        return None
    s = np.clip((phi-pb)/span, 0., 1.)
    delta = span*s**2*(1-s)**4*inside
    gz, gr = np.gradient(phi, z, r)
    dz, dr = np.gradient(delta, z, r)
    c = (gr*gr+gz*gz)/rr**2
    b = 2*(gr*dr+gz*dz)/rr**2
    a = (dr*dr+dz*dz)/rr**2
    volume_weight = 2*np.pi*rr*float(np.mean(np.diff(r)))*float(np.mean(np.diff(z)))*inside
    volume = volume_weight.sum()
    numerator = np.array([np.sum(v*volume_weight)/volume for v in (c, b, a)])
    cr, cz = contour[:, 0], contour[:, 1]
    fr, fz = np.interp(cr, r, np.arange(len(r))), np.interp(cz, z, np.arange(len(z)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(r)-2)
    z0 = np.clip(np.floor(fz).astype(int), 0, len(z)-2)
    tr, tz = fr-r0, fz-z0
    seg = np.linalg.norm(np.roll(contour, -1, axis=0)-contour, axis=1)
    vertex = .5*(seg+np.roll(seg, 1))/seg.sum()
    weights = np.zeros(phi.size)
    for rx, zx, w in [(0, 0, (1-tr)*(1-tz)), (1, 0, tr*(1-tz)), (0, 1, (1-tr)*tz), (1, 1, tr*tz)]:
        np.add.at(weights, (z0+zx)*len(r)+r0+rx, vertex*w)
    keep = weights != 0
    coefficients = np.stack([v.ravel()[keep] for v in (c, b, a)])
    weights = weights[keep]
    li = float(numerator[0]/np.sum(weights*np.sqrt(np.maximum(coefficients[0], 0.)))**2)
    exact = internal_inductance(p, r, z, 'DIII-D', contour)
    if not np.isfinite(exact) or not .1 <= exact <= 5:
        return None
    assert abs(li-exact) < 1e-10, (li, exact)
    return dict(delta=-delta, coefficients=coefficients, weights=weights, numerator=numerator,
                li=li, contour=contour)


@torch.inference_mode()
def solve(prepared, desired, max_amount=1., iterations=12):
    if not prepared:
        return np.empty(0), np.empty(0)
    count, width = len(prepared), max(len(p['weights']) for p in prepared)
    c, w = np.zeros((count, 3, width)), np.zeros((count, width))
    for i, p in enumerate(prepared):
        n = len(p['weights'])
        c[i, :, :n], w[i, :n] = p['coefficients'], p['weights']
    c, w, numerator, target = [torch.as_tensor(x, dtype=torch.float64, device='cuda') for x in
                               (c, w, np.stack([p['numerator'] for p in prepared]), np.asarray(desired))]
    def evaluate(amount):
        square = c[:, 0]+amount[:, None]*c[:, 1]+amount[:, None]**2*c[:, 2]
        bp = square.clamp_min(1e-30).sqrt()
        perimeter = (w*bp).sum(1).clamp_min(1e-30)
        dp = (w*(c[:, 1]+2*amount[:, None]*c[:, 2])/(2*bp)).sum(1)
        nom = numerator[:, 0]+amount*numerator[:, 1]+amount**2*numerator[:, 2]
        dn = numerator[:, 1]+2*amount*numerator[:, 2]
        li = nom/perimeter**2
        derivative = dn/perimeter**2-2*nom*dp/perimeter**3
        return li, derivative
    amount = torch.zeros(count, dtype=torch.float64, device='cuda')
    base, _ = evaluate(amount)
    target = torch.where(torch.isfinite(target), target.clamp(.1, 5.), base)
    best, best_li, error = amount.clone(), base.clone(), (base-target).square()
    def record(candidate):
        nonlocal best, best_li, error
        li, derivative = evaluate(candidate)
        loss = (li-target).square()
        good = torch.isfinite(loss)&(loss < error)
        best = torch.where(good, candidate, best)
        best_li = torch.where(good, li, best_li)
        error = torch.where(good, loss, error)
        return li, derivative
    record(amount+max_amount)
    record(amount-max_amount)
    for _ in range(iterations):
        li, derivative = evaluate(amount)
        step = torch.where(derivative.abs()>1e-9, (target-li)/derivative, torch.zeros_like(amount)).clamp(-.25, .25)
        amount = (amount+step).clamp(-max_amount, max_amount)
        record(amount)
    return best.cpu().numpy(), best_li.cpu().numpy()


def apply(phi, prepared, amounts):
    return np.stack([(np.asarray(p, float)+float(a)*s['delta']).astype(np.float32)
                     for p, s, a in zip(phi, prepared, amounts)])
