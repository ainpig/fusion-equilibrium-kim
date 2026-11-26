"""Independent shape targets and boundary-preserving field corrections for v12.

Targets are functionals of official DIII-D training flux maps. No predicted
scalar is submitted separately: a correction has to change the submitted map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from nm1.model import ResidualMLPBlock


NAMES = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")


def raw_features(base, ext, cds, cdsg):
    from nm1.features2 import context_block, seg_block
    if base.shape[1] < 272:
        raise ValueError("The shape head requires the 272-column base block")
    return np.ascontiguousarray(np.column_stack([
        base[:, :272], ext, context_block(ext, base[:, 136]),
        cds[:, :41], cdsg[:, :41], seg_block(base[:, 136]),
    ]), dtype=np.float32)


class ShapeHead(nn.Module):
    def __init__(self, n_features, width=512, depth=4, dropout=0.02):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(width, dropout) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 7))

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


class ShapePredictor:
    def __init__(self, paths, device="cuda", encoders=None):
        self.device = device
        self.members = []
        for path in paths:
            p = torch.load(path, map_location="cpu", weights_only=False)
            m = ShapeHead(**p["model"])
            m.load_state_dict(p["state_dict"])
            if p.get("encoder_tfm"):
                # a head read off a frozen whole-shot transformer (nm1.seqmodel.TfmRegressor.encode): its latent
                # sees the entire discharge, so every call must pass one complete shot (the predictor does)
                from nm1 import ARTIFACTS
                from nm1.seqmodel import load_seq
                enc = Path(p["encoder_tfm"])
                p["_tfm"] = load_seq(enc if enc.is_absolute() else ARTIFACTS / enc, device)
            elif p.get("encoder"):
                from nm1 import ARTIFACTS
                from nm1.model import load_member
                if encoders and p["encoder"] in encoders:
                    pair = encoders[p["encoder"]]
                else:
                    pair = load_member(ARTIFACTS/p["encoder"],device,keep_state=False)
                p["_encoder_model"],p["_encoder_payload"]=pair
            self.members.append((m.to(device).eval(), p))

    @torch.inference_mode()
    def predict_raw(self, raw, batch=8192):
        individual = self.predict_members_raw(raw,batch)
        out = np.zeros((len(raw),7),np.float64)
        for values in individual:out+=values
        return out/len(individual)

    @torch.inference_mode()
    def predict_members_raw(self, raw, batch=8192):
        individual = []
        for model, p in self.members:
            out = np.zeros((len(raw),7),np.float64)
            if "_tfm" in p:
                features = tfm_latent_raw(raw, *p["_tfm"], self.device)
            else:
                features=(latent_features(raw,p["_encoder_model"],p["_encoder_payload"],self.device)
                          if p.get("encoder") else raw)
            for a in range(0, len(features), batch):
                x = (features[a:a+batch] - p["feature_mean"]) / p["feature_std"]
                y = model(torch.as_tensor(x, device=self.device)).float().cpu().numpy()
                out[a:a+len(y)] += y * p["target_std"] + p["target_mean"]
            individual.append(out)
        return individual

    def predict(self, base, ext, cds, cdsg):
        return self.predict_raw(raw_features(base, ext, cds, cdsg))


@torch.inference_mode()
def tfm_latent_raw(raw, model, payload, device="cuda"):
    """The frozen transformer's 1024-dim latent for ONE whole shot given in the raw_features layout
    (base 0:272, ext 272:570, ctx 570:586, cds 586:627, cdsg 627:668, seg)."""
    from nm1.seqmodel import seq_inputs
    base = raw[:, :272]
    x = seq_inputs(base, raw[:, 272:570], base[:, 136].astype(np.float64) * 1000.0, payload,
                   raw[:, 586:627], raw[:, 627:668])
    dev = torch.device(device)
    xt = torch.from_numpy(x).to(dev)[None]
    t = torch.from_numpy(base[:, 136].astype(np.float32)).to(dev)[None]
    m = torch.ones((1, len(x)), dtype=torch.bool, device=dev)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        h = model.encode(xt, m, t)[0]
    return h.float().cpu().numpy()


@torch.inference_mode()
def latent_features(raw, model, p, device="cuda", batch=4096):
    """Read a frozen, train-only flux member's hidden representation.

    The input layout is the same raw_features layout at train and inference.
    The encoder weights are fixed; only the new seven-output readout is fitted.
    """
    if p.get("temporal_offsets") or p.get("vac_features"):
        raise ValueError("Latent encoder must use the base+profile feature blocks")
    from nm1.model import base_width
    parts=[(raw[:,:base_width(p)]-p["feature_mean"])/p["feature_std"]]
    for key,lo,hi in (("ext",272,570),("ctx",570,586),("cds",586,627),("cdsg",627,668)):
        n=int(p.get("n_"+key) or 0)
        if n: parts.append((raw[:,lo:lo+n]-p[key+"_mean"])/p[key+"_std"])
    x=np.ascontiguousarray(np.column_stack(parts),np.float32)
    out=np.empty((len(x),int(p["model"]["width"])),np.float32)
    offloaded = getattr(model, "_fusion_offload", False)
    if offloaded:
        model.to(device)
    for a in range(0,len(x),batch):
        xb=torch.as_tensor(x[a:a+batch],device=device)
        h=model.norm(model.blocks(model.input(xb)))
        out[a:a+len(h)]=h.float().cpu().numpy()
    if offloaded:
        model.to("cpu")
    return out


def relabel_li(psi, R, Z, target, strength, max_change=0.15):
    """Match a learned li by a monotone relabeling inside the inferred LCFS.

    The smooth relabeling is zero on the boundary. The original exterior stays
    fixed. Every candidate is checked with the published functional; a failed
    extraction or a worse target match keeps the input field.
    """
    from nm1.guard import box_mask, _relabel
    from derive import internal_inductance, _bilinear, _points_in_polygon
    from lcfs import extract_lcfs
    from o_point import find_o_point
    mask = box_mask(R, Z)
    mf = mask.astype(float)
    C = extract_lcfs(psi, R, Z, "DIII-D", mask, mf, n_points=512)
    if C is None or not np.isfinite(target):
        return psi, {"status": "no_target"}
    li = internal_inductance(psi, R, Z, "DIII-D", C)
    if not np.isfinite(li) or not 0.1 < li < 5.2:
        return psi, {"status": "no_li"}
    delta = float(np.clip(strength * (target-li), -max_change, max_change))
    want = float(np.clip(li + delta, 0.15, 4.85))
    if abs(want-li) < 0.001:
        return psi, {"status": "small", "li": li}
    phi = -np.asarray(psi, np.float64)
    _, _, iz, ir, _ = find_o_point(phi, R, Z, mask)
    level = float(np.median(_bilinear(phi, R, Z, C[:, 0], C[:, 1])))
    span = float(phi[iz, ir]-level)
    if span <= 1e-8:
        return psi, {"status": "no_span"}
    rr, zz = np.meshgrid(R, Z)
    inside = _points_in_polygon(rr.ravel(), zz.ravel(), C).reshape(rr.shape)
    # Finite differences of the map make li a quadratic in alpha for a fixed
    # contour; fit that quadratic exactly, then validate with a new contour.
    def candidate(alpha):
        return _relabel(psi, -1.0, level, float(phi[iz, ir]), inside, alpha)
    h = 0.1
    lm = internal_inductance(candidate(-h), R, Z, "DIII-D", C)
    lp = internal_inductance(candidate(h), R, Z, "DIII-D", C)
    q = (lp+lm-2*li)/(2*h*h)
    b = (lp-lm)/(2*h)
    if not np.isfinite(q+b) or abs(b) < 1e-8:
        return psi, {"status": "no_derivative"}
    alpha = (want-li)/b
    for _ in range(4):
        alpha -= (q*alpha*alpha+b*alpha+li-want)/(b+2*q*alpha)
    alpha = float(np.clip(alpha, -0.25, 0.5))
    out = candidate(alpha)
    C2 = extract_lcfs(out, R, Z, "DIII-D", mask, mf, n_points=512)
    got = internal_inductance(out, R, Z, "DIII-D", C2)
    if not np.isfinite(got) or abs(got-want) >= abs(li-want):
        return psi, {"status": "rejected", "li": li}
    return out, {"status": "corrected", "li": li, "new_li": got,
                 "target": want, "alpha": alpha}


def move_axis(psi, R, Z, target, strength, max_change=0.012):
    """Move the interior O-point with a smooth local tilt, keeping the LCFS fixed.

    Uses the published subpixel axis functional in a two-dimensional Newton
    solve. No resampling is performed, so grid interpolation cannot blur li.
    """
    from nm1.guard import box_mask
    from derive import magnetic_axis, _bilinear, _points_in_polygon
    from lcfs import extract_lcfs
    mask = box_mask(R, Z)
    C = extract_lcfs(psi, R, Z, "DIII-D", mask, mask.astype(float), n_points=256)
    axis = np.asarray(magnetic_axis(psi, R, Z, "DIII-D", mask)[:2])
    if C is None or not np.isfinite(axis).all() or not np.isfinite(target).all():
        return psi
    desired = axis + np.clip(np.asarray(strength)*(target-axis),-max_change,max_change)
    if np.linalg.norm(desired-axis) < 1e-5: return psi
    phi = -np.asarray(psi,float)
    level = float(np.median(_bilinear(phi,R,Z,C[:,0],C[:,1])))
    peak = float(_bilinear(phi,R,Z,axis[:1],axis[1:])[0])
    if peak-level <= 1e-8: return psi
    rr,zz = np.meshgrid(R,Z)
    inside = _points_in_polygon(rr.ravel(),zz.ravel(),C).reshape(rr.shape)
    s = np.where(inside,np.clip((phi-level)/(peak-level),0,1),0)
    radial = np.exp(-0.5*((rr-axis[0])**2+(zz-axis[1])**2)/0.15**2)
    taper = radial*s*s*(3-2*s)
    bases = np.stack([(rr-axis[0])*taper,(zz-axis[1])*taper])
    amplitudes = np.zeros(2)
    step = 0.001
    for _ in range(3):
        field = psi+np.einsum('i,ijk->jk',amplitudes,bases)
        got = np.asarray(magnetic_axis(field,R,Z,"DIII-D",mask)[:2])
        if np.linalg.norm(got-desired) < 1e-5: break
        jac = np.zeros((2,2))
        for j in range(2):
            pos = np.asarray(magnetic_axis(field+step*bases[j],R,Z,"DIII-D",mask)[:2])
            neg = np.asarray(magnetic_axis(field-step*bases[j],R,Z,"DIII-D",mask)[:2])
            jac[:,j] = (pos-neg)/(2*step)
        if not np.isfinite(jac).all() or abs(np.linalg.det(jac)) < 1e-8: return psi
        amplitudes += np.linalg.solve(jac,desired-got)
        amplitudes = np.clip(amplitudes,-0.05,0.05)
    field = psi+np.einsum('i,ijk->jk',amplitudes,bases)
    got = np.asarray(magnetic_axis(field,R,Z,"DIII-D",mask)[:2])
    if np.linalg.norm(got-desired) >= np.linalg.norm(axis-desired): return psi
    return field


def morph_boundary(psi,R,Z,target,strength,max_displacement=0.008):
    """Small, smooth coordinate pushforward towards learned boundary geometry.

    The R/Z stretches hold area fixed when only elongation changes. Core flux
    is frozen. A second-order Taylor pullback avoids resampling blur, and a
    positive affine fit returns the map's level and amplitude to the original.
    """
    from nm1.guard import box_mask
    from derive import _shape_from_contour,magnetic_axis,_bilinear
    from lcfs import extract_lcfs
    from nm1.predict import projected_flux
    mask=box_mask(R,Z)
    C=extract_lcfs(psi,R,Z,"DIII-D",mask,mask.astype(float),n_points=512)
    if C is None: return psi
    shape=_shape_from_contour(C)
    if not np.isfinite([shape[k] for k in ("kappa","tri_top","tri_bot","volume")]).all():return psi
    rgeo=(C[:,0].max()+C[:,0].min())/2
    zgeo=(C[:,1].max()+C[:,1].min())/2
    a=np.ptp(C[:,0])/2;b=np.ptp(C[:,1])/2
    if min(a,b)<0.1:return psi
    deltas=np.asarray(strength)*np.array([target[2]-shape["kappa"],target[3]-shape["tri_top"],
                                        target[4]-shape["tri_bot"],target[5]-shape["volume"]])
    if not np.isfinite(deltas).all():return psi
    kr=float(np.clip(1+deltas[0]/shape["kappa"],0.96,1.04))
    vr=float(np.clip(1+deltas[3]/max(shape["volume"],1e-6),0.98,1.02))
    sr=np.sqrt(vr/kr);sz=np.sqrt(vr*kr)
    rr,zz=np.meshgrid(R,Z)
    uz=(zz-zgeo)/b
    dr=(sr-1)*(rr-rgeo)-a*(deltas[1]*np.maximum(uz,0)**2+deltas[2]*np.minimum(uz,0)**2)
    dz=(sz-1)*(zz-zgeo)
    phi=-np.asarray(psi,float)
    axis=magnetic_axis(psi,R,Z,"DIII-D",mask)
    peak=float(phi[axis[2],axis[3]])
    level=float(np.median(_bilinear(phi,R,Z,C[:,0],C[:,1])))
    if peak-level<1e-8:return psi
    u=(phi-level)/(peak-level)
    w=np.clip((0.6-u)/0.4,0,1);w=w*w*(3-2*w)
    radius=np.maximum(np.abs((rr-rgeo)/a),np.abs(uz))
    w*=np.exp(-(np.maximum(radius-1.2,0)/0.6)**4)
    dr=np.clip(dr*w,-max_displacement,max_displacement)
    dz=np.clip(dz*w,-max_displacement,max_displacement)
    pz,pr=np.gradient(np.asarray(psi,float),Z,R,edge_order=2)
    pzz,pzr=np.gradient(pz,Z,R,edge_order=2)
    _,prr=np.gradient(pr,Z,R,edge_order=2)
    result=np.asarray(psi,float)-dr*pr-dz*pz+0.5*(dr*dr*prr+2*dr*dz*pzr+dz*dz*pzz)
    return projected_flux(result[None],np.asarray(psi,float)[None],1.,1.)[0]


def match_boundary(psi, R, Z, target, strength, max_displacement=0.008):
    """Match four measured LCFS functionals with a local numerical response.

    The Jacobian is measured on this frame's own predicted flux. A line search
    accepts a field only when the published contour functionals move closer to
    the requested blend. No training or held-out target is read at inference.
    """
    from nm1.guard import box_mask
    from derive import _shape_from_contour, magnetic_axis, _bilinear
    from lcfs import extract_lcfs
    from nm1.predict import projected_flux
    p = np.asarray(psi, float)
    mask = box_mask(R, Z)
    mf = mask.astype(float)
    C = extract_lcfs(p, R, Z, "DIII-D", mask, mf, n_points=512)
    if C is None: return psi
    keys = ("kappa", "tri_top", "tri_bot", "volume")
    shape = _shape_from_contour(C)
    base = np.array([shape[k] for k in keys])
    if not np.isfinite(base).all(): return psi
    wanted = np.asarray(strength) * (np.asarray(target)[2:6] - base)
    wanted = np.clip(wanted, [-.04,-.03,-.03,-.15], [.04,.03,.03,.15])
    if not np.isfinite(wanted).all() or np.linalg.norm(wanted) < 1e-5: return psi
    rgeo = (C[:,0].max()+C[:,0].min())/2
    zgeo = (C[:,1].max()+C[:,1].min())/2
    a, b = np.ptp(C[:,0])/2, np.ptp(C[:,1])/2
    if min(a,b) < .1: return psi
    rr, zz = np.meshgrid(R,Z)
    uz = (zz-zgeo)/b
    phi = -p
    axis = magnetic_axis(p,R,Z,"DIII-D",mask)
    peak = float(phi[axis[2],axis[3]])
    level = float(np.median(_bilinear(phi,R,Z,C[:,0],C[:,1])))
    if peak-level < 1e-8: return psi
    u = (phi-level)/(peak-level)
    w = np.clip((.6-u)/.4,0,1);w=w*w*(3-2*w)
    radius = np.maximum(np.abs((rr-rgeo)/a),np.abs(uz))
    w *= np.exp(-(np.maximum(radius-1.2,0)/.6)**4)
    pz,pr = np.gradient(p,Z,R,edge_order=2)
    pzz,pzr = np.gradient(pz,Z,R,edge_order=2)
    _,prr = np.gradient(pr,Z,R,edge_order=2)
    def warp(d):
        kr = float(np.clip(1+d[0]/base[0],.96,1.04))
        vr = float(np.clip(1+d[3]/max(base[3],1e-6),.98,1.02))
        sr,sz = np.sqrt(vr/kr),np.sqrt(vr*kr)
        dr = (sr-1)*(rr-rgeo)-a*(d[1]*np.maximum(uz,0)**2+d[2]*np.minimum(uz,0)**2)
        dz = (sz-1)*(zz-zgeo)
        dr = np.clip(dr*w,-max_displacement,max_displacement)
        dz = np.clip(dz*w,-max_displacement,max_displacement)
        field = p-dr*pr-dz*pz+.5*(dr*dr*prr+2*dr*dz*pzr+dz*dz*pzz)
        return projected_flux(field[None],p[None],1.,1.)[0]
    def measured(field):
        contour = extract_lcfs(field,R,Z,"DIII-D",mask,mf,n_points=512)
        if contour is None: return None
        s = _shape_from_contour(contour)
        v = np.array([s[k] for k in keys])
        return v if np.isfinite(v).all() else None
    step = np.array([.005,.003,.003,.025])
    scale = np.array([.25,.25,.25,1.5])
    jac = np.empty((4,4))
    for j in range(4):
        d = np.zeros(4);d[j]=step[j]
        pos,neg = measured(warp(d)),measured(warp(-d))
        if pos is None or neg is None: return psi
        jac[:,j] = (pos-neg)/(2*scale)
    rhs = wanted/scale
    try:
        controls = step*np.linalg.solve(jac.T@jac+1e-7*np.eye(4),jac.T@rhs)
    except np.linalg.LinAlgError:
        return psi
    controls = np.clip(controls,[-.04,-.03,-.03,-.15],[.04,.03,.03,.15])
    goal = base+wanted
    best_error = float(np.dot(rhs,rhs))
    best = psi
    for fraction in (1.,.5,.25):
        field = warp(fraction*controls)
        got = measured(field)
        if got is None: continue
        error = float(np.sum(((got-goal)/scale)**2))
        if error < best_error*.999:
            best_error,best = error,field
    return best


def correct_shot(psi, R, Z, targets, strength, method="legacy", groups=None):
    """Apply the frozen correction recipe to an already guarded flux sequence."""
    strength = np.asarray(strength, float)
    targets = np.asarray(targets, float)
    if strength.shape != (7,) or targets.shape != (len(psi), 7):
        raise ValueError("Expected seven correction strengths and seven targets per frame")
    out = np.asarray(psi, np.float64).copy()
    if np.any(strength[2:6]):
        if method=="mixture":
            from nm1.v12_mix import mix_frame
            if groups is None:raise ValueError("The mixture correction needs five flux groups")
            for k in range(len(out)):
                out[k]=mix_frame(out[k],R,Z,targets[k],strength,np.asarray(groups)[:,k])
        else:
            boundary = match_boundary if method == "feedback" else morph_boundary
            for k in range(len(out)):
                out[k] = boundary(out[k], R, Z, targets[k], strength[2:6])
    if np.any(strength[:2]):
        for k in range(len(out)):
            out[k] = move_axis(out[k], R, Z, targets[k, :2], strength[:2])
    counts = {}
    if strength[6]:
        for k in range(len(out)):
            out[k], note = relabel_li(out[k], R, Z, targets[k, 6], strength[6])
            status = note["status"]
            counts[status] = counts.get(status, 0) + 1
    if not np.isfinite(out).all():
        raise ValueError("A field correction produced non-finite flux")
    return out.astype(np.float32), counts


def _field_job(args):
    from nm1.guard import guard_shot
    psi, R, Z, targets, strength, guard_params, method, groups = args
    # The experiment starts from the float32 submission field, guards it, then
    # rounds to float32 before the new correction. Preserve that same order.
    guarded, notes = guard_shot(np.asarray(psi, np.float32), R, Z, params=guard_params)
    out, counts = correct_shot(guarded.astype(np.float32), R, Z, targets, strength, method, groups)
    notes.append({"status": "field_correction", "li_status_counts": counts})
    return out, notes


from nm1.guard import ShotGuard


class FieldCorrectionPool(ShotGuard):
    """Guard and correct whole shots on CPUs while the GPU predicts inputs."""

    def __init__(self, strength, workers=6, depth=12, params=None, method="legacy"):
        super().__init__(workers=workers, depth=depth, params=params)
        self.strength = list(strength)
        self.method = method

    def submit(self, key, psi, R, Z, payload, targets, groups=None):
        args = (psi, R, Z, targets, self.strength, self.params, self.method, groups)
        self.pending.append((key, self.pool.submit(_field_job, args), payload))
        done = []
        while self.pending and (len(self.pending) > self.depth or self.pending[0][1].done()):
            k, fut, pl = self.pending.pop(0)
            psi_g, notes = fut.result()
            self.notes.extend({"shot": k, **n} for n in notes)
            done.append((k, psi_g, pl))
        return done