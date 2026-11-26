"""v13 field correction: ensemble-of-functionals shape targets, realized by re-mixing the flux groups.

The scorer derives kappa, the two triangularities and the volume from a contour it extracts at the
critical flux level, and the X-point end of that contour lands on the grid in a way that moves
in jumps as the saddle moves inside a cell. Evaluating the functional on each flux group's map and
averaging the results estimates the expected scored value better than evaluating it once on the
averaged map: on the held-out selection half kappa R2 0.98034 -> 0.98189, tri_bot 0.99490 ->
0.99551 (evaluation 0.97971 -> 0.98090, 0.99326 -> 0.99409). The v12 shape heads, blended in at
0.35, add to that (kappa 0.98293 / 0.98247).

The submitted map has to imply those values, since every scalar is derived from it. Each frame
is moved by a Gauss-Newton step in the span of its own flux groups' deviations from the
submitted map (the directions the ensemble is actually uncertain in), with the axis and the volume
held at their current values, then the axis is put back with v12's interior tilt and li is set by
the interior relabel the li guard uses (contours fixed). Nothing here reads a label: the
targets are the ensemble's and the heads' own outputs.
"""
from __future__ import annotations

import numpy as np

NAMES = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li")
GROUPS = ("dprod", "xd", "ctx", "cds", "cdsg", "prod", "ext", "dlq")
# training-target spreads used as the objective's units (R2 weights each scalar by 1/variance)
SCALES = np.array([0.0434, 0.0461, 0.1193, 0.2487, 0.2398, 1.5243, 0.26])

DEFAULT = {
    # per scalar: [source, temporal smoothing width (frames), head weight]
    "recipe": {"R_axis": ["base", 0.0, 0.0], "Z_axis": ["base", 0.0, 0.0],
               "kappa": ["groups", 0.0, 0.35], "tri_top": ["groups", 0.0, 0.35],
               "tri_bot": ["groups", 0.0, 0.35], "volume": ["base", 0.0, 0.35],
               "li": ["base", 0.5, 0.5]},
    # the axis is left free in the mixing step and put back by the tilt afterwards
    "objective_weights": [0, 0, 1, 1, 1, 1, 0],
    "strength": 1.0, "eps": 0.5, "iters": 2, "ridge": 1e-3, "clip": 0.4,
    "fractions": [1.0, 0.6, 0.3], "skip_below": 5e-4,
    "axis_strength": 1.0, "axis_max_change": 0.012,
    "li_strength": 1.0, "li_max_change": 0.15,
    # A frame's flux level moves no contour and no derived scalar. The dilation-trained q95 group
    # (dlq, flux weight 0 in the blend) predicts that level best of all groups, on all three
    # held-out parts: the level is set halfway between the projected blend's and dlq's.
    "offset_group": "dlq", "offset_weight": 0.5,
    "group_names": list(GROUPS),
}


def _grid_mask(R, Z):
    from nm1.guard import box_mask
    m = box_mask(R, Z)
    return m, m.astype(float)


def functionals(psi, R, Z, mask, mf, with_li=True):
    """The scorer's seven derived scalars of one frame (box mask, as everywhere in this project)."""
    from derive import derive_frame
    v = derive_frame(np.asarray(psi, float), R, Z, "DIII-D", mask, mf, with_li=with_li)
    return np.array([v[k] for k in NAMES], float)


def _measure(field, R, Z, mask, mf):
    from derive import _shape_from_contour, magnetic_axis
    from lcfs import extract_lcfs
    C = extract_lcfs(field, R, Z, "DIII-D", mask, mf, n_points=512)
    if C is None:
        return None
    s = _shape_from_contour(C)
    ax = magnetic_axis(field, R, Z, "DIII-D", mask)
    v = np.array([ax[0], ax[1], s["kappa"], s["tri_top"], s["tri_bot"], s["volume"]], float)
    return v if np.isfinite(v).all() else None


def _align(g, p):
    """A group map with the submitted map's frame mean and amplitude (its contours unchanged)."""
    mg, mp = g.mean(), p.mean()
    cg, cp = g - mg, p - mp
    amp = np.sum(cg * cp) / max(np.sum(cg * cg), 1e-20)
    if not np.isfinite(amp) or amp <= 0:
        amp = 1.0
    return mp + amp * cg


def _smooth(values, times, width):
    """Gaussian smoothing along the shot's own frame times (width in 20 ms frames)."""
    v = np.asarray(values, float)
    if width <= 0 or len(v) < 2:
        return v.copy()
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return v.copy()
    d = (np.asarray(times, float)[:, None] - np.asarray(times, float)[None, :]) / 0.02
    W = np.exp(-0.5 * (d / width) ** 2) * ok[None, :]
    W /= W.sum(1, keepdims=True)
    return W @ np.where(ok, v, 0.0)


def shot_targets(base_f, group_f, heads, times, recipe, target_stack=None):
    """(n, 7) targets from the submitted map's functionals, the group functionals and the heads.

    A scalar whose source is "stack" takes a linear combination of [base, group 1..G, head] fitted on the
    calibration third (``target_stack[name]`` = {mu, w, ym}; research/t3_target_stack.py)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns fall back below
        gm = np.nanmean(group_f, axis=0)
    gm = np.where(np.isfinite(gm), gm, base_f)
    out = np.empty_like(base_f)
    for j, name in enumerate(NAMES):
        source, width, w_head = recipe[name]
        if source == "stack":
            ts = target_stack[name]
            gj = np.where(np.isfinite(group_f[:, :, j]), group_f[:, :, j], base_f[None, :, j]).T      # (n, G)
            hj = np.asarray(heads, float)[:, j] if heads is not None else base_f[:, j]
            X = np.column_stack([base_f[:, j], gj, hj])
            if ts.get("features") == "disagreement":
                # group spread, blend minus group mean, position in the shot (frame index), |head - blend|
                n = len(base_f)
                X = np.column_stack([X, gj.std(1), base_f[:, j] - gj.mean(1),
                                     np.arange(n) / max(1, n - 1), np.abs(hj - base_f[:, j])])
            v = (X - np.asarray(ts["mu"], float)) @ np.asarray(ts["w"], float) + float(ts["ym"])
            v = _smooth(v, times, float(width))
            out[:, j] = np.where(np.isfinite(v), v, base_f[:, j])
            continue
        v = gm[:, j] if source == "groups" else base_f[:, j]
        v = _smooth(v, times, float(width))
        if w_head and heads is not None:
            v = (1 - w_head) * v + w_head * np.asarray(heads, float)[:, j]
        out[:, j] = np.where(np.isfinite(v), v, base_f[:, j])
    return out


def realize(psi, groups, target, cfg, R, Z, mask, mf):
    """One frame: Gauss-Newton over group-mixing weights toward the six contour/axis targets."""
    wts = np.asarray(cfg["objective_weights"][:6], float)
    t = np.asarray(target[:6], float)
    use = np.isfinite(t) & (wts > 0)
    p = np.asarray(psi, float)
    f0 = _measure(p, R, Z, mask, mf)
    if f0 is None or not use.any():
        return psi, {"status": "skip"}
    D = np.stack([_align(np.asarray(g, float), p) - p for g in groups])
    dn = np.sqrt(np.mean(D ** 2, axis=(1, 2))) + 1e-12
    goal = np.where(use, f0 + float(cfg["strength"]) * (t - f0), f0)
    scale = SCALES[:6]

    def objective(f):
        r = (f - goal) / scale
        return float(np.sum(wts * use * r * r))

    x, w, fx = p, np.zeros(len(groups)), f0
    best = objective(f0)
    first = best
    if best < float(cfg.get("skip_below", 0.0)):
        # the targets already sit on this frame (the quarter of frames below 5e-4 holds <1% of the gain)
        return psi, {"status": "small", "obj0": first, "obj": best}
    eps = float(cfg["eps"])
    for _ in range(int(cfg["iters"])):
        J = np.empty((6, len(groups)))
        for g in range(len(groups)):
            fg = _measure(x + eps * D[g], R, Z, mask, mf)
            if fg is None:
                return x, {"status": "partial", "obj0": first, "obj": best}
            J[:, g] = (fg - fx) / eps
        sw = np.sqrt(wts * use) / scale
        Jw = J * sw[:, None]
        rhs = (goal - fx) * sw
        A = Jw.T @ Jw + float(cfg["ridge"]) * np.diag((dn / dn.mean()) ** 2)
        try:
            dw = np.linalg.solve(A, Jw.T @ rhs)
        except np.linalg.LinAlgError:
            break
        dw = np.clip(w + dw, -cfg["clip"], cfg["clip"]) - w
        moved = False
        for frac in cfg["fractions"]:
            cand = x + frac * np.einsum("g,gzr->zr", dw, D)
            fc = _measure(cand, R, Z, mask, mf)
            if fc is None:
                continue
            o = objective(fc)
            if o < best * 0.999:
                best, x, fx, w = o, cand, fc, w + frac * dw
                moved = True
                break
        if not moved:
            break
    return x, {"status": "ok", "obj0": first, "obj": best, "w": w.tolist()}


def correct_shot(psi, R, Z, groups, heads, times, cfg=None):
    """Correct one (guarded) shot. ``groups`` is (G, n, 65, 65), ``heads`` (n, 7)."""
    from nm1.v12_shape import move_axis, relabel_li
    c = {**DEFAULT, **(cfg or {})}
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    mask, mf = _grid_mask(R, Z)
    psi = np.asarray(psi, np.float64)
    n = len(psi)
    base_f = np.stack([functionals(psi[k], R, Z, mask, mf) for k in range(n)])
    # only the contour scalars are read from the groups; their li is never used
    group_li = any(src in ("groups", "stack") for src, _, _ in [c["recipe"]["li"]])
    group_f = np.stack([[functionals(groups[g, k], R, Z, mask, mf, with_li=group_li)
                         for k in range(n)] for g in range(len(groups))])
    targets = shot_targets(base_f, group_f, heads, times, c["recipe"], c.get("target_stack"))
    level = psi.mean(axis=(1, 2))
    if c.get("offset_weight"):
        if len(c["group_names"]) != len(groups) or c["offset_group"] not in c["group_names"]:
            raise ValueError("offset_group must name one of the flux groups passed in")
        g = c["group_names"].index(c["offset_group"])
        level = level + float(c["offset_weight"]) * (np.asarray(groups[g], float).mean(axis=(1, 2)) - level)
    out = psi.copy()
    counts = {}
    for k in range(n):
        out[k], info = realize(out[k], groups[:, k], targets[k], c, R, Z, mask, mf)
        counts["mix_" + info["status"]] = counts.get("mix_" + info["status"], 0) + 1
        if c["axis_strength"] > 0:
            out[k] = move_axis(out[k], R, Z, targets[k, :2], [c["axis_strength"]] * 2,
                               c["axis_max_change"])
        if c["li_strength"] > 0:
            out[k], note = relabel_li(out[k], R, Z, targets[k, 6], c["li_strength"],
                                      c["li_max_change"])
            counts["li_" + note["status"]] = counts.get("li_" + note["status"], 0) + 1
    # last: the free per-frame level (a constant moves no contour, axis or li)
    out += (level - out.mean(axis=(1, 2)))[:, None, None]
    if not np.isfinite(out).all():
        raise ValueError("v13 correction produced non-finite flux")
    return out.astype(np.float32), counts, targets


def _field_job(args):
    """Worker: li guard (as every version since v7), then the v13 correction."""
    from nm1.guard import guard_shot
    psi, R, Z, heads, groups, times, guard_params, cfg = args
    guarded, notes = guard_shot(np.asarray(psi, np.float32), R, Z, params=guard_params)
    out, counts, _ = correct_shot(guarded.astype(np.float32), R, Z, np.asarray(groups, np.float64),
                                  heads, times, cfg)
    notes.append({"status": "field_correction", "counts": counts})
    return out, notes


from nm1.guard import ShotGuard  # noqa: E402


class V13CorrectionPool(ShotGuard):
    """Guard and correct whole shots on CPU workers while the GPU predicts the next ones."""

    def __init__(self, cfg, workers=6, depth=12, params=None, group_names=None):
        super().__init__(workers=workers, depth=depth, params=params)
        self.cfg = dict(cfg)
        if group_names is not None:
            self.cfg["group_names"] = list(group_names)

    def submit(self, key, psi, R, Z, payload, targets=None, groups=None, times=None):
        if targets is None or groups is None or times is None:
            raise ValueError("v13 correction needs head targets, flux groups and frame times")
        args = (psi, R, Z, targets, groups, times, self.params, self.cfg)
        self.pending.append((key, self.pool.submit(_field_job, args), payload))
        done = []
        while self.pending and (len(self.pending) > self.depth or self.pending[0][1].done()):
            k, fut, pl = self.pending.pop(0)
            psi_g, notes = fut.result()
            self.notes.extend({"shot": k, **n} for n in notes)
            done.append((k, psi_g, pl))
        return done