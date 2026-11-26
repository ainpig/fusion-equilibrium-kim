"""MAST leg: free-boundary equilibria from the measured coil currents, Ip and Thomson pressure.

No MAST equilibrium, no MAST archive and no MAST-trained weight is read anywhere (competition
rule 3).  The inputs are the shipped MAST parquet columns (coil currents, Ip, TF current, Thomson,
conductor and grid geometry); the solver (``nm1.gsfb``) is machine-agnostic and was validated on
DIII-D, whose labels are allowed.  Every constant below is in ``docs/CONSTANTS_LEDGER.md``.

    python -m nm1.mast --fold public --out <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nm1 import OUTPUTS, gsfb, gsread, inverse
from nm1.features import interp
from nm1.fetch import listing, local_path

MU0 = gsfb.MU0
COLUMNS = ["magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current", "magnetics_p3u_current",
           "magnetics_p4l_current", "magnetics_p4u_current", "magnetics_p5l_current", "magnetics_p5u_current",
           "magnetics_p6l_current", "magnetics_p6u_current", "magnetics_sol_current"]
INPUTS = (["efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time", "magnetics_plasma_current",
           "magnetics_tf_current"] + COLUMNS
          + ["thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
             "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial",
             "coil_input_column", "coil_R", "coil_Z"])

# ---- constants (each has a ledger entry) --------------------------------------------------------
CONFIG = dict(
    tf_conductors=24,          # machine spec: ampere-turns of the TF centre column = 24 x tf_current
    sol_scale=1.0,             # solenoid response; dataset documents "2x" as an open question
    box=(0.20, 1.90, -1.95, 1.95),   # region the plasma may occupy (grid extent minus a margin)
    keep_out=0.08,             # m, cells this close to a conductor are not part of the plasma region
    alpha=1.0, gamma=1.25,     # profile exponents P(x) = (1 - x^alpha)^gamma, DIII-D fit
    f_p=3.0,                   # total / electron pressure of this development-phase solver (not the submitted MAST leg)
    iters=100, iters_warm=40, relax=0.6,     # solver numerics (cold start / warm start)
    flux_dtype="float32",     # stored precision of the flux maps (float16 keeps relative precision and halves the file)
    shrink=1.0,                # c in [0, 1]: plasma flux = template + c (solver - template), template = median over the run
    q95_scale=1.0, betan_scale=1.0,   # output scale factors applied when the file is written (readout conventions)
    assim=False, assim_lr=0.08, assim_reg=0.02,   # Thomson isotherm-pair assimilation (variant)
    closure="static",          # "static": one (alpha, gamma) for every frame; "anchored": gamma fitted per frame to the Thomson anchors (nm1.inverse)
    anchor_use=(True, True),   # which anchors enter the fit: (axis = Te maximum, edge = Te 100 eV); the other one validates
    anchor_sigma=(0.06, 0.04), # m, assumed scatter of the two anchors
    gamma_sigma=0.35,          # prior width of log gamma around the previous frame's value
    refine_rounds=2,           # Gauss-Newton rounds per frame
    sol_post=1.0,              # solenoid flux multiplied after the solve (post-processing): psi += (sol_post - 1) x solenoid flux
    pf_post=1.0,               # PF-coil (P2..P6) flux multiplied after the solve: psi += (pf_post - 1) x PF flux
    template_gain=1.0,         # median plasma flux per ampere T multiplied after the solve: psi += (template_gain - 1) x Ip T
    q95_method="readout",      # "readout": toroidal-flux readout of the solve; "shape": ITER/Uckan shape formula on the LCFS of the written flux
    q95_shape_scale=1.0,       # factor on the shape formula (DIII-D calibration)
    bad_frames="previous",     # plasma flux of a frame that failed the good-frame rule: "previous" good frame scaled by Ip, or the run's "template" (Ip T)
    gamma_bounds=(0.8, 2.6),   # the fitted gamma stays inside this range
    refine_iters=20,           # Picard iterations after each gamma update
    col_scale=None,            # per input column response scale (calibration), None = all ones
    tol=8e-2, max_jump=0.25, min_area=150, max_area=2200,   # a frame counts as converged inside these bounds
    te_min=10.0, ne_min=1e17,  # Thomson point validity: Te [eV] and ne [m^-3] lower bounds
    psin_max=0.98, min_pts=5,  # pressure matching uses points with psi_N below psin_max, and only with at least min_pts of them
    vb_span=0.25, vb_n=26,     # vertical force balance: shift grid +-vb_span [m] with vb_n points (root linearly interpolated)
    axis_half=(0.45, 1.0),     # (R, Z) half-widths [m] of the axis search window around the previous axis
    axis_keep=False,           # a warm start's previous axis bounds the first axis search too (False: first search unbounded)
    assim_w=1.0,               # weight of the iso-flux pair mismatch against the prior pull (assimilation variant)
    smooth_s=0.0,              # Gaussian time scale [s] of the plasma flux per ampere and the scalars (0 = no smoothing)
    ts_rel=None,               # relative Thomson rules {"q", "hot", "support", "pair": [lo, hi]} (None: the absolute rules above);
                               #   q: peak Te = q-quantile of the frame's valid Te; hot: Te-weighted centroid of Te >= hot x peak is
                               #   the axis radius splitting the inboard / outboard pair branches (and the cold-start radius only
                               #   with "hot_start": True); support: only Te >= support x peak enters the pressure matching;
                               #   pair: equal-Te levels lo..hi x peak
    vnull_win=0.0,             # m: a cold start is placed at the height of the vacuum B_R null within +-vnull_win (0: at Z = 0)
    init_a_ts=False,           # cold-start blob width = Thomson half extent of Te > 50 eV, clamped 0.2-0.8 m
    ip_conserve=False,         # rescale the force-balance-shifted current to the measured Ip every iteration
    axis_adapt=False,          # double the axis search window where the maximum sits on its edge (at most twice)
    restart_failed=False,      # one restart of failed frames from the measurement-based start (hot centroid, null, a_ts)
    median_frames=0,           # running median over this many frames before the Gaussian smoothing
    q95_clip=None,             # [lo, hi] physical range of the written q95
)
TIME_UNIT_S = 1e-3             # efit_times / magnetics_time / thomson times of the shipped MAST files are in ms


def read_inputs(path) -> pd.Series:
    """Input columns only: the MAST test parquets carry no targets, and none is requested."""
    return pd.read_parquet(path, columns=INPUTS).iloc[0]


_G_CACHE: dict = {}


def machine(row, cfg=CONFIG, device="cuda") -> gsfb.Machine:
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    geom = {k: np.asarray(row[k]) for k in ("coil_input_column", "coil_R", "coil_Z")}
    key = (R.tobytes(), Z.tobytes(), device)
    if key not in _G_CACHE:
        _G_CACHE[key] = (gsfb.coil_basis(geom, COLUMNS, R, Z),
                         torch.as_tensor(gsfb.cell_green(R, Z), dtype=torch.float32, device=device))
    (psi0, br0), G = _G_CACHE[key]
    scale = np.ones(len(COLUMNS)) if cfg.get("col_scale") is None else np.asarray(cfg["col_scale"], float)
    scale = scale.copy()
    scale[-1] *= cfg["sol_scale"]
    psi, br = psi0 * scale[:, None, None], br0 * scale[:, None, None]
    cond = np.column_stack([np.asarray(row["coil_R"], float), np.asarray(row["coil_Z"], float)])
    return gsfb.Machine(R, Z, COLUMNS, psi, br, G, cond, tuple(cfg["box"]), keep_out=cfg["keep_out"], device=device)


def frame_inputs(row, te_min: float = 10.0, ne_min: float = 1e17, ts_rel: dict | None = None) -> dict:
    """Everything the solver needs at the EFIT times: currents [A], Ip [A], TF [A], Thomson points."""
    t = np.asarray(row["efit_times"], float)
    tm = np.asarray(row["magnetics_time"], float)
    cur = np.column_stack([interp(tm, row[c], t) for c in COLUMNS]).astype(np.float64) * 1000.0
    ip = interp(tm, row["magnetics_plasma_current"], t).astype(np.float64) * 1000.0
    tf = interp(tm, row["magnetics_tf_current"], t).astype(np.float64) * 1000.0
    R_all, TE_all, NE_all = [], [], []
    for system, rkey in (("core", "thomson_core_R"), ("edge", "thomson_edge_spatial")):
        try:
            R = np.asarray(row[rkey], float)
            tt = np.asarray(row[f"thomson_{system}_times"], float)
            Te = np.stack([np.asarray(p, float) for p in row[f"thomson_{system}_Te"]])
            ne = np.stack([np.asarray(p, float) for p in row[f"thomson_{system}_ne"]])
            if Te.shape[1] == len(R) + 1:                  # the shipped core grid has one ghost channel
                Te, ne = Te[:, 1:], ne[:, 1:]
            n = min(len(tt), len(Te))
            TE = np.full((len(t), len(R)), np.nan)
            NE = TE.copy()
            for j in range(len(R)):
                te, nn = np.interp(t, tt[:n], Te[:n, j]), np.interp(t, tt[:n], ne[:n, j])
                good = (te > te_min) & (nn > ne_min)
                TE[:, j], NE[:, j] = np.where(good, te, np.nan), np.where(good, nn, np.nan)
            R_all.append(np.tile(R, (len(t), 1)))
            TE_all.append(TE)
            NE_all.append(NE)
        except Exception:                                  # noqa: BLE001 - a missing subsystem
            continue
    Rt = np.concatenate(R_all, 1)
    TE, NE = np.concatenate(TE_all, 1), np.concatenate(NE_all, 1)
    pe = 1.602176634e-19 * TE * NE
    rhot = np.full(len(t), np.nan)
    rsplit = np.full(len(t), np.nan)
    # Thomson half extent of the points hotter than 50 eV -> cold-start blob width
    hot50 = np.where(np.isfinite(TE) & (TE > 50.0), Rt, np.nan)
    n50 = np.isfinite(hot50).sum(1)
    with np.errstate(all="ignore"):
        a_ts = np.where(n50 >= 5, 0.5 * (np.nanmax(hot50, 1) - np.nanmin(hot50, 1)), np.nan)
    if ts_rel:
        pairs = np.full((len(t), N_LEVELS, 2), np.nan)
        for k in range(len(t)):
            ok = np.isfinite(TE[k])
            if ok.sum() < 20:
                continue
            tpk = float(np.quantile(TE[k][ok], ts_rel["q"]))
            hot = ok & (TE[k] >= ts_rel["hot"] * tpk)
            rsplit[k] = float(np.sum(TE[k][hot] * Rt[k][hot]) / np.sum(TE[k][hot]))
            if ts_rel.get("hot_start", False):         # off in the recipes
                rhot[k] = rsplit[k]
            pe[k] = np.where(TE[k] >= ts_rel["support"] * tpk, pe[k], np.nan)
            pairs[k] = iso_pairs(Rt[k], TE[k], tpk=tpk, rpk=rsplit[k], levels=tuple(ts_rel["pair"]))
    else:
        pairs = np.stack([iso_pairs(Rt[k], TE[k]) for k in range(len(t))])
    return dict(t=t, cur=cur, ip=ip, tf=tf, Rt=Rt, pe=pe, pairs=pairs, te=TE, rhot=rhot, rsplit=rsplit, a_ts=a_ts,
                anch=inverse.frame_anchors(Rt, TE))


N_LEVELS = 12


def iso_pairs(R: np.ndarray, Te: np.ndarray, n_lev: int = N_LEVELS, tpk: float | None = None, rpk: float | None = None,
              levels: tuple = (0.15, 0.75)) -> np.ndarray:
    """``(n_lev, 2)`` major radii (inboard, outboard) of equal electron temperature on the midplane.

    Te is a flux function, so the two points of a pair must carry the same flux.  The split between
    the inboard and the outboard branch is the temperature maximum (smoothed) unless ``rpk`` is given,
    the peak is the smoothed maximum unless ``tpk`` is given, and the levels span ``levels`` x peak;
    NaN rows where a level is not crossed on both branches.
    """
    out = np.full((n_lev, 2), np.nan)
    ok = np.isfinite(Te)
    if ok.sum() < 20:
        return out
    r, te = R[ok], Te[ok]
    o = np.argsort(r)
    r, te = r[o], te[o]
    k = np.convolve(te, np.ones(5) / 5, mode="same")
    ipk = int(np.argmax(k[2:-2])) + 2
    if rpk is None or not np.isfinite(rpk):
        rpk = r[ipk]
    inb, outb = r < rpk - 0.03, r > rpk + 0.03
    if inb.sum() < 4 or outb.sum() < 4:
        return out
    ri, ti = r[inb], np.maximum.accumulate(te[inb])                 # non-decreasing towards the peak
    ro, to = r[outb], np.minimum.accumulate(te[outb])               # non-increasing away from it
    if tpk is None:
        tpk = float(k[ipk])
    for j, level in enumerate(np.linspace(levels[0] * tpk, levels[1] * tpk, n_lev)):
        if ti[0] < level < ti[-1] and to[0] > level > to[-1]:
            out[j, 0] = np.interp(level, ti + 1e-9 * np.arange(len(ti)), ri)
            out[j, 1] = np.interp(level, to[::-1] + 1e-9 * np.arange(len(to)), ro[::-1])
    return out


def _pad(arrs, fill=np.nan):
    """Stack ragged (T_s, ...) arrays into (S, T_max, ...)."""
    T = max(len(a) for a in arrs)
    out = np.full((len(arrs), T) + arrs[0].shape[1:], fill, dtype=np.float64)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def solve_sequences(M, prof, cfg, shots: list, device="cuda", log=None, diag: list | None = None) -> list:
    """All frames of ``shots``, solved in time order with warm starts, batched over shots.

    Frame k of every shot is solved together; its start is the previous frame's plasma flux scaled
    by the plasma-current ratio on top of this frame's own coil flux, so only the slow evolution of
    the equilibrium has to be iterated.  A frame whose iteration does not settle (a transient the
    quasi-static model cannot describe) keeps the previous good plasma current distribution,
    rescaled to its own Ip and placed in its own coil field; its scalars are carried forward.
    """
    dev = torch.device(device)
    S = len(shots)
    nz, nr = M.shape
    Tn = np.array([len(s["t"]) for s in shots])
    Tmax = int(Tn.max())
    npt = max(s["Rt"].shape[1] for s in shots)
    pad_pts = lambda a: np.pad(a, ((0, 0), (0, npt - a.shape[1])), constant_values=np.nan)   # noqa: E731
    Rt = _pad([pad_pts(s["Rt"]) for s in shots])
    pe = _pad([pad_pts(s["pe"]) for s in shots])
    pairs = _pad([s["pairs"] for s in shots])
    cur = _pad([s["cur"] for s in shots], 0.0)
    ip = _pad([s["ip"] for s in shots], 0.0)
    tf = _pad([s["tf"] for s in shots], 0.0)
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)   # noqa: E731
    out_psi = np.zeros((S, Tmax, nz, nr), np.float32)
    out_q = np.full((S, Tmax), np.nan, np.float32)
    out_b = np.full((S, Tmax), np.nan, np.float32)
    out_good = np.zeros((S, Tmax), bool)
    psi_p = torch.zeros((S, nz, nr), device=dev)                # last good plasma flux
    ip_prev = torch.zeros(S, device=dev)
    beta_prev = torch.full((S,), 0.2, device=dev)
    axis_prev = torch.zeros((S, 2), device=dev)
    have = torch.zeros(S, dtype=torch.bool, device=dev)
    raw_prev = torch.zeros((S, 4), device=dev)
    raw_prev[:] = torch.tensor([gsfb._unbounded(v, lo, hi) for v, (lo, hi) in zip(gsfb.THETA0, gsfb.THETA_RANGE)], device=dev)
    q_prev = torch.full((S,), 5.0, device=dev)
    b_prev = torch.full((S,), 1.0, device=dev)
    anchored = cfg.get("closure") == "anchored"
    # solver numerics shared by every solve call (CONFIG defaults)
    num = dict(vb_span=float(cfg.get("vb_span", 0.25)), vb_n=int(cfg.get("vb_n", 26)),
               axis_half=tuple(float(v) for v in cfg.get("axis_half", (0.45, 1.0))),
               p_psin_max=float(cfg.get("psin_max", 0.98)), p_min_pts=int(cfg.get("min_pts", 5)),
               ip_conserve=bool(cfg.get("ip_conserve", False)), axis_adapt=bool(cfg.get("axis_adapt", False)))
    use_ats = bool(cfg.get("init_a_ts", False))
    restart = bool(cfg.get("restart_failed", False)) and cfg.get("closure") != "anchored"
    ats = _pad([s.get("a_ts", np.full(len(s["t"]), np.nan)) for s in shots])
    rsp = _pad([s.get("rsplit", np.full(len(s["t"]), np.nan)) for s in shots])
    a_box = 0.25 * (M.box[1] - M.box[0])
    keep = bool(cfg.get("axis_keep", False))
    w_pair = float(cfg.get("assim_w", 1.0))
    vwin = float(cfg.get("vnull_win", 0.0))
    rhot = _pad([s.get("rhot", np.full(len(s["t"]), np.nan)) for s in shots])
    Zg = np.asarray(M.Z, float)
    Rg = np.asarray(M.R, float)

    def vacuum_null(br, rc):
        return vacuum_null_win(br, rc, vwin)

    n_restart = [0, 0]                                    # failed frames restarted, of which recovered

    def vacuum_null_win(br, rc, win):
        """Height of the vacuum B_R null nearest Z = 0 within +-win on the column at each rc; 0 where none."""
        out = np.zeros(len(rc))
        cols = np.abs(Rg[None, :] - rc[:, None]).argmin(1)
        inw = np.abs(Zg) <= win
        zz = Zg[inw]
        for i, c in enumerate(cols):
            b = br[i, inw, c]
            s = np.flatnonzero(np.sign(b[:-1]) * np.sign(b[1:]) <= 0)
            if len(s):
                m = s[np.argmin(np.abs(zz[s]))]
                out[i] = zz[m] - b[m] * (zz[m + 1] - zz[m]) / (b[m + 1] - b[m] + 1e-30)
        return out
    anch = _pad([s["anch"] for s in shots]) if anchored else None
    gamma_state = torch.full((S,), float(cfg["gamma"]), device=dev)
    out_gamma = np.full((S, Tmax), float(cfg["gamma"]), np.float32)
    for k in range(Tmax):
        act = np.flatnonzero(Tn > k)
        a_t = torch.as_tensor(act, device=dev)
        cur_k, ip_k, tf_k = f32(cur[act, k]), f32(np.abs(ip[act, k])), f32(np.abs(tf[act, k]))
        therm = dict(R=f32(Rt[act, k]), Z=f32(np.zeros_like(Rt[act, k])), p=f32(pe[act, k]))
        psi_vac, br_vac = M.external(cur_k)
        warm = have[a_t]
        for group, mask in (("warm", warm), ("cold", ~warm)):
            sel = torch.nonzero(mask).flatten()
            if not len(sel):
                continue
            gi = a_t[sel]
            ratio = (ip_k[sel] / ip_prev[gi].clamp_min(1.0)).clamp(0.5, 2.0)
            th = {kk: vv[sel] for kk, vv in therm.items()}
            pr = f32(pairs[act, k])[sel]
            prof_g = (inverse.BatchProfile(torch.full((len(sel),), float(cfg["alpha"]), device=dev), gamma_state[gi].clone(), cfg["f_p"])
                      if anchored else prof)
            if group == "warm":
                init = psi_vac[sel] + psi_p[gi] * ratio[:, None, None]
                if cfg.get("assim"):
                    res = gsfb.solve_assim(M, cur_k[sel], ip_k[sel], th, pr, f_p=cfg["f_p"], init_psi=init,
                                           init_center=(axis_prev[gi, 0], axis_prev[gi, 1]), iters=cfg["iters_warm"],
                                           relax=cfg["relax"], init_theta=raw_prev[gi], lr=cfg["assim_lr"], reg=cfg["assim_reg"],
                                           w_pair=w_pair, **num)
                else:
                    res = gsfb.solve(M, cur_k[sel], ip_k[sel], th, prof_g, init_psi=init,
                                     init_center=(axis_prev[gi, 0], axis_prev[gi, 1]), iters=cfg["iters_warm"],
                                     relax=cfg["relax"], init_beta0=beta_prev[gi], keep_center=keep, **num)
            else:
                pm = torch.nan_to_num(th["p"], nan=0.0)
                rc0 = th["R"].gather(1, pm.argmax(1, keepdim=True))[:, 0].clamp(0.5, 1.2)
                rh = torch.as_tensor(rhot[act, k], dtype=torch.float32, device=dev)[sel]
                rc0 = torch.where(torch.isfinite(rh), rh.clamp(0.5, 1.2), rc0)   # Thomson hot-part centroid when given
                zc0 = torch.zeros(len(sel), device=dev)
                if vwin > 0:
                    zc0 = torch.as_tensor(vacuum_null(br_vac[sel].cpu().numpy(), rc0.cpu().numpy()), dtype=torch.float32, device=dev)
                ia = None
                if use_ats:                                   # cold-start blob width from the Thomson half extent
                    av = torch.as_tensor(ats[act, k], dtype=torch.float32, device=dev)[sel]
                    ia = torch.where(torch.isfinite(av), av.clamp(0.2, 0.8), torch.full_like(av, a_box))
                if cfg.get("assim"):
                    res = gsfb.solve_assim(M, cur_k[sel], ip_k[sel], th, pr, f_p=cfg["f_p"],
                                           init_center=(rc0, zc0), iters=cfg["iters"],
                                           relax=cfg["relax"], lr=cfg["assim_lr"], reg=cfg["assim_reg"], w_pair=w_pair,
                                           init_a=ia, **num)
                else:
                    res = gsfb.solve(M, cur_k[sel], ip_k[sel], th, prof_g,
                                     init_center=(rc0, zc0), iters=cfg["iters"],
                                     relax=cfg["relax"], init_a=ia, **num)
            if anchored:
                a_k = anch[act, k][sel.cpu().numpy()]
                for _ in range(cfg["refine_rounds"]):
                    usable = (res.residual < 2 * cfg["tol"]) & (res.inside.sum((1, 2)) >= cfg["min_area"])
                    if not bool(usable.any()):
                        break
                    g_new, _info = inverse.refine(M, cur_k[sel], ip_k[sel], th, res, prof_g.alpha, prof_g.gamma, cfg["f_p"], cfg["relax"], a_k,
                                                  prior_mu=torch.log(gamma_state[gi]), prior_sigma=cfg["gamma_sigma"],
                                                  sigma=tuple(cfg["anchor_sigma"]), use=tuple(bool(u) for u in cfg["anchor_use"]))
                    g_new = torch.where(usable, g_new.clamp(*cfg["gamma_bounds"]), prof_g.gamma)
                    prof_g = inverse.BatchProfile(prof_g.alpha, g_new.detach(), cfg["f_p"])
                    res = gsfb.solve(M, cur_k[sel], ip_k[sel], th, prof_g, init_psi=res.psi, init_center=(res.Rax, res.Zax),
                                     iters=cfg["refine_iters"], relax=cfg["relax"], init_beta0=res.beta0, keep_center=keep, **num)
            F0 = MU0 * cfg["tf_conductors"] * tf_k[sel] / (2 * np.pi)

            def judge(res, gi, F0, ipk, prof):
                """Readout, closed area, axis jump and the good-frame rule of one batch of solutions."""
                def p_of_x(x, res=res, prof=prof):
                    if hasattr(res, "theta"):                         # assimilated: p(x) = span * Ap * int_x^1 Pp
                        ap, gp = res.theta[:, 0], res.theta[:, 1]
                        xs = (torch.arange(64, device=x.device) + 0.5) / 64
                        tab = (1 - xs[None] ** ap[:, None]).clamp_min(1e-6) ** gp[:, None]
                        T_tab = torch.flip(torch.cumsum(torch.flip(tab, [1]), 1), [1]) / 64
                        B_ = x.shape[0]
                        idx = (x.reshape(B_, -1) * 64).clamp(0, 62.999)
                        i0 = idx.floor().long()
                        v = torch.gather(T_tab, 1, i0) * (1 - (idx - i0)) + torch.gather(T_tab, 1, i0 + 1) * (idx - i0)
                        return (((res.psi_ax - res.psi_b) * res.beta0)[:, None] * v).reshape(x.shape)
                    q = res.lam * res.beta0 * (res.psi_ax - res.psi_b) / res.Rax
                    return q.view(-1, *([1] * (x.dim() - 1))) * prof.tail(x)
                rd = gsread.readout(M, res, F0, ipk, p_of_x)
                area = res.inside.sum((1, 2))
                jump = ((res.Rax - axis_prev[gi, 0]) ** 2 + (res.Zax - axis_prev[gi, 1]) ** 2).sqrt()
                jump = torch.where(have[gi], jump, torch.zeros_like(jump))
                good = ((res.residual < cfg["tol"]) & (jump < cfg["max_jump"]) & (area >= cfg["min_area"]) & (area <= cfg["max_area"])
                        & torch.isfinite(rd["q95"]) & torch.isfinite(rd["betaN"]) & (rd["q95"] > 1.0)
                        & (rd["betaN"] < 15.0) & torch.isfinite(res.psi).all(2).all(1))
                return rd, area, jump, good

            rd, area, jump, good = judge(res, gi, F0, ip_k[sel], prof_g)
            if restart and bool((~good).any()):
                # one restart of the failed frames from a measurement-based start: R = Thomson hot-part centroid
                # (pressure maximum where none), Z = vacuum B_R null, blob width = Thomson half extent; the frame
                # takes the restart only where the restart passes the same good-frame rule
                bad = torch.nonzero(~good).flatten()
                gb, thb = gi[bad], {kk: vv[bad] for kk, vv in th.items()}
                pmb = torch.nan_to_num(thb["p"], nan=0.0)
                rcb = thb["R"].gather(1, pmb.argmax(1, keepdim=True))[:, 0].clamp(0.5, 1.2)
                rs = torch.as_tensor(rsp[act, k], dtype=torch.float32, device=dev)[sel][bad]
                rcb = torch.where(torch.isfinite(rs), rs.clamp(0.5, 1.2), rcb)
                vw = vwin if vwin > 0 else num["vb_span"]
                zcb = torch.as_tensor(vacuum_null_win(br_vac[sel][bad].cpu().numpy(), rcb.cpu().numpy(), vw),
                                      dtype=torch.float32, device=dev)
                av = torch.as_tensor(ats[act, k], dtype=torch.float32, device=dev)[sel][bad]
                iab = torch.where(torch.isfinite(av), av.clamp(0.2, 0.8), torch.full_like(av, a_box))
                if cfg.get("assim"):
                    res2 = gsfb.solve_assim(M, cur_k[sel][bad], ip_k[sel][bad], thb, pr[bad], f_p=cfg["f_p"],
                                            init_center=(rcb, zcb), iters=cfg["iters"], relax=cfg["relax"], lr=cfg["assim_lr"],
                                            reg=cfg["assim_reg"], w_pair=w_pair, init_a=iab, **num)
                else:
                    res2 = gsfb.solve(M, cur_k[sel][bad], ip_k[sel][bad], thb, prof_g, init_center=(rcb, zcb),
                                      iters=cfg["iters"], relax=cfg["relax"], init_a=iab, **num)
                rd2, area2, jump2, good2 = judge(res2, gb, F0[bad], ip_k[sel][bad], prof_g)
                if bool(good2.any()):
                    idx, ok = bad[good2], good2
                    for name in ("psi", "inside", "psi_n", "psi_ax", "psi_b", "Rax", "Zax", "Rx", "Zx", "found_x",
                                 "beta0", "lam", "i_ctrl", "residual", "J"):
                        getattr(res, name)[idx] = getattr(res2, name)[ok].to(getattr(res, name).dtype)
                    for name in ("theta", "raw"):
                        if hasattr(res, name) and hasattr(res2, name):
                            getattr(res, name)[idx] = getattr(res2, name)[ok]
                    for kk, vv in rd.items():
                        if torch.is_tensor(vv) and vv.shape[:1] == good.shape:
                            vv[idx] = rd2[kk][ok].to(vv.dtype)
                    area[idx], jump[idx], good[idx] = area2[ok], jump2[ok], True
                n_restart[0] += int(len(bad)); n_restart[1] += int(good2.sum())
            plasma = res.psi - psi_vac[sel]
            fallback = psi_vac[sel] + psi_p[gi] * ratio[:, None, None]
            use_fb = (~good) & have[gi]
            psi_final = torch.where(good[:, None, None], res.psi, torch.where(use_fb[:, None, None], fallback, res.psi))
            gi_u = gi[good]
            if anchored:
                gamma_state[gi_u] = prof_g.gamma[good].detach()
            psi_p[gi_u] = plasma[good]
            ip_prev[gi_u] = ip_k[sel][good]
            beta_prev[gi_u] = res.beta0[good]
            axis_prev[gi_u, 0], axis_prev[gi_u, 1] = res.Rax[good], res.Zax[good]
            have[gi_u] = True
            if hasattr(res, "raw"):
                raw_prev[gi_u] = res.raw[good]
            q_prev[gi_u], b_prev[gi_u] = rd["q95"][good], rd["betaN"][good]
            q_f = torch.where(good, rd["q95"], q_prev[gi])
            b_f = torch.where(good, rd["betaN"], b_prev[gi])
            gi_np = gi.cpu().numpy()
            if diag is not None:
                z0 = int(np.abs(M.Z).argmin())
                row_in = res.inside[:, z0, :]
                r_ax_ = torch.as_tensor(M.R, device=dev, dtype=torch.float32)[None]
                r_out = torch.where(row_in, r_ax_, torch.full_like(row_in, float("nan"), dtype=torch.float32)).nan_to_num(nan=-1).amax(1)
                r_in = torch.where(row_in, r_ax_, torch.full_like(row_in, float("nan"), dtype=torch.float32)).nan_to_num(nan=9).amin(1)
                for j_, g_ in enumerate(gi_np):
                    diag.append(dict(shot=int(g_), k=k, group=group, res=float(res.residual[j_]), area=int(area[j_]), q95=float(rd['q95'][j_]),
                                     betaN=float(rd['betaN'][j_]), beta0=float(res.beta0[j_]), shift=float(res.i_ctrl[j_]), ax=(float(res.Rax[j_]), float(res.Zax[j_])),
                                     good=bool(good[j_]), ip=float(ip_k[sel][j_])/1e3, rout=float(r_out[j_]), rin=float(r_in[j_]),
                                     gamma=float(prof_g.gamma[j_]) if anchored else float(cfg["gamma"])))
            if anchored:
                out_gamma[gi_np, k] = prof_g.gamma.detach().cpu().numpy()
            out_psi[gi_np, k] = psi_final.cpu().numpy()
            out_q[gi_np, k], out_b[gi_np, k] = q_f.cpu().numpy(), b_f.cpu().numpy()
            out_good[gi_np, k] = good.cpu().numpy()
        if log and (k % 10 == 0 or k == Tmax - 1):
            log(f"    step {k + 1}/{Tmax}: {len(act)} shots, converged {int(out_good[act, k].sum())}")
    if log and restart:
        log(f"    restarts: {n_restart[1]} of {n_restart[0]} failed frames recovered from the measurement-based start")
    res_list = []
    for i in range(S):
        n = int(Tn[i])
        good = out_good[i, :n]
        psi, q, b = out_psi[i, :n], out_q[i, :n], out_b[i, :n]
        if not good.any():
            q[:], b[:] = 5.0, 1.0
        elif not good[0]:
            first = int(np.flatnonzero(good)[0])
            psi[:first] = psi[first]
            q[:first], b[:first] = q[first], b[first]
        res_list.append(dict(psi=psi, q95=q, betaN=b, good=good, gamma=out_gamma[i, :n]))
    return res_list


def _vac_and_ip(M, row):
    inp = frame_inputs(row)
    dev = torch.device(M.device)
    vac, _ = M.external(torch.as_tensor(inp["cur"], dtype=torch.float32, device=dev))
    return vac.cpu().numpy(), np.maximum(np.abs(inp["ip"]), 1e4)


def _sol_field(M, row) -> np.ndarray:
    """Flux of the solenoid column alone at every EFIT time, ``(T, nz, nr)``."""
    return _coil_fields(M, row)[1]


def _coil_fields(M, row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(PF-coil flux, solenoid flux, Ip)`` at every EFIT time; the two fluxes are ``(T, nz, nr)``."""
    inp = frame_inputs(row)
    cur = torch.as_tensor(inp["cur"], dtype=torch.float32, device=torch.device(M.device))
    pf = torch.einsum("bc,czr->bzr", cur[:, :-1], M.t_psi_cols[:-1])
    sol = cur[:, -1, None, None] * M.t_psi_cols[-1][None]
    return pf.cpu().numpy(), sol.cpu().numpy(), np.maximum(np.abs(inp["ip"]), 1e4)


def plasma_template(M, paths, chunk_dir, ident, every: int = 12, max_frames: int = 6000) -> np.ndarray:
    """Median over the run's good frames of the plasma flux per ampere of plasma current.

    Built from this run's own solver output and the shipped inputs only (no equilibrium data): it
    is the "typical plasma" the shrinkage pulls the noisy per-frame solutions toward.
    """
    acc = []
    for j in range(0, len(paths), every):
        psi = np.load(chunk_dir / f"s{j:04d}_psi_{ident}.npy")
        sc = np.load(chunk_dir / f"s{j:04d}_sc_{ident}.npy")
        good = sc[2] > 0.5
        if not good.any():
            continue
        vac, ip = _vac_and_ip(M, read_inputs(paths[j]))
        acc.append(((psi - vac) / ip[:, None, None])[good])
    cat = np.concatenate(acc)
    if len(cat) > max_frames:
        cat = cat[np.linspace(0, len(cat) - 1, max_frames).astype(int)]
    return np.median(cat, axis=0).astype(np.float32)


def _q95_shape_job(args):
    """q95 from the LCFS of our own flux: 5 a^2 B0 / (R0 Ip[MA]) (1 + k^2 (1 + 2 d^2 - 1.2 d^3)) / 2 (1.17 - 0.65 e) / (1 - e^2)^2.

    The LCFS comes from the official extraction code run on our prediction inside our own region (the
    solver box of CONFIG), never the scorer's MAST envelope mask.  NaN where no LCFS is found."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fusion_scoring"))
    from lcfs import extract_lcfs
    psi, R, Z, box, tf_conductors, tf, ip = args
    rr, zz = np.meshgrid(R, Z)
    m = (rr >= box[0]) & (rr <= box[1]) & (zz >= box[2]) & (zz <= box[3])
    out = np.full(len(psi), np.nan)
    for k in range(len(psi)):
        C = extract_lcfs(np.asarray(psi[k], np.float64), R, Z, "MAST", m, m.astype(float), n_points=256)
        if C is None:
            continue
        Rc, Zc = C[:, 0], C[:, 1]
        a = 0.5 * (Rc.max() - Rc.min()); R0 = 0.5 * (Rc.max() + Rc.min())
        if a <= 0 or R0 <= a:
            continue
        kap = (Zc.max() - Zc.min()) / (2 * a)
        d = 0.5 * ((R0 - Rc[np.argmax(Zc)]) + (R0 - Rc[np.argmin(Zc)])) / a
        e = a / R0
        B0 = MU0 * tf_conductors * abs(tf[k]) / (2 * np.pi) / R0
        out[k] = (5 * a ** 2 * B0 / (R0 * abs(ip[k]) / 1e6) * (1 + kap ** 2 * (1 + 2 * d ** 2 - 1.2 * d ** 3)) / 2
                  * (1.17 - 0.65 * e) / (1 - e ** 2) ** 2)
    return out


def shrink_shot(M, row, psi, template, c: float, bad: np.ndarray | None = None) -> np.ndarray:
    """psi = vac + Ip (T + c (plasma/Ip - T)): the noisy part of the solution is damped by c.

    ``bad`` (optional, per frame): frames that failed the good-frame rule take the template itself
    (on DIII-D labels this beats re-using the previous good frame)."""
    vac, ip = _vac_and_ip(M, row)
    plasma = (psi - vac) / ip[:, None, None]
    mix = template[None] + c * (plasma - template[None])
    if bad is not None:
        mix[bad] = template
    return (vac + ip[:, None, None] * mix).astype(np.float32)


def smooth_shot(M, row, psi, sc, sigma_s: float, median: int = 0):
    """Time filtering of the plasma flux per ampere and of q95 / beta_N: first a running median over ``median``
    frames (odd; removes isolated failed frames), then a Gaussian of time scale ``sigma_s`` [s].

    The coil (vacuum) flux and Ip stay exact frame by frame: only the solved plasma part, (psi - vac) / Ip, and the
    two solver scalars are filtered over neighbouring EFIT frames (Gaussian weights exp(-dt^2 / 2 sigma^2), cut at
    3 sigma, renormalised at the shot ends; the median window is shortened at the shot ends)."""
    vac, ip = _vac_and_ip(M, row)
    t = np.asarray(row["efit_times"], float) * TIME_UNIT_S
    plasma = ((psi - vac) / ip[:, None, None]).reshape(len(t), -1)
    sc = sc.copy()
    if median and median > 1 and len(t) >= 3:
        h = median // 2
        win = [slice(max(0, i - h), min(len(t), i + h + 1)) for i in range(len(t))]
        plasma = np.stack([np.median(plasma[s], 0) for s in win])
        sc[0] = np.array([np.median(sc[0][s]) for s in win])
        sc[1] = np.array([np.median(sc[1][s]) for s in win])
    if sigma_s and sigma_s > 0:
        d = (t[:, None] - t[None, :]) / sigma_s
        w = np.where(np.abs(d) <= 3.0, np.exp(-0.5 * d ** 2), 0.0)
        w /= w.sum(1, keepdims=True)
        plasma = w @ plasma
        sc[0], sc[1] = w @ sc[0], w @ sc[1]
    psi = (vac + ip[:, None, None] * plasma.reshape(psi.shape)).astype(np.float32)
    return psi, sc


def build(out: Path, fold: str = "public", limit: int = 0, cfg=CONFIG, prefetch: int = 4, chunk: int = 300,
          chunk_ident: str | None = None, chunks_from: Path | None = None, device: str = "cuda") -> Path:
    from nm1.input_policy import install
    from nm1.predict import NpzStreamWriter
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["FUSION_RUN_ROOT"] = str(out.resolve())
    policy = install(out)
    config = f"mast_{fold}_test"
    files = listing(config)
    if limit:
        files = files[:limit]
    paths = [local_path(f) for f in files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(paths)} MAST parquets are missing; run nm1.fetch --config {config}")
    row0 = read_inputs(paths[0])
    M = machine(row0, cfg, device=device)
    prof = gsfb.Profile(cfg["alpha"], cfg["gamma"], cfg["f_p"])
    import hashlib
    chunk_dir = Path(chunks_from) / "chunks" if chunks_from else out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    solver_cfg = {k: v for k, v in cfg.items() if k not in ("q95_scale", "betan_scale", "shrink", "flux_dtype", "sol_post", "pf_post", "template_gain", "bad_frames",
                                                         "q95_method", "q95_shape_scale", "smooth_s", "median_frames",
                                                         "q95_clip")}   # output steps do not change the solve
    ident = chunk_ident or hashlib.sha256(json.dumps([solver_cfg, len(paths), chunk], sort_keys=True, default=str).encode()).hexdigest()[:16]
    started, frames, good_frames = time.perf_counter(), 0, 0

    def log(msg):
        print(msg, flush=True)

    # Each chunk of shots is solved once and written as plain .npy files; a run that dies (a GPU driver
    # reset is the usual cause) is restarted with the same command and picks the finished chunks up.
    for c0 in range(0, len(paths), chunk):
        idx = list(range(c0, min(len(paths), c0 + chunk)))
        done_flag = chunk_dir / f"chunk_{c0:05d}_{ident}.done"
        if done_flag.exists():
            log(f"  chunk {c0}: already solved, skipping")
            continue
        with ThreadPoolExecutor(max_workers=prefetch) as pool:
            shots = list(pool.map(lambda j: frame_inputs(read_inputs(paths[j]), cfg.get("te_min", 10.0), cfg.get("ne_min", 1e17),
                                                         cfg.get("ts_rel")), idx))
        results = solve_sequences(M, prof, cfg, shots, device=device, log=log)
        for j, r in zip(idx, results):
            np.save(chunk_dir / f"s{j:04d}_psi_{ident}.npy", r["psi"].astype(np.float32))
            np.save(chunk_dir / f"s{j:04d}_sc_{ident}.npy", np.stack([r["q95"], r["betaN"], r["good"].astype(np.float32), r["gamma"]]).astype(np.float32))
        done_flag.write_text("ok")
        log(f"  {idx[-1] + 1}/{len(paths)} shots solved, {(time.perf_counter() - started) / 60:.1f} min")
    template = None
    to_template = cfg["bad_frames"] == "template"
    if cfg["shrink"] < 1.0 or cfg["template_gain"] != 1.0 or to_template:
        template = plasma_template(M, paths, chunk_dir, ident)
    writer = NpzStreamWriter(out / f"{config}.npz")
    shape_q = cfg["q95_method"] == "shape"
    pool = None
    if shape_q:
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor
        n_workers = max(1, (os.cpu_count() or 2) // 2)
        pool = ProcessPoolExecutor(n_workers)
        queue = deque()

    def emit(j, psi, sc, q95):
        nonlocal frames, good_frames
        if cfg.get("q95_clip"):                                  # physical range of q95
            q95 = np.clip(q95, float(cfg["q95_clip"][0]), float(cfg["q95_clip"][1])).astype(np.float32)
        writer.write(f"shot_{j:04d}_psirz", psi.astype(cfg["flux_dtype"]))
        writer.write(f"shot_{j:04d}_q95", q95 * np.float32(cfg["q95_scale"]))
        writer.write(f"shot_{j:04d}_betaN", sc[1] * np.float32(cfg["betan_scale"]))
        frames += len(sc[0])
        good_frames += int(sc[2].sum())

    def emit_shape(item):
        j, psi, sc, fut = item
        qs = fut.result()
        q95 = np.where(np.isfinite(qs) & (qs > 1.0) & (qs < 50.0), np.float32(cfg["q95_shape_scale"]) * qs,
                       sc[0]).astype(np.float32)
        emit(j, psi, sc, q95)

    try:
        for j in range(len(paths)):
            psi = np.load(chunk_dir / f"s{j:04d}_psi_{ident}.npy")
            sc = np.load(chunk_dir / f"s{j:04d}_sc_{ident}.npy")
            if cfg["shrink"] < 1.0 or to_template:
                psi = shrink_shot(M, read_inputs(paths[j]), psi, template, cfg["shrink"],
                                  bad=(sc[2] < 0.5) if to_template else None)
            if cfg.get("smooth_s", 0.0) > 0.0 or int(cfg.get("median_frames", 0)) > 1:
                psi, sc = smooth_shot(M, read_inputs(paths[j]), psi, sc, float(cfg.get("smooth_s", 0.0)),
                                      int(cfg.get("median_frames", 0)))
            if cfg["sol_post"] != 1.0 or cfg["pf_post"] != 1.0 or cfg["template_gain"] != 1.0:
                # linear post-scaling of fixed fields (solenoid x328 in the recipe; the PF / template gains stay 1.0)
                pf, sol, ipa = _coil_fields(M, read_inputs(paths[j]))
                psi = (psi + (cfg["sol_post"] - 1.0) * sol + (cfg["pf_post"] - 1.0) * pf
                       + (cfg["template_gain"] - 1.0) * ipa[:, None, None] * template[None]).astype(np.float32)
            if not shape_q:
                emit(j, psi, sc, sc[0])
                continue
            # the LCFS-based q95 runs in worker processes; shots are written in order through a bounded window
            inp = frame_inputs(read_inputs(paths[j]))
            fut = pool.submit(_q95_shape_job, (psi.astype(np.float64), np.asarray(M.R, float), np.asarray(M.Z, float),
                                               cfg["box"], cfg["tf_conductors"], inp["tf"], inp["ip"]))
            queue.append((j, psi, sc, fut))
            while len(queue) > 2 * n_workers:
                emit_shape(queue.popleft())
        while shape_q and queue:
            emit_shape(queue.popleft())
    finally:
        writer.close()
        if pool is not None:
            pool.shutdown()
    if policy.report["denied"] or policy.report["network_attempts"]:
        raise PermissionError("MAST run read outside its declared inputs")
    (out / f"mast_run_{fold}.json").write_text(json.dumps(
        {"config": cfg, "shots": len(paths), "frames": frames, "converged_fraction": good_frames / max(frames, 1),
         "seconds": time.perf_counter() - started, "input_policy": policy.report}, indent=2, default=list),
        encoding="utf-8")
    policy.save()
    return out / f"{config}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", choices=("public", "private"), default="public")
    ap.add_argument("--out", type=Path, default=OUTPUTS / f"mast_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--recipe", type=Path, help="a recipe file: its config first, then the options below")
    ap.add_argument("--chunks-from", type=Path, help="output dir of an earlier run whose chunks are re-assembled")
    ap.add_argument("--chunk-ident", help="reuse the finished chunks of a run with this identity (re-assemble with other output options)")
    ap.add_argument("--f-p", type=float)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--sol-scale", type=float)
    ap.add_argument("--q95-scale", type=float)
    ap.add_argument("--shrink", type=float)
    ap.add_argument("--bad-frames", choices=("previous", "template"))
    ap.add_argument("--q95-method", choices=("readout", "shape"))
    ap.add_argument("--q95-shape-scale", type=float)
    ap.add_argument("--flux-dtype", choices=("float16", "float32"))
    ap.add_argument("--betan-scale", type=float)
    ap.add_argument("--assim", action="store_true")
    ap.add_argument("--closure", choices=("static", "anchored"))
    ap.add_argument("--sol-post", type=float, help="multiply the solenoid flux after the solve")
    ap.add_argument("--pf-post", type=float, help="multiply the PF-coil flux after the solve")
    ap.add_argument("--template-gain", type=float, help="multiply the median plasma flux per ampere after the solve")
    ap.add_argument("--anchor-use", help="axis,edge as 0/1, e.g. 1,1")
    ap.add_argument("--anchor-sigma", help="axis,edge scatter in m, e.g. 0.05,0.05")
    ap.add_argument("--col-scale", help="response scale per coil group P2,P3,P4,P5,P6,SOL (six comma-separated numbers)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu leaves the GPU to a concurrent training run)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for --device cpu (0: torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = dict(CONFIG)
    chunk = a.chunk
    if a.recipe:
        recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
        cfg.update(recipe["config"])
        chunk = recipe.get("chunk", chunk)
    if a.closure:
        cfg["closure"] = a.closure
    if a.sol_post is not None:
        cfg["sol_post"] = a.sol_post
    if a.pf_post is not None:
        cfg["pf_post"] = a.pf_post
    if a.template_gain is not None:
        cfg["template_gain"] = a.template_gain
    if a.anchor_use:
        cfg["anchor_use"] = tuple(bool(int(v)) for v in a.anchor_use.split(","))
    if a.anchor_sigma:
        cfg["anchor_sigma"] = tuple(float(v) for v in a.anchor_sigma.split(","))
    if a.col_scale:
        g = [float(v) for v in a.col_scale.split(",")]
        if len(g) != 6:
            raise SystemExit("--col-scale needs six numbers: P2,P3,P4,P5,P6,SOL")
        cfg["col_scale"] = [g[0], g[0], g[1], g[1], g[2], g[2], g[3], g[3], g[4], g[4], g[5]]
    if a.f_p is not None:
        cfg["f_p"] = a.f_p
    if a.alpha is not None:
        cfg["alpha"] = a.alpha
    if a.gamma is not None:
        cfg["gamma"] = a.gamma
    if a.sol_scale is not None:
        cfg["sol_scale"] = a.sol_scale
    if a.q95_scale is not None:
        cfg["q95_scale"] = a.q95_scale
    if a.shrink is not None:
        cfg["shrink"] = a.shrink
    if a.bad_frames is not None:
        cfg["bad_frames"] = a.bad_frames
    if a.q95_method is not None:
        cfg["q95_method"] = a.q95_method
    if a.q95_shape_scale is not None:
        cfg["q95_shape_scale"] = a.q95_shape_scale
    if a.flux_dtype:
        cfg["flux_dtype"] = a.flux_dtype
    if a.betan_scale is not None:
        cfg["betan_scale"] = a.betan_scale
    if a.assim:
        cfg["assim"] = True
    print(build(a.out, a.fold, a.limit, cfg, chunk=chunk, chunk_ident=a.chunk_ident, chunks_from=a.chunks_from, device=a.device))


if __name__ == "__main__":
    main()
