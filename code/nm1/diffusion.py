"""A 1-D resistive current-diffusion simulation of each discharge, as an input block.

Why: the members' `li` errors are concentrated where the current profile is out of equilibrium
with the plasma -- current ramp-down and the end of a discharge carry 43-51% of `li`'s residual
on held-out shots -- and they are anti-correlated with the `betaN` error (-0.41 to -0.47) and
the `R_axis` error (-0.19 to -0.23). That is the signature of one unresolved quantity: the coil
currents fix the vertical field and therefore Lambda = beta_p + li/2, and the model does not know
how to split it. The current profile is set by the *history* of the discharge -- resistive
diffusion with a time constant of ~1 s at flat-top temperatures -- and a frame's own inputs do
not carry it.

So each shot is run through the poloidal-field diffusion equation, in a cylinder of fixed shape,
with the plasma current as the boundary condition and a Spitzer-neoclassical resistivity built
from the Thomson electron temperature:

    dI/dt = x d/dx[ (eta / x) dI/dx ] / (mu0 a^2),    I(0) = 0,  I(1) = Ip(t)

where ``I(x, t)`` is the current inside normalised radius ``x``. Sawteeth are a flattening of the
current density inside the q = 1 radius. The outputs, read at the EFIT times, are the
cylindrical internal inductance of the simulated profile, the enclosed-current fractions, the
central safety factor and the distance from the relaxed (steady-state) profile -- for a few
resistivity scales, since Zeff is not measured and the diffusion time is proportional to it.

Only test-legal columns are read: the plasma current, the toroidal-field coil current and the
two Thomson systems. The geometry is fixed (DIII-D's typical R0, a and elongation); what is
learned downstream is how these numbers relate to the scored ones.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from nm1 import CACHE_DATA
from nm1.data_fixes import fix_d3d_ip_times

CDS_CACHE = CACHE_DATA / "d3d_cds.h5"

MU0 = 4e-7 * np.pi
R0, R_AX, A_MINOR, KAPPA = 1.70, 1.75, 0.60, 1.75
N_TF = 144                          # DIII-D toroidal-field turns (d3d_pressure_prior_v13 source)
NX = 41
X = np.linspace(0.0, 1.0, NX)
DT = 0.005                           # s
TE_EDGE = 0.04                       # keV at x = 1
ZEFF_SCALES = (1.0, 3.0)             # resistivity multipliers (Zeff ~ 1.5 and ~ 4.5)
COLUMNS = ("magnetics_plasma_current", "magnetics_plasma_current_times", "magnetics_time",
           "magnetics_bcoil", "thomson_core_times", "thomson_core_Te", "thomson_edge_times",
           "thomson_edge_Te", "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
           "efit_times", "magnetics_ECOILA", "thomson_core_ne", "thomson_edge_ne")
XB = np.linspace(0.0, 1.0, 11)       # radial bins for the Te profile fit
FRACS = (0.3, 0.5, 0.7)
N_PER_SCALE = 6 + len(FRACS)
# Linear-response kernels of the current profile to changes of Ip: exponential memories of
# d(ln Ip)/dt, in clock time and in resistive time (dt scaled by Te(x=0.5)^-1.5, 1 keV = 1),
# and two anticipating ones (the whole shot is known offline).
TAUS = (0.03, 0.1, 0.3, 1.0, 3.0)
LEADS = (0.1, 0.3)
N_EMA = 2 * len(TAUS) + len(LEADS)
N_CDS_V1 = len(ZEFF_SCALES) * N_PER_SCALE + 4 + N_EMA
# v2 appends the flux the plasma has consumed. The within-shot variation of the EFIT flux level
# follows the simulated resistive consumption, integral of V_loop dt, at correlation 0.91 on held-out
# shots (clock time 0.83, ECOILA -0.76): the ohmic flux is only partly visible in the one E-coil
# current the data carries. Per scale: resistive and inductive flux [Wb/rad]; then ECOILA at
# breakdown and its change since, and the time since breakdown. v1 members read the first
# N_CDS_V1 columns, so one cache serves both.
N_FLUX = 2 * len(ZEFF_SCALES) + 3
N_CDS_V2 = N_CDS_V1 + N_FLUX
# v3 appends the pressure side of Lambda = beta_p + li/2. The frame errors of betaN and li are
# anti-correlated (-0.45): the coils fix the sum, not the split. From the Thomson electron density
# and temperature on the same grid: the electron beta_p and beta_N, the pressure peaking, the
# density, two memories of beta_N (fast ions build up over ~100 ms), and a second diffusion run
# with a bootstrap current j_bs = -C sqrt(eps) (dp/dr) / B_theta as a non-inductive source
# (E = eta (j - j_bs)): its li, the change against the run without it, and the bootstrap fraction.
NE_EDGE = 0.3                        # 1e19 m^-3 at x = 1
C_BS = 1.5
N_V3 = 11
N_CDS = N_CDS_V2 + N_V3


def _stack(value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2:
            return arr
    except (TypeError, ValueError):
        pass
    rows = [np.asarray(v, dtype=np.float64) for v in value]
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _chords(row, prefix: str) -> np.ndarray:
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    return np.flatnonzero(np.char.startswith(names, prefix))


def _sample_x(R: np.ndarray, Z: np.ndarray, times_s: np.ndarray, xmap) -> np.ndarray:
    """``(len(times), C)`` normalised radius of every channel at every laser pulse.

    Without ``xmap`` it is the fixed ellipse; with one (``FluxMap``), the square root of the
    normalised poloidal flux of the nearest EFIT frame's flux map, falling back to the ellipse
    where no frame is within reach (the limited ramp-up has no EFIT frames at all).
    """
    x0 = np.sqrt(((R - R_AX) / A_MINOR) ** 2 + (Z / (KAPPA * A_MINOR)) ** 2)
    x = np.broadcast_to(x0, (len(times_s), len(R))).copy()
    if xmap is not None:
        xf = xmap(times_s, R, Z)
        x = np.where(np.isfinite(xf), xf, x)
    return x


def _te_points(row, xmap=None):
    """Every Thomson Te sample as (time, x, Te[keV]), both systems, valid readings only."""
    out_t, out_x, out_v = [], [], []
    for system, prefix in (("core", "TS_core"), ("edge", "TS_tangential")):
        try:
            times = np.asarray(row[f"thomson_{system}_times"], dtype=np.float64) / 1000.0
            te = _stack(row[f"thomson_{system}_Te"])
            idx = _chords(row, prefix)
            R = np.asarray(row["thomson_chord_R"], dtype=np.float64)[idx]
            Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)[idx]
            C = min(len(R), te.shape[1])
            n = min(len(times), len(te))
            R, Z, te, times = R[:C], Z[:C], te[:n, :C], times[:n]
        except Exception:                        # noqa: BLE001 - a missing subsystem
            continue
        x = _sample_x(R, Z, times, xmap)
        ok = np.isfinite(te) & (te > 5.0) & (te < 2.0e4) & np.isfinite(times)[:, None]
        ti, ci = np.nonzero(ok)
        out_t.append(times[ti])
        out_x.append(x[ti, ci])
        out_v.append(te[ti, ci] / 1000.0)
    if not out_t:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(out_t), np.concatenate(out_x), np.concatenate(out_v)


class FluxMap:
    """Thomson channel -> sqrt(psi_N) through a sequence of flux maps (one per EFIT frame).

    psi_N = (phi_axis - phi) / (phi_axis - phi_LCFS) with phi the flux oriented so the axis is a
    maximum; the axis is the scorer's O-point and the LCFS level is read on the contour the
    scorer's own extraction returns, inside a fixed vessel box. psi_N is invariant to the flux
    offset and scale, which is where most of a predicted map's error is, so a predicted map
    serves as well as the true one: the simulated li tracks EFIT's li at within-shot correlation
    0.70 through either, against 0.14 through the fixed ellipse (40 held-out shots).
    """

    def __init__(self, efit_times_ms: np.ndarray, psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                 machine: str = "DIII-D", reach_ms: float = 30.0):
        import sys
        from pathlib import Path
        scoring = Path(__file__).resolve().parents[1] / "fusion_scoring"
        if str(scoring) not in sys.path:
            sys.path.insert(0, str(scoring))
        from common import AXIS_SIGN
        from lcfs import extract_lcfs
        from o_point import find_o_point
        from nm1.guard import box_mask
        self.t = np.asarray(efit_times_ms, dtype=np.float64) / 1000.0
        self.R, self.Z = np.asarray(R, float), np.asarray(Z, float)
        self.reach = reach_ms / 1000.0
        mask = box_mask(self.R, self.Z)
        psi = np.asarray(psi, dtype=np.float64)
        self.phi = AXIS_SIGN[machine] * psi
        n = len(psi)
        self.axis = np.full(n, np.nan)
        self.edge = np.full(n, np.nan)
        for k in range(n):
            _, _, iz, ir, _ = find_o_point(self.phi[k], self.R, self.Z, mask)
            self.axis[k] = self.phi[k][iz, ir]
            c = extract_lcfs(psi[k], self.R, self.Z, machine, mask, mask.astype(float),
                             n_iter=12, n_points=128)
            if c is not None:
                self.edge[k] = float(np.median(self._bilinear(self.phi[k], c[:, 0], c[:, 1])))

    def _bilinear(self, field, rq, zq):
        R, Z = self.R, self.Z
        fr = np.interp(rq, R, np.arange(len(R)))
        fz = np.interp(zq, Z, np.arange(len(Z)))
        r0 = np.clip(np.floor(fr).astype(int), 0, len(R) - 2)
        z0 = np.clip(np.floor(fz).astype(int), 0, len(Z) - 2)
        tr, tz = fr - r0, fz - z0
        return (field[z0, r0] * (1 - tr) * (1 - tz) + field[z0, r0 + 1] * tr * (1 - tz)
                + field[z0 + 1, r0] * (1 - tr) * tz + field[z0 + 1, r0 + 1] * tr * tz)

    def __call__(self, times_s: np.ndarray, Rc: np.ndarray, Zc: np.ndarray) -> np.ndarray:
        out = np.full((len(times_s), len(Rc)), np.nan)
        if len(self.t) == 0:
            return out
        k = np.clip(np.searchsorted(self.t, times_s), 0, len(self.t) - 1)
        km = np.clip(k - 1, 0, len(self.t) - 1)
        k = np.where(np.abs(self.t[km] - times_s) < np.abs(self.t[k] - times_s), km, k)
        near = (np.abs(self.t[k] - times_s) <= self.reach) & np.isfinite(times_s)
        for i in np.flatnonzero(near):
            f = k[i]
            span = self.axis[f] - self.edge[f]
            if not np.isfinite(span) or span <= 0:
                continue
            psin = (self.axis[f] - self._bilinear(self.phi[f], Rc, Zc)) / span
            out[i] = np.sqrt(np.clip(psin, 0.0, 4.0))
        return out


def te_profiles(row, grid_t: np.ndarray, xmap=None) -> np.ndarray:
    """``(len(grid_t), NX)`` electron temperature [keV] on the diffusion grid.

    Samples are binned by radius per laser pulse window (a 20 ms box around each grid time),
    medianed, interpolated across empty bins, held flat inward of the innermost bin and brought
    down linearly to TE_EDGE at the boundary beyond the outermost one.
    """
    t, x, v = _te_points(row, xmap)
    nT = len(grid_t)
    out = np.full((nT, NX), np.nan)
    if len(t) == 0:
        out[:] = np.maximum(0.3 * (1 - X ** 2), TE_EDGE)[None, :]
        return out
    order = np.argsort(t, kind="stable")
    t, x, v = t[order], x[order], v[order]
    lo = np.searchsorted(t, grid_t - 0.010, side="left")
    hi = np.searchsorted(t, grid_t + 0.010, side="right")
    centers = 0.5 * (XB[:-1] + XB[1:])
    bin_of = np.clip(np.digitize(x, XB) - 1, 0, len(XB) - 2)
    last = None
    for k in range(nT):
        a, b = lo[k], hi[k]
        if b - a < 4:
            if last is not None:
                out[k] = last
            continue
        med = np.full(len(centers), np.nan)
        bb, vv = bin_of[a:b], v[a:b]
        for j in range(len(centers)):
            sel = vv[bb == j]
            if len(sel):
                med[j] = np.median(sel)
        good = np.isfinite(med) & (centers <= 1.0)
        if good.sum() < 2:
            if last is not None:
                out[k] = last
            continue
        cx, cv = centers[good], med[good]
        # beyond the outermost populated bin: straight down to the edge value
        cx = np.concatenate([cx, [1.0]])
        cv = np.concatenate([cv, [TE_EDGE]])
        prof = np.interp(X, cx, cv, left=cv[0])
        last = np.maximum(prof, TE_EDGE)
        out[k] = last
    first = np.flatnonzero(np.isfinite(out[:, 0]))
    if len(first):
        out[:first[0]] = out[first[0]]
    else:
        out[:] = np.maximum(0.3 * (1 - X ** 2), TE_EDGE)[None, :]
    return out


def _ne_points(row, xmap=None):
    """Every Thomson ne sample as (time, x, ne[1e19 m^-3]), both systems, valid readings only."""
    out_t, out_x, out_v = [], [], []
    for system, prefix in (("core", "TS_core"), ("edge", "TS_tangential")):
        try:
            times = np.asarray(row[f"thomson_{system}_times"], dtype=np.float64) / 1000.0
            ne = _stack(row[f"thomson_{system}_ne"])
            idx = _chords(row, prefix)
            R = np.asarray(row["thomson_chord_R"], dtype=np.float64)[idx]
            Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)[idx]
            C = min(len(R), ne.shape[1])
            n = min(len(times), len(ne))
            R, Z, ne, times = R[:C], Z[:C], ne[:n, :C], times[:n]
        except Exception:                        # noqa: BLE001 - a missing subsystem
            continue
        x = _sample_x(R, Z, times, xmap)
        ok = np.isfinite(ne) & (ne > 1e17) & (ne < 3e21) & np.isfinite(times)[:, None]
        ti, ci = np.nonzero(ok)
        out_t.append(times[ti])
        out_x.append(x[ti, ci])
        out_v.append(ne[ti, ci] / 1e19)
    if not out_t:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return np.concatenate(out_t), np.concatenate(out_x), np.concatenate(out_v)


def ne_profiles(row, grid_t: np.ndarray, xmap=None) -> np.ndarray:
    """``(len(grid_t), NX)`` electron density [1e19 m^-3], binned exactly like ``te_profiles``."""
    t, x, v = _ne_points(row, xmap)
    nT = len(grid_t)
    out = np.full((nT, NX), np.nan)
    flat = np.maximum(3.0 * (1 - X ** 2) ** 0.5, NE_EDGE)
    if len(t) == 0:
        out[:] = flat[None, :]
        return out
    order = np.argsort(t, kind="stable")
    t, x, v = t[order], x[order], v[order]
    lo = np.searchsorted(t, grid_t - 0.010, side="left")
    hi = np.searchsorted(t, grid_t + 0.010, side="right")
    centers = 0.5 * (XB[:-1] + XB[1:])
    bin_of = np.clip(np.digitize(x, XB) - 1, 0, len(XB) - 2)
    last = None
    for k in range(nT):
        a, b = lo[k], hi[k]
        if b - a < 4:
            if last is not None:
                out[k] = last
            continue
        med = np.full(len(centers), np.nan)
        bb, vv = bin_of[a:b], v[a:b]
        for j in range(len(centers)):
            sel = vv[bb == j]
            if len(sel):
                med[j] = np.median(sel)
        good = np.isfinite(med)
        if good.sum() < 2:
            if last is not None:
                out[k] = last
            continue
        cx = np.concatenate([centers[good], [1.0]])
        cv = np.concatenate([med[good], [NE_EDGE]])
        last = np.maximum(np.interp(X, cx, cv, left=cv[0]), 0.05)
        out[k] = last
    first = np.flatnonzero(np.isfinite(out[:, 0]))
    if len(first):
        out[:first[0]] = out[first[0]]
    else:
        out[:] = flat[None, :]
    return out


def _resistivity(te_kev: np.ndarray, scale: float) -> np.ndarray:
    """Spitzer (Zeff ~ 1.5 at scale 1, lnLambda 17) with a neoclassical trapped-particle factor."""
    eps = np.clip(A_MINOR * X / R0, 0.0, 0.9)
    ft = np.clip(1.46 * np.sqrt(eps) - 0.46 * eps, 0.0, 0.95)
    neo = 1.0 / np.maximum((1 - ft) * (1 - 0.4 * ft), 0.05)
    return scale * 1.65e-9 * 17.0 * 1.5 * neo / np.maximum(te_kev, 0.01) ** 1.5


def _q_profile(I: np.ndarray, b0: float) -> np.ndarray:
    """Cylindrical safety factor with elongation, q(x) = 2 pi (a x)^2 B0 (1+k^2)/2 / (mu0 R0 I)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (2 * np.pi * (A_MINOR * X) ** 2 * b0 * 0.5 * (1 + KAPPA ** 2)
             / (MU0 * R0 * np.maximum(np.abs(I), 1.0)))
    q[0] = q[1] if len(q) > 1 else q[0]
    return q


def _sawtooth(I: np.ndarray, b0: float) -> tuple[np.ndarray, float]:
    """Flatten the current density inside 1.3x the q = 1 radius (current conserved)."""
    q = _q_profile(I, b0)
    below = np.flatnonzero(q[1:] < 1.0) + 1
    if not len(below):
        return I, 0.0
    x1 = X[below.max()]
    xm = min(1.0, 1.3 * x1)
    k = int(np.searchsorted(X, xm))
    if k < 2:
        return I, x1
    I = I.copy()
    I[:k] = I[k] * (X[:k] / X[k]) ** 2
    return I, x1


def _li_cyl(I: np.ndarray) -> float:
    ip = I[-1]
    if abs(ip) < 1.0:
        return np.nan
    f = (I[1:] / ip) ** 2 / X[1:]
    return float(2.0 * np.trapezoid(np.concatenate([[0.0], f]), X))


def _steady(eta: np.ndarray, ip: float, b0: float) -> np.ndarray:
    """Relaxed profile j ~ 1/eta (E uniform), sawtooth-clamped, scaled to Ip."""
    j = 1.0 / eta
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (j[1:] * X[1:] + j[:-1] * X[:-1]) * np.diff(X))])
    I = ip * cum / max(cum[-1], 1e-30)
    I, _ = _sawtooth(I, b0)
    return I


def simulate(ip_t: np.ndarray, ip_a: np.ndarray, te: np.ndarray, grid_t: np.ndarray,
             b0: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    """Implicit-Euler diffusion of the enclosed current on ``grid_t`` (uniform, DT)."""
    nT = len(grid_t)
    xh = 0.5 * (X[1:] + X[:-1])                  # half points
    dx = X[1] - X[0]
    I = ip_a[0] * X ** 2                         # flat current density at breakdown
    rec = {k: np.zeros(nT) for k in ("li", "li_ss", "q0", "x1", "vloop")}
    rec.update({f"f{int(f * 10)}": np.zeros(nT) for f in FRACS})
    for n in range(nT):
        eta = _resistivity(te[n], scale)
        etah = 0.5 * (eta[1:] + eta[:-1])
        cfl = DT / (MU0 * A_MINOR ** 2)
        # interior nodes 1..NX-2; I[0] = 0, I[-1] = Ip
        cw = etah / xh / dx                     # flux coefficient at half points
        lower = -cfl * X[1:-1] * cw[:-1] / dx
        upper = -cfl * X[1:-1] * cw[1:] / dx
        diag = 1.0 + cfl * X[1:-1] * (cw[:-1] + cw[1:]) / dx
        rhs = I[1:-1].copy()
        rhs[-1] -= upper[-1] * ip_a[n]
        ab = np.zeros((3, len(diag)))
        ab[0, 1:] = upper[:-1]
        ab[1] = diag
        ab[2, :-1] = lower[1:]
        sol = solve_banded((1, 1), ab, rhs, check_finite=False)
        I = np.concatenate([[0.0], sol, [ip_a[n]]])
        I, x1 = _sawtooth(I, b0[n])
        rec["li"][n] = _li_cyl(I)
        rec["li_ss"][n] = _li_cyl(_steady(eta, ip_a[n], b0[n]))
        q = _q_profile(I, b0[n])
        rec["q0"][n] = float(np.clip(q[1], 0.0, 20.0))
        rec["x1"][n] = x1
        # loop voltage at the boundary: E = eta j, j(1) from the edge gradient of I
        j1 = (I[-1] - I[-2]) / dx / (2 * np.pi * A_MINOR ** 2 * X[-1])
        rec["vloop"][n] = 2 * np.pi * R0 * eta[-1] * j1
        ip = ip_a[n] if abs(ip_a[n]) > 1.0 else 1.0
        for f in FRACS:
            rec[f"f{int(f * 10)}"][n] = np.interp(f, X, I) / ip
    return rec


def simulate_bs(ip_a: np.ndarray, te: np.ndarray, ne: np.ndarray, b0: np.ndarray,
                scale: float = 1.0) -> dict[str, np.ndarray]:
    """The same diffusion with a bootstrap current as a non-inductive source, E = eta (j - j_bs).

    j_bs = -C_BS sqrt(eps) (dp/dr) / B_theta, p = 2 ne Te (Ti = Te), B_theta from the previous
    step's enclosed current; the source enters explicitly, -(2 pi x / mu0) d(eta j_bs)/dx.
    """
    nT = len(ip_a)
    xh = 0.5 * (X[1:] + X[:-1])
    dx = X[1] - X[0]
    eps = np.clip(A_MINOR * X / R0, 0.0, 0.9)
    I = ip_a[0] * X ** 2
    rec = {"li": np.zeros(nT), "fbs": np.zeros(nT)}
    for n in range(nT):
        eta = _resistivity(te[n], scale)
        p = 2.0 * ne[n] * 1e19 * te[n] * 1.602e-16                  # Pa
        dpdr = np.gradient(p, X) / A_MINOR
        with np.errstate(divide="ignore", invalid="ignore"):
            btheta = MU0 * np.abs(I) / (2 * np.pi * A_MINOR * np.maximum(X, dx))
        btheta = np.maximum(btheta, 0.02)
        jbs = -C_BS * np.sqrt(eps) * dpdr / btheta
        jbs[0] = 0.0
        src = -(2 * np.pi * X / MU0) * np.gradient(eta * jbs, X)
        etah = 0.5 * (eta[1:] + eta[:-1])
        cfl = DT / (MU0 * A_MINOR ** 2)
        cw = etah / xh / dx
        lower = -cfl * X[1:-1] * cw[:-1] / dx
        upper = -cfl * X[1:-1] * cw[1:] / dx
        diag = 1.0 + cfl * X[1:-1] * (cw[:-1] + cw[1:]) / dx
        rhs = I[1:-1] + DT * src[1:-1]
        rhs[-1] -= upper[-1] * ip_a[n]
        ab = np.zeros((3, len(diag)))
        ab[0, 1:] = upper[:-1]
        ab[1] = diag
        ab[2, :-1] = lower[1:]
        sol = solve_banded((1, 1), ab, rhs, check_finite=False)
        I = np.concatenate([[0.0], sol, [ip_a[n]]])
        I, _ = _sawtooth(I, b0[n])
        rec["li"][n] = _li_cyl(I)
        ibs = float(np.trapezoid(jbs * 2 * np.pi * A_MINOR ** 2 * X, X))
        rec["fbs"][n] = ibs / max(abs(ip_a[n]), 1.0)
    return rec


def pressure_block(ip_g, te, ne, b0, grid, t_eval, li_plain, li_bs, fbs) -> np.ndarray:
    """``(T, N_V3)``: electron beta_p / beta_N, peaking, density, beta_N memories, bootstrap run."""
    pe = ne * 1e19 * te * 1.602e-16                                 # Pa, electrons
    pe_avg = np.trapezoid(pe * 2 * X[None, :], X, axis=1)           # cylinder volume average
    bpa = MU0 * ip_g / (2 * np.pi * A_MINOR * np.sqrt(0.5 * (1 + KAPPA ** 2)))
    betap = 2 * MU0 * pe_avg / np.maximum(bpa, 1e-3) ** 2
    betat = 2 * MU0 * pe_avg / np.maximum(b0, 0.3) ** 2
    betan = 100 * betat * A_MINOR * b0 / np.maximum(ip_g / 1e6, 0.05)
    peaking = pe[:, 0] / np.maximum(pe_avg, 1.0)
    ne0 = ne[:, 0]
    neavg = np.trapezoid(ne, X, axis=1)
    mems = []
    for tau in (0.1, 0.3):
        a = np.exp(-DT / tau)
        acc = np.empty(len(betan))
        s = betan[0]
        for k in range(len(betan)):
            s = a * s + (1 - a) * betan[k]
            acc[k] = s
        mems.append(acc)
    cols = [betap, betan, peaking, ne0, neavg, mems[0], mems[1], li_bs, li_bs - li_plain, fbs,
            betap + 0.5 * li_bs]
    return np.stack([np.interp(t_eval, grid, np.nan_to_num(c)) for c in cols], 1)


def diffusion_features(row, efit_times_ms: np.ndarray, xmap=None) -> np.ndarray:
    """``(T, N_CDS)`` simulated current-profile features at the EFIT times.

    ``xmap`` (a ``FluxMap``) places the Thomson channels on the flux coordinate of a given
    equilibrium instead of the fixed ellipse; everything else is unchanged.
    """
    t_eval = np.asarray(efit_times_ms, dtype=np.float64) / 1000.0
    T = len(t_eval)
    out = np.zeros((T, N_CDS), dtype=np.float64)
    if T == 0:
        return out.astype(np.float32)
    ipt = fix_d3d_ip_times(row) / 1000.0
    ip = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64)) * 1000.0  # A
    ok = np.isfinite(ipt) & np.isfinite(ip)
    ipt, ip = ipt[ok], ip[ok]
    order = np.argsort(ipt, kind="stable")
    ipt, ip = ipt[order], ip[order]
    ipmax = np.percentile(ip, 99.5) if len(ip) else 0.0
    if len(ip) < 10 or ipmax < 1e4:
        return out.astype(np.float32)
    t0 = ipt[np.argmax(ip > 0.05 * ipmax)]
    t1 = max(t_eval.max(), t0 + 0.05) + DT
    grid = np.arange(t0, t1 + DT / 2, DT)
    # Ip smoothed over the step so the boundary condition does not alias ripple
    csum = np.concatenate([[0.0], np.cumsum(ip)])
    lo = np.searchsorted(ipt, grid - DT / 2)
    hi = np.searchsorted(ipt, grid + DT / 2)
    n = hi - lo
    ip_g = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.interp(grid, ipt, ip))
    ip_g = np.maximum(ip_g, 1e3)
    mt = np.asarray(row["magnetics_time"], dtype=np.float64) / 1000.0
    bc = np.abs(np.asarray(row["magnetics_bcoil"], dtype=np.float64)) * 1000.0
    m = min(len(mt), len(bc))
    good = np.isfinite(mt[:m]) & np.isfinite(bc[:m])
    b0 = (2e-7 * N_TF * np.interp(grid, mt[:m][good], bc[:m][good]) / R0
          if good.sum() > 2 else np.full(len(grid), 2.0))
    b0 = np.maximum(b0, 0.3)
    te = te_profiles(row, grid, xmap)
    col = 0
    fluxes = []
    lp0 = MU0 * R0 * (np.log(8 * R0 / A_MINOR) - 2.0)
    li_plain = None
    for scale in ZEFF_SCALES:
        rec = simulate(grid, ip_g, te, grid, b0, scale)
        if li_plain is None:
            li_plain = rec["li"]
        at = {k: np.interp(t_eval, grid, v) for k, v in rec.items()}
        out[:, col + 0] = at["li"]
        out[:, col + 1] = at["li_ss"]
        out[:, col + 2] = at["li"] - at["li_ss"]
        out[:, col + 3] = at["q0"]
        out[:, col + 4] = at["x1"]
        out[:, col + 5] = np.sign(at["vloop"]) * np.log1p(np.abs(at["vloop"]))
        for k, f in enumerate(FRACS):
            out[:, col + 6 + k] = at[f"f{int(f * 10)}"]
        col += N_PER_SCALE
        v = np.nan_to_num(rec["vloop"])
        consumed = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])
        inductive = (lp0 + MU0 * R0 * 0.5 * np.nan_to_num(rec["li"], nan=1.0)) * ip_g
        fluxes.append(np.interp(t_eval, grid, consumed) / (2 * np.pi))
        fluxes.append(np.interp(t_eval, grid, inductive) / (2 * np.pi))
    # the temperature profile the simulation ran on, at the EFIT times
    te_at = np.stack([np.interp(t_eval, grid, te[:, j]) for j in (0, 10, 20, 30)], 1)
    out[:, col:col + 4] = np.log(np.maximum(te_at, 1e-3))
    col += 4
    out[:, col:col + N_EMA] = ema_block(grid, ip_g, te[:, 20], t_eval)
    col += N_EMA
    # v2: flux consumed since breakdown, and the one E-coil current's state then and now
    out[:, col:col + len(fluxes)] = np.column_stack(fluxes)
    col += len(fluxes)
    try:
        ec = np.asarray(row["magnetics_ECOILA"], dtype=np.float64)
        k = min(len(mt), len(ec))
        g2 = np.isfinite(mt[:k]) & np.isfinite(ec[:k])
        e0 = float(np.interp(t0, mt[:k][g2], ec[:k][g2]))
        out[:, col] = e0
        out[:, col + 1] = np.interp(t_eval, mt[:k][g2], ec[:k][g2]) - e0
    except Exception:                                # noqa: BLE001 - column missing
        pass
    out[:, col + 2] = t_eval - t0
    col += 3
    # v3: the pressure side and a bootstrap-current run
    ne = ne_profiles(row, grid, xmap)
    bs = simulate_bs(ip_g, te, ne, b0, ZEFF_SCALES[0])
    out[:, col:col + N_V3] = pressure_block(ip_g, te, ne, b0, grid, t_eval, li_plain,
                                            bs["li"], bs["fbs"])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def read_inputs(path):
    """Only the columns this block reads -- all of them test-legal."""
    import pandas as pd
    return pd.read_parquet(path, columns=list(COLUMNS)).iloc[0]


def _cache_job(job):
    index, path, times_s = job
    try:
        row = read_inputs(path)
        return index, diffusion_features(row, np.asarray(times_s, np.float64) * 1000.0), ""
    except Exception as exc:                        # noqa: BLE001 - reported by the parent
        return index, None, f"{type(exc).__name__}: {exc}"


def build_cache(out=None, splits=("train", "val"), workers: int = 10):
    """``d3d_cds.h5``: this block for every frame of the coefficient cache, row-aligned with it.

    The frames are the cache's own (its stored EFIT times), so the rows line up by construction.
    """
    import json
    import time
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    import h5py

    from nm1.cache import CACHE
    from nm1.fetch import listing, local_path
    out = Path(out) if out else CDS_CACHE
    files = listing("diii_d_train")
    started = time.perf_counter()
    with h5py.File(CACHE, "r") as h5, h5py.File(out, "a") as o:
        for split in splits:
            shot = h5[split]["shot"][:]
            clock = h5[split]["features"][:, 136].astype(np.float64)
            b = np.flatnonzero(np.diff(shot)) + 1
            st = np.concatenate([[0], b])
            en = np.concatenate([b, [len(shot)]])
            if split in o:
                del o[split]
            d = o.create_group(split).create_dataset("cds", (len(shot), N_CDS), dtype="f4")
            jobs = [(int(k), local_path(files[int(shot[s])]), clock[s:e])
                    for k, (s, e) in enumerate(zip(st, en))]
            bad = []
            with ProcessPoolExecutor(workers) as pool:
                for done, (k, res, err) in enumerate(pool.map(_cache_job, jobs, chunksize=4), 1):
                    if res is None:
                        bad.append((k, err))
                        continue
                    d[st[k]:en[k]] = res
                    if done % 250 == 0 or done == len(jobs):
                        print(f"  {split}: {done}/{len(jobs)} shots, "
                              f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
            if bad:
                for k, err in bad[:20]:
                    print(f"  FAILED shot {k}: {err}", flush=True)
                raise SystemExit(f"{len(bad)} {split} shots failed")
        o.attrs["n_cds"] = N_CDS
    out.with_suffix(".json").write_text(json.dumps({"n_cds": N_CDS}), encoding="utf-8")
    print(f"wrote {out} in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return out


def ema_block(grid: np.ndarray, ip_g: np.ndarray, te_mid: np.ndarray,
              t_eval: np.ndarray) -> np.ndarray:
    """Exponential memories of d(ln Ip)/dt [1/s] on the uniform grid, read at ``t_eval``."""
    u = np.log(np.maximum(ip_g, 1e3))
    du = np.gradient(u, DT)
    warp = DT * np.clip(1.0 / np.maximum(te_mid, 0.02), 0.0, 50.0) ** 1.5
    cols = []
    for clock in (np.full(len(grid), DT), warp):
        for tau in TAUS:
            a = np.exp(-clock / tau)
            acc = np.empty(len(du))
            s = 0.0
            for k in range(len(du)):
                s = a[k] * s + (1 - a[k]) * du[k]
                acc[k] = s
            cols.append(acc)
    for tau in LEADS:
        a = np.exp(-DT / tau)
        acc = np.empty(len(du))
        s = 0.0
        for k in range(len(du) - 1, -1, -1):
            s = a * s + (1 - a) * du[k]
            acc[k] = s
        cols.append(acc)
    return np.stack([np.interp(t_eval, grid, c) for c in cols], 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build the current-diffusion block cache")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    build_cache(a.out, tuple(a.splits.split(",")), a.workers)