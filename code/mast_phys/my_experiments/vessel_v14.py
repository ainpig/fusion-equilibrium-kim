"""v14: the vessel's induced toroidal current, as one more circuit of the MAST vacuum field.

The v3-1 .. v13 MAST equilibrium has coils + plasma only. MAST's stainless vessel, a closed toroidal
conductor around the whole plasma, carries the current that the loop voltage drives in it: a solenoid
ramp of 300 kA/s (the ramp-up of every shot) puts ~1-3 V around the vessel (computed below from the
measured currents), and a thin stainless shell of the vessel's size conducts tens of kA at that voltage.
Without it a part of the frames -- almost all in the fast ramp-up -- come out of the seed solve as
inboard-limited equilibria (the X-point separatrix does not close before the solenoid surface),
although the frames of the challenge are diverted.

Model (official inputs only):
* geometry: the boundary of the released flux grid stands for the vessel (the EFIT grid spans the
  vessel interior): a side wall at R = R_grid_max and two end plates at Z = +-Z_grid_max from the
  solenoid's outer radius outwards. A thin uniform shell: current per unit length ~ E_phi ~ 1/R.
* drive: the EMF around the shell, -2 pi d(psi)/dt of the flux that the measured coil currents and the
  measured plasma current (a filament at R = 0.9 m, Z = 0) put on it, shell-averaged with the same
  weights; smoothed over 2 ms before the derivative.
* response: a first-order L/R filter with time constant TAU_MS and a conductance G_KA_PER_V,
  I_vessel(t) = G * lowpass(EMF, TAU). Mutual shielding inside the shell beyond the filter is ignored.

The vessel current enters the runtime as the column VESSEL_COLUMN, added to each input row by
`kinetic_equilibrium.read_inputs` on the row's own magnetics time base, with this module's flux map as
its circuit basis (`mast_equilibrium_v12.circuit_basis`). Every stage that builds a vacuum field from
`MAST_COILS` therefore sees it. G and TAU are machine constants, set label-free (FUSION_MAST_VESSEL_G,
FUSION_MAST_VESSEL_TAU; G = 0 switches the vessel off and reproduces the coils-only vacuum field).
No MAST equilibrium, label or score is read.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.special import ellipk, ellipe

VESSEL_COLUMN = "vessel_current_v14"
N_SIDE = 31
N_END = 24
SMOOTH_MS = 2.0
PLASMA_R = 0.9
COILS = ("magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p3l_current",
         "magnetics_p3u_current", "magnetics_p4l_current", "magnetics_p4u_current",
         "magnetics_p5l_current", "magnetics_p5u_current", "magnetics_p6l_current",
         "magnetics_p6u_current", "magnetics_sol_current")


def settings():
    g = float(os.environ.get("FUSION_MAST_VESSEL_G", "0"))
    tau = float(os.environ.get("FUSION_MAST_VESSEL_TAU", "10"))
    if g < 0 or tau <= 0:
        raise ValueError("vessel conductance must be >= 0 and its time constant > 0")
    return g, tau


def _green(ro, zo, rs, zs):
    m = np.clip(4 * ro * rs / ((ro + rs) ** 2 + (zo - zs) ** 2), 1e-12, 1 - 1e-14)
    return 4e-7 * np.sqrt(ro * rs / m) * ((1 - m / 2) * ellipk(m) - ellipe(m))


def shell(row):
    """(R, Z, weight) of the shell loops; weights sum to 1."""
    r = np.asarray(row["efit_grid_R"], float); z = np.asarray(row["efit_grid_Z"], float)
    names = np.asarray(row["coil_input_column"]).astype(str)
    cr, cw = np.asarray(row["coil_R"], float), np.asarray(row["coil_width"], float)
    sol = names == "magnetics_sol_current"
    r_in = float(np.max(cr[sol] + cw[sol] / 2)) if sol.any() else float(r[0])
    rmax, zmax = float(r[-1]), float(min(abs(z[0]), abs(z[-1])))
    loops = []
    dz = 2 * zmax / N_SIDE
    for zv in -zmax + dz * (np.arange(N_SIDE) + 0.5):
        loops.append((rmax, zv, dz / rmax))
    dr = (rmax - r_in) / N_END
    for rv in r_in + dr * (np.arange(N_END) + 0.5):
        for s in (-1.0, 1.0):
            loops.append((rv, s * zmax, dr / rv))
    a = np.array(loops)
    a[:, 2] /= a[:, 2].sum()
    return a


def basis(row, r, z):
    """psi (Wb/rad) on the (z, r) grid per ampere of total shell current."""
    rr, zz = np.meshgrid(np.asarray(r, float), np.asarray(z, float))
    out = np.zeros_like(rr)
    for R0, Z0, w in shell(row):
        out += w * _green(rr, zz, R0, Z0)
    return out


def coupling(row, sol_divisor):
    """Shell-averaged flux (Wb/rad) per kA of each coil column (order of COILS) and per kA of Ip."""
    loops = shell(row)
    names = np.asarray(row["coil_input_column"]).astype(str)
    cr, cz = np.asarray(row["coil_R"], float), np.asarray(row["coil_Z"], float)
    out = []
    for name in COILS:
        ids = np.flatnonzero(names == name)
        g = sum(w * np.mean(_green(R0, Z0, cr[ids], cz[ids])) for R0, Z0, w in loops) if len(ids) else 0.0
        if name == "magnetics_sol_current":
            g = g * len(ids) / sol_divisor
        out.append(1e3 * g)
    gp = sum(w * _green(R0, Z0, PLASMA_R, 0.0) for R0, Z0, w in loops)
    return np.asarray(out), 1e3 * float(gp)


def lowpass(t_ms, x, tau_ms):
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        a = 1.0 - np.exp(-max(t_ms[i] - t_ms[i - 1], 0.0) / tau_ms)
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _time_base(row):
    """Sorted times at which the plasma current is measured, near-repeats merged.

    Some MAST shots carry the union of two acquisition time bases (1 ms over
    -2.5..5.5 s, all NaN, and 0.2 ms over the discharge): every column is NaN on the other base's
    samples, and the merged base has stamps 1e-5 ms apart. Smoothing over a fixed number of samples
    and differentiating across such a pair makes the EMF explode. The circuit is
    therefore integrated on the measured samples only.
    """
    t0 = np.asarray(row["magnetics_time"], float)
    ip = np.asarray(row["magnetics_plasma_current"], float)
    t = np.unique(t0[np.isfinite(t0) & np.isfinite(ip)])
    if len(t) < 2:
        return t
    step = float(np.median(np.diff(t)))
    return t[np.r_[True, np.diff(t) > 0.01 * step]]


def _column(row, name, t):
    """One measured current on the time base `t`, from its finite samples only (as `_interp`)."""
    t0 = np.asarray(row["magnetics_time"], float)
    v = np.asarray(row[name], float)
    ok = np.isfinite(t0) & np.isfinite(v)
    if not ok.any():
        return np.zeros(len(t))
    if ok.sum() == 1:
        return np.full(len(t), v[ok][0])
    order = np.argsort(t0[ok], kind="stable")
    return np.interp(t, t0[ok][order], v[ok][order])


def current(row, g_ka_per_v, tau_ms, sol_divisor):
    """Shell current (kA) aligned element-wise with row['magnetics_time']."""
    t0 = np.asarray(row["magnetics_time"], float)
    if g_ka_per_v == 0:
        return np.zeros(len(t0))
    t = _time_base(row)
    if len(t) < 2:
        return np.zeros(len(t0))
    m_coil, m_ip = coupling(row, sol_divisor)
    psi = np.zeros(len(t))
    for name, m in zip(COILS, m_coil):
        psi += m * _column(row, name, t)
    psi += m_ip * _column(row, "magnetics_plasma_current", t)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(SMOOTH_MS / max(dt, 1e-6))))
    sm = np.convolve(psi, np.ones(k) / k, mode="same")
    emf = -2 * np.pi * np.gradient(sm, t) * 1e3
    cur = g_ka_per_v * lowpass(t, emf, tau_ms)
    return np.interp(t0, t, cur)


def add_column(row):
    """The input row with the vessel current column (kA on the magnetics time base) added in place."""
    from my_experiments.mast_equilibrium_v12 import SOLENOID_PARALLEL_CIRCUITS
    g, tau = settings()
    row[VESSEL_COLUMN] = current(row, g, tau, SOLENOID_PARALLEL_CIRCUITS)
    return row
