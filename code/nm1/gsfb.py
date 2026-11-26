"""Free-boundary Grad-Shafranov equilibria by a batched Picard iteration -- machine-agnostic.

Everything machine-specific is an *input*: the coil conductor geometry, the measured coil
currents, the flux grid and the Thomson chord positions.  Nothing here is fitted to a machine's
equilibrium reconstructions; the few profile constants are fitted on DIII-D (``fit_profiles``) or
fixed by physics, and every one of them is listed in ``docs/CONSTANTS_LEDGER.md``.

Physical sign convention throughout: psi is the poloidal flux per radian, positive current makes
the magnetic axis a MAXIMUM (EFIT stores DIII-D with the opposite sign; the scorer is sign
invariant, ``to_efit_sign`` converts when a machine needs it).

    psi = psi_ext + G J ,      J = lambda [ beta0 R/Rax + (1 - beta0) Rax/R ] P(psi_N)

* ``psi_ext``  Green-function flux of the measured coil currents (+ one fictitious antisymmetric
               coil pair, see below).
* ``G``        the grid-to-grid flux response of a toroidal current cell.
* ``J``        a flux-function current density inside the closed-flux region (flood fill from the
               axis above the separatrix level), normalised to the measured plasma current.
* ``beta0``    the pressure-driven share of the current; chosen each iteration so the model's
               pressure profile matches the Thomson electron pressure times a fixed factor.
* vertical force balance.  The plasma is vertically unstable, so a plain Picard iteration drifts
               up or down.  The exact condition is that the external field exerts no net vertical
               force on J: ``sum J dpsi_ext/dZ = 0``.  A fictitious up-down antisymmetric coil
               pair supplies a uniform radial field; its current is solved *linearly* each
               iteration so the condition holds.  Its final current is a free diagnostic: small
               means the measured coil currents are consistent with an equilibrium.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * math.pi
E_CHARGE = 1.602176634e-19


# ------------------------------------------------------------------ Green functions (numpy, once)
def filament_psi_br(R: np.ndarray, Z: np.ndarray, rc: float, zc: float):
    """Flux per radian (Wb/rad/A) and radial field (T/A) of one circular filament, on the grid."""
    r, z = np.meshgrid(R.astype(np.float64), Z.astype(np.float64))
    dz = z - zc
    denom = (r + rc) ** 2 + dz ** 2
    m = np.clip(4.0 * r * rc / np.maximum(denom, 1e-16), 1e-12, 1.0 - 1e-12)
    K, E = ellipk(m), ellipe(m)
    psi = MU0 / math.pi * np.sqrt(r * rc) * ((1.0 - m / 2.0) * K - E) / np.sqrt(m)
    d2 = (rc - r) ** 2 + dz ** 2
    br = (MU0 / (2 * math.pi)) * dz / (r * np.sqrt(denom)) * (-K + (rc ** 2 + r ** 2 + dz ** 2) / np.maximum(d2, 1e-16) * E)
    return psi, br


def coil_basis(geom: dict, columns: list[str], R: np.ndarray, Z: np.ndarray):
    """Per input column: psi and B_R maps per unit of the column's current (A or A-turn).

    A column's current is an ampere-turn total shared by its conductor elements, so its response
    is the mean of the element maps (dataset documentation, Data page).
    """
    names = np.asarray(geom["coil_input_column"]).astype(str)
    cr, cz = np.asarray(geom["coil_R"], float), np.asarray(geom["coil_Z"], float)
    psi = np.zeros((len(columns), len(Z), len(R)))
    br = np.zeros_like(psi)
    for k, col in enumerate(columns):
        idx = np.flatnonzero(names == col)
        if not len(idx):
            continue
        for j in idx:
            p, b = filament_psi_br(R, Z, float(cr[j]), float(cz[j]))
            psi[k] += p / len(idx)
            br[k] += b / len(idx)
    return psi, br


def cell_green(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``G[i, j]``: flux per radian at grid cell i from unit toroidal current in cell j."""
    r, z = np.meshgrid(R.astype(np.float64), Z.astype(np.float64))
    rr, zz = r.ravel(), z.ravel()
    n = len(rr)
    G = np.empty((n, n), np.float64)
    for j0 in range(0, n, 256):
        j1 = min(n, j0 + 256)
        rj, zj = rr[None, j0:j1], zz[None, j0:j1]
        denom = (rr[:, None] + rj) ** 2 + (zz[:, None] - zj) ** 2
        m = np.clip(4 * rr[:, None] * rj / denom, 1e-12, 1 - 1e-12)
        G[:, j0:j1] = MU0 / math.pi * np.sqrt(rr[:, None] * rj) * ((1 - m / 2) * ellipk(m) - ellipe(m)) / np.sqrt(m)
    # self term: uniform rectangular conductor, equivalent (geometric-mean-distance) radius
    rho = 0.2235 * ((R[1] - R[0]) + (Z[1] - Z[0]))
    G[np.arange(n), np.arange(n)] = MU0 * rr / (2 * math.pi) * (np.log(8 * rr / rho) - 1.75) * 1.0
    return G


# ------------------------------------------------------------------ machine
@dataclass
class Machine:
    """Everything the solver needs to know about one machine, built from one shot's input columns."""
    R: np.ndarray
    Z: np.ndarray
    columns: list[str]
    psi_cols: np.ndarray                  # (n_col, nz, nr)
    br_cols: np.ndarray                   # (n_col, nz, nr)
    G: torch.Tensor                       # (nz*nr, nz*nr)
    conductors: np.ndarray                # (n, 2) R, Z of every conductor element
    box: tuple                            # (Rmin, Rmax, Zmin, Zmax) region the plasma may occupy
    keep_out: float = 0.12                # m, no axis / X-point search this close to a conductor
    pair: tuple = (1.0, 1.2)              # (R, |Z|) of the fictitious control pair
    device: str = "cuda"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        nz, nr = len(self.Z), len(self.R)
        rr, zz = np.meshgrid(self.R, self.Z)
        in_box = (rr >= self.box[0]) & (rr <= self.box[1]) & (zz >= self.box[2]) & (zz <= self.box[3])
        d = np.min(np.hypot(rr[..., None] - self.conductors[:, 0], zz[..., None] - self.conductors[:, 1]), axis=-1)
        self.ok = in_box & (d > self.keep_out)
        self.ok_saddle = in_box & (d > 0.5 * self.keep_out)
        dev = torch.device(self.device)
        self.t_ok = torch.as_tensor(self.ok, device=dev)
        self.t_ok_saddle = torch.as_tensor(self.ok_saddle, device=dev)
        self.t_R = torch.as_tensor(self.R, dtype=torch.float32, device=dev)
        self.t_Z = torch.as_tensor(self.Z, dtype=torch.float32, device=dev)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        rp, zp = self.pair
        p_up, b_up = filament_psi_br(self.R, self.Z, rp, zp)
        p_dn, b_dn = filament_psi_br(self.R, self.Z, rp, -zp)
        self.pair_psi = torch.as_tensor(p_up - p_dn, dtype=torch.float32, device=dev)
        self.pair_br = torch.as_tensor(b_up - b_dn, dtype=torch.float32, device=dev)
        self.t_psi_cols = torch.as_tensor(self.psi_cols, dtype=torch.float32, device=dev)
        self.t_br_cols = torch.as_tensor(self.br_cols, dtype=torch.float32, device=dev)

    @property
    def shape(self):
        return len(self.Z), len(self.R)

    def external(self, currents: torch.Tensor):
        """``psi_vac, br_vac`` for ``currents (B, n_col)`` in A (A-turn)."""
        psi = torch.einsum("bc,czr->bzr", currents, self.t_psi_cols)
        br = torch.einsum("bc,czr->bzr", currents, self.t_br_cols)
        return psi, br


def build_machine(row, columns: list[str], box, *, device="cuda", keep_out=0.12, pair=(1.0, 1.2),
                  G: np.ndarray | None = None) -> Machine:
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    geom = {k: np.asarray(row[k]) for k in ("coil_input_column", "coil_R", "coil_Z")}
    psi, br = coil_basis(geom, columns, R, Z)
    G = cell_green(R, Z) if G is None else G
    cond = np.column_stack([np.asarray(row["coil_R"], float), np.asarray(row["coil_Z"], float)])
    return Machine(R, Z, columns, psi, br, torch.as_tensor(G, dtype=torch.float32, device=device),
                   cond, box, keep_out, pair, device)


# ------------------------------------------------------------------ geometry on the grid (batched)
def _shift(x: torch.Tensor, dz: int, dr: int, fill: float) -> torch.Tensor:
    """x shifted so that out[..., i, j] = x[..., i + dz, j + dr] (fill outside)."""
    out = torch.full_like(x, fill)
    nz, nr = x.shape[-2:]
    zs, zd = (slice(max(0, dz), nz + min(0, dz)), slice(max(0, -dz), nz - max(0, dz)))
    rs, rd = (slice(max(0, dr), nr + min(0, dr)), slice(max(0, -dr), nr - max(0, dr)))
    out[..., zd, rd] = x[..., zs, rs]
    return out


def find_axis(M: Machine, psi: torch.Tensor, center=None, half=(0.45, 1.0), adapt: bool = False):
    """Maximum of psi over the allowed cells (inside a window around ``center`` if given) with
    parabolic refinement. Returns (iz, ir, Rax, Zax, psi_ax).

    ``adapt``: where the maximum lies on the window edge (the true axis may be outside), the window is
    doubled for that frame, at most twice; the allowed region (plasma box minus the conductors) is kept."""
    B = psi.shape[0]
    nz, nr = M.shape
    allowed0 = M.t_ok[None].expand(B, nz, nr)

    def pick(hr, hz):
        allowed = allowed0
        if center is not None:
            Rc, Zc = center
            win = ((M.t_R[None, None, :] - Rc[:, None, None]).abs() < hr[:, None, None]) & \
                  ((M.t_Z[None, :, None] - Zc[:, None, None]).abs() < hz[:, None, None])
            allowed = allowed & win
        masked = torch.where(allowed, psi, torch.full_like(psi, -1e30)).reshape(B, -1)
        return masked.argmax(1)

    hr = torch.full((B,), float(half[0]), device=psi.device)
    hz = torch.full((B,), float(half[1]), device=psi.device)
    idx = pick(hr, hz)
    if adapt and center is not None:
        for _ in range(2):
            iz_, ir_ = idx // nr, idx % nr
            edge = (((M.t_R[ir_] - center[0]).abs() >= hr - 1.5 * M.dR) | ((M.t_Z[iz_] - center[1]).abs() >= hz - 1.5 * M.dZ))
            if not bool(edge.any()):
                break
            hr = torch.where(edge, 2 * hr, hr)
            hz = torch.where(edge, 2 * hz, hz)
            idx = torch.where(edge, pick(hr, hz), idx)
    iz, ir = idx // nr, idx % nr
    iz_c, ir_c = iz.clamp(1, nz - 2), ir.clamp(1, nr - 2)
    b = torch.arange(B, device=psi.device)
    c = psi[b, iz_c, ir_c]
    l, r_ = psi[b, iz_c, ir_c - 1], psi[b, iz_c, ir_c + 1]
    d, u = psi[b, iz_c - 1, ir_c], psi[b, iz_c + 1, ir_c]
    den_r = (l - 2 * c + r_).clamp(max=-1e-12)
    den_z = (d - 2 * c + u).clamp(max=-1e-12)
    sr = (0.5 * (l - r_) / den_r).clamp(-1, 1)
    sz = (0.5 * (d - u) / den_z).clamp(-1, 1)
    psi_ax = c - 0.25 * ((r_ - l) * sr + (u - d) * sz)
    Rax = M.t_R[ir_c] + sr * M.dR
    Zax = M.t_Z[iz_c] + sz * M.dZ
    return iz_c, ir_c, Rax, Zax, psi_ax


def find_xpoint(M: Machine, psi: torch.Tensor, psi_ax: torch.Tensor, Rax: torch.Tensor):
    """The saddle point with the highest flux below the axis level. Returns (found, R, Z, psi_x)."""
    B = psi.shape[0]
    nz, nr = M.shape
    dR, dZ = M.dR, M.dZ
    gr = (_shift(psi, 0, 1, 0.0) - _shift(psi, 0, -1, 0.0)) / (2 * dR)
    gz = (_shift(psi, 1, 0, 0.0) - _shift(psi, -1, 0, 0.0)) / (2 * dZ)
    g2 = gr ** 2 + gz ** 2
    hrr = (_shift(psi, 0, 1, 0.0) - 2 * psi + _shift(psi, 0, -1, 0.0)) / dR ** 2
    hzz = (_shift(psi, 1, 0, 0.0) - 2 * psi + _shift(psi, -1, 0, 0.0)) / dZ ** 2
    hrz = (_shift(psi, 1, 1, 0.0) - _shift(psi, 1, -1, 0.0) - _shift(psi, -1, 1, 0.0)
           + _shift(psi, -1, -1, 0.0)) / (4 * dR * dZ)
    det = hrr * hzz - hrz ** 2
    interior = torch.zeros_like(M.t_ok_saddle)
    interior[1:-1, 1:-1] = True
    # local minimum of |grad psi|^2 over the 3x3 neighbourhood
    neigh = torch.stack([_shift(g2, a, b, 1e30) for a in (-1, 0, 1) for b in (-1, 0, 1)], 0)
    is_min = g2 <= neigh.min(0).values
    cand = is_min & (det < 0) & M.t_ok_saddle & interior & (psi < psi_ax[:, None, None] - 1e-9)
    score = torch.where(cand, psi, torch.full_like(psi, -1e30)).reshape(B, -1)
    best, idx = score.max(1)
    found = best > -1e29
    iz, ir = (idx // nr).clamp(1, nz - 2), (idx % nr).clamp(1, nr - 2)
    b = torch.arange(B, device=psi.device)
    g = torch.stack([gr[b, iz, ir], gz[b, iz, ir]], 1)
    H = torch.stack([torch.stack([hrr[b, iz, ir], hrz[b, iz, ir]], 1),
                     torch.stack([hrz[b, iz, ir], hzz[b, iz, ir]], 1)], 1)
    H = H + 1e-12 * torch.eye(2, device=psi.device)
    step = -torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
    step = torch.stack([step[:, 0].clamp(-dR, dR), step[:, 1].clamp(-dZ, dZ)], 1)
    psi_x = psi[b, iz, ir] + 0.5 * (g * step).sum(1)
    return found, M.t_R[ir] + step[:, 0], M.t_Z[iz] + step[:, 1], psi_x


def bottleneck(M: Machine, psi: torch.Tensor, iz: torch.Tensor, ir: torch.Tensor, up: int = 1,
               return_cell: bool = False):
    """Flooding level of every cell from the axis, and the limiting flux ``L``.

    ``m(x)`` is the highest level L such that x lies in the component of {psi > L} that contains
    the axis (the best path's lowest flux).  ``L`` is the highest level at which that component
    reaches a cell the plasma cannot occupy (outside the box, or inside a coil's keep-out zone):
    for a diverted plasma that is the separatrix flux (the SOL surfaces just below it run down
    the legs to the wall), for a limited plasma the flux at the limiter.  Both topologies and
    stray hills of flux around the coils are handled by the same rule, and no saddle search is
    needed.  Paths run through allowed cells only.  Returns (m, L); closed region = ``m > L``.
    """
    B = psi.shape[0]
    ok = M.t_ok
    neg = torch.full_like(psi, -1e30)
    m = neg.clone()
    b = torch.arange(B, device=psi.device)
    m[b, iz, ir] = psi[b, iz, ir]

    def nbr_max(x):
        return torch.nn.functional.max_pool2d(x[:, None], 3, 1, 1)[:, 0]
    for it in range(2 * max(M.shape)):
        new = torch.minimum(psi, nbr_max(m))
        new = torch.where(ok, new, neg)
        new = torch.maximum(new, m)
        if it % 8 == 7 and bool((new == m).all()):
            m = new
            break
        m = new
    lb = torch.minimum(psi, nbr_max(m))
    L, cell = torch.where(~ok, lb, neg).reshape(B, -1).max(1)
    if return_cell:
        nr = M.shape[1]
        return m, L, (M.t_R[cell % nr], M.t_Z[cell // nr])
    return m, L


def closed_region(M: Machine, psi: torch.Tensor, level: torch.Tensor, iz, ir) -> torch.Tensor:
    """Cells with psi above ``level`` connected (4-neighbour) to the axis cell."""
    m, _ = bottleneck(M, psi, iz, ir)
    return m > level[:, None, None]


# ------------------------------------------------------------------ profiles
@dataclass
class Profile:
    """P(x) = (1 - x^alpha)^gamma on psi_N in [0, 1]; its integral table for the pressure."""
    alpha: float = 1.0
    gamma: float = 1.0
    f_p: float = 1.0                   # total / electron pressure at the same flux (fixed factor)

    def P(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x.clamp(0, 1) ** self.alpha).clamp_min(0.0) ** self.gamma

    def tail(self, x: torch.Tensor, n: int = 200) -> torch.Tensor:
        """integral_x^1 P(s) ds, by a fixed table."""
        s = torch.linspace(0, 1, n + 1, device=x.device)
        P = self.P(s)
        cum = torch.cat([torch.zeros(1, device=x.device), torch.cumsum(0.5 * (P[1:] + P[:-1]) / n, 0)])
        total = cum[-1]
        idx = (x.clamp(0, 1) * n)
        i0 = idx.floor().long().clamp(0, n - 1)
        frac = idx - i0
        c = cum[i0] * (1 - frac) + cum[i0 + 1] * frac
        return total - c


def interp_grid(M: Machine, f: torch.Tensor, Rp: torch.Tensor, Zp: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of ``f (B, nz, nr)`` at points ``(B, N)``."""
    B, N = Rp.shape
    nz, nr = M.shape
    fr = ((Rp - M.t_R[0]) / M.dR).clamp(0, nr - 1 - 1e-4)
    fz = ((Zp - M.t_Z[0]) / M.dZ).clamp(0, nz - 1 - 1e-4)
    r0, z0 = fr.floor().long(), fz.floor().long()
    wr, wz = fr - r0, fz - z0
    b = torch.arange(B, device=f.device)[:, None]
    return (f[b, z0, r0] * (1 - wr) * (1 - wz) + f[b, z0, r0 + 1] * wr * (1 - wz)
            + f[b, z0 + 1, r0] * (1 - wr) * wz + f[b, z0 + 1, r0 + 1] * wr * wz)



def shift_z(M: Machine, J: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """J(R, Z - delta) by bilinear resampling along Z. ``delta`` is (B,) or (B, K); returns (B[*K], nz, nr)."""
    B = J.shape[0]
    nz, nr = M.shape
    if delta.dim() == 1:
        delta = delta[:, None]
    K = delta.shape[1]
    zr = float(M.Z[-1] - M.Z[0])
    yb = 2 * (M.t_Z - float(M.Z[0])) / zr - 1                              # (nz,)
    ys = yb[None, None, :, None] - 2 * delta[:, :, None, None] / zr        # (B, K, nz, 1)
    xs = torch.linspace(-1, 1, nr, device=J.device)[None, None, None, :].expand(B, K, nz, nr)
    grid = torch.stack([xs, ys.expand(B, K, nz, nr)], -1).reshape(B * K, nz, nr, 2)
    src = J[:, None].expand(B, K, nz, nr).reshape(B * K, 1, nz, nr)
    out = torch.nn.functional.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(B, K, nz, nr)


def _vb_root(M, J, rr, br_vac, cellA, span, n):
    B = J.shape[0]
    deltas = torch.linspace(-span, span, n, device=J.device)[None, :].expand(B, n)
    Js = shift_z(M, J, deltas)                                              # (B, n, nz, nr)
    F = (Js * (rr * br_vac)[:, None] * cellA).sum((2, 3))                  # (B, n)
    # root: first sign change nearest the centre, else the minimum |F|
    sign = torch.sign(F)
    change = (sign[:, :-1] * sign[:, 1:]) <= 0
    centre = torch.abs(torch.arange(n - 1, device=J.device) - (n - 2) / 2)[None, :]
    score = torch.where(change, centre, torch.full_like(centre, 1e9).expand(B, n - 1))
    k = score.argmin(1)
    has = change.any(1)
    b = torch.arange(B, device=J.device)
    f0, f1 = F[b, k], F[b, k + 1]
    d0, d1 = deltas[b, k], deltas[b, k + 1]
    root = d0 - f0 * (d1 - d0) / torch.where((f1 - f0).abs() < 1e-30, torch.full_like(f1, 1e-30), f1 - f0)
    fallback = deltas[b, F.abs().argmin(1)]
    delta = torch.where(has, root, fallback)
    edge = (~has) | (k == 0) | (k == n - 2)                                 # no root, or the root in an end interval
    return delta, edge


def vertical_balance(M: Machine, J: torch.Tensor, rr: torch.Tensor, br_vac: torch.Tensor, cellA: float,
                     span: float = 0.25, n: int = 26, expand: int = 0, info: dict | None = None):
    """Shift J vertically so the external field exerts no net vertical force on it.

    The vertical force is ``2 pi sum J R B_R,ext dA``; the plasma's own field contributes none, so
    this is exact vertical force balance. Returns (shifted J, shift in m).

    ``expand``: frames whose root sits at the edge of the shift grid (or has none) are searched again on a grid
    extended by one more ``span`` on each side, at the same step, up to ``expand`` times; the other frames keep
    the first answer. ``info`` (optional dict) receives per frame: ``edge0`` (edge hit on the first grid),
    ``edge`` (after the extensions) and ``cut`` (share of the current pushed off the grid by the shift).
    """
    delta, edge0 = _vb_root(M, J, rr, br_vac, cellA, span, n)
    edge = edge0.clone()
    step = 2 * span / (n - 1)
    for e in range(1, int(expand) + 1):
        if not bool(edge.any()):
            break
        idx = torch.nonzero(edge).flatten()
        sp = span * (1 + e)
        ne = int(round(2 * sp / step)) + 1
        d2, e2 = _vb_root(M, J[idx], rr[idx], br_vac[idx], cellA, sp, ne)
        delta[idx] = torch.where(e2, delta[idx], d2)
        edge[idx] = e2
    Js = shift_z(M, J, delta)[:, 0]
    if info is not None:
        info["edge0"], info["edge"] = edge0, edge
        info["cut"] = 1 - Js.sum((1, 2)) / J.sum((1, 2)).clamp_min(1e-30)
    return Js, delta


@dataclass
class Result:
    psi: torch.Tensor            # (B, nz, nr) physical sign, total flux
    inside: torch.Tensor         # (B, nz, nr) closed-flux region
    psi_n: torch.Tensor
    psi_ax: torch.Tensor
    psi_b: torch.Tensor
    Rax: torch.Tensor
    Zax: torch.Tensor
    Rx: torch.Tensor
    Zx: torch.Tensor
    found_x: torch.Tensor
    beta0: torch.Tensor
    lam: torch.Tensor
    i_ctrl: torch.Tensor         # fictitious control-pair current (A): the consistency diagnostic
    residual: torch.Tensor       # last Picard update, relative to the flux span
    J: torch.Tensor


def solve(M: Machine, currents: torch.Tensor, ip: torch.Tensor, therm: dict | None, prof: Profile,
          init_center: tuple[torch.Tensor, torch.Tensor] | None = None, iters: int = 80,
          relax: float = 0.6, init_psi: torch.Tensor | None = None, beta0_fixed: float | None = None,
          pressure_gain: float = 1.0, verbose: bool = False, tab: dict | None = None,
          init_beta0: torch.Tensor | None = None, vb_span: float = 0.25, vb_n: int = 26,
          axis_half: tuple[float, float] = (0.45, 1.0), keep_center: bool = False,
          p_psin_max: float = 0.98, p_min_pts: int = 5, init_a: torch.Tensor | None = None,
          ip_conserve: bool = False, axis_adapt: bool = False, vb_expand: int = 0) -> Result:
    """Batched free-boundary solve for ``currents (B, n_col)`` [A], plasma current ``ip (B,)`` [A].

    ``therm`` (optional): dict(R (B,N), Z (B,N), p (B,N)) Thomson electron pressure points [Pa]
    with ``p`` NaN where a point is invalid. Without it ``beta0_fixed`` (default 0.5) is used.
    ``vb_span`` / ``vb_n``: the vertical-shift grid of the force balance; ``axis_half``: the (R, Z)
    half-widths of the axis search window around the previous axis; ``keep_center``: a warm start's
    ``init_center`` bounds the first iteration's axis search too (False: the first search is
    unbounded); ``p_psin_max`` / ``p_min_pts``: Thomson points used by the pressure matching;
    ``init_a`` (B,): width [m] of the cold-start current blob (default a quarter of the box width);
    ``ip_conserve``: the current shifted by the force balance is rescaled to the measured Ip (cells pushed
    off the grid would otherwise lose current); ``axis_adapt``: see ``find_axis``.
    """
    dev = torch.device(M.device)
    B = currents.shape[0]
    nz, nr = M.shape
    psi_vac, br_vac = M.external(currents)
    rr = M.t_R[None, None, :].expand(B, nz, nr)
    cellA = M.dR * M.dZ
    ip = ip.to(dev).float()
    # --- initial plasma: a broad blob at the requested centre (default: grid centre of the box)
    if init_psi is None:
        if init_center is None:
            Rc = torch.full((B,), 0.5 * (M.box[0] + M.box[1]), device=dev)
            Zc = torch.zeros(B, device=dev)
        else:
            Rc, Zc = init_center
        a = init_a[:, None, None] if init_a is not None else 0.25 * (M.box[1] - M.box[0])
        zz = M.t_Z[None, :, None]
        blob = torch.exp(-0.5 * (((rr - Rc[:, None, None]) / a) ** 2 + ((zz - Zc[:, None, None]) / (1.6 * a)) ** 2))
        J0 = blob / (blob.sum((1, 2), keepdim=True) * cellA) * ip[:, None, None]
        psi = psi_vac + ((J0 * cellA).reshape(B, -1) @ M.G.T).reshape(B, nz, nr)
        center = (Rc, Zc)
    else:
        psi = init_psi.clone()
        if init_center is not None:
            center = init_center
    i_ctrl = torch.zeros(B, device=dev)
    if not (keep_center and init_psi is not None and init_center is not None):
        center = None
    cen_prev = None
    beta0 = (init_beta0.clone() if init_beta0 is not None
             else torch.full((B,), 0.5 if beta0_fixed is None else beta0_fixed, device=dev))
    lam = torch.zeros(B, device=dev)
    last = None
    for it in range(iters):
        iz, ir, Rax, Zax, psi_ax = find_axis(M, psi, center, axis_half, axis_adapt)
        center = (Rax, Zax)
        m_fl, psi_b = bottleneck(M, psi, iz, ir)
        span = (psi_ax - psi_b).clamp_min(1e-9)
        inside = m_fl > psi_b[:, None, None]
        x = ((psi_ax[:, None, None] - psi) / span[:, None, None])
        P = prof.P(x) * inside
        w1 = (rr / Rax[:, None, None])
        w2 = 1.0 / w1.clamp_min(1e-3)
        A1 = (w1 * P).sum((1, 2)) * cellA
        A2 = (w2 * P).sum((1, 2)) * cellA
        if therm is not None and beta0_fixed is None:
            # pressure matching: q = lam * beta0 * span / Rax scales p(psi_N) = q * tail(psi_N)
            xp = interp_grid(M, x, therm["R"], therm["Z"])
            ok = torch.isfinite(therm["p"]) & (xp >= 0) & (xp < p_psin_max) & (therm["p"] > 0)
            tail = prof.tail(xp)
            target = torch.where(ok, therm["p"] * prof.f_p * pressure_gain, torch.zeros_like(tail))
            num = (tail * target * ok).sum(1)
            den = (tail * tail * ok).sum(1).clamp_min(1e-30)
            q = (num / den).clamp_min(1e-9)
            have = ok.sum(1) >= p_min_pts
            need = ip * span / (q * Rax)                       # = A1 + A2 (1 - beta0)/beta0
            ratio = ((need - A1) / A2.clamp_min(1e-12)).clamp_min(0.0)
            b_new = 1.0 / (1.0 + ratio)
            b_new = b_new.clamp(0.05, 0.85)
            beta0 = torch.where(have, beta0 + (0.3 if (it > 0 or init_beta0 is not None) else 1.0) * (b_new - beta0), beta0)
        lam = ip / (beta0 * A1 + (1 - beta0) * A2).clamp_min(1e-12)
        J = lam[:, None, None] * (beta0[:, None, None] * w1 + (1 - beta0)[:, None, None] * w2) * P
        if tab is not None:
            # tabulated flux functions on psi_N in [0, 1]: J = R p'(x) + FF'(x) / (mu0 R)
            K = tab["pp"].shape[1]
            xi = (x.clamp(0, 1) * (K - 1))
            i0 = xi.floor().long().clamp(0, K - 2)
            fr = xi - i0
            def look(t):
                g0 = torch.gather(t, 1, i0.reshape(B, -1)).reshape(x.shape)
                g1 = torch.gather(t, 1, (i0 + 1).reshape(B, -1)).reshape(x.shape)
                return g0 * (1 - fr) + g1 * fr
            if "ffs" in tab:
                # pressure-driven part fixed by the table; the poloidal-current part is a shape with
                # an amplitude chosen so the total current is the measured one
                Jp = rr * look(tab["pp"]) * inside
                Jf = look(tab["ffs"]) / rr * inside
                c = (ip - Jp.sum((1, 2)) * cellA) / (Jf.sum((1, 2)) * cellA).clamp_min(1e-12)
                scale_p = torch.where(c < 0, ip / (Jp.sum((1, 2)) * cellA).clamp_min(1e-9), torch.ones_like(c))
                J = Jp * scale_p[:, None, None] + Jf * c.clamp_min(0)[:, None, None]
            else:
                J = (rr * look(tab["pp"]) + look(tab["ff"]) / rr) * inside
                if tab.get("ip_scale", True):
                    J = J * (ip / (J.sum((1, 2)) * cellA).clamp_min(1e-9))[:, None, None]
        # vertical force balance: shift J along Z by the delta that zeroes sum(J R B_R,ext) dA
        vbi = {}
        J, delta = vertical_balance(M, J, rr, br_vac, cellA, vb_span, vb_n, vb_expand, vbi)
        if ip_conserve:
            J = J * (ip / (J.sum((1, 2)) * cellA).clamp_min(1e-9))[:, None, None]
        i_ctrl = delta
        psi_new = psi_vac + ((J * cellA).reshape(B, -1) @ M.G.T).reshape(B, nz, nr)
        res = (psi_new - psi).abs().amax((1, 2)) / span
        # limit how fast the current centroid may move between iterations (runaway guard)
        Jsum = J.sum((1, 2)).clamp_min(1e-9)
        cen = torch.stack([(J * rr).sum((1, 2)) / Jsum, (J * M.t_Z[None, :, None]).sum((1, 2)) / Jsum], 1)
        if cen_prev is not None:
            moved = (cen - cen_prev).norm(dim=1)
            omega = (relax * torch.clamp(0.03 / moved.clamp_min(1e-6), max=1.0))[:, None, None]
        else:
            omega = relax
        cen_prev = cen
        psi = psi + omega * (psi_new - psi)
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  iter {it}: median residual {float(res.median()):.2e}",
                  flush=True)
    iz, ir, Rax, Zax, psi_ax = find_axis(M, psi, center, axis_half, axis_adapt)
    found, Rx, Zx, psi_x = find_xpoint(M, psi, psi_ax, Rax)
    m_fl, psi_b = bottleneck(M, psi, iz, ir)
    inside = m_fl > psi_b[:, None, None]
    x = ((psi_ax[:, None, None] - psi) / (psi_ax - psi_b).clamp_min(1e-9)[:, None, None])
    out = Result(psi, inside, x, psi_ax, psi_b, Rax, Zax, Rx, Zx, found, beta0, lam, i_ctrl, res, J)
    if iters:                                    # force-balance diagnostics of the last iteration (label-free)
        out.vb_edge0, out.vb_edge, out.cut = vbi["edge0"], vbi["edge"], vbi["cut"]
    return out


# ------------------------------------------------------------------ assimilation of iso-flux pairs
def _bounded(raw, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(raw)


def _unbounded(v, lo, hi):
    v = (v - lo) / (hi - lo)
    return math.log(v / (1 - v))


THETA0 = (1.0, 1.25, 1.0, 1.5)          # alpha_p, gamma_p, alpha_f, gamma_f (DIII-D fit of the flux-function shapes)
THETA_RANGE = ((0.5, 4.0), (0.3, 3.0), (0.5, 4.0), (0.3, 3.0))


def solve_assim(M: Machine, currents: torch.Tensor, ip: torch.Tensor, therm: dict, pairs: torch.Tensor,
                f_p: float = 1.5, theta0=THETA0, init_center=None, iters: int = 80, relax: float = 0.6,
                init_psi: torch.Tensor | None = None, init_theta: torch.Tensor | None = None,
                lr: float = 0.08, reg: float = 0.02, w_press: float = 0.0, n_tab: int = 64, verbose: bool = False,
                huber: float = 0.3, w_pair: float = 1.0, vb_span: float = 0.25, vb_n: int = 26,
                axis_half: tuple[float, float] = (0.45, 1.0), p_psin_max: float = 0.98, p_min_pts: int = 5,
                init_a: torch.Tensor | None = None, ip_conserve: bool = False, axis_adapt: bool = False,
                vb_expand: int = 0) -> Result:
    """Free-boundary solve with the flux-function shapes fitted, while iterating, to iso-flux pairs.

    ``pairs`` (B, L, 2): major radii on the midplane of points that must carry the same flux (for
    MAST: the inboard and outboard points of equal electron temperature -- Te is a flux function).
    NaN rows are ignored.  The four shape exponents (alpha_p, gamma_p, alpha_f, gamma_f) of the
    pressure-gradient and poloidal-current profiles are the free parameters; the pressure *amplitude*
    comes from the Thomson pressure (closed form each iteration), the poloidal-current amplitude
    from the measured Ip.  Each iteration takes one Adam step on the mismatch of the one-step
    look-ahead flux psi_ext + G J(theta) at the pairs, then the usual relaxed Picard update.
    """
    dev = torch.device(M.device)
    B = currents.shape[0]
    nz, nr = M.shape
    psi_vac, br_vac = M.external(currents)
    rr = M.t_R[None, None, :].expand(B, nz, nr)
    cellA = M.dR * M.dZ
    ip = ip.to(dev).float()
    center = None
    if init_psi is None:
        Rc, Zc = init_center if init_center is not None else (torch.full((B,), 0.5 * (M.box[0] + M.box[1]), device=dev), torch.zeros(B, device=dev))
        a = init_a[:, None, None] if init_a is not None else 0.25 * (M.box[1] - M.box[0])
        zz = M.t_Z[None, :, None]
        blob = torch.exp(-0.5 * (((rr - Rc[:, None, None]) / a) ** 2 + ((zz - Zc[:, None, None]) / (1.6 * a)) ** 2))
        J0 = blob / (blob.sum((1, 2), keepdim=True) * cellA) * ip[:, None, None]
        psi = psi_vac + ((J0 * cellA).reshape(B, -1) @ M.G.T).reshape(B, nz, nr)
        center = (Rc, Zc)
    else:
        psi = init_psi.clone()
        if init_center is not None:
            center = init_center
    raw0 = torch.tensor([_unbounded(v, lo, hi) for v, (lo, hi) in zip(theta0, THETA_RANGE)], device=dev)
    raw = (init_theta.clone() if init_theta is not None else raw0[None].repeat(B, 1)).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    xs = (torch.arange(n_tab, device=dev) + 0.5) / n_tab
    pr_in, pr_out = pairs[..., 0], pairs[..., 1]
    pair_ok = torch.isfinite(pr_in) & torch.isfinite(pr_out)
    z0 = torch.zeros_like(pr_in)
    cen_prev = None
    res_last = torch.zeros(B, device=dev)
    for it in range(iters):
        with torch.no_grad():
            iz, ir, Rax, Zax, psi_ax = find_axis(M, psi, center, axis_half, axis_adapt)
            center = (Rax, Zax)
            m_fl, psi_b = bottleneck(M, psi, iz, ir)
            span = (psi_ax - psi_b).clamp_min(1e-9)
            inside = m_fl > psi_b[:, None, None]
            x = ((psi_ax[:, None, None] - psi) / span[:, None, None]).clamp(0, 1)
        th = torch.stack([_bounded(raw[:, k], *THETA_RANGE[k]) for k in range(4)], 1)
        ap, gp, af, gf = (th[:, k] for k in range(4))
        Pp = (1 - x ** ap[:, None, None]).clamp_min(1e-6) ** gp[:, None, None] * inside
        Pf = (1 - x ** af[:, None, None]).clamp_min(1e-6) ** gf[:, None, None] * inside
        # pressure amplitude: p(x) = span * Ap * T(x), T(x) = integral_x^1 Pp
        tab_p = (1 - xs[None] ** ap[:, None]).clamp_min(1e-6) ** gp[:, None]            # (B, n_tab)
        T_tab = torch.flip(torch.cumsum(torch.flip(tab_p, [1]), 1), [1]) / n_tab
        if therm is not None:
            xp = interp_grid(M, x, therm["R"], therm["Z"])
            okp = torch.isfinite(therm["p"]) & (xp >= 0) & (xp < p_psin_max) & (therm["p"] > 0)
            idx = (xp * n_tab).clamp(0, n_tab - 1.001)
            i0 = idx.floor().long()
            Ti = torch.gather(T_tab, 1, i0) * (1 - (idx - i0)) + torch.gather(T_tab, 1, i0 + 1) * (idx - i0)
            y = torch.where(okp, therm["p"] * f_p, torch.zeros_like(xp))
            Ap = ((Ti * y * okp).sum(1) / (span * (Ti * Ti * okp).sum(1)).clamp_min(1e-30))
            Ap = torch.where(okp.sum(1) >= p_min_pts, Ap, torch.full_like(Ap, 1e3))
        else:
            Ap = torch.full((B,), 1e3, device=dev)
        Jp = Ap[:, None, None] * rr * Pp
        Ip_p = Jp.sum((1, 2)) * cellA
        scale_p = torch.where(Ip_p > 0.9 * ip, 0.9 * ip / Ip_p.clamp_min(1e-9), torch.ones_like(Ip_p))
        Jp = Jp * scale_p[:, None, None]
        Jf_shape = Pf / rr
        Af = (ip - Jp.sum((1, 2)) * cellA) / (Jf_shape.sum((1, 2)) * cellA).clamp_min(1e-12)
        J = Jp + Af[:, None, None] * Jf_shape
        vbi = {}
        with torch.no_grad():
            _, delta = vertical_balance(M, J.detach(), rr, br_vac, cellA, vb_span, vb_n, vb_expand, vbi)
        Js = shift_z(M, J, delta)[:, 0]
        if ip_conserve:
            Js = Js * (ip / (Js.sum((1, 2)) * cellA).clamp_min(1e-9))[:, None, None]
        psi_new = psi_vac + ((Js * cellA).reshape(B, -1) @ M.G.T).reshape(B, nz, nr)
        # iso-flux pair mismatch of the look-ahead flux on the midplane
        fi = interp_grid(M, psi_new, torch.nan_to_num(pr_in), z0)
        fo = interp_grid(M, psi_new, torch.nan_to_num(pr_out), z0)
        dx = (fi - fo) / span[:, None]
        hub = torch.where(dx.abs() < huber, dx ** 2, huber * (2 * dx.abs() - huber))
        mis = (hub * pair_ok).sum(1) / pair_ok.sum(1).clamp_min(1)
        loss = (w_pair * mis + reg * ((raw - raw0) ** 2).mean(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"  it {it}: pair loss median {float(mis.median()):.4f} mean {float(mis.mean()):.4f}  theta {th.median(0).values.detach().cpu().numpy().round(2)} Ap {float(Ap.median()):.3g}", flush=True)
        with torch.no_grad():
            res_last = (psi_new - psi).abs().amax((1, 2)) / span
            Jsum = J.sum((1, 2)).clamp_min(1e-9)
            cen = torch.stack([(J * rr).sum((1, 2)) / Jsum, (J * M.t_Z[None, :, None]).sum((1, 2)) / Jsum], 1)
            omega = relax if cen_prev is None else (relax * torch.clamp(0.03 / (cen - cen_prev).norm(dim=1).clamp_min(1e-6), max=1.0))[:, None, None]
            cen_prev = cen
            psi = psi + omega * (psi_new.detach() - psi)
    with torch.no_grad():
        iz, ir, Rax, Zax, psi_ax = find_axis(M, psi, center, axis_half, axis_adapt)
        found, Rx, Zx, psi_x = find_xpoint(M, psi, psi_ax, Rax)
        m_fl, psi_b = bottleneck(M, psi, iz, ir)
        inside = m_fl > psi_b[:, None, None]
        x = ((psi_ax[:, None, None] - psi) / (psi_ax - psi_b).clamp_min(1e-9)[:, None, None])
    out = Result(psi.detach(), inside, x.detach(), psi_ax, psi_b, Rax, Zax, Rx, Zx, found, Ap.detach(), Af.detach(), delta.detach(), res_last, J.detach())
    out.theta = th.detach()
    out.raw = raw.detach()
    out.pair_mismatch = mis.detach()
    if iters:
        out.vb_edge0, out.vb_edge, out.cut = vbi["edge0"], vbi["edge"], vbi["cut"].detach()
    return out


def to_efit_sign(psi: torch.Tensor, machine: str) -> torch.Tensor:
    """DIII-D's EFIT stores the flux with its axis at the minimum; MAST's at the maximum."""
    return -psi if machine == "DIII-D" else psi
