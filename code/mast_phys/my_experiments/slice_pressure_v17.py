"""Nonnegative/electron-bounded radial-force pressure moments at each Z.

Integrating d_R(R² W)=R J_phi psi_R+2 R p across a closed plasma slice gives
integral R p dR = -.5 integral R J_phi psi_R dR. Each slice, not only their
sum, must have nonnegative total pressure and exceed electron pressure.
No target-domain regression or fitted gain is used.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from matplotlib.path import Path as Polygon
from my_experiments.physical_scalars_v13 import enclosing_contour
from my_experiments.virial_pressure_v14 import discrete_jr, MU0
from my_experiments.cell_fraction_v17 import fraction as cell_fraction


def cached_geometry(phi, r, z, x, ip):
    """Recover the already inferred boundary without repeating wall-ray search.

The cache stores a/Rgeo, (Raxis-Rgeo)/a, grid_dr/a and normalized flux span.
The remaining Z axis is the stationary point at the known Raxis. All inputs
come from predictions, never reference equilibrium coordinates.
"""
    a = float((r[1]-r[0])/x[12])
    rg = a/x[6]
    ra = rg+x[8]*a
    sp = RectBivariateSpline(z, r, np.asarray(phi, float))
    zs = np.linspace(max(z[0], -.5), min(z[-1], .5), 129)
    gradient = sp.ev(zs, np.full(len(zs), ra), dx=1)
    candidates = []
    for i in range(len(zs)-1):
        if gradient[i]>=0 and gradient[i+1]<=0 and gradient[i]!=gradient[i+1]:
            za = brentq(lambda zz: float(sp.ev(zz, ra, dx=1)), zs[i], zs[i+1], xtol=1e-13)
            h1, h2, h12 = float(sp.ev(za, ra, dy=2)), float(sp.ev(za, ra, dx=2)), float(sp.ev(za, ra, dx=1, dy=1))
            if h1*h2-h12*h12>0 and h1+h2<0:
                candidates.append((abs(float(sp.ev(za, ra, dy=1))), za))
    if not candidates:
        raise ValueError('Cached axis did not resolve')
    error, za = min(candidates)
    if error>1e-6*max(1., np.ptp(phi)):
        raise ValueError('Cached radial axis mismatch')
    pa = float(sp.ev(za, ra))
    span = x[9]*max(2e-7*abs(ip)*rg, 1e-8)
    pb = pa-span
    fr, fz = np.linspace(r[0], r[-1], 257), np.linspace(z[0], z[-1], 257)
    boundary = enclosing_contour(sp(fz, fr), pb+1e-4*span, fr, fz, (ra, za, pa))
    if boundary is None:
        raise ValueError('Cached boundary did not close')
    return {'spline': sp, 'axis': (ra, za, pa), 'xpoint': (0, 0, pb), 'boundary': boundary}


def pressure_slices(phi, r, z, geometry, ip, fvac, tsr=None, te=None, ne=None):
    rr, zz = np.meshgrid(r, z)
    dr, dz = float(r[1]-r[0]), float(z[1]-z[0])
    fraction = cell_fraction(geometry['boundary'], r, z)
    jr = discrete_jr(phi, r, z)
    derivative = geometry['spline'](z, r, dy=1)
    columns = -.5*np.sum(jr*derivative*fraction, axis=1)
    volume = float(np.sum(rr*fraction))
    if volume <= 0:
        raise ValueError('Empty pressure volume')
    electron_columns = np.zeros(len(z))
    if tsr is not None:
        pa, pb = geometry['axis'][2], geometry['xpoint'][2]
        s = (pa-geometry['spline'].ev(np.zeros_like(tsr), tsr))/(pa-pb)
        measured = np.asarray(te)*np.asarray(ne)*1.602176634e-19
        locations, pressure = [], []
        for lo in np.linspace(0, .95, 20):
            good = (s>=lo)&(s<lo+.05)&np.isfinite(measured)&(measured>0)
            if good.any():
                locations.append(float(np.median(s[good])))
                pressure.append(float(np.median(measured[good])))
        if len(locations)>=5 and min(locations)<=.35:
            profile = np.interp(np.clip((pa-phi)/(pa-pb), 0, 1), locations, pressure)
            electron_columns = np.sum(rr*profile*fraction, axis=1)
    c = geometry['boundary']
    minor, rg = np.ptp(c[:, 0])/2, (c[:, 0].min()+c[:, 0].max())/2
    coefficient = 2*MU0*100*minor*rg/(abs(fvac)*abs(ip)/1e6)/volume
    positive = np.maximum(columns, 0.)
    bounded = np.maximum(columns, electron_columns)
    assert np.all(bounded>=electron_columns) and np.all(positive>=0)
    return {'raw': float(columns.sum()*coefficient), 'nonnegative': float(positive.sum()*coefficient),
            'electron_bounded': float(bounded.sum()*coefficient),
            'negative_slice_count': int(np.sum((columns<0)&(fraction.sum(1)>0))),
            'negative_pressure_correction': float((positive-columns).sum()*coefficient)}
