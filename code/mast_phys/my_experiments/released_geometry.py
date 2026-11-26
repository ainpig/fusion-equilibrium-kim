"""Numerical source support derived only from released coil rectangles.

This is not a measured vessel wall. It cannot supply limiter-contact surfaces.
Every process is configured with the same official input geometry by the driver.
"""
from __future__ import annotations
import numpy as np

_GEOMETRY = None


def from_row(row):
    r, z = (np.asarray(row[k], dtype=np.float64) for k in ("efit_grid_R", "efit_grid_Z"))
    names = np.asarray(row["coil_input_column"]).astype(str)
    cr, cz, cw, ch = (np.asarray(row[k], dtype=np.float64) for k in
                       ("coil_R", "coil_Z", "coil_width", "coil_height"))
    if r.ndim != 1 or z.ndim != 1 or min(len(r), len(z)) < 3:
        raise ValueError("Missing released flux grid")
    if not np.all(np.diff(r) > 0) or not np.all(np.diff(z) > 0):
        raise ValueError("Released flux coordinates must increase")
    if any(a.shape != names.shape or not np.isfinite(a).all() for a in (cr, cz, cw, ch)):
        raise ValueError("Invalid released conductor rectangles")
    sol = names == "magnetics_sol_current"
    outer = np.isin(names, ("magnetics_p5l_current", "magnetics_p5u_current"))
    ends = np.isin(names, ("magnetics_p2l_current", "magnetics_p2u_current"))
    if not sol.any():
        raise ValueError("Released solenoid geometry is required")
    dr = float(r[1] - r[0])
    import os
    gap = float(os.environ.get("FUSION_MAST_INNER_GAP_M", "0") or 0.0)   # centre-column tube + armour (recipe)
    rmin = max(float(r[0]), float(np.max(cr[sol] + cw[sol] / 2) + dr / 2) + gap)
    rmax = min(float(r[-1]), float(np.min(cr[outer] - cw[outer] / 2))) if outer.any() else float(r[-2])
    zmax = min(float(max(abs(z))), float(np.min(abs(cz[ends]) - ch[ends] / 2))) if ends.any() else float(max(abs(z)))
    rr, zz = np.meshgrid(r, z)
    mask = (rr > rmin) & (rr < rmax) & (abs(zz) < zmax)
    if mask.sum() < 16:
        raise ValueError("Released conductors leave no resolved plasma support")
    return r.copy(), z.copy(), mask, dict(rmin=rmin, rmax=rmax, zmax=zmax,
        source="official input coil rectangles and flux grid", measured_wall=False)


def configure(row):
    global _GEOMETRY
    geometry = from_row(row)
    if _GEOMETRY is not None:
        if any(not np.array_equal(a, b) for a, b in zip(_GEOMETRY[:3], geometry[:3])):
            raise ValueError("Geometry changed within one MAST run")
        if _GEOMETRY[3] != geometry[3]:
            raise ValueError("Conductor bounds changed within one MAST run")
    _GEOMETRY = geometry
    return geometry


def grid(row=None):
    geometry = configure(row) if row is not None else _GEOMETRY
    if geometry is None:
        raise RuntimeError("Configure MAST geometry from an official input row first")
    return tuple(a.copy() for a in geometry[:3])


def source_fraction(r, z, row):
    _, _, _, limits = from_row(row)
    dr, dz = float(r[1] - r[0]), float(z[1] - z[0])
    rr, zz = np.meshgrid(r, z)
    fraction = np.zeros(rr.shape, dtype=np.float64)
    for a in (-.375, -.125, .125, .375):
        for b in (-.375, -.125, .125, .375):
            fraction += ((rr + a * dr > limits["rmin"]) &
                         (rr + a * dr < limits["rmax"]) &
                         (abs(zz + b * dz) < limits["zmax"])) / 16
    return fraction


def diverted_geometry(phi, r, z, mask, resolution=257):
    from my_experiments.physical_scalars_v13 import critical_points, enclosing_contour
    from my_experiments.cartesian_polygon_v17 import grid_mask
    sp, axes, saddles = critical_points(phi, r, z, mask)
    axes = [p for p in axes if abs(p[1]) < .5 and .3 < p[0] < 1.5]
    if not axes:
        raise ValueError("No physical plasma O-point")
    axis = max(axes, key=lambda p: p[2])
    good = []
    for point in saddles:
        span = axis[2] - point[2]
        if span <= 1e-7:
            continue
        ir, iz = np.argmin(abs(r - point[0])), np.argmin(abs(z - point[1]))
        if not mask[iz, ir]:
            continue
        line = np.linspace(axis[:2], point[:2], 100)
        values = sp.ev(line[:, 1], line[:, 0])
        if (values.min() < point[2] - .005 * span or values.max() > axis[2] + .005 * span
                or np.maximum(np.diff(values), 0).sum() > .03 * span):
            continue
        good.append(point)
    if not good:
        raise ValueError("No accessible X-point; a numerical support edge is not a limiter")
    # Keep v3's flux ordering and contour offset; remove only wall contacts.
    xpoint = max(good, key=lambda p: p[2])
    fr, fz = np.linspace(r[0], r[-1], resolution), np.linspace(z[0], z[-1], resolution)
    flux = sp(fz, fr)
    span = axis[2] - xpoint[2]
    boundary = enclosing_contour(flux, xpoint[2] + 1e-4 * span, fr, fz, axis)
    qcurve = enclosing_contour(flux, axis[2] - .95 * span, fr, fz, axis)
    if boundary is None or qcurve is None:
        raise ValueError("Physical flux surface did not close")
    inside = grid_mask(boundary, fr, fz)
    if inside.sum() < 16:
        raise ValueError("Closed plasma volume unresolved")
    return dict(spline=sp, axis=axis, xpoint=xpoint, R=fr, Z=fz, flux=flux,
                boundary=boundary, qcurve=qcurve, inside=inside)
