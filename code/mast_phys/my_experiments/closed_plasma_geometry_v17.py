"""Use the same closed LCFS for pressure fitting and volume integration.

Flood-filling psi > psi_separatrix can numerically bridge an X-point and include
private-flux regions. The contour used by current fitting is already closed;
v3-1 integrates pressure over that contour without an external vessel mask.
"""
import numpy as np
from my_experiments.physical_scalars_v13 import physical_geometry as previous_geometry
from my_experiments.cartesian_polygon_v17 import grid_mask


def close_domain(geometry, r=None, z=None):
    result = dict(geometry)
    if 'R' not in result:
        result['R'] = np.linspace(r[0], r[-1], 257)
        result['Z'] = np.linspace(z[0], z[-1], 257)
        result['flux'] = result['spline'](result['Z'], result['R'])
    result['inside'] = grid_mask(result['boundary'], result['R'], result['Z'])
    if result['inside'].sum()<16:
        raise ValueError('Closed plasma volume unresolved')
    return result


def physical_geometry(phi, r, z, mask, resolution=257):
    return close_domain(previous_geometry(phi, r, z, mask, resolution))
