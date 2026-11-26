"""Exact 4x4 subcell area fractions on a single structured sampling grid."""
import numpy as np
from my_experiments.cartesian_polygon_v17 import grid_mask


def fraction(boundary, r, z):
    r, z = np.asarray(r), np.asarray(z)
    offsets = np.array([-.375, -.125, .125, .375])
    fr = (r[:, None]+offsets[None]*float(r[1]-r[0])).reshape(-1)
    fz = (z[:, None]+offsets[None]*float(z[1]-z[0])).reshape(-1)
    mask = grid_mask(boundary, fr, fz)
    return mask.reshape(len(z), 4, len(r), 4).mean((1, 3))
