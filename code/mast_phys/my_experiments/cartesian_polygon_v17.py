"""Exact structured-grid polygon masking with boundary-point compatibility."""
import numpy as np
from matplotlib.path import Path as Polygon


def grid_mask(vertices, r, z):
    p = np.asarray(vertices, float)
    r, z = np.asarray(r, float), np.asarray(z, float)
    if len(p)<3:
        return np.zeros((len(z), len(r)), bool)
    q = np.roll(p, -1, axis=0)
    dy = q[:, 1]-p[:, 1]
    crossing = (p[None, :, 1]>z[:, None]) != (q[None, :, 1]>z[:, None])
    intersections = p[None, :, 0]+(z[:, None]-p[None, :, 1])*(q[None, :, 0]-p[None, :, 0])/np.where(dy!=0, dy, 1)[None]
    intersections = np.sort(np.where(crossing, intersections, np.inf), axis=1)
    output = np.zeros((len(z), len(r)), bool)
    polygon = Polygon(p)
    tolerance = 1e-12*max(np.ptp(r), np.ptp(z), 1.)
    for j, zz in enumerate(z):
        xx = intersections[j, np.isfinite(intersections[j])]
        if len(xx)%2:
            output[j] = polygon.contains_points(np.column_stack([r, np.full(len(r), zz)]))
            continue
        output[j] = np.searchsorted(xx, r, side='right')%2 == 1
        near = np.zeros(len(r), bool)
        for value in xx:
            k = int(np.searchsorted(r, value))
            for index in (k-1, k):
                if 0<=index<len(r) and abs(r[index]-value)<=tolerance:
                    near[index] = True
        flat = (abs(dy)<=tolerance)&(abs(p[:, 1]-zz)<=tolerance)
        for a, b in zip(p[flat, 0], q[flat, 0]):
            near |= (r>=min(a,b)-tolerance)&(r<=max(a,b)+tolerance)
        if near.any():
            output[j, near] = polygon.contains_points(np.column_stack([r[near], np.full(near.sum(), zz)]))
    return output
