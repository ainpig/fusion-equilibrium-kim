"""Keep the density field's gauge while adjusting its internal flux profile."""
import numpy as np


def preserve_mean(base,density,anchored):
    base,density,anchored=[np.asarray(v,np.float32) for v in (base,density,anchored)]
    active=np.any(anchored!=base,axis=(1,2))
    shift=density.astype(float).mean((1,2))-anchored.astype(float).mean((1,2))
    shift[~active]=0.
    result=(anchored.astype(float)+shift[:,None,None]).astype(np.float32)
    assert np.isfinite(result).all()
    return result,dict(active_frames=int(active.sum()),max_gauge_shift=float(np.max(np.abs(shift))))
