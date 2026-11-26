"""Density-informed MAST geometry with retained baseline scalar/profile estimates."""
import numpy as np
from my_experiments.pipeline_v17 import ROOT
from my_experiments.mast_profile_anchor_v29 import stage_shot,anchor
from my_experiments.profile_gauge_v29 import preserve_mean


class MastRefinement:
    def __init__(self):
        from my_experiments.released_geometry import grid
        self.r, self.z, self.mask = grid()

    def predict_many(self,baselines,density_fields,pool,enabled=True):
        if not enabled:return [(dict(b),dict(enabled=False)) for b in baselines]
        jobs=[(b['psirz'],p,self.r,self.z,self.mask) for b,p in zip(baselines,density_fields)]
        results=[]
        for base,field,staged in zip(baselines,density_fields,pool.map(stage_shot,jobs)):
            adjusted,detail=anchor(base['psirz'],field,staged)
            result,gauge=preserve_mean(base['psirz'],field,adjusted)
            results.append((dict(base,psirz=result),dict(detail,**gauge,enabled=True)))
        return results
