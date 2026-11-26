"""Raw-input pressure-constrained field branch; scalars come from the baseline.

Identical to density_field_runtime_v29 except for the constraint set: the
electron-pressure equal-value pairs of pressure_constraints_v30 are appended to
the temperature and density pairs. Stage names and numerics are unchanged so the
two branches differ only by the extra observation.
"""
from my_experiments import reproduce_from_inputs_v25 as runner
from my_experiments import reproduction_io_v25 as audit
from my_experiments.pressure_constraints_v30 import prepare_kinetic
from my_experiments.fusion_pipeline import MAST_COILS,_interp
from my_experiments.pipeline_v17 import MastPredictor,ROOT
from my_experiments.pipeline_v15 import infer_fields
from my_experiments.pipeline_v16 import project
from my_experiments.vertical_transfer_v21 import VerticalTransfer,GAIN_LIMIT
from my_experiments.mast_correction_v22 import MastCorrection
from my_experiments.positive_current_v23 import refine
from my_experiments.reproduction_sampler_v25 import precision
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np
import torch
import time
import json


def prepare_seed(path):
    with threadpool_limits(limits=2,user_api='blas'):
        case=runner.load_input(path)
        exponent=json.loads((runner.ART/'kinetic_prior_v11.json').read_text())['te_exponent']
        kin,thermal=prepare_kinetic(case['row'],case['times'],exponent)
        from my_experiments import edge_isotherms_v14
        if edge_isotherms_v14.enabled():
            # v14: the same edge pairs as the baseline branch, after the T / n_e / p_e pairs
            from my_experiments.mast_inputs_v15 import profiles
            r,te,_,counts=profiles(case['row'],case['times'])
            thermal=edge_isotherms_v14.append_edge_pairs(r,te,counts,kin,thermal)
        curr=np.column_stack([_interp(case['row']['magnetics_time'],case['row'][k],case['times']) for k in MAST_COILS])
        if audit.POLICY is not None:
            valid=thermal[1][:,18:27]
            audit.atomic_json(audit.POLICY.output/'pressure_evidence'/(Path(path).stem+'.json'),
                 dict(frames=len(kin),pressure_pairs=int(valid.sum()),total_pairs=int(thermal[1].sum()),frames_with_pressure_pairs=int(np.any(valid>0,axis=1).sum())))
        runner.worker_audit()
        return dict(case,kin=kin,thermal=thermal,curr=curr)


def fields(run,paths,config,specification,workers,stop_after):
    torch.set_num_threads(2);start=time.perf_counter()
    with threadpool_limits(limits=2,user_api='blas'):
        first=runner.load_input(paths[0]);model=MastPredictor(first['row']);eq=model.eq
        with ProcessPoolExecutor(max_workers=workers,initializer=runner.worker_init,initargs=(specification,)) as pool:
            batch=config['mast_seed_batch_shots']
            for begin in range(0,len(paths),batch):
                ids=list(range(begin,min(begin+batch,len(paths))))
                if all(run.exists('density_seed',i) for i in ids):continue
                cases=list(pool.map(prepare_seed,[paths[i] for i in ids]))
                print(f'Density field seed {begin}/{len(paths)}: {sum(len(c["times"]) for c in cases)} frames',flush=True)
                with precision(config.get('stable_sampler',False)):
                    values,history=infer_fields(eq,cases,steps=config['mast_seed_steps'])
                for i,(phi,state) in zip(ids,values):run.save('density_seed',i,dict(psirz=phi.astype(np.float32)),dict(state=state,history=history))
                run.progress('Density seed',ids[-1]+1,len(paths),start)
            if stop_after=='seed':return dict(stopped_after='seed',shots=len(paths))
            for i,path in enumerate(paths):
                if run.exists('density_force',i):continue
                case=runner.load_input(path);phi=run.read('density_seed',i)['psirz']
                updated=project(eq,case['row'],case['times'],phi)[0]
                run.save('density_force',i,dict(psirz=updated))
                if (i+1)%100==0 or i+1==len(paths):run.progress('Density force',i+1,len(paths),start)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1,user_api='blas'):
        for begin in range(0,len(paths),18):
            ids=list(range(begin,min(begin+18,len(paths))))
            if all(run.exists('density_temporal',i) for i in ids):continue
            cases=[runner.load_input(paths[i]) for i in ids]
            bases=[run.read('density_force',i) for i in ids]
            values,_=model.fields(cases,bases)
            for i,phi in zip(ids,values):run.save('density_temporal',i,dict(psirz=phi))
            run.progress('Density temporal',ids[-1]+1,len(paths),start)
        from my_experiments.released_geometry import grid
        r, z, mask = grid(first['row'])
        vertical=VerticalTransfer(first['row'],r,z,mask)
        for i,path in enumerate(paths):
            case=runner.load_input(path)
            vertical.observe(run.read('density_temporal',i)['psirz'],case['row'],case['times'])
        fold=vertical.solve();audit.atomic_json(run.output/'mast_density_fold.json',dict(fold=fold))
        coil=MastCorrection(first['row'],r,z,mask,fold)
        for i,path in enumerate(paths):
            if run.exists('density_final',i):continue
            case=runner.load_input(path);phi=run.read('density_temporal',i)['psirz']
            moved,delta=vertical.apply(i,phi)
            phi=coil.predict(case['row'],case['times'],moved,delta)
            currents=np.column_stack([_interp(case['row']['magnetics_time'],case['row'][k],case['times']) for k in MAST_COILS])
            times=np.asarray(case['times'],float)
            row=dict(magnetics_time=times,**dict(zip(MAST_COILS,currents.T)))
            enabled=fold['gain']<GAIN_LIMIT
            once,_=refine(eq,row,times,phi,enabled)
            final,_=refine(eq,case['row'],case['times'],once,enabled)
            run.save('density_final',i,dict(psirz=final))
            if (i+1)%100==0 or i+1==len(paths):run.progress('Density final field',i+1,len(paths),start)
    result=run.assemble('mast','density_final',paths)
    result['scope']='density flux only; q95 and betaN are retained from the baseline'
    return result


def main():
    runner.mast=fields
    runner.main()


if __name__=='__main__':main()
