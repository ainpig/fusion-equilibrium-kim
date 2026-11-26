"""Run v25 from raw input Parquets and fixed trained assets in one command.

No previous prediction, experiment cache, score or release manifest is read.
The fresh output directory holds intermediate products of THIS run only.
"""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['FUSION_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import gc
import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from my_experiments.reproduction_io_v25 import sha256, atomic_json, install, worker_init, worker_audit, replace_retry

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'my_experiments/artifacts'
KEYS = ('psirz', 'q95', 'betaN')


def load_input(path):
    import pyarrow.parquet as pq
    from my_experiments.kinetic_equilibrium import read_inputs
    row = read_inputs(path)
    from my_experiments.released_geometry import configure
    configure(row)
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    dtype = metadata.get(b'fusion_solver_time_dtype', b'float64').decode('ascii')
    if dtype not in ('float32','float64'):
        raise ValueError('Unsupported solver timestamp dtype')
    t = np.asarray(row['efit_times'], dtype=dtype)
    if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Expected finite, strictly increasing native EFIT times')
    forbidden = {'efit_psirz', 'efit_q95', 'efit_beta_n', 'magnetics_dsep'}
    assert not forbidden.intersection(row.keys())
    return dict(shot=str(path), row=row, times=t)


def prepare_seed(path):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2, user_api='blas'):
        return prepare_seed_two_blas(path)


def prepare_seed_two_blas(path):
    from my_experiments import edge_isotherms_v14
    if edge_isotherms_v14.enabled():
        # v14: the core isotherm pairs plus equal-temperature pairs at the plasma edge
        prepare_kinetic = edge_isotherms_v14.prepare_kinetic
    else:
        from my_experiments.mast_inputs_v15 import prepare_kinetic
    from my_experiments.fusion_pipeline import MAST_COILS, _interp
    case = load_input(path)
    exponent = json.loads((ART/'kinetic_prior_v11.json').read_text(encoding='utf-8'))['te_exponent']
    kin, thermal = prepare_kinetic(case['row'], case['times'], exponent)
    curr = np.column_stack([_interp(case['row']['magnetics_time'], case['row'][k], case['times']) for k in MAST_COILS])
    worker_audit()
    return dict(case, kin=kin, thermal=thermal, curr=curr)


def stage_readout(job):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2 if job[0] == 'original' else 1, user_api='blas'):
        return stage_readout_limited(job)


def stage_readout_limited(job):
    mode, case, phi = job
    if mode == 'closed':
        from my_experiments.pipeline_v17 import prepare_mast
        result = prepare_mast(case['row'], case['times'], phi)
    else:
        from my_experiments.gpu_readout_v15 import prepare_shot
        result = prepare_shot(case['row'], case['times'], phi)
    worker_audit()
    return result


def stage_final(job):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        return stage_final_single_blas(job)


def stage_final_single_blas(job):
    case, prediction, gain = job
    from my_experiments.pipeline_v23 import prepare
    result = prepare(case['row'], case['times'], prediction['psirz'], prediction['q95'], gain)
    worker_audit()
    return result


def solve_joint(job):
    # The original joint-pressure producer explicitly used one BLAS thread;
    # the other native producers used this host's recorded 12-thread default.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        return solve_joint_single_blas(job)


def solve_joint_single_blas(job):
    path, force, temporal, readout = job
    from my_experiments.mast_joint_v20 import prepare_joint
    from my_experiments.joint_beta_confidence_v20 import joint_estimate, refine
    case = load_input(path)
    prior = json.loads((ART/'d3d_uncertainty_calibration_v14.json').read_text(encoding='utf-8'))['prior']
    x, scale, floor, good = prepare_joint(case['row'], case['times'], force['psirz'])
    joint = joint_estimate(x, scale, floor, good, case['times'], force['betaN'])
    beta, gate = refine(temporal['betaN'], joint, x, scale, good, readout, prior)
    worker_audit()
    return dict(temporal, betaN=beta), dict(resolved=int(good.sum()), mean_gate=float(gate.mean()))


class Run:
    def __init__(self, output, identity, resume):
        self.output = output
        self.identity = identity
        if resume:
            old = json.loads((output/'run.json').read_text(encoding='utf-8'))
            assert old['identity'] == identity, 'Resume requires identical code/assets/inputs and settings'
            self.run_id = old['run_id']
        else:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('Choose a fresh output directory, or explicitly resume this same run')
            output.mkdir(parents=True, exist_ok=True)
            self.run_id = uuid.uuid4().hex
            atomic_json(output/'run.json', dict(identity=identity, run_id=self.run_id, complete=False,
                previous_prediction_cache_used=False))

    def path(self, stage, i):
        return self.output/'intermediate'/stage/f'{i:04d}.npz'

    def exists(self, stage, i):
        p = self.path(stage, i)
        if not (p.exists() and p.with_suffix('.json').exists()):
            return False
        m = json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        assert m['run_id'] == self.run_id and m['sha256'] == sha256(p)
        return True

    def save(self, stage, i, arrays, details=None):
        from my_experiments.reproduction_io_v25 import NpzStreamWriter
        p = self.path(stage, i)
        if self.exists(stage, i):
            # saved by an earlier process of this same run before it was stopped inside a batch; the batch is
            # recomputed from the same inputs and the stored shot is kept as it is
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        writer = NpzStreamWriter(p.with_suffix('.npz.partial'))
        try:
            for k, v in arrays.items():
                assert np.isfinite(v).all() and v.dtype == np.float32, (stage, i, k)
                writer.write(k, v)
        finally:
            writer.close()
        replace_retry(p.with_suffix('.npz.partial'), p)
        atomic_json(p.with_suffix('.json'), dict(run_id=self.run_id, sha256=sha256(p), details=details or {}))

    def read(self, stage, i):
        assert self.exists(stage, i)
        with np.load(self.path(stage, i)) as f:
            return {k: f[k] for k in KEYS}

    def details(self, stage, i):
        assert self.exists(stage, i)
        return json.loads(self.path(stage, i).with_suffix('.json').read_text(encoding='utf-8'))['details']

    def progress(self, stage, done, total, start):
        value = dict(run_id=self.run_id, stage=stage, done=done, total=total, seconds=time.perf_counter()-start,
            peak_allocated_gpu_GiB=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.)
        atomic_json(self.output/'progress.json', value)
        print(f'{stage}: {done}/{total}; {value["seconds"]:.1f}s', flush=True)

    def assemble(self, machine, stage, paths):
        from my_experiments.reproduction_io_v25 import NpzStreamWriter
        # The final phase swaps the public fold for a private one; the scorer finds files by name.
        fold = self.identity.get('fold', 'public')
        output = self.output/f'{machine}_{fold}_test.npz'
        writer = NpzStreamWriter(output.with_suffix('.npz.partial'))
        frames = 0
        try:
            for i in range(len(paths)):
                pred = self.read(stage, i)
                frames += len(pred['psirz'])
                for k, v in pred.items():
                    writer.write(f'shot_{i:04d}_{k}', v)
        finally:
            writer.close()
        replace_retry(output.with_suffix('.npz.partial'), output)
        result = dict(machine=machine, shots=len(paths), frames=frames, sha256=sha256(output), bytes=output.stat().st_size)
        atomic_json(output.with_suffix('.json'), result)
        return result


def d3d(run, paths, config):
    """Not part of this project.

    DIII-D is predicted by the trained ensemble in ``fusion_eq``; the stacked legacy
    chain that used to live here is not shipped and its checkpoints are not present.
    """
    raise NotImplementedError("DIII-D is predicted by nm1.predict, not here")


def mast(run, paths, config, specification, workers, stop_after):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2, user_api='blas'):
        return mast_stages(run, paths, config, specification, workers, stop_after)


def mast_stages(run, paths, config, specification, workers, stop_after):
    from threadpoolctl import threadpool_limits
    from my_experiments.pipeline_v17 import MastPredictor
    from my_experiments.pipeline_v15 import infer_fields
    from my_experiments.pipeline_v16 import project, finish
    from my_experiments.gpu_readout_v15 import readout_staged
    from my_experiments.beta_readout_v18 import refine_beta
    from my_experiments.vertical_transfer_v21 import VerticalTransfer
    from my_experiments.mast_correction_v22 import MastCorrection
    from my_experiments import pipeline_v23
    from my_experiments.positive_current_v23 import refine as current_projection
    torch.set_num_threads(2)
    start = time.perf_counter()
    geometry = load_input(paths[0])['row']
    model = MastPredictor(geometry)
    eq = model.eq
    prior = json.loads((ART/'d3d_uncertainty_calibration_v14.json').read_text(encoding='utf-8'))['prior']
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(specification,)) as pool:
        batch = config['mast_seed_batch_shots']
        for first in range(0, len(paths), batch):
            ids = list(range(first, min(first+batch, len(paths))))
            if all(run.exists('mast_seed', i) for i in ids):
                continue
            cases = list(pool.map(prepare_seed, [paths[i] for i in ids]))
            print(f'MAST seed: optimize {len(ids)} shots / {sum(len(c["times"]) for c in cases)} frames, 600 + 300 iterations', flush=True)
            from my_experiments.reproduction_sampler_v25 import precision
            with precision(config.get('stable_sampler', False)):
                fields, history = infer_fields(eq, cases, steps=config['mast_seed_steps'])
            staged = list(pool.map(stage_readout, [('original', c, f[0]) for c, f in zip(cases, fields)]))
            readout = readout_staged(staged, [np.full(len(c['times']),5.213475704193115) for c in cases],
                [np.full(len(c['times']),1.252995408430349) for c in cases], batch_frames=4096, q_method='conditioned')
            for i, (phi, state), (q, b, read) in zip(ids, fields, readout):
                run.save('mast_seed', i, dict(psirz=phi.astype(np.float32), q95=q, betaN=b),
                         dict(state=state, history=history))
            run.progress('MAST raw -> seed', ids[-1]+1, len(paths), start)
        if stop_after == 'seed':
            return dict(stopped_after='seed', shots=len(paths))
        batch = config['mast_force_batch_shots']
        for first in range(0, len(paths), batch):
            ids = list(range(first, min(first+batch, len(paths))))
            if all(run.exists('mast_force', i) for i in ids):
                continue
            cases = list(pool.map(load_input, [paths[i] for i in ids]))
            bases = [run.read('mast_seed', i) for i in ids]
            projected = [project(eq, c['row'], c['times'], b['psirz']) for c, b in zip(cases, bases)]
            jobs = [pool.submit(stage_readout, ('original', c, value[0])) if value[1].any() else None for c, value in zip(cases, projected)]
            staged = [j.result() if j is not None else None for j in jobs]
            results = finish([c['times'] for c in cases], [p[1] for p in projected], staged,
                [b['q95'] for b in bases], [b['betaN'] for b in bases])
            for i, value, (q, b, record) in zip(ids, projected, results):
                run.save('mast_force', i, dict(psirz=value[0], q95=q, betaN=b))
            run.progress('MAST force repair', ids[-1]+1, len(paths), start)
        torch.set_num_threads(1)
        late_blas = threadpool_limits(limits=1, user_api='blas')
        batch = config['mast_temporal_batch_shots']
        for first in range(0, len(paths), batch):
            ids = list(range(first, min(first+batch, len(paths))))
            if all(run.exists('mast_temporal', i) for i in ids):
                continue
            cases = list(pool.map(load_input, [paths[i] for i in ids]))
            bases = [run.read('mast_force', i) for i in ids]
            fields, _ = model.fields(cases, bases)
            staged = list(pool.map(stage_readout, [('closed', c, f) for c, f in zip(cases, fields)]))
            results = model.finish(staged, bases)
            for i, phi, b16, (q, b17, record) in zip(ids, fields, bases, results):
                b18, _ = refine_beta(b17, b16['betaN'], record, prior)
                run.save('mast_temporal', i, dict(psirz=phi, q95=q, betaN=b18), dict(readout=record))
            run.progress('MAST temporal current / pressure', ids[-1]+1, len(paths), start)
        for first in range(0, len(paths), 8):
            ids = [i for i in range(first, min(first+8,len(paths))) if not run.exists('mast_joint', i)]
            jobs = [(paths[i], run.read('mast_force', i), run.read('mast_temporal', i), run.details('mast_temporal', i)['readout']) for i in ids]
            for i, (pred, detail) in zip(ids, pool.map(solve_joint, jobs)):
                run.save('mast_joint', i, pred, detail)
            run.progress('MAST joint pressure', min(first+8,len(paths)), len(paths), start)
        if stop_after == 'joint':
            return dict(stopped_after='joint', shots=len(paths))
        from my_experiments.released_geometry import grid
        r, z, mask = grid(geometry)
        vertical = VerticalTransfer(geometry, r, z, mask)
        for i, path in enumerate(paths):
            c, b = load_input(path), run.read('mast_joint', i)
            vertical.observe(b['psirz'], c['row'], c['times'])
        fold = vertical.solve()
        atomic_json(run.output/'mast_fold.json', dict(fold=fold, computed_from_this_runs_predictions=True))
        coil = MastCorrection(geometry, r, z, mask, fold)
        for i, path in enumerate(paths):
            if run.exists('mast_coils', i):
                continue
            c, b = load_input(path), run.read('mast_joint', i)
            moved, displacement = vertical.apply(i, b['psirz'])
            phi = coil.predict(c['row'], c['times'], moved, displacement)
            run.save('mast_coils', i, dict(b, psirz=phi))
            if (i+1)%100 == 0 or i+1 == len(paths):
                run.progress('MAST fold / coil correction', i+1, len(paths), start)
        projector = pipeline_v23.CurrentCorrection.__new__(pipeline_v23.CurrentCorrection)
        projector.eq = eq
        batch = config['mast_final_batch_shots']
        for first in range(0, len(paths), batch):
            ids = list(range(first, min(first+batch,len(paths))))
            if all(run.exists('mast_final', i) for i in ids):
                continue
            cases = list(pool.map(load_input, [paths[i] for i in ids]))
            bases = [run.read('mast_coils', i) for i in ids]
            b16 = [run.read('mast_force', i)['betaN'] for i in ids]
            b18 = [run.read('mast_temporal', i)['betaN'] for i in ids]
            staged = list(pool.map(stage_final, [(c, b, fold['gain']) for c, b in zip(cases, bases)]))
            predictions, _ = pipeline_v23.finish(staged, bases, b18, b16, config['mast_final_batch_frames'])
            predictions, _ = projector.apply(staged, predictions)
            for i, c, stage, pred in zip(ids, cases, staged, predictions):
                phi, _ = current_projection(eq, c['row'], c['times'], pred['psirz'], stage['enabled'])
                run.save('mast_final', i, dict(pred, psirz=phi))
            run.progress('MAST final v25 readout', ids[-1]+1, len(paths), start)
    late_blas.restore_original_limits()
    del model, eq
    gc.collect()
    torch.cuda.empty_cache()
    return run.assemble('mast', 'mast_final', paths)


def listing(path, data_root, limit):
    values = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(values, list):
        raise ValueError('Input listing must be an ordered JSON list')
    paths = [(data_root/(v if isinstance(v, str) else v['path'])).resolve() for v in values]
    if limit:
        paths = paths[:limit]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Input listing is empty or contains duplicate files')
    for p in paths:
        if p.suffix.lower() != '.parquet' or not p.is_file():
            raise ValueError('Expected an existing input Parquet: '+str(p))
    return paths


def main():
    """Use ``python -m nm1.mast_phys`` -- it builds the shot list and the read policy."""
    raise SystemExit("run python -m nm1.mast_phys")


if __name__ == "__main__":
    main()
