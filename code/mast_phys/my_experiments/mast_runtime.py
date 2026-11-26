import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor

import torch
from threadpoolctl import threadpool_limits

from my_experiments import reproduce_from_inputs_v25 as runner
from my_experiments.reproduce_from_inputs_v25 import mast as baseline_mast
from my_experiments.intermediate_io_v29 import BranchRun
from my_experiments.pressure_field_runtime_v30 import fields as pressure_fields
from my_experiments.mast_refinement_v29 import MastRefinement
from my_experiments.vertical_transfer_v21 import GAIN_LIMIT
from my_experiments.reproduction_io_v25 import atomic_json

def mast(run, paths, config, specification, workers, stop_after):
    result = baseline_mast(BranchRun(run), paths, config, specification, workers, stop_after)
    if stop_after != 'all':
        return result
    base_fold = json.loads((run.output/'mast_fold.json').read_text())['fold']

    enabled = base_fold['gain'] < GAIN_LIMIT and not config.get('mast_no_pressure_branch', False)
    pressure_fold = None
    if enabled:
        pressure_fields(BranchRun(run, ('psirz',)), paths, config, specification, workers, 'all')
        pressure_fold = json.loads((run.output/'mast_density_fold.json').read_text())['fold']
        enabled = pressure_fold['gain'] < GAIN_LIMIT
    atomic_json(run.output/'mast_method.json', dict(baseline_fold=base_fold,
                pressure_fold=pressure_fold, pressure_geometry_enabled=enabled,
                q95_betaN_from_this_runs_baseline=True, li_target_from_this_runs_baseline=True,
                target_labels_used=False, method='pressure branch'))
    torch.set_num_threads(1)
    model, start = MastRefinement(), time.perf_counter()
    missing = [i for i in range(len(paths)) if not run.exists('mast_v31', i)]
    with threadpool_limits(limits=1, user_api='blas'):
        with ProcessPoolExecutor(max_workers=workers, initializer=runner.worker_init,
                                 initargs=(specification,)) as pool:
            for begin in range(0, len(missing), 12):
                ids = missing[begin:begin+12]
                bases = [run.read('mast_final', i) for i in ids]
                fields = ([BranchRun(run, ('psirz',)).read('density_final', i)['psirz']
                           for i in ids] if enabled else [])
                for i, (prediction, detail) in zip(ids, model.predict_many(bases, fields, pool,
                                                                          enabled)):
                    run.save('mast_v31', i, prediction, detail)
                run.progress('MAST pressure geometry / retained profile',
                             min(begin+12, len(missing)), len(missing), start)
    gc.collect()
    torch.cuda.empty_cache()
    if config.get('q95_physical', False):
        limited = _physical_q95(run, paths, start, bool(config.get('q95_limited_fix', False)), workers,
                                specification)
        beta_fill = bool(config.get('betaN_limited_fill', False))
        q95_fill = bool(config.get('q95_limited_fill', False))
        stage = 'mast_v11'
        if beta_fill or q95_fill:
            _limited_fill(run, paths, limited, beta_fill, q95_fill, workers, specification, start,
                          float(config.get('betaN_diverted_blend', 0.0)), float(config.get('q95_diverted_blend', 0.0)),
                          str(config.get('fill_model', 'catboost')))
            stage = 'mast_v17'
        if config.get('axis_from_baseline', False):
            _axis_from_baseline(run, paths, stage, workers, specification, start)
            stage = 'mast_axis'
        return run.assemble('mast', stage, paths)
    return run.assemble('mast', 'mast_v31', paths)

def _limited_job(job):

    from my_experiments.q_physical_v14 import limited_flags
    psi, r, z, support = job
    flags = limited_flags(psi, r, z, support)
    runner.worker_audit()
    return flags

def _physical_q95(run, paths, start, limited_fix=False, workers=4, specification=None):

    import numpy as np
    from my_experiments import q_physical_v11 as qp
    from my_experiments import q_physical_v14 as qp14
    from my_experiments.fusion_pipeline import MAST_COILS, _interp
    from my_experiments.mast_constants import MAST_TF_TURNS
    from my_experiments.reproduce_from_inputs_v25 import load_input
    limited = [None] * len(paths)
    if limited_fix:
        from my_experiments.released_geometry import from_row
        geometry = load_input(paths[0])['row']
        r, z, _, bounds = from_row(geometry)
        support = (bounds['rmin'], bounds['rmax'], bounds['zmax'])
        with ProcessPoolExecutor(max_workers=workers, initializer=runner.worker_init,
                                 initargs=(specification,)) as pool:
            jobs = [(run.read('mast_final', i)['psirz'], r, z, support) for i in range(len(paths))]
            limited = list(pool.map(_limited_job, jobs, chunksize=4))
    currents, references = [], []
    for i, path in enumerate(paths):
        case = load_input(path)
        curr = np.column_stack([_interp(case['row']['magnetics_time'], case['row'][k], case['times'])
                                for k in MAST_COILS])
        ip = np.abs(curr[:, -1] * 1000.0)
        fv = np.abs(curr[:, 11] * 1000.0 * MAST_TF_TURNS * 2e-7)
        q = run.read('mast_v31', i)['q95']
        currents.append((ip, fv))
        references.append(qp14.flat_top_reference(q, ip, fv, limited[i]) if limited_fix
                          else qp.flat_top_reference(q, ip, fv))
    references = np.asarray(references, float)
    default = float(np.nanmedian(references)) if np.isfinite(references).any() else float('nan')
    summary = dict(shots=len(paths), shots_without_reference=int((~np.isfinite(references)).sum()),
                   default_reference=default)
    for i in range(len(paths)):
        if run.exists('mast_v11', i):
            continue
        base = run.read('mast_v31', i)
        ip, fv = currents[i]
        reference = references[i] if np.isfinite(references[i]) else default
        if not np.isfinite(reference):
            record = dict(skipped='no resolved frame in the run')
            q = base['q95']
        elif limited_fix:
            q, record = qp14.regularize(base['q95'], ip, fv, reference, limited[i])
        else:
            q, record = qp.regularize(base['q95'], ip, fv, reference)
        run.save('mast_v11', i, dict(psirz=base['psirz'], q95=np.asarray(q, np.float32),
                                     betaN=base['betaN']), record)
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            run.progress('MAST q95 physical regularization', i + 1, len(paths), start)
    if limited_fix:
        summary['limited_frames'] = int(sum(int(np.sum(f)) for f in limited))
    atomic_json(run.output/'mast_q95_physical.json', dict(summary, target_labels_used=False,
                limited_frames_unresolved=bool(limited_fix),
                method='q95 = y F / Ip, shape factor y compressed to the shot flat-top (q_physical_v11.py)'
                       + ('; inboard-limited frames unresolved (q_physical_v14.py)' if limited_fix else '')))
    return limited

def _axis_job(job):

    from my_experiments.axis_from_baseline import shot
    i, submitted, baseline, r, z, rmax, zmax = job
    flux, shifts = shot(submitted, baseline, r, z, rmax, zmax)
    runner.worker_audit()
    return i, flux, shifts

def _axis_from_baseline(run, paths, source, workers, specification, start):

    import numpy as np
    from my_experiments.released_geometry import from_row
    from my_experiments.reproduce_from_inputs_v25 import load_input
    r, z, _, bounds = from_row(load_input(paths[0])['row'])
    shifts_all = []
    with ProcessPoolExecutor(max_workers=workers, initializer=runner.worker_init,
                             initargs=(specification,)) as pool:
        for first in range(0, len(paths), 48):
            ids = [i for i in range(first, min(first + 48, len(paths))) if not run.exists('mast_axis', i)]
            bases = {i: run.read(source, i) for i in ids}
            jobs = [(i, bases[i]['psirz'], run.read('mast_final', i)['psirz'], r, z, bounds['rmax'], bounds['zmax'])
                    for i in ids]
            for i, flux, shifts in pool.map(_axis_job, jobs, chunksize=2):
                run.save('mast_axis', i, dict(bases[i], psirz=flux),
                         dict(axis_moved=int(np.sum(np.abs(shifts) > 0)), shift_rms_m=float(np.sqrt(np.mean(shifts ** 2)))))
                shifts_all.append(shifts)
            run.progress('MAST axis radius from the baseline branch', min(first + 48, len(paths)), len(paths), start)
    s = np.concatenate(shifts_all) if shifts_all else np.zeros(0)
    atomic_json(run.output/'mast_axis_from_baseline.json', dict(
        frames=int(len(s)), moved=int(np.sum(np.abs(s) > 0)), shift_rms_m=float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0,
        shift_mean_m=float(np.mean(s)) if len(s) else 0.0, source=source, target_labels_used=False,
        method='axis radius moved to the baseline branch (mast_final) of this run, LCFS kept (axis_from_baseline.py)'))

def _features_job(path):

    from my_experiments.limited_fill_v17 import frame_features
    from my_experiments.reproduce_from_inputs_v25 import load_input
    features = frame_features(load_input(path)['row'])
    runner.worker_audit()
    return features

def _limited_fill(run, paths, limited, beta_fill, q95_fill, workers, specification, start,
                  beta_blend=0.0, q95_blend=0.0, fill_model='catboost'):

    import numpy as np
    from my_experiments import limited_fill_v17 as lf
    from my_experiments import q_physical_v11 as qp
    from my_experiments.mast_constants import MAST_TF_TURNS
    if any(f is None for f in limited):
        raise ValueError('the v17 limited-frame fill needs the v14 limited flags (q95_limited_fix)')
    with ProcessPoolExecutor(max_workers=workers, initializer=runner.worker_init,
                             initargs=(specification,)) as pool:
        features = list(pool.map(_features_job, paths, chunksize=4))
    bases = [run.read('mast_v11', i) for i in range(len(paths))]
    sizes = [len(b['q95']) for b in bases]
    if any(len(f) != n for f, n in zip(features, sizes)):
        raise ValueError('feature frames do not match the run')
    x = np.vstack(features)
    lim = np.concatenate([np.asarray(f, bool) for f in limited])
    fold = lf.shot_folds(len(paths))[np.repeat(np.arange(len(paths)), sizes)]
    beta = np.concatenate([b['betaN'].astype(float) for b in bases])
    q95 = np.concatenate([b['q95'].astype(float) for b in bases])
    ip = np.abs(x[:, lf.FEATURES.index('ip')] * 1000.0)
    fv = np.abs(x[:, lf.FEATURES.index('tf')] * 1000.0 * MAST_TF_TURNS * 2e-7)
    ip_eff = np.maximum(np.where(np.isfinite(ip), ip, qp.MIN_IP), qp.MIN_IP)
    y = q95 * (ip_eff / 1.0e6) / np.maximum(fv, 1e-12)
    if fill_model == 'transformer':
        # whole-shot transformer instead of per-frame trees: same features, targets, masks and shot folds
        from my_experiments import limited_fill_tfm as lft
        m_beta, m_y = lft.out_of_fold(x, [beta, y], [~lim, ~lim & np.isfinite(y) & (fv > qp.MIN_F)], fold, sizes,
                                      x[:, lf.FEATURES.index('t')])
        m_beta = m_beta if beta_fill else None
        m_y = m_y if q95_fill else None
        run.progress('MAST limited-frame fill: whole-shot transformer', 2, 2, start)
    else:
        m_beta = lf.out_of_fold(x, beta, ~lim, fold, lf.SEEDS['betaN']) if beta_fill else None
        run.progress('MAST limited-frame fill: betaN trees', 1, 2, start)
        m_y = (lf.out_of_fold(x, y, ~lim & np.isfinite(y) & (fv > qp.MIN_F), fold, lf.SEEDS['y'])
               if q95_fill else None)
        run.progress('MAST limited-frame fill: shape-factor trees', 2, 2, start)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    changes = {'betaN': [], 'q95': []}
    for i in range(len(paths)):
        s, e = starts[i], starts[i + 1]
        L = lim[s:e]
        b = lf.fill(beta[s:e], m_beta[s:e], L, lf.FLOOR['betaN']) if beta_fill else beta[s:e]
        if beta_fill and beta_blend > 0:
            b = np.where(L, b, (1.0 - beta_blend) * beta[s:e] + beta_blend * m_beta[s:e])
        q = q95[s:e]
        if q95_fill:
            y_new = lf.fill(y[s:e], m_y[s:e], L, lf.FLOOR['y'])
            f = fv[s:e]
            f_eff = np.where(np.isfinite(f) & (f > qp.MIN_F), f,
                             np.nanmedian(f[f > qp.MIN_F]) if (f > qp.MIN_F).any() else 0.4)
            q = np.where(L, np.clip(y_new * f_eff / (ip_eff[s:e] / 1.0e6), qp.Q_MIN, qp.Q_MAX), q)
            if q95_blend > 0:
                ok = ~L & np.isfinite(m_y[s:e]) & (f > qp.MIN_F)
                y_mix = (1.0 - q95_blend) * y[s:e] + q95_blend * m_y[s:e]
                q = np.where(ok, np.clip(y_mix * f_eff / (ip_eff[s:e] / 1.0e6), qp.Q_MIN, qp.Q_MAX), q)
        changes['betaN'].append((b - beta[s:e])[L])
        changes['q95'].append((q - q95[s:e])[L])
        run.save('mast_v17', i, dict(psirz=bases[i]['psirz'], q95=np.asarray(q, np.float32),
                                     betaN=np.asarray(b, np.float32)),
                 dict(limited=int(L.sum()), betaN_filled=bool(beta_fill), q95_filled=bool(q95_fill)))
    rms = {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if lim.any() else 0.0 for k, v in changes.items()}
    atomic_json(run.output/'mast_limited_fill.json', dict(
        frames=int(len(lim)), limited_frames=int(lim.sum()), betaN_filled=bool(beta_fill),
        q95_filled=bool(q95_fill), betaN_change_rms=rms['betaN'], q95_change_rms=rms['q95'],
        betaN_diverted_blend=float(beta_blend), q95_diverted_blend=float(q95_blend),
        trees=lf.TREES, folds=lf.FOLDS, target_labels_used=False, fill_model=fill_model,
        method=('limited frames: ' + ('whole-shot transformer (limited_fill_tfm.py)' if fill_model == 'transformer'
                                       else 'CatBoost (limited_fill_v17.py)') +
                ' on this run\'s diverted frames (shot-grouped out-of-fold) x the shot\'s own ratio, '
                '5-frame running median'),
        trees_used=fill_model != 'transformer'))
