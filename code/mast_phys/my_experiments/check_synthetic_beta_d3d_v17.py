"""Evaluate synthetic-only corrections on the existing DIII-D validation split."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RectBivariateSpline
from matplotlib.path import Path as Polygon
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from my_experiments.fusion_pipeline import CACHE_ROOT, stack_psirz, _interp, stack_2d, _interp_profile
from my_experiments.calibrate_pressure_moments_v14 import moments, MU0
from my_experiments.gpu_profiles_v15 import prepare_geometry
from my_experiments.beta_features_v17 import features
from my_experiments.synthetic_beta_model_v17 import predict
from data_fixes import fix_d3d_ip_times


def build_cache(art, part='validation', with_thermal=False):
    split = json.loads((art/'d3d_pressure_prior_v13.json').read_text())
    columns = ['efit_times', 'efit_psirz', 'efit_grid_R', 'efit_grid_Z', 'efit_lcfs_n', 'efit_lcfs_r',
               'efit_lcfs_z', 'efit_r_axis', 'efit_z_axis', 'efit_beta_n', 'magnetics_time',
               'magnetics_plasma_current_times', 'magnetics_plasma_current', 'magnetics_bcoil']
    batches, labels, electron, sample_ids = [], [], [], []
    for name in split[part+'_files']:
        path = CACHE_ROOT/'data/diii_d_train'/name
        selected_columns = columns
        if with_thermal:
            import pyarrow.parquet as pq
            selected_columns = list(dict.fromkeys(columns+[k for k in pq.ParquetFile(path).schema_arrow.names if k.startswith('thomson_')]))
        row = pd.read_parquet(path, columns=selected_columns).iloc[0]
        times = np.asarray(row['efit_times'], float)
        r, z = np.asarray(row['efit_grid_R']), np.asarray(row['efit_grid_Z'])
        psi = -stack_psirz(row['efit_psirz']).astype(float)
        ip = abs(_interp(fix_d3d_ip_times(row), row['magnetics_plasma_current'], times))*1000
        fv = abs(_interp(row['magnetics_time'], row['magnetics_bcoil'], times))*1000*144*2e-7
        temperatures, densities, locations = [], [], []
        if with_thermal:
            chords = np.asarray(row['thomson_chord_name']).astype(str)
            for system, prefix in [('core', 'TS_core'), ('edge', 'TS_tangential')]:
                if f'thomson_{system}_Te' not in row:
                    continue
                te, ne = stack_2d(row[f'thomson_{system}_Te']), stack_2d(row[f'thomson_{system}_ne'])
                keep = np.char.startswith(chords, prefix)
                if te.shape!=ne.shape or keep.sum()!=te.shape[1]:
                    continue
                temperatures.append(_interp_profile(row[f'thomson_{system}_times'], te, times))
                densities.append(_interp_profile(row[f'thomson_{system}_times'], ne, times))
                locations.append(np.column_stack([np.asarray(row['thomson_chord_Z'])[keep], np.asarray(row['thomson_chord_R'])[keep]]))
            if temperatures:
                temperatures, densities, locations = np.concatenate(temperatures, 1), np.concatenate(densities, 1), np.concatenate(locations)
        fr, fz = np.linspace(r[0], r[-1], 129), np.linspace(z[0], z[-1], 129)
        rr, zz = np.meshgrid(fr, fz)
        cr, cz = np.meshgrid(r, z)
        for i in range(0, len(times), 34):
            target = float(row['efit_beta_n'][i])
            if ip[i]<3e5 or not .05<target<8 or not np.isfinite(psi[i]).all():
                continue
            n = int(row['efit_lcfs_n'][i])
            lr, lz = np.asarray(row['efit_lcfs_r'][i])[:n], np.asarray(row['efit_lcfs_z'][i])[:n]
            good = np.isfinite(lr)&np.isfinite(lz)
            if good.sum()<12:
                continue
            boundary = np.column_stack([lr[good], lz[good]])
            ra, za = float(row['efit_r_axis'][i]), float(row['efit_z_axis'][i])
            sp = RectBivariateSpline(z, r, psi[i])
            pa, pb = float(sp.ev(za, ra)), float(np.median(sp.ev(lz[good], lr[good])))
            if pa-pb<1e-5:
                continue
            coarse = Polygon(boundary).contains_points(np.column_stack([cr.ravel(), cz.ravel()])).reshape(cr.shape)
            pn = (pa-psi[i])/(pa-pb)
            if np.sum(coarse&(pn>.015)&(pn<.985))<60:
                continue
            g = {'axis': (ra, za, pa), 'xpoint': (0, 0, pb), 'boundary': boundary, 'R': fr, 'Z': fz,
                 'flux': sp(fz, fr), 'inside': Polygon(boundary).contains_points(np.column_stack([rr.ravel(), zz.ravel()])).reshape(rr.shape)}
            batches.append(prepare_geometry(psi[i:i+1], r, z, [g], ip[i:i+1], fv[i:i+1]))
            labels.append(target)
            sample_ids.append(name+':'+str(i))
            e = 0.
            if with_thermal and isinstance(temperatures, np.ndarray):
                pn_ts = (pa-sp.ev(locations[:, 0], locations[:, 1]))/(pa-pb)
                try:
                    pe = moments(pn_ts, temperatures[i], densities[i], (pa-g['flux'])/(pa-pb), rr*g['inside'])[0]
                    a, rg = np.ptp(boundary[:, 0])/2, (boundary[:, 0].max()+boundary[:, 0].min())/2
                    e = float(2*MU0*100*a*rg*pe/(fv[i]*ip[i]/1e6))
                except ValueError:
                    pass
            electron.append(e)
    merged = {k: batches[0][k] if k in ('r', 'z') else np.concatenate([b[k] for b in batches]) for k in batches[0]}
    for b in batches:
        assert np.array_equal(b['r'], merged['r']) and np.array_equal(b['z'], merged['z'])
    x, scale = features(merged)
    suffix = '_kinetic' if with_thermal else ''
    np.savez_compressed(art/f'd3d_{part}_beta{suffix}_features_v17.npz', X=x, beta_scale=scale, betaN=np.asarray(labels),
                        electron_beta=np.asarray(electron), sample_ids=np.asarray(sample_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prepare-calibration', action='store_true')
    parser.add_argument('--prepare-kinetic', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    art = ROOT/'my_experiments/artifacts'
    if args.prepare_kinetic:
        for part in ('calibration', 'validation'):
            if not (art/f'd3d_{part}_beta_kinetic_features_v17.npz').exists():
                build_cache(art, part, True)
    if args.prepare_calibration and not (art/'d3d_calibration_beta_features_v17.npz').exists():
        build_cache(art, 'calibration')
    if not (art/'d3d_validation_beta_features_v17.npz').exists():
        build_cache(art)
    data = np.load(art/'d3d_validation_beta_features_v17.npz')
    checkpoint = torch.load(art/f'{args.model}.pt', map_location='cpu', weights_only=False)
    x = data['X']
    if checkpoint.get('raw_feature_count', x.shape[1]) == x.shape[1]+1:
        kinetic = np.load(art/'d3d_validation_beta_kinetic_features_v17.npz')
        assert np.array_equal(data['betaN'], kinetic['betaN'])
        x = np.column_stack([x, kinetic['electron_beta']/data['beta_scale']])
    corrected, sigma, delta = predict(checkpoint, x, data['beta_scale'])
    truth = data['betaN']
    raw = data['X'][:, 0]*data['beta_scale']
    def r2(p):
        return float(1-np.sum((p-truth)**2)/np.sum((truth-truth.mean())**2))
    report = {'MAST_files_opened': False, 'validation_only': True, 'model': args.model,
              'frames': len(truth), 'baseline_raw_r2': r2(raw), 'corrected_r2': r2(corrected),
              'released_DIII_D_predictions_changed': False}
    (art/f'{args.model}_d3d_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
