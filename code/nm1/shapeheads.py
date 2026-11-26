"""Audited v12 experiments. Reads official Parquets, source H5 and weights only.

All predictions and derived targets are regenerated. Existing NPZ inputs are
blocked by input_policy. The scorer uses the documented vessel rectangle,
constructed from grid arrays, because reading the supplied NPZ mask is forbidden.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py
import numpy as np
import torch

from nm1 import ARTIFACTS, CACHE_DATA, OUTPUTS, ROOT
from nm1.cache import BASIS, CACHE
from nm1.guard import box_mask
from nm1.model import load_member, member_inputs
from nm1.v12_shape import NAMES, ShapeHead, raw_features

LAB = OUTPUTS / "shape_heads_lab"
_G = {}


def install_policy(out):
    from nm1.input_policy import install
    os.environ["FUSION_RUN_ROOT"] = str(out.resolve())
    return install(out)


def load_inputs(split):
    from nm1.cachepred import CacheInputs
    return CacheInputs(split)


def whole_input(inputs, payload):
    n = len(inputs.shot)
    out = np.empty((n, payload["model"]["n_features"]), np.float32)
    for s, e in zip(inputs.starts, inputs.ends):
        out[s:e] = member_inputs(inputs.base[s:e], inputs.currents[s:e], payload,
            inputs.green, inputs.ext[s:e] if payload.get("n_ext") else None,
            inputs.cds[s:e] if payload.get("n_cds") else None,
            inputs.cdsg[s:e] if payload.get("n_cdsg") else None)
    return out


@torch.inference_mode()
def batched_member(path, inputs, batch=4096):
    model, p = load_member(path, "cuda")
    x = whole_input(inputs, p)
    c = np.empty((len(x), len(p["coeff_std"])), np.float32)
    scal = np.empty((len(x), 2), np.float32)
    for a in range(0, len(x), batch):
        z, y = model(torch.as_tensor(x[a:a+batch], device="cuda"))
        c[a:a+len(z)] = z.float().cpu().numpy() * p["coeff_std"]
        scal[a:a+len(z)] = y.float().cpu().numpy()*p["scalar_std"]+p["scalar_mean"]
    return c, scal


def baseline(out):
    target = out / "baseline.h5"
    if target.exists():
        raise FileExistsError(target)
    spec = json.loads((ROOT/"recipes/v11-1/members.json").read_text(encoding="utf-8"))
    inp = load_inputs("val")
    n = len(inp.shot)
    coeff = np.zeros((n, 256), np.float64)
    scal = np.zeros((n, 2), np.float64)
    t0 = time.perf_counter()
    with h5py.File(target, "w") as f:
        f["shot"] = inp.shot
        f["times_s"] = inp.times_ms/1000
        group = {}
        for i, rel in enumerate(spec["members"]):
            c, s = batched_member(ARTIFACTS/rel, inp)
            coeff += spec["weights"][i]*c
            scal[:, 0] += spec["weights_q95"][i]*s[:, 0]
            scal[:, 1] += spec.get("weights_betaN", spec["weights"])[i]*s[:, 1]
            for name, rels in spec["groups"].items():
                if rel in rels:
                    if name not in group:
                        group[name] = [np.zeros_like(coeff), np.zeros_like(scal)]
                    group[name][0] += c/len(rels)
                    group[name][1] += s/len(rels)
            print(f"baseline {i+1}/{len(spec['members'])}: {Path(rel).stem} {time.perf_counter()-t0:.1f}s", flush=True)
        c, _ = batched_member(ARTIFACTS/spec["specialist"], inp)
        f["coeff"] = coeff.astype(np.float32)
        f["scal"] = scal.astype(np.float32)
        f["specialist"] = c
        for name, (cg, sg) in group.items():
            g = f.create_group("groups/"+name); g["coeff"] = cg.astype(np.float32); g["scal"] = sg.astype(np.float32)
        f.attrs["source"] = "fresh v11-1 checkpoint inference; no NPZ reads"
    print(f"baseline completed in {time.perf_counter()-t0:.1f}s", flush=True)


def grid():
    from nm1.cache import training_shots
    from nm1.features import read_row
    row = read_row(training_shots("train")[0], drop_targets=True)
    return np.asarray(row["efit_grid_R"], float), np.asarray(row["efit_grid_Z"], float)


def init_worker(R, Z, mean=None, comps=None, out=None):
    if out:
        install_policy(Path(out))
    sys.path.insert(0, str(ROOT/"fusion_scoring"))
    import local_score
    local_score.MACHINE = "DIII-D"
    m = box_mask(R, Z)
    _G.update(R=R, Z=Z, mc=m, mf=m.astype(float), mean=mean, comps=comps)


def derive_job(job):
    from derive import derive_frame
    i, c = job
    maps = (_G["mean"] + c.astype(float)@_G["comps"]).reshape(-1, 65, 65)
    ys = []
    for psi in maps:
        v = derive_frame(psi, _G["R"], _G["Z"], "DIII-D", _G["mc"], _G["mf"])
        ys.append([v[k] for k in NAMES])
    return i, np.asarray(ys)


def labels(out, per_shot=16, workers=6, calibration_only=False):
    from nm1.score import held_out_ids
    p = torch.load(BASIS, map_location="cpu", weights_only=False)
    R, Z = grid()
    t0 = time.perf_counter()
    target = out/"labels.h5"
    if target.exists() and not calibration_only: raise FileExistsError(target)
    with h5py.File(CACHE, "r") as f, h5py.File(target, "a" if calibration_only else "w") as dst:
        for split in (("val",) if calibration_only else ("train", "val")):
            if split in dst: raise FileExistsError(f"Completed labels already exist: {split}")
            shot = f[split]["shot"][:]
            if split == "val":
                cal = np.asarray(sorted(held_out_ids("calibration")))
            bounds = np.r_[0, np.flatnonzero(np.diff(shot))+1, len(shot)]
            idxs = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                if split == "val" and shot[s] not in cal: continue
                idxs.extend(np.unique(np.r_[np.linspace(s, e-1, min(per_shot, e-s)).astype(int),
                                           np.arange(s, min(s+3, e)), np.arange(max(s,e-3),e)]))
            idx = np.asarray(idxs, np.int64)
            coeff = f[split]["coeff"][idx]
            y = np.full((len(idx), 7), np.nan)
            jobs = [(i, coeff[i:i+16]) for i in range(0, len(idx), 16)]
            print(f"labels {split}: {len(idx):,} frames, {len(jobs)} blocks", flush=True)
            with ProcessPoolExecutor(workers, initializer=init_worker,
                    initargs=(R,Z,p["pca_mean"].astype(float),p["pca_components"].astype(float),str(out))) as pool:
                for done, (i, values) in enumerate(pool.map(derive_job, jobs, chunksize=2), 1):
                    y[i:i+len(values)] = values
                    if done % 100 == 0 or done == len(jobs):
                        print(f"labels {split} {done}/{len(jobs)} {time.perf_counter()-t0:.1f}s", flush=True)
            g = dst.create_group(split); g["idx"] = idx; g["target"] = y.astype(np.float32)
            print("label std", np.nanstd(y, axis=0).tolist(), flush=True)
        if "grid_R" not in dst: dst["grid_R"] = R; dst["grid_Z"] = Z
        dst.attrs["source"] = "new published functionals on official DIII-D train PCA coefficients; no pre-existing NPZ mask"


def shape_inputs(inputs):
    out = np.empty((len(inputs.shot), 676), np.float32)
    for s, e in zip(inputs.starts, inputs.ends):
        x = raw_features(inputs.base[s:e], inputs.ext[s:e], inputs.cds[s:e], inputs.cdsg[s:e])
        if x.shape[1] != out.shape[1]:
            if s == 0: out = np.empty((len(inputs.shot), x.shape[1]), np.float32)
            else: raise ValueError("shape feature width changed")
        out[s:e] = x
    return out


def exact_label_job(job):
    import pyarrow.parquet as pq
    from nm1.features import stack_psirz
    from derive import derive_frame
    path, expected, offsets, positions = job
    row = pq.read_table(path, columns=["efit_psirz"]).to_pylist()[0]
    maps = stack_psirz(row["efit_psirz"])
    maps = maps[np.isfinite(maps).all(axis=(1, 2))]
    if len(maps) != expected:
        return positions, np.full((len(positions), 7), np.nan), str(path)
    ys = []
    for psi in maps[offsets]:
        v = derive_frame(psi, _G["R"], _G["Z"], "DIII-D", _G["mc"], _G["mf"])
        ys.append([v[k] for k in NAMES])
    return positions, np.asarray(ys), None


def exact_labels(out, workers=3):
    """Replace PCA target approximations with functionals of exact official maps."""
    from nm1.fetch import listing, local_path
    target = out / "labels_exact.h5"
    if target.exists():
        raise FileExistsError(target)
    R, Z = grid()
    paths = [local_path(f) for f in listing("diii_d_train")]
    t0 = time.perf_counter()
    with h5py.File(out / "labels.h5", "r") as chosen, h5py.File(CACHE, "r") as cache, h5py.File(target, "w") as dst:
        for split in ("train", "val"):
            idx = chosen[split]["idx"][:]
            shot = cache[split]["shot"][:]
            bounds = np.r_[0, np.flatnonzero(np.diff(shot)) + 1, len(shot)]
            jobs = []
            for s, e in zip(bounds[:-1], bounds[1:]):
                a, b = np.searchsorted(idx, [s, e])
                if b > a:
                    jobs.append((paths[int(shot[s])], int(e-s), idx[a:b]-s, np.arange(a,b)))
            y = np.full((len(idx), 7), np.nan)
            skipped = []
            with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(R,Z,None,None,str(out))) as pool:
                for done, (positions, values, problem) in enumerate(pool.map(exact_label_job,jobs,chunksize=1),1):
                    y[positions] = values
                    if problem: skipped.append(problem)
                    if done % 100 == 0 or done == len(jobs):
                        print(f"exact labels {split}: {done}/{len(jobs)}, {time.perf_counter()-t0:.1f}s",flush=True)
            g = dst.create_group(split)
            g["idx"] = idx
            g["target"] = y.astype(np.float32)
            g.attrs["skipped"] = json.dumps(skipped)
        dst["grid_R"] = R
        dst["grid_Z"] = Z
        dst.attrs["source"] = "published functionals of exact official DIII-D Parquet psirz; train-only and calibration-only partitions"
    print("exact label generation complete",flush=True)


def sample_frame_weights(split, idx):
    """Voronoi quadrature weights of selected frames inside each training shot."""
    with h5py.File(CACHE,"r") as f:shot=f[split]["shot"][:]
    bounds=np.r_[0,np.flatnonzero(np.diff(shot))+1,len(shot)]
    w=np.ones(len(idx),np.float32)
    for s,e in zip(bounds[:-1],bounds[1:]):
        a,b=np.searchsorted(idx,[s,e])
        if b<=a:continue
        chosen=idx[a:b]
        edges=np.r_[s-.5,(chosen[:-1]+chosen[1:])/2,e-.5]
        w[a:b]=np.diff(edges)
    return w/w.mean()


def train_shapes(out, seed=12001, epochs=100, encoder="", labels_source="labels.h5", stream_training=False,
                 frame_weighted=False,target_weights=None):
    from nm1.train import _stats
    torch.manual_seed(seed); np.random.seed(seed)
    with h5py.File(out/labels_source, "r") as f:
        ti, ty = f["train/idx"][:], f["train/target"][:]
        vi, vy = f["val/idx"][:], f["val/target"][:]
    tx = shape_inputs(load_inputs("train"))[ti]
    vx = shape_inputs(load_inputs("val"))[vi]
    if encoder:
        from nm1.v12_shape import latent_features
        em,ep=load_member(ARTIFACTS/encoder,"cuda",keep_state=False)
        tx=latent_features(tx,em,ep,batch=512 if stream_training else 4096)
        vx=latent_features(vx,em,ep,batch=512 if stream_training else 4096)
        del em,ep;torch.cuda.empty_cache()
    mu, sd = _stats(tx)
    valid = np.isfinite(ty); vvalid = np.isfinite(vy)
    tw=sample_frame_weights("train",ti) if frame_weighted else np.ones(len(ti),np.float32)
    vw=sample_frame_weights("val",vi) if frame_weighted else np.ones(len(vi),np.float32)
    weights=valid*tw[:,None]
    ym=(np.nan_to_num(ty)*weights).sum(0)/weights.sum(0)
    ys=np.maximum(np.sqrt((np.nan_to_num(ty-ym)**2*weights).sum(0)/weights.sum(0)),1e-4)
    y = np.nan_to_num((ty-ym)/ys); yv = np.nan_to_num((vy-ym)/ys)
    store_device = "cpu" if stream_training else "cuda"
    x = torch.as_tensor((tx-mu)/sd, device=store_device)
    xv = torch.as_tensor((vx-mu)/sd, device=store_device)
    y = torch.as_tensor(y, device=store_device); yv = torch.as_tensor(yv, device="cuda")
    mask = torch.as_tensor(valid, device=store_device); mv = torch.as_tensor(vvalid, device="cuda")
    tw=torch.as_tensor(tw,device=store_device)
    vw=torch.as_tensor(vw[:,None],device="cuda")
    loss_weights=np.asarray(target_weights if target_weights is not None else [1.]*7,np.float32)
    if loss_weights.shape!=(7,) or not np.isfinite(loss_weights).all() or (loss_weights<=0).any():
        raise ValueError("Seven positive, finite target weights are required")
    lw=torch.as_tensor(loss_weights,device="cuda")
    config = dict(n_features=x.shape[1], width=512, depth=4, dropout=0.02)
    model = ShapeHead(**config).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = float("inf"); path = out/f"shape_s{seed}.pt"; t0 = time.perf_counter()
    amp = torch.amp.GradScaler("cuda")
    for ep in range(1, epochs+1):
        model.train(); perm = torch.randperm(len(x), device=store_device)
        lr = 0.001*(0.1+0.9*0.5*(1+np.cos(np.pi*(ep-1)/epochs)))
        for g in opt.param_groups: g["lr"] = lr
        train_loss = 0.0
        for a in range(0,len(x),2048):
            ii = perm[a:a+2048]; opt.zero_grad(set_to_none=True)
            xb,yb,mb = x[ii].to("cuda"),y[ii].to("cuda"),mask[ii].to("cuda")
            wb=tw[ii,None].to("cuda")*mb*lw
            with torch.autocast("cuda",dtype=torch.float16):
                pred = model(xb); err = (pred.float()-yb)**2
                loss = (err*wb).sum()/wb.sum()
            amp.scale(loss).backward(); amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            amp.step(opt); amp.update(); train_loss += float(loss.detach())*len(ii)
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model(xv[a:a+4096].to("cuda")) for a in range(0,len(xv),4096)])
            mse = (((pred-yv)**2)*mv*vw).sum(0)/(mv*vw).sum(0)
            score = float((mse*lw).sum()/lw.sum())
        if score < best:
            best = score
            torch.save(dict(model=config,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                feature_mean=mu,feature_std=sd,target_mean=ym,target_std=ys,seed=seed,epoch=ep,
                calibration_mse=mse.cpu().tolist(),train_source="official train split only",extra_train=False,
                calibration_use="checkpoint selection only",target_source="fresh field functionals",
                label_file=labels_source,encoder=encoder,stream_training=stream_training,
                frame_weighted=frame_weighted,target_weights=loss_weights.tolist()),path)
        if ep % 5 == 0 or ep == 1:
            print(f"shape ep {ep} train={train_loss/len(x):.6f} cal={score:.6f} best={best:.6f} r2={(1-mse).cpu().tolist()} {time.perf_counter()-t0:.1f}s",flush=True)
    print("trained",path,flush=True)


def train_tree(out, seed=12017, iterations=5000):
    from catboost import CatBoostRegressor, Pool
    with h5py.File(out/"labels.h5","r") as f:
        ti,ty=f["train/idx"][:],f["train/target"][:]
        vi,vy=f["val/idx"][:],f["val/target"][:]
    tx=shape_inputs(load_inputs("train"))[ti]
    vx=shape_inputs(load_inputs("val"))[vi]
    ym=np.nanmean(ty,0);ys=np.maximum(np.nanstd(ty,0),1e-4)
    good=np.isfinite(ty).all(1);vgood=np.isfinite(vy).all(1)
    model=CatBoostRegressor(loss_function="MultiRMSE",iterations=iterations,
        depth=8,learning_rate=0.04,l2_leaf_reg=5,random_seed=seed,task_type="GPU",
        devices="0",gpu_ram_part=0.65,thread_count=4,verbose=200,
        early_stopping_rounds=400,allow_writing_files=False)
    model.fit(Pool(tx[good],(ty[good]-ym)/ys),eval_set=Pool(vx[vgood],(vy[vgood]-ym)/ys))
    path=out/f"shape_tree_s{seed}.cbm";model.save_model(os.path.relpath(path,Path.cwd()))
    pred=model.predict(vx)*ys+ym
    result=dict(target_mean=ym.tolist(),target_std=ys.tolist(),seed=seed,
        target_names=list(NAMES),training_split="official train only; 88% shot split",
        calibration_use="early stopping only",iterations=model.tree_count_,
        calibration_r2=(1-np.nanmean(((pred-vy)/ys)**2,0)).tolist())
    path.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("baseline","labels","labels-calibration","exact-labels","train","tree"))
    ap.add_argument("--out", type=Path, default=LAB)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-shot", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12001)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iterations",type=int,default=5000)
    ap.add_argument("--encoder",default="")
    ap.add_argument("--labels-source",default="labels.h5")
    ap.add_argument("--stream-training",action="store_true",help="keep training features in RAM and send one minibatch to CUDA")
    ap.add_argument("--frame-weighted",action="store_true",help="weight sparse functional labels by the frames they represent")
    ap.add_argument("--target-weights",default="1,1,1,1,1,1,1")
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    install_policy(a.out); torch.set_num_threads(2)
    if a.stage=="baseline": baseline(a.out)
    elif a.stage=="labels": labels(a.out,a.per_shot,a.workers)
    elif a.stage=="labels-calibration": labels(a.out,a.per_shot,a.workers,True)
    elif a.stage=="exact-labels": exact_labels(a.out,a.workers)
    elif a.stage=="tree":train_tree(a.out,a.seed,a.iterations)
    else: train_shapes(a.out,a.seed,a.epochs,a.encoder,a.labels_source,a.stream_training,
                      a.frame_weighted,[float(v) for v in a.target_weights.split(',')])


if __name__=="__main__": main()
