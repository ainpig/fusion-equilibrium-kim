"""Train one whole-shot transformer member (``nm1.seqmodel.TfmRegressor``) on the coefficient caches.

Every frame of a discharge attends to every other frame of the same discharge (both directions; the
reconstruction is offline, the whole shot's inputs exist before the first frame is predicted).  The
member returns the same whitened flux coefficients and (q95, betaN) as the per-frame members, so it
joins the blend through the manifest (``nm1.predict`` blends sequence members in the two-stage path).

Inputs per frame: the base block (272), the frame spacing (2, ``seqmodel.dt_block``), the features2
block (298), the current-diffusion block and its flux-coordinate twin (41 + 41) -- every block the
per-frame second-stage members read.  Loss and the epoch-selection surrogate are the per-frame
trainer's (``nm1.train``): the flux term weighted per mode, the Consistency surrogate through the
measured mode Jacobian (Gram form), and the (q95, betaN) squared error.

    python -m nm1.train_tfm --output <member.pt> [--all-data --save-epoch N] [--epochs 60] [--seed 1]

``--all-data`` trains on every labelled shot and saves epoch ``--save-epoch`` (picked by a development run
on the calibration third, as for the per-frame final members).
"""
from __future__ import annotations

import argparse
import os
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from nm1.cache import BASIS, CACHE
from nm1.cache2 import CACHE2
from nm1.cds_geo import CDSG_CACHE
from nm1.diffusion import CDS_CACHE
from nm1.jacobian import OUT as JACOBIAN
from nm1.seqmodel import TfmRegressor, dt_block
from nm1.train import W_CONS, W_PSI, W_QB, build_weights

TIME_COL = 136          # the base block's discharge clock [s] (features.d3d_input_features)


def _segments(shot: np.ndarray) -> np.ndarray:
    """(start, end) of each shot's contiguous rows."""
    b = np.flatnonzero(np.diff(shot)) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(shot)]])
    return np.stack([st, en], 1)


def _read(ds, rows: np.ndarray | None) -> np.ndarray:
    if rows is None:
        return ds[:]
    if len(rows) and rows[-1] - rows[0] + 1 == len(rows):        # one contiguous range: read only that
        return ds[int(rows[0]):int(rows[-1]) + 1]
    return ds[:][rows]


def _load(group: str, rows: np.ndarray | None, n_features: int, n_cds: int, n_cdsg: int, use_ext: bool,
          ctx: bool = False):
    """Raw input blocks, coefficients, scalars, shot ids and frame times of one cache group.

    ``ctx`` appends the per-frame members' shot-context block (``features2.context_block``, from the
    features2 block and the frame clock, per shot) last, as ``seqmodel.seq_inputs`` does."""
    with h5py.File(CACHE, "r") as h:
        g = h[group]
        feat = _read(g["features"], rows)[:, :n_features].astype(np.float32)
        coeff = _read(g["coeff"], rows)
        scal = _read(g["scalars"], rows)
        shot = _read(g["shot"], rows)
    blocks = [feat]
    t_s = feat[:, TIME_COL].astype(np.float64)
    dt = np.zeros((len(shot), 2), np.float32)
    for s, e in _segments(shot):
        dt[s:e] = dt_block(t_s[s:e] * 1000.0)
    blocks.append(dt)
    ext = None
    if use_ext or ctx:
        with h5py.File(CACHE2, "r") as h:
            ext = _read(h[group]["ext"], rows)
    if use_ext:
        blocks.append(ext)
    for path, k in ((CDS_CACHE, n_cds), (CDSG_CACHE, n_cdsg)):
        if k:
            with h5py.File(path, "r") as h:
                blk = _read(h[group]["cds"], rows)
                blocks.append(blk[:, :k])
    if ctx:
        from nm1.features2 import context_block
        c = np.zeros((len(shot), 0), np.float32)
        parts = [context_block(ext[s:e], t_s[s:e]) for s, e in _segments(shot)]
        blocks.append(np.concatenate(parts) if parts else c)
    del ext
    x = np.hstack([np.asarray(b, np.float32) for b in blocks])
    return x, coeff.astype(np.float32), scal.astype(np.float32), shot, t_s.astype(np.float32)


def train(output: Path, all_data: bool = False, save_epoch: int = 0, epochs: int = 60, seed: int = 1,
          width: int = 512, layers: int = 4, heads: int = 8, ffn: int = 1024, dropout: float = 0.05,
          batch_shots: int = 16, lr: float = 1e-3, warmup: int = 300, weight_decay: float = 1e-5,
          weights: str = "gram", n_features: int = 272, n_cds: int = 41, n_cdsg: int = 41,
          use_ext: bool = True, val_every: int = 1, max_shots: int = 0, device: str = "",
          init: Path | None = None, pre_blocks: int = 2, post_blocks: int = 2, attn_width: int = 0,
          ctx: bool = False, init_member: Path | None = None, trunk_lr: float = 1.0,
          layer_scale: float = 0.0, freeze_epochs: int = 0, diff_weight: float = 0.0,
          scalar_weights: str = "", init_seq: Path | None = None, bnd_file: Path | None = None,
          bnd_weight: float = 0.0, resume: bool = False, ip_weight: float = 0.0, bnd_p: float = 2.0,
          grow: bool = False, chan_tokens: int = 0, chan_layers: int = 2, chan_width: int = 128,
          n_experts: int = 0, ramp_weight: float = 1.0, grad_ckpt: bool = False) -> Path:
    from nm1.score import held_out_ids
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()
    state = torch.load(BASIS, map_location="cpu", weights_only=False)
    coeff_std = np.asarray(state["coeff_std"], np.float32)
    n_comp = len(coeff_std)

    print("loading the caches ...", flush=True)
    with h5py.File(CACHE, "r") as h:
        vshot = h["val"]["shot"][:]
        tshot = h["train"]["shot"][:]
    trows = None
    if max_shots:                                            # smoke test: the first shots only
        seg = _segments(tshot)[:max_shots]
        trows = np.arange(seg[0, 0], seg[-1, 1])
    xtr, ctr, strn, shtr, ttr = _load("train", trows, n_features, n_cds, n_cdsg, use_ext, ctx)
    cal = np.array(sorted(held_out_ids("calibration")), dtype=vshot.dtype)
    sel = np.array(sorted(held_out_ids("selection")), dtype=vshot.dtype)
    if all_data:
        if save_epoch <= 0:
            raise SystemExit("--all-data needs --save-epoch")
        xa, ca, sa, sha, ta = _load("val", None, n_features, n_cds, n_cdsg, use_ext, ctx)
        xtr = np.concatenate([xtr, xa]); ctr = np.concatenate([ctr, ca]); strn = np.concatenate([strn, sa])
        shtr = np.concatenate([shtr, sha]); ttr = np.concatenate([ttr, ta])
        del xa, ca, sa, ta
        vrows = np.flatnonzero(np.isin(vshot, sel))          # monitored only; the epoch is given
    else:
        vrows = np.flatnonzero(np.isin(vshot, cal))          # the per-frame members' epoch-selection pool
    if max_shots:
        keep = np.unique(vshot[vrows])[:max(2, max_shots // 4)]
        vrows = vrows[np.isin(vshot[vrows], keep)]
    xva, cva, sva, shva, tva = _load("val", vrows, n_features, n_cds, n_cdsg, use_ext, ctx)
    print(f"  train rows {len(xtr):,} ({len(np.unique(shtr))} shots), monitor rows {len(xva):,} "
          f"({len(np.unique(shva))} shots), {xtr.shape[1]} inputs  [{time.perf_counter() - t0:.0f}s]", flush=True)

    pre = None
    if init is not None:
        # a self-supervised trunk (nm1.pretrain_tfm): its input whitening is kept so the trunk sees the
        # numbers it was pretrained on
        pre = torch.load(init, map_location="cpu", weights_only=False)
        if pre.get("kind") != "tfm_trunk" or int(pre["model"]["n_features"]) != xtr.shape[1]:
            raise SystemExit("--init needs a trunk from nm1.pretrain_tfm with the same input blocks")
        for k in ("width", "layers", "heads", "ffn", "pre", "post", "attn_width"):
            if int(pre["model"].get(k, 0)) != int({"width": width, "layers": layers, "heads": heads, "ffn": ffn,
                                                   "pre": pre_blocks, "post": post_blocks, "attn_width": attn_width}[k]):
                raise SystemExit(f"--init trunk has {k}={pre['model'][k]}")
        fmean, fstd = pre["feature_mean"], pre["feature_std"]
    elif init_seq is not None:
        # fine-tuning an existing transformer member: keep its input and output whitening
        seq0 = torch.load(init_seq, map_location="cpu", weights_only=False)
        if seq0.get("arch") != "tfm" or int(seq0["model"]["n_features"]) != xtr.shape[1]:
            raise SystemExit("--init-seq needs a transformer member with the same input blocks")
        fmean, fstd = seq0["feature_mean"], seq0["feature_std"]
    else:
        # in row chunks: a float64 copy of the whole matrix (all data: 1.56 M x 670, 7.8 GB) does not fit in 16 GB
        s1 = np.zeros(xtr.shape[1]); s2 = np.zeros(xtr.shape[1])
        for a_ in range(0, len(xtr), 200_000):
            blk = xtr[a_:a_ + 200_000].astype(np.float64)
            s1 += blk.sum(0); s2 += (blk * blk).sum(0)
        mu_ = s1 / len(xtr)
        fmean = mu_.astype(np.float32)
        fstd = np.maximum(np.sqrt(np.maximum(s2 / len(xtr) - mu_ ** 2, 0.0)), 1e-6).astype(np.float32)
    smean = np.nanmean(strn, 0).astype(np.float32); sstd = np.maximum(np.nanstd(strn, 0), 1e-4).astype(np.float32)
    if init_seq is not None:
        smean, sstd = seq0["scalar_mean"], seq0["scalar_std"]
    n_rows_tr = len(xtr)

    Wtr = None
    if ramp_weight != 1.0:
        # as nm1.train --ramp-weight: Ip over its own shot's maximum, from the base block's instantaneous Ip column;
        # the frames below 90 % of it (ramp-up, ramp-down, the end of a discharge) carry most of li's held-out
        # error, so they weigh ramp_weight in the flux, Consistency and scalar terms (batch-normalised, so
        # the loss keeps its scale)
        ip = np.abs(xtr[:, 20].astype(np.float64))
        b_ = np.flatnonzero(np.diff(shtr)) + 1
        peak = np.empty_like(ip)
        for s_, e_ in zip(np.concatenate([[0], b_]), np.concatenate([b_, [len(shtr)]])):
            peak[s_:e_] = max(float(ip[s_:e_].max()), 1.0)
        fw = np.where(ip / peak < 0.9, float(ramp_weight), 1.0).astype(np.float32)
        print(f"  ramp weight x{ramp_weight:g} on {float((fw != 1).mean()):.1%} of the training frames", flush=True)
        Wtr = torch.from_numpy(fw).to(dev)

    def to_dev(x, c, s, t):
        xw = torch.from_numpy(((x - fmean) / fstd).astype(np.float16)).to(dev)
        z = torch.from_numpy(c / coeff_std).to(dev)
        sw = torch.from_numpy((s - smean) / sstd).to(dev)
        return xw, z, sw, torch.from_numpy(t).to(dev)

    Xtr, Ztr, Str, Ttr = to_dev(xtr, ctr, strn, ttr)
    Xva, Zva, Sva, Tva = to_dev(xva, cva, sva, tva)
    seg_tr = _segments(shtr); seg_va = _segments(shva)
    del xtr, ctr, strn, xva, cva, sva
    print(f"  on the device: {torch.cuda.memory_allocated() / 1e9 if dev.type == 'cuda' else 0:.2f} GB", flush=True)

    sw = np.asarray([float(v) for v in scalar_weights.split(",")]) if scalar_weights else None
    w_psi_np, w_cons_np, gram_np = build_weights(weights, coeff_std, JACOBIAN, n_comp, sw)
    if sw is not None:
        print(f"  Consistency surrogate weights per scalar (R_axis..li): {sw.tolist()}", flush=True)
    w_psi = torch.from_numpy(w_psi_np).float().to(dev)
    gram = torch.from_numpy(gram_np).float().to(dev) if gram_np is not None else None
    w_cons = torch.from_numpy(w_cons_np).float().to(dev)

    def cons(dz):
        return ((dz @ gram) * dz).sum(1) if gram is not None else (w_cons * dz.pow(2)).sum(1)

    def batch(segs, idx_shots, X, Z, S, T):
        lens = segs[idx_shots, 1] - segs[idx_shots, 0]
        L = int(lens.max())
        ar = np.arange(L)[None, :]
        rows = segs[idx_shots, 0][:, None] + np.minimum(ar, lens[:, None] - 1)
        mask = torch.from_numpy(ar < lens[:, None]).to(dev)
        r = torch.from_numpy(rows).to(dev)
        batch.rows = r                                 # cache rows of this batch (for the boundary term)
        return X[r].float(), Z[r], S[r], T[r], mask      # stored half precision; autocast picks the compute type

    model = TfmRegressor(Xtr.shape[1], n_comp, width, pre_blocks, post_blocks, layers, heads, ffn, dropout,
                         attn_width, layer_scale, chan_tokens, chan_layers, chan_width, n_experts).to(dev)
    model.grad_ckpt = bool(grad_ckpt)          # recompute attention activations in backward (memory only, same values)
    if pre is not None:
        missing, unexpected = model.load_state_dict(pre["trunk"], strict=False)
        if unexpected or set(k.split(".")[0] for k in missing) - {"coeff_head", "scalar_head"}:
            raise SystemExit(f"trunk does not fit: missing {missing}, unexpected {unexpected}")
        print(f"  trunk from {init} (pretrained on {pre.get('shots')} shots, inputs only)", flush=True)
    if init_member is not None:
        # the per-frame trunk starts as a trained per-frame member (DIII-D labels only, same split); the
        # attention residual is zero at initialisation, so epoch 0 below scores that member itself
        from nm1.seqmodel import tfm_layout, warm_from_member
        if pre is not None:
            raise SystemExit("--init and --init-member are exclusive")
        member = torch.load(init_member, map_location="cpu", weights_only=False)
        if bool(member.get("all_data")) != bool(all_data):
            raise SystemExit("--init-member must come from a run on the same split (--all-data with --all-data)")
        layout = tfm_layout(n_features, 298 if use_ext else 0, n_cds, n_cdsg, 16 if ctx else 0)
        zero = warm_from_member(model, member, layout, fmean, fstd, smean, sstd, coeff_std)
        print(f"  per-frame trunk from {init_member} (epoch {member.get('epoch')}, S^ "
              f"{(member.get('val') or {}).get('S_hat', float('nan')):.5f}); blocks starting at zero: {zero}", flush=True)
    if init_seq is not None:
        if init_member is not None or pre is not None:
            raise SystemExit("--init-seq excludes --init and --init-member")
        for k in ("width", "pre", "post", "layers", "heads", "ffn", "attn_width"):
            want = int({"width": width, "layers": layers, "heads": heads, "ffn": ffn,
                        "pre": pre_blocks, "post": post_blocks, "attn_width": attn_width}[k])
            have = int(seq0["model"].get(k, 0))
            if have != want and not (k in ("layers", "ffn") and grow and want > have):
                raise SystemExit(f"--init-seq member has {k}={seq0['model'][k]}")
        if float(seq0["model"].get("layer_scale", 0.0)) != float(layer_scale):
            raise SystemExit("--init-seq member was trained with a different --layer-scale")
        n_old, f_old = int(seq0["model"]["layers"]), int(seq0["model"]["ffn"])
        if grow and (layers > n_old or ffn > f_old or chan_tokens or n_experts):
            # extension of a trained transformer that starts exactly as it (output projections of every new part
            # at zero): depth growth appends pre-norm attention layers whose attention and feed-forward outputs
            # are zero; feed-forward widening (Net2Net-style) keeps the trained units and adds units whose
            # outgoing weights are zero; the channel axis and the phase experts start at (almost) zero
            sd = dict(seq0["state_dict"])
            ffw = {k: sd.pop(k) for k in list(sd) if ".linear1." in k or ".linear2." in k} if ffn > f_old else {}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            new = {f"encoder.layers.{i}." for i in range(n_old, layers)} | {"chan_", "moe_"}
            if unexpected or any(not (any(k.startswith(p) for p in new) or k in ffw) for k in missing):
                raise SystemExit(f"growth does not fit: missing {missing[:4]}, unexpected {unexpected[:4]}")
            with torch.no_grad():
                for k, v in ffw.items():
                    p = dict(model.named_parameters())[k]
                    if k.endswith("linear1.weight") or k.endswith("linear1.bias"):
                        p[:f_old].copy_(v)                     # trained units; the new rows keep their init
                    elif k.endswith("linear2.weight"):
                        p.zero_(); p[:, :f_old].copy_(v)       # new units' outgoing weights at zero
                    else:
                        p.copy_(v)                             # linear2.bias
                for i in range(n_old, layers):
                    lyr = model.encoder.layers[i]
                    for lin in (lyr.self_attn.out_proj, lyr.linear2):
                        lin.weight.zero_(); lin.bias.zero_()
            print(f"  start: transformer member {init_seq} (epoch {seq0.get('epoch')}), extended: attention "
                  f"{n_old} -> {layers} layers, feed-forward {f_old} -> {ffn}, channel-axis tokens "
                  f"{chan_tokens or 'off'}, phase experts {n_experts or 'off'} (new parts start as the identity)",
                  flush=True)
        else:
            model.load_state_dict(seq0["state_dict"])
            print(f"  start: transformer member {init_seq} (epoch {seq0.get('epoch')})", flush=True)
    bnd = None
    if bnd_file is not None and bnd_weight > 0:
        # boundary-sensitivity term: the LCFS moves by (flux error at a boundary point - flux error at the
        # point where the boundary flux gradient is smallest, the x-point side) / |grad psi|; frames sampled
        # as for the shape heads (research/t29_lcfs_points.py), labels from the training flux only
        with h5py.File(bnd_file, "r") as hb:
            b_idx = hb["idx"][:]; b_pts = hb["pts"][:]; b_x = hb["xref"][:]; b_g = hb["grad"][:]; b_rg = hb["rgeo"][:]
        bmap = np.full(n_rows_tr, -1, np.int64)
        okb = b_idx < n_rows_tr
        bmap[b_idx[okb]] = np.arange(okb.sum())
        to = lambda a: torch.as_tensor(a, device=dev)
        pts_all = np.concatenate([b_pts[okb], b_x[okb][:, None, :]], 1)              # (n, K+1, 2) rows, cols
        grid = torch.as_tensor(np.stack([pts_all[..., 1] / 32.0 - 1.0, pts_all[..., 0] / 32.0 - 1.0], -1),
                               dtype=torch.float32, device=dev)                   # grid_sample x = col, y = row
        # the boundary points' major radius (the DIII-D grid: R 0.84..2.54 m over 65 columns)
        r_pts = 0.84 + b_pts[okb][..., 1] * (1.70 / 64.0)
        bnd = dict(map=to(bmap), grid=grid, g=to(np.maximum(b_g[okb], 1e-3)).float(), rg=to(b_rg[okb]).float(),
                   comps=torch.as_tensor(np.asarray(state["pca_components"], np.float32), device=dev),
                   mean=torch.as_tensor(np.asarray(state["pca_mean"], np.float32), device=dev),
                   cstd=torch.as_tensor(coeff_std, device=dev), rpts=to(r_pts.astype(np.float32)))
        print(f"  boundary term: {int(okb.sum()):,} labelled training frames, weight {bnd_weight}", flush=True)

    def bnd_loss(zp, zb, mb):
        rows = batch.rows[mb]
        ids = bnd["map"][rows]
        sel = ids >= 0
        if not bool(sel.any()):
            return None
        ids = ids[sel]
        e = ((zp.float()[mb])[sel] - zb[mb][sel]) * bnd["cstd"]                     # raw coefficient error
        emap = (e @ bnd["comps"]).view(-1, 1, 65, 65)
        gsel = bnd["grid"][ids][:, :, None, :]
        v = F.grid_sample(emap, gsel, align_corners=True)[:, 0, :, 0]               # (n, K+1)
        d = (v[:, :-1] - v[:, -1:]) / bnd["g"][ids] / bnd["rg"][ids, None]          # displacement / R_geo
        if bnd_p == 2.0:
            lb = d.pow(2).mean()
        else:
            # D_LCFS is the symmetric Hausdorff distance (the frame's largest deviation): a p-norm over the frame's
            # points leans towards that maximum; computed relative to the frame's largest |d| (a constant scale, so
            # the value and its gradient are the p-norm's exactly) to stay clear of underflow at high p
            ad = d.abs()
            dm = ad.amax(1, keepdim=True).detach().clamp_min(1e-12)
            lb = (dm[:, 0] * (ad / dm).pow(bnd_p).mean(1).pow(1.0 / bnd_p)).pow(2).mean()
        if ip_weight > 0:
            # li is computed from the map as (inner integral of |grad psi|^2) / (the map's own Ip)^2, and that Ip is
            # Ampere's loop integral of |grad psi| / R along the LCFS: a boundary term that bends the gradient there
            # moves li through the denominator.  Keep the loop integral of the prediction on the true LCFS equal to
            # the truth's (points equally spaced in arc length, so the sum is the integral up to a constant).
            ct = zb[mb][sel] * bnd["cstd"]
            ptrue = (bnd["mean"] + ct @ bnd["comps"]).view(-1, 1, 65, 65)
            ppred = ptrue + emap
            def gmag(m):
                gz, gr = torch.gradient(m[:, 0], spacing=(0.05, 1.70 / 64.0), dim=(1, 2))
                gg = torch.sqrt(gz * gz + gr * gr + 1e-12)[:, None]
                return F.grid_sample(gg, gsel[:, :-1], align_corners=True)[:, 0, :, 0]
            r = bnd["rpts"][ids]
            it = (gmag(ptrue) / r).sum(1); ip_ = (gmag(ppred) / r).sum(1)
            lb = lb + (ip_weight / max(bnd_weight, 1e-9)) * (ip_ / it.clamp_min(1e-9) - 1.0).pow(2).mean()
        return lb

    print(f"  transformer member: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    trunk_names = ("inp.", "pre.", "post.", "norm.", "coeff_head.", "scalar_head.")
    trunk = [p for n, p in model.named_parameters() if n.startswith(trunk_names)]
    if trunk_lr != 1.0:
        # the per-frame trunk and heads train at trunk_lr x lr, the attention path at lr
        attn = [p for n, p in model.named_parameters() if not n.startswith(trunk_names)]
        opt = torch.optim.AdamW([{"params": trunk, "lr": lr * trunk_lr}, {"params": attn, "lr": lr}],
                                weight_decay=weight_decay)
        print(f"  learning rate: trunk {lr * trunk_lr:.2e}, attention {lr:.2e}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(seg_tr) / batch_shots)
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    sv_fin = torch.isfinite(Sva)

    def evaluate():
        model.eval()
        zp_all, sp_all, z_all, s_all = [], [], [], []
        with torch.inference_mode():
            for a in range(0, len(seg_va), batch_shots):
                ids = np.arange(a, min(a + batch_shots, len(seg_va)))
                xb, zb, sb, tb, mb = batch(seg_va, ids, Xva, Zva, Sva, Tva)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    zp, sp = model(xb, mb, tb)
                zp_all.append(zp.float()[mb]); sp_all.append(sp.float()[mb]); z_all.append(zb[mb]); s_all.append(sb[mb])
        zp, sp, z, s = (torch.cat(v) for v in (zp_all, sp_all, z_all, s_all))
        dz = zp - z
        r2_psi = 1.0 - float((w_psi * dz.pow(2)).sum(1).mean())
        cons_hat = 1.0 - float(cons(dz).mean())
        fin = torch.isfinite(s); s0 = torch.nan_to_num(s)
        ss_res = ((sp - s0).pow(2) * fin).sum(0)
        mu = (s0 * fin).sum(0) / fin.sum(0).clamp_min(1)
        ss_tot = (((s0 - mu).pow(2)) * fin).sum(0)
        r2_qb = float((1 - ss_res / ss_tot.clamp_min(1e-9)).mean())
        model.train()
        return {"r2_psi": r2_psi, "cons_hat": cons_hat, "r2_qb": r2_qb,
                "S_hat": W_PSI * r2_psi + W_QB * r2_qb + W_CONS * cons_hat + 0.10 * 0.97}

    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, step = -1e9, 0, 0
    rng = np.random.default_rng(seed)

    def save(epoch, m):
        torch.save({"kind": "seq", "arch": "tfm", "state_dict": model.state_dict(),
                    "model": {"n_features": int(Xtr.shape[1]), "n_components": n_comp, "width": width,
                              "pre": pre_blocks, "post": post_blocks, "layers": layers, "heads": heads,
                              "ffn": ffn, "dropout": dropout, "attn_width": attn_width,
                              **({"layer_scale": layer_scale} if layer_scale > 0 else {}),
                              **({"chan_tokens": chan_tokens, "chan_layers": chan_layers, "chan_width": chan_width}
                                 if chan_tokens else {}),
                              **({"n_experts": n_experts} if n_experts else {})},
                    "feature_mean": fmean, "feature_std": fstd,
                    "pca_mean": state["pca_mean"], "pca_components": state["pca_components"],
                    "coeff_std": coeff_std, "scalar_mean": smean, "scalar_std": sstd,
                    "n_base_features": n_features, "n_ext": 298 if use_ext else 0,
                    "n_cds": n_cds, "n_cdsg": n_cdsg, "n_ctx": 16 if ctx else 0, "epoch": epoch, "seed": seed,
                    "val": m, "all_data": bool(all_data), "weights": weights,
                    "init": str(init) if init else None,
                    "init_member": str(init_member) if init_member else None, "trunk_lr": trunk_lr,
                    "freeze_epochs": freeze_epochs, "diff_weight": diff_weight,
                    "scalar_weights": scalar_weights or None, "init_seq": str(init_seq) if init_seq else None,
                    "bnd_weight": bnd_weight if bnd is not None else 0.0, "ip_weight": ip_weight,
                    **({"ramp_weight": ramp_weight} if ramp_weight != 1.0 else {}),
                    **({"bnd_p": bnd_p} if bnd_p != 2.0 else {})}, output)
        output.with_suffix(".json").write_text(json.dumps(
            {"best": best, "epoch": epoch, "source": "raw", "all_data": bool(all_data), "seed": seed,
             "arch": "tfm", "width": width, "layers": layers, "heads": heads, "epochs": epochs,
             "init_member": str(init_member) if init_member else None,
             "selected_on": "S_hat" if not save_epoch else "given"}, indent=2), encoding="utf-8")

    if init_member is not None:
        m = evaluate()
        print(f"epoch 000: warm start  r2psi={m['r2_psi']:.6f} cons^={m['cons_hat']:.5f} r2qb={m['r2_qb']:.5f} "
              f"S^={m['S_hat']:.5f}", flush=True)
        if not save_epoch:
            best = m["S_hat"]
            save(0, m)
    start_epoch = 1
    last = output.with_name(output.stem + "_last.pt")
    ck = None
    if resume and save_epoch and last.exists():
        try:
            ck = torch.load(last, map_location=dev, weights_only=False)
        except Exception as e:  # noqa: BLE001 - a file cut by a crash: start over instead of failing
            bad = last.with_name(last.name + ".corrupt")
            last.replace(bad)
            print(f"  resume file {last.name} unreadable ({e!r}); moved to {bad.name}, starting from epoch 1", flush=True)
    if ck is not None:
        # continue a fixed-epoch run from its last finished epoch: weights, optimiser, schedule position and the
        # shot order of the epochs already done (so the remaining epochs see the same batches as an unbroken run)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        done_ep = int(ck["epoch"])
        for _ in range(done_ep):
            rng.permutation(len(seg_tr))
        for _ in range(done_ep * steps_per_epoch):
            sched.step()
        start_epoch = done_ep + 1
        print(f"  resume from {last.name}: epoch {done_ep} done, continuing at {start_epoch}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        te = time.perf_counter()
        if freeze_epochs:
            # LP-FT (Kumar et al. 2022): the new attention path trains alone first, on the trunk it was put in,
            # so the pretrained per-frame features are not pulled about while the new part is still random
            frozen = epoch <= freeze_epochs
            for p_ in trunk:
                p_.requires_grad_(not frozen)
            if epoch == 1 or epoch == freeze_epochs + 1:
                print(f"  epoch {epoch}: per-frame trunk {'frozen' if frozen else 'trainable'}", flush=True)
        order = rng.permutation(len(seg_tr))
        tot = 0.0; tot_b = 0.0; n_b = 0
        for a in range(0, len(order), batch_shots):
            xb, zb, sb, tb, mb = batch(seg_tr, order[a:a + batch_shots], Xtr, Ztr, Str, Ttr)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                zp, sp = model(xb, mb, tb)
            dz = (zp.float() - zb)[mb]
            sp_, sb_ = sp.float()[mb], sb[mb]
            fin = torch.isfinite(sb_)
            if Wtr is None:
                loss = (W_PSI * (w_psi * dz.pow(2)).sum(1).mean() + W_CONS * cons(dz).mean()
                        + W_QB * (F.mse_loss(sp_[fin], sb_[fin]) if fin.any() else dz.sum() * 0))
            else:
                wn = Wtr[batch.rows][mb]; wn = wn / wn.mean()
                qf = fin.all(1) if fin.dim() > 1 else fin
                loss = (W_PSI * (wn * (w_psi * dz.pow(2)).sum(1)).mean() + W_CONS * (wn * cons(dz)).mean()
                        + W_QB * ((wn[qf, None] * (sp_[qf] - sb_[qf]).pow(2)).mean() if qf.any() else dz.sum() * 0))
            if bnd is not None:
                lb = bnd_loss(zp, zb, mb)
                if lb is not None:
                    loss = loss + bnd_weight * lb
                    tot_b += float(lb); n_b += 1
            if diff_weight:
                # increments between consecutive frames of a shot (PaMMA-Net-style incremental supervision): the
                # change from one EFIT frame to the next, where the ramp-down / end-of-shot errors concentrate
                pair = mb[:, 1:] & mb[:, :-1]
                ddz = ((zp[:, 1:] - zp[:, :-1]).float() - (zb[:, 1:] - zb[:, :-1]))[pair]
                if len(ddz):
                    loss = loss + diff_weight * (W_PSI * (w_psi * ddz.pow(2)).sum(1).mean() + W_CONS * cons(ddz).mean())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += float(loss)
        m = evaluate() if (epoch % val_every == 0 or epoch == epochs) else None
        line = f"epoch {epoch:03d}: loss={tot / steps_per_epoch:.6f}"
        if n_b:
            line += f" bnd={tot_b / n_b:.3e}"
        if m:
            line += f"  r2psi={m['r2_psi']:.6f} cons^={m['cons_hat']:.5f} r2qb={m['r2_qb']:.5f} S^={m['S_hat']:.5f}"
        print(line + f"  {time.perf_counter() - te:.1f}s", flush=True)
        if save_epoch and epoch < save_epoch:
            # a run that saves one given epoch keeps the latest state too, so a crash before that epoch
            # loses at most one epoch (the file is replaced every epoch; the submitted member is `output`);
            # written to a temporary name and renamed, so a crash mid-write never leaves a cut resume file
            tmp = last.with_name(last.name + ".tmp")
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, tmp)
            tmp.replace(last)
        keep = (epoch == save_epoch) if save_epoch else (m is not None and m["S_hat"] > best)
        if keep:
            if m is not None:
                best = m["S_hat"]
            best_epoch = epoch
            save(epoch, m)
        if save_epoch and epoch >= save_epoch:
            break
    if save_epoch and best_epoch == save_epoch and output.exists() and os.environ.get("NM1_KEEP_LAST") != "1":
        # the resume file only serves a run that stopped before its save epoch; the member is written, so the
        # resume file (about three times the member's size: it holds the optimiser state) is removed
        last.unlink(missing_ok=True)
    print(f"saved {output} (epoch {best_epoch}, S^ {best:.5f}) in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--save-epoch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-shots", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--weights", default="gram", choices=("gram", "jacobian"))
    ap.add_argument("--max-shots", type=int, default=0, help="smoke test: use only this many training shots")
    ap.add_argument("--device", default="", help="cuda / cpu (default: cuda when available)")
    ap.add_argument("--init", type=Path, help="start from a self-supervised trunk (nm1.pretrain_tfm)")
    ap.add_argument("--pre-blocks", type=int, default=2)
    ap.add_argument("--post-blocks", type=int, default=2)
    ap.add_argument("--attn-width", type=int, default=0, help="attention width (0 = the trunk width)")
    ap.add_argument("--ctx", action="store_true", help="append the per-frame members' shot-context block (16)")
    ap.add_argument("--init-member", type=Path,
                    help="start the per-frame trunk from a trained per-frame member (width x pre+post blocks)")
    ap.add_argument("--trunk-lr", type=float, default=1.0, help="per-frame trunk learning rate, as a multiple of --lr")
    ap.add_argument("--layer-scale", type=float, default=0.0,
                    help="LayerScale gain on the attention output (e.g. 1e-4); 0 = zero-initialised projection")
    ap.add_argument("--freeze-epochs", type=int, default=0, help="train only the attention path for the first N epochs")
    ap.add_argument("--diff-weight", type=float, default=0.0, help="weight of the frame-to-frame increment loss")
    ap.add_argument("--scalar-weights", default="", help="7 weights of R_axis..li in the Consistency surrogate (nm1.train)")
    ap.add_argument("--init-seq", type=Path, help="fine-tune an existing transformer member (same architecture)")
    ap.add_argument("--bnd-file", type=Path, help="LCFS points of training frames (research/t29_lcfs_points.py)")
    ap.add_argument("--bnd-weight", type=float, default=0.0, help="weight of the boundary-sensitivity term")
    ap.add_argument("--resume", action="store_true", help="with --save-epoch: continue from <output>_last.pt")
    ap.add_argument("--ip-weight", type=float, default=0.0,
                    help="with the boundary term: keep the LCFS loop integral of |grad psi|/R (the map's Ip, li's denominator)")
    ap.add_argument("--bnd-p", type=float, default=2.0,
                    help="boundary term: p-norm of the frame's point displacements (2 = mean square; higher leans to the "
                         "largest deviation, as D_LCFS's Hausdorff distance)")
    ap.add_argument("--chan-tokens", type=int, default=0,
                    help="second attention axis inside each frame: input columns per token (0 = off)")
    ap.add_argument("--chan-layers", type=int, default=2)
    ap.add_argument("--chan-width", type=int, default=128)
    ap.add_argument("--experts", type=int, default=0, help="discharge-phase experts on the flux coefficients (0 = off)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing of the attention layers (fits a deeper stack in GPU memory; same values)")
    ap.add_argument("--ramp-weight", type=float, default=1.0,
                    help="weight of the frames below 90 %% of their shot's peak Ip (as nm1.train; 1 = off)")
    ap.add_argument("--grow", action="store_true",
                    help="with --init-seq and a larger --layers: keep the trained attention layers, append identity-"
                         "initialised ones (depth growth)")
    a = ap.parse_args()
    train(a.output, a.all_data, a.save_epoch, a.epochs, a.seed, a.width, a.layers, a.heads, a.ffn,
          a.dropout, a.batch_shots, a.lr, a.warmup, weights=a.weights, max_shots=a.max_shots, device=a.device,
          init=a.init, pre_blocks=a.pre_blocks, post_blocks=a.post_blocks, attn_width=a.attn_width,
          ctx=a.ctx, init_member=a.init_member, trunk_lr=a.trunk_lr,
          layer_scale=a.layer_scale, freeze_epochs=a.freeze_epochs, diff_weight=a.diff_weight,
          scalar_weights=a.scalar_weights, init_seq=a.init_seq, bnd_file=a.bnd_file, bnd_weight=a.bnd_weight,
          resume=a.resume, ip_weight=a.ip_weight, bnd_p=a.bnd_p, grow=a.grow, chan_tokens=a.chan_tokens,
          chan_layers=a.chan_layers, chan_width=a.chan_width, n_experts=a.experts, ramp_weight=a.ramp_weight,
          grad_ckpt=a.grad_ckpt)


if __name__ == "__main__":
    main()