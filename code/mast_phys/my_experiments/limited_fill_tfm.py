"""Whole-shot transformer for the transductive limited-frame fill and diverted blend (alternative to the per-frame
CatBoost of limited_fill_v17.py).

Same task, same data, same shot-grouped 5-fold out-of-fold protocol as the trees: inputs are the 28 per-frame input
features of limited_fill_v17.frame_features (coil currents, Ip and its derivative, TF, solenoid, Thomson summaries,
discharge clock), the targets are this run's OWN physics outputs on its resolved (diverted) frames -- beta_N and the
q95 shape factor y = q95 Ip / F -- so no label is read (label-free test-time adaptation, rule 3).  The difference is
the receptive field: the trees see one frame; the encoder attends over the whole discharge (bidirectional, padding
masked), so a frame is predicted from the evolution of Ip, the coils and the Thomson profile before and after it.

Deterministic: CPU, fixed seeds and a fixed thread count, so a rerun gives the same arrays (rule 6 reproduction).
"""
import math

import numpy as np

D_MODEL, LAYERS, HEADS, FFN, DROPOUT = 96, 3, 4, 192, 0.1
EPOCHS, LR, WD, BATCH = 40, 1e-3, 1e-4, 16
THREADS = 4


def _normalise(X):
    """Robust per-feature scaling over all frames of the fold's inputs (inputs only); NaN -> 0 with a flag column."""
    med = np.nanmedian(X, 0)
    iqr = np.nanpercentile(X, 75, 0) - np.nanpercentile(X, 25, 0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
    Z = (X - med) / iqr
    nan = ~np.isfinite(Z)
    Z = np.clip(np.where(nan, 0.0, Z), -10, 10)
    flags = nan[:, nan.any(0)].astype(float)
    return np.hstack([Z, flags]).astype(np.float32)


def _model(n_in, n_out):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_in, D_MODEL)
            layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FFN, DROPOUT, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(D_MODEL, n_out)

        def forward(self, x, tpos, pad):
            # sinusoidal encoding of the frame time within the shot (ms), added to the projected inputs
            k = torch.arange(D_MODEL // 2, dtype=x.dtype)
            freq = torch.exp(-math.log(1000.0) * 2 * k / D_MODEL)
            ang = tpos[..., None] * freq
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], -1)
            h = self.enc(self.inp(x) + pe, src_key_padding_mask=pad)
            return self.out(h)

    return Net()


def _batches(idx, rng):
    idx = np.array(idx)
    rng.shuffle(idx)
    return [idx[i:i + BATCH] for i in range(0, len(idx), BATCH)]


def _pack(shots, Z, T, Y, M):
    import torch
    L = max(len(Z[s]) for s in shots)
    x = torch.zeros(len(shots), L, Z[shots[0]].shape[1])
    t = torch.zeros(len(shots), L)
    y = torch.zeros(len(shots), L, Y[shots[0]].shape[1])
    m = torch.zeros(len(shots), L, Y[shots[0]].shape[1], dtype=torch.bool)
    pad = torch.ones(len(shots), L, dtype=torch.bool)
    for j, s in enumerate(shots):
        n = len(Z[s])
        x[j, :n] = torch.from_numpy(Z[s]); t[j, :n] = torch.from_numpy(T[s])
        y[j, :n] = torch.from_numpy(Y[s]); m[j, :n] = torch.from_numpy(M[s]); pad[j, :n] = False
    return x, t, y, m, pad


def out_of_fold(X, targets, train_masks, fold, sizes, times_ms, seed=0):
    """``targets``: list of per-frame target arrays; ``train_masks``: matching boolean masks of the frames each target
    may be learned from.  Returns the out-of-fold predictions (one array per target), shot-grouped by ``fold``."""
    import torch
    torch.set_num_threads(THREADS)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    Zall = _normalise(np.asarray(X, float))
    Yall = np.column_stack(targets).astype(float)
    Mall = np.column_stack(train_masks) & np.isfinite(Yall)
    shot_fold = np.array([fold[starts[i]] for i in range(len(sizes))])
    preds = np.full(Yall.shape, np.nan)
    for k in sorted(set(shot_fold)):
        tr_shots = [i for i in range(len(sizes)) if shot_fold[i] != k]
        te_shots = [i for i in range(len(sizes)) if shot_fold[i] == k]
        mu = np.array([np.mean(Yall[np.concatenate([np.arange(starts[i], starts[i + 1]) for i in tr_shots])][:, c][
            Mall[np.concatenate([np.arange(starts[i], starts[i + 1]) for i in tr_shots])][:, c]]) for c in range(Yall.shape[1])])
        sd = np.array([np.std(Yall[np.concatenate([np.arange(starts[i], starts[i + 1]) for i in tr_shots])][:, c][
            Mall[np.concatenate([np.arange(starts[i], starts[i + 1]) for i in tr_shots])][:, c]]) + 1e-9 for c in range(Yall.shape[1])])
        Z = {i: Zall[starts[i]:starts[i + 1]] for i in range(len(sizes))}
        T = {i: ((np.asarray(times_ms[starts[i]:starts[i + 1]], float) - float(times_ms[starts[i]])) / 10.0).astype(np.float32)
             for i in range(len(sizes))}
        Y = {i: np.nan_to_num((Yall[starts[i]:starts[i + 1]] - mu) / sd).astype(np.float32) for i in range(len(sizes))}
        M = {i: Mall[starts[i]:starts[i + 1]] for i in range(len(sizes))}
        torch.manual_seed(seed + k)
        rng = np.random.default_rng(seed + k)
        net = _model(Zall.shape[1], Yall.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        steps = EPOCHS * math.ceil(len(tr_shots) / BATCH)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
        net.train()
        for _ in range(EPOCHS):
            for b in _batches(tr_shots, rng):
                x, t, y, m, pad = _pack(list(b), Z, T, Y, M)
                err = (net(x, t, pad) - y) ** 2
                loss = (err * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            for b in [te_shots[i:i + 64] for i in range(0, len(te_shots), 64)]:
                x, t, y, m, pad = _pack(list(b), Z, T, Y, M)
                p = net(x, t, pad).numpy()
                for j, s in enumerate(b):
                    preds[starts[s]:starts[s + 1]] = p[j, :sizes[s]] * sd + mu
    return [preds[:, c] for c in range(Yall.shape[1])]
