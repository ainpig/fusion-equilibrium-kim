"""A differentiable emulator of what the scorer does to a flux-map error.

Given a frame's true flux coefficients ``c`` and an error ``d`` on them, the scorer changes the
seven derived scalars by ``s(c + d) - s(c)`` and moves the LCFS by a Hausdorff distance ``H``.
Both are discrete procedures (marching squares, a level bisection that stops at the X-point *or*
at the envelope mask, a 512-point resampling), so no frame-averaged quadratic form describes them
well.  This network is trained on the scorer's own outputs (``emul_data.py``) to predict

    ds_j(c, d) = f_j(c, d) - f_j(c, 0)          exactly zero at d = 0, for any c
    H(c, d)    = || g(c, d) - g(c, 0) ||        non-negative, exactly zero at d = 0

and a member trained through it gets per-frame sensitivities of the actual metric.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from nm1 import ARTIFACTS
from nm1.cache import BASIS, CACHE
from nm1.emul_data import OUT as DATA_H5

OUT = ARTIFACTS / "emulator.pt"
NAMES = ("R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li", "hausdorff")


class Emulator(nn.Module):
    def __init__(self, n_comp: int = 256, width: int = 1024, depth: int = 4, h_dim: int = 16):
        super().__init__()
        self.n_comp = n_comp
        layers = [nn.Linear(2 * n_comp, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 7 + h_dim)

    def _f(self, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([z, d], -1)))

    def forward(self, z: torch.Tensor, d: torch.Tensor):
        """``z``, ``d`` whitened true coefficients and whitened error -> (ds (.., 7), H (..))."""
        a = self._f(z, d) - self._f(z, torch.zeros_like(d))
        ds = a[..., :7]
        h = torch.sqrt((a[..., 7:] ** 2).sum(-1) + 1e-12) - 1e-6
        return ds, h


def load(path: Path = OUT, device="cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = Emulator(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload


def train(data: Path = DATA_H5, out: Path = OUT, epochs: int = 60, batch: int = 4096,
          lr: float = 1e-3, seed: int = 0, width: int = 1024, depth: int = 4,
          device: str = "cuda") -> Path:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state = torch.load(BASIS, map_location="cpu", weights_only=False)
    cstd = state["coeff_std"].astype(np.float64)
    with h5py.File(data, "r") as h5:
        idx = h5["idx"][:]
        delta = h5["delta"][:].astype(np.float64)              # (F, V, 256)
        o = h5["out"][:]                                        # (F, 1+V, 8)
    with h5py.File(CACHE, "r") as h5c:
        coeff = h5c["train"]["coeff"][idx].astype(np.float64)
        shot = h5c["train"]["shot"][idx]
    base = o[:, 0, :7]
    ds = o[:, 1:, :7] - base[:, None, :]
    H = o[:, 1:, 7]
    ok = np.isfinite(ds).all(-1) & np.isfinite(H) & np.isfinite(base).all(-1)[:, None]
    var = np.nanvar(base, 0)                                    # pooled-R2 denominators
    scale = np.sqrt(var)
    h_scale = float(np.nanstd(H))
    # shot-level split of the sampled frames for an honest fidelity number
    ushots = np.unique(shot)
    test_shots = set(rng.choice(ushots, max(1, len(ushots) // 10), replace=False).tolist())
    is_test = np.array([s in test_shots for s in shot])
    F_, V = delta.shape[:2]
    z = (coeff / cstd).astype(np.float32)
    d = (delta / cstd).astype(np.float32)
    y = (ds / scale).astype(np.float32)
    h = (H / h_scale).astype(np.float32)

    def flat(mask_frames):
        fi, vi = np.nonzero(ok & mask_frames[:, None])
        return fi, vi

    tr_f, tr_v = flat(~is_test)
    te_f, te_v = flat(is_test)
    dev = torch.device(device)
    Z, D, Y, Hh = (torch.from_numpy(a).to(dev) for a in (z, d, y, h))
    model = Emulator(z.shape[1], width, depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    n = len(tr_f)
    steps = math.ceil(n / batch) * epochs
    step = 0
    tf, tv = torch.from_numpy(tr_f).to(dev), torch.from_numpy(tr_v).to(dev)
    ef, evv = torch.from_numpy(te_f).to(dev), torch.from_numpy(te_v).to(dev)
    print(f"emulator: {n:,} train samples, {len(te_f):,} test samples "
          f"({len(test_shots)} held-out shots)", flush=True)

    def fidelity():
        model.eval()
        with torch.no_grad():
            p, ph = [], []
            for i in range(0, len(ef), 16384):
                a, b = model(Z[ef[i:i + 16384]], D[ef[i:i + 16384], evv[i:i + 16384]])
                p.append(a)
                ph.append(b)
            p, ph = torch.cat(p), torch.cat(ph)
            yt, ht = Y[ef, evv], Hh[ef, evv]
            corr = [float(torch.corrcoef(torch.stack([p[:, j], yt[:, j]]))[0, 1])
                    for j in range(7)]
            corr.append(float(torch.corrcoef(torch.stack([ph, ht]))[0, 1]))
            r2 = [1 - float(((p[:, j] - yt[:, j]) ** 2).mean() / (yt[:, j] ** 2).mean())
                  for j in range(7)]
            r2.append(1 - float(((ph - ht) ** 2).mean() / ((ht - ht.mean()) ** 2).mean()))
        model.train()
        return corr, r2

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            fi, vi = tf[b], tv[b]
            p, ph = model(Z[fi], D[fi, vi])
            loss = ((p - Y[fi, vi]) ** 2).mean() + ((ph - Hh[fi, vi]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr * min(1, step / 200) * 0.5 * (1 + math.cos(math.pi * step / steps))
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == epochs:
            corr, r2 = fidelity()
            print(f"epoch {ep:03d} loss={tot / n:.5f} | held-out corr "
                  + " ".join(f"{k}={c:.3f}" for k, c in zip(NAMES, corr))
                  + f" | R2 " + " ".join(f"{r:.3f}" for r in r2)
                  + f" | {time.perf_counter() - t0:.0f}s", flush=True)
    corr, r2 = fidelity()
    torch.save({"state_dict": model.state_dict(),
                "model": {"n_comp": int(z.shape[1]), "width": width, "depth": depth},
                "scale": scale, "var": var, "h_scale": h_scale, "coeff_std": cstd,
                "fidelity_corr": dict(zip(NAMES, corr)), "fidelity_r2": dict(zip(NAMES, r2))},
               out)
    out.with_suffix(".json").write_text(json.dumps(
        {"fidelity_corr": dict(zip(NAMES, corr)), "fidelity_r2": dict(zip(NAMES, r2)),
         "var": dict(zip(NAMES[:7], var.tolist())), "h_scale": h_scale}, indent=2),
        encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_H5)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    train(a.data, a.out, a.epochs, width=a.width, depth=a.depth, lr=a.lr)


if __name__ == "__main__":
    main()
