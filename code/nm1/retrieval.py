"""Retrieval shape correction: borrow flux-surface shape from the nearest training equilibria.

DIII-D runs the same scenario many times, so a test frame usually has near-twins among the
training frames.  A regressor averages over them and returns the smooth answer; their ground
truth still carries the scenario's own boundary detail.  So:

1. embed every training frame with a trained member -- its final hidden layer, after the last
   LayerNorm, the space its linear head reads coefficients from;
2. for each frame to predict, take the ``k`` nearest training frames in that space and average
   their TRUE flux coefficients;
3. blend that into the ensemble's flux map with weight ``w`` -- and then refit the blend's
   amplitude and level back onto the ensemble's (``predict.projected_flux`` at full strength), so
   what is borrowed is the *shape* of the flux surfaces and not the flux level, which the
   retrieval gets worse than the regressor does.

Measured on the `selection` half with one member (gram_s4001, k=8): w=0.2 moves S from 0.995079 to
0.995223 -- R2_psi unchanged, 1 - D_LCFS +3.2e-04, Consistency +5.7e-04; w=0.35 is already worse.
Training frames only are ever retrieved; the held-out and test shots never enter the index.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from nm1.cache import CACHE
from nm1.model import load_member, member_inputs


class Retrieval:
    def __init__(self, member: Path, k: int = 8, device: torch.device | str = "cuda",
                 chunk: int = 65536):
        self.k = int(k)
        self.device = torch.device(device)
        self.model, self.payload = load_member(member, self.device)
        with h5py.File(CACHE, "r") as h5:
            tr = h5["train"]
            green = h5["green"][:]
            ext = None
            if int(self.payload.get("n_ext") or 0):
                from nm1.cache2 import CACHE2
                with h5py.File(CACHE2, "r") as h5e:
                    ext = h5e["train"]["ext"][:]
            feats = tr["features"][:]
            cur = tr["currents"][:]
            shot = tr["shot"][:]
            self.coeff = tr["coeff"][:]                   # stays on the CPU: 1.4 GB
        # A shot-context block or temporal offsets make the input a whole-shot function, so
        # those members are embedded shot by shot; everything else in large chunks.
        per_shot = (bool(self.payload.get("temporal_offsets"))
                    or bool(int(self.payload.get("n_ctx") or 0)))
        if per_shot:
            b = np.flatnonzero(np.diff(shot)) + 1
            bounds = zip(np.concatenate([[0], b]), np.concatenate([b, [len(shot)]]))
        else:
            bounds = [(s, min(s + 65536, len(shot))) for s in range(0, len(shot), 65536)]
        parts = []
        for s, e in bounds:
            parts.append(self._embed(member_inputs(feats[s:e], cur[s:e], self.payload, green,
                                                   None if ext is None else ext[s:e])))
        self.emb = torch.cat(parts)                       # (N, width) fp16
        self.emb_sq = (self.emb.float() ** 2).sum(1)
        self.chunk = chunk
        self.green = green

    @torch.inference_mode()
    def _embed(self, x: np.ndarray) -> torch.Tensor:
        m = self.model
        xb = torch.from_numpy(x).to(self.device)
        return m.norm(m.blocks(m.input(xb))).half()

    @torch.inference_mode()
    def neighbour_coeff(self, base, currents, ext=None) -> np.ndarray:
        """``(T, n_comp)`` mean true coefficients of each frame's k nearest training frames."""
        q = self._embed(member_inputs(base, currents, self.payload, self.green, ext)).float()
        qn = (q ** 2).sum(1, keepdim=True)
        best_d = torch.full((len(q), self.k), float("inf"), device=self.device)
        best_i = torch.zeros((len(q), self.k), dtype=torch.long, device=self.device)
        for j in range(0, len(self.emb), self.chunk):
            t = self.emb[j:j + self.chunk].float()
            d = qn + self.emb_sq[j:j + self.chunk][None, :] - 2 * q @ t.T
            dv, di = torch.topk(d, self.k, dim=1, largest=False)
            cd = torch.cat([best_d, dv], 1)
            ci = torch.cat([best_i, di + j], 1)
            best_d, sel = torch.topk(cd, self.k, dim=1, largest=False)
            best_i = ci.gather(1, sel)
        return self.coeff[best_i.cpu().numpy()].astype(np.float64).mean(1)
