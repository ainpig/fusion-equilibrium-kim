"""Cache-row predictions of any member on the held-out (``val``) frames, stored once per member.

Choosing an ensemble means scoring many combinations of the same members.  Running every member
over the held-out frames once and keeping the coefficients makes each combination a weighted mean
and one official-functional scoring pass (``fastscore.Reference.score``).

The inputs come from the caches, which are bit-identical to the parquet-derived features
(VERIFICATION.md 1 for the base block, ``cache2.py``'s alignment check for the second block), so
these are the predictions the deployed chain makes -- before the stack.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from nm1 import ARTIFACTS
from nm1.cache import CACHE

PRED_DIR = ARTIFACTS / "valpreds"


class CacheInputs:
    """Everything a member might read for one split of the caches, loaded lazily.

    ``rows`` (a boolean mask over the split's rows) keeps a subset of whole shots -- the training
    shots of one fold, for the out-of-fold first stage (stage1.py).
    """

    def __init__(self, split: str = "val", rows: np.ndarray | None = None):
        self.split, self.rows = split, rows
        with h5py.File(CACHE, "r") as h5:
            g = h5[split]
            self.base = self._take(g["features"])
            self.currents = self._take(g["currents"])
            self.shot = self._take(g["shot"])
            self.green = h5["green"][:]
        self.times_ms = self.base[:, 136].astype(np.float64) * 1000.0
        b = np.flatnonzero(np.diff(self.shot)) + 1
        self.starts = np.concatenate([[0], b])
        self.ends = np.concatenate([b, [len(self.shot)]])
        self._ext = None

    def _take(self, ds) -> np.ndarray:
        return ds[:] if self.rows is None else ds[:][self.rows]

    @property
    def ext(self) -> np.ndarray:
        if self._ext is None:
            from nm1.cache2 import CACHE2
            with h5py.File(CACHE2, "r") as h5e:
                self._ext = self._take(h5e[self.split]["ext"])
        return self._ext

    @property
    def cds(self) -> np.ndarray:
        if getattr(self, "_cds", None) is None:
            from nm1.diffusion import CDS_CACHE
            with h5py.File(CDS_CACHE, "r") as h5d:
                self._cds = self._take(h5d[self.split]["cds"])
        return self._cds

    @property
    def cdsg(self) -> np.ndarray:
        """The flux-coordinate block (first-stage equilibrium, cds_geo.py)."""
        if getattr(self, "_cdsg", None) is None:
            from nm1.cds_geo import CDSG_CACHE
            with h5py.File(CDSG_CACHE, "r") as h5g:
                self._cdsg = self._take(h5g[self.split]["cds"])
        return self._cdsg


class ValInputs(CacheInputs):
    """Everything a member might read for the cache's ``val`` rows, loaded lazily."""

    def __init__(self):
        super().__init__("val")


@torch.inference_mode()
def predict_member(path: Path, inputs: CacheInputs,
                   device="cuda") -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n = len(inputs.shot)
    if payload.get("kind") == "seq":
        from nm1.seqmodel import load_seq, predict_seq
        model, payload = load_seq(path, device)
        coeff = np.zeros((n, len(payload["coeff_std"])))
        scal = np.zeros((n, 2))
        ext = inputs.ext if (int(payload.get("n_ext") or 0) or int(payload.get("n_ctx") or 0)) else None
        cds = inputs.cds if int(payload.get("n_cds") or 0) else None
        cdsg = inputs.cdsg if int(payload.get("n_cdsg") or 0) else None
        for s, e in zip(inputs.starts, inputs.ends):
            c, q = predict_seq(model, payload, inputs.base[s:e],
                               None if ext is None else ext[s:e], inputs.times_ms[s:e], device,
                               cds=None if cds is None else cds[s:e], cdsg=None if cdsg is None else cdsg[s:e])
            coeff[s:e], scal[s:e] = c, q
        return coeff, scal
    from nm1.model import load_member, member_inputs
    model, payload = load_member(path, device)
    coeff = np.zeros((n, len(payload["coeff_std"])))
    scal = np.zeros((n, 2))
    ext = inputs.ext if int(payload.get("n_ext") or 0) else None
    cds = inputs.cds if int(payload.get("n_cds") or 0) else None
    cdsg = inputs.cdsg if int(payload.get("n_cdsg") or 0) else None
    for s, e in zip(inputs.starts, inputs.ends):
        x = member_inputs(inputs.base[s:e], inputs.currents[s:e], payload, inputs.green,
                          None if ext is None else ext[s:e], None if cds is None else cds[s:e],
                          None if cdsg is None else cdsg[s:e])
        c, q = model(torch.from_numpy(x).to(device))
        coeff[s:e] = c.double().cpu().numpy() * payload["coeff_std"]
        scal[s:e] = q.double().cpu().numpy() * payload["scalar_std"] + payload["scalar_mean"]
    return coeff, scal


def cached(path: Path, inputs: ValInputs | None = None, device="cuda",
           refresh: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Member predictions on the val rows, from disk when the checkpoint has not changed."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    out = PRED_DIR / (path.stem + ".npz")
    stamp = path.stat().st_mtime
    if out.exists() and not refresh:
        z = np.load(out)
        if float(z["stamp"]) == stamp:
            return z["coeff"].astype(np.float64), z["scal"].astype(np.float64)
    inputs = inputs or ValInputs()
    coeff, scal = predict_member(path, inputs, device)
    np.savez(out, coeff=coeff.astype(np.float32), scal=scal.astype(np.float32), stamp=stamp)
    return coeff, scal


def main() -> None:
    """``python -m nm1.cachepred [--half selection] member.pt ...`` -- score each and the mean."""
    import argparse

    from nm1.cache import BASIS
    from nm1.fastscore import Reference, fmt

    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+", type=Path)
    ap.add_argument("--half", default="selection")
    ap.add_argument("--solo", action="store_true", help="also score every member alone")
    ap.add_argument("--weights", default="", help="comma-separated ensemble weights")
    a = ap.parse_args()
    inputs = ValInputs()
    preds = [cached(p, inputs) for p in a.members]
    state = torch.load(BASIS, map_location="cpu", weights_only=False)
    ref = Reference(a.half)
    if a.solo:
        for p, (c, s) in zip(a.members, preds):
            res = ref.score(c, s[:, 0], s[:, 1], state["pca_mean"], state["pca_components"])
            print(f"{p.stem}: {fmt(res)}", flush=True)
    w = (np.array([float(v) for v in a.weights.split(",")]) if a.weights
         else np.ones(len(preds)))
    w = w / w.sum()
    coeff = sum(wi * c for wi, (c, _) in zip(w, preds))
    scal = sum(wi * s for wi, (_, s) in zip(w, preds))
    res = ref.score(coeff, scal[:, 0], scal[:, 1], state["pca_mean"], state["pca_components"])
    print(f"MEAN of {len(preds)}: {fmt(res)}", flush=True)
    print(json.dumps(res), file=sys.stderr)


if __name__ == "__main__":
    main()