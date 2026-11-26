"""The first-stage equilibrium of the cache rows, computed in this process from checkpoints.

The flux-coordinate diffusion block (cds_geo.py) needs, for the held-out (`val`) rows, the flux
the first-stage ensemble predicts -- exactly what a test shot gets at inference. It is computed
here from the first-stage recipe's own members and weights (v9's first stage is the v8 flux
blend) and handed to the cache builder in memory, so a rebuild reads nothing but the official
parquets, the caches the pipeline builds and the checkpoints it trains -- no prediction file.

The training rows need a *predicted* equilibrium too. Built on the stored flux, the block hands
the second stage the true geometry during training and only a prediction at test time: scored
with true-geometry held-out features, two second-stage members reach S 0.99624/0.99638, and with
the first stage's own 0.99581/0.99599 -- worse than their twins without the block. So the
training shots get an out-of-fold first stage (``oof_coefficients``): members trained on every
training shot outside one fold (train.py --folds/--fold) predict that fold's shots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nm1 import ARTIFACTS, ROOT


def val_coefficients(version: str = "v8", key: str = "weights",
                     verbose: bool = True) -> np.ndarray:
    """``(val rows, n_components)`` float32: the ``key`` blend of recipes/<version>'s members.

    Each member's prediction is rounded to float32 before the float64 blend, the precision the
    member caches of the pipeline have always used, so the result is reproducible byte for byte.
    """
    import torch
    from nm1.cachepred import ValInputs, predict_member
    spec = _spec(version)
    w = np.asarray(spec[key], dtype=np.float64)
    if w.shape != (len(spec["members"]),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"recipes/{version}: {key} must be one weight per member")
    w = w / w.sum()
    inputs = ValInputs()
    total = None
    for rel, wi in zip(spec["members"], w):
        if wi == 0:
            continue
        c, _ = predict_member(ARTIFACTS / rel, inputs)
        c = c.astype(np.float32).astype(np.float64)
        total = wi * c if total is None else total + wi * c
        if verbose:
            print(f"  stage 1: {Path(rel).stem} (w={wi:.4f})", flush=True)
        torch.cuda.empty_cache()
    return total.astype(np.float32)


def _spec(manifest) -> dict:
    """A members.json given as a path, or the name of a recipe folder under recipes/."""
    path = Path(manifest)
    if not path.suffix:
        path = ROOT / "recipes" / str(manifest) / "members.json"
    return json.loads(path.read_text(encoding="utf-8"))


def oof_members(manifest) -> list[list[str]]:
    """The manifest's ``stage1_oof``: one list of member paths per fold."""
    return [list(f) for f in _spec(manifest)["stage1_oof"]]


def val_oof_coefficients(folds: list[list[str]], verbose: bool = True) -> np.ndarray:
    """``(val rows, n_components)`` float32: the held-out rows' flux, predicted by the equal-weight blend
    of every out-of-fold member (none of them saw a held-out shot). Precision as in ``val_coefficients``."""
    import torch
    from nm1.cachepred import ValInputs, predict_member
    inputs = ValInputs()
    members = [rel for fold in folds for rel in fold]
    total = None
    for rel in members:
        c, _ = predict_member(ARTIFACTS / rel, inputs)
        c = c.astype(np.float32).astype(np.float64)
        total = c if total is None else total + c
        if verbose:
            print(f"  out-of-fold stage 1 on the held-out rows: {Path(rel).stem}", flush=True)
        torch.cuda.empty_cache()
    return (total / len(members)).astype(np.float32)


def oof_coefficients(folds: list[list[str]], verbose: bool = True) -> np.ndarray:
    """``(train rows, n_components)`` float32: every training shot's flux, predicted by the
    equal-weight blend of the members that never saw it.

    ``folds[k]`` were trained with ``--folds len(folds) --fold k``, i.e. on the training shots
    whose cache index is not k modulo the number of folds; they predict exactly the shots that
    are. Precision as in ``val_coefficients``.
    """
    import h5py
    import torch
    from nm1.cache import CACHE
    from nm1.cachepred import CacheInputs, predict_member
    with h5py.File(CACHE, "r") as h5:
        shot = h5["train"]["shot"][:]
    n = len(folds)
    out = None
    for k, members in enumerate(folds):
        rows = (shot % n) == k
        inputs = CacheInputs("train", rows)
        total = None
        for rel in members:
            meta = torch.load(ARTIFACTS / rel, map_location="cpu", weights_only=False)
            if (int(meta.get("folds") or 0), int(meta.get("fold") or 0)) != (n, k):
                raise ValueError(f"{rel} was trained for fold {meta.get('fold')}/"
                                 f"{meta.get('folds')}, not {k}/{n}")
            del meta
            c, _ = predict_member(ARTIFACTS / rel, inputs)
            c = c.astype(np.float32).astype(np.float64)
            total = c if total is None else total + c
            if verbose:
                print(f"  out-of-fold stage 1, fold {k}/{n}: {Path(rel).stem} on "
                      f"{int(rows.sum()):,} rows", flush=True)
            torch.cuda.empty_cache()
        if out is None:
            out = np.empty((len(shot), total.shape[1]), dtype=np.float32)
        out[rows] = (total / len(members)).astype(np.float32)
        del inputs, total
    return out
