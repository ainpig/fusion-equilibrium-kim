"""Up-down mirror symmetry of the DIII-D equilibrium problem, used as training-data augmentation.

The Grad-Shafranov problem is invariant under Z -> -Z: a plasma with the coil currents of the upper
set and the lower set exchanged has the mirrored flux map.  DIII-D's F-coils are built in
mirrored pairs (F1A/F1B ... F9A/F9B agree to a few mm, the Ohmic coil sits on the midplane), so
for the inputs and targets of one frame

    currents   swap F_kA <-> F_kB (ECOILA, bcoil and Ip stay)
    flux map   psi(R, Z) -> psi(R, -Z)  (a row reversal of the 65x65 grid, which is symmetric)

is a physically consistent second frame.  Upper-single-null plasmas are 8% of the training frames
and the extreme ones (zmax + zmin of the LCFS above +0.45 m) are 0.03%; those frames carry 13% of
the Consistency loss of the v10 ensemble on `evaluation` (73x the average frame).  Their mirror
images -- deep lower-single-null plasmas -- are 1.9% of the frames.

Only the base block is mirrored here.  Its Thomson entries are quantile summaries over channels
and do not depend on which side of the midplane a channel sits; the coil entries are exchanged
pairwise.  The features2 block reads the vertical Thomson chord on a fixed grid that only covers
the upper half and cannot be mirrored.
"""
from __future__ import annotations

import numpy as np

# columns of the 21 currents: ECOILA, bcoil, F1A, F1B, ..., F9A, F9B, Ip
PERM21 = np.arange(21)
for _a in range(2, 20, 2):
    PERM21[_a], PERM21[_a + 1] = _a + 1, _a


def base_perm(n_base: int = 140) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, sign)`` with ``mirrored[:, j] = sign[j] * frame[:, perm[j]]`` for the base block.

    Layout (features.d3d_magnetic_features): currents, their derivative, the 20 ms and the 100 ms
    causal change (four blocks of 21), then top+bottom coil sum, top-bottom, coil rms, bcoil/Ip;
    then 48 Thomson summaries and 4 clock terms, which do not change.
    """
    perm = np.arange(n_base)
    sign = np.ones(n_base)
    for block in range(4):
        perm[21 * block:21 * (block + 1)] = 21 * block + PERM21
    sign[85] = -1.0                       # top - bottom
    return perm, sign


def ext_perm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(perm, sign, fill)`` for the 298 columns of the features2 block.

    The coil-current windows (21 currents x 6 statistics) exchange F_kA <-> F_kB. The tangential
    chord is horizontal just below the midplane and is left as it is. The core block reads the
    vertical Thomson chord on a grid that only exists in the upper half, so a mirrored frame has
    nothing to put there: `fill` marks those columns, which take the value the block already
    uses for a chord that is missing (features2.UNSEEN).
    """
    from nm1.features2 import N_CORE, N_EXT
    perm = np.arange(N_EXT)
    for c in range(21):
        perm[6 * c:6 * c + 6] = 6 * PERM21[c] + np.arange(6)
    fill = np.zeros(N_EXT, dtype=bool)
    fill[126:126 + N_CORE] = True
    return perm, np.ones(N_EXT), fill


def mirror_stats(mean: np.ndarray, std: np.ndarray, perm: np.ndarray, sign: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Whitening statistics of the union of the frames and their mirror images.

    With them, a mirrored frame in whitened space is exactly the signed permutation of the real
    one: ``xw_mirror = sign * xw[:, perm]``.
    """
    m1, v1 = mean.astype(np.float64), std.astype(np.float64) ** 2
    m2, v2 = m1[perm] * sign, v1[perm]
    mu = 0.5 * (m1 + m2)
    var = 0.5 * (v1 + v2) + (0.5 * (m1 - m2)) ** 2
    return mu.astype(np.float32), np.maximum(np.sqrt(var), 1e-6).astype(np.float32)


def _flip(x: np.ndarray) -> np.ndarray:
    """Reverse the Z axis of flattened 65x65 maps."""
    return x.reshape(-1, 65, 65)[:, ::-1, :].reshape(len(x), -1)


def flux_flip_maps(pca_mean: np.ndarray, comps: np.ndarray, green: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients of the mirrored flux map: ``c' = A c + b + I[:20] @ E``.

    ``A`` and ``b`` project the flipped map back onto the basis (relative residual 4e-5 of the
    variance on training frames). ``E`` is the exact correction for the small asymmetry of the
    coil set: the vacuum flux of the exchanged currents minus the flipped vacuum flux of the
    original ones, from the Green functions (``green``: 20 x 4225, kA in, Wb/rad out).
    """
    V = comps.astype(np.float64)
    m = pca_mean.astype(np.float64)
    A = V @ _flip(V).T
    b = V @ (_flip(m[None])[0] - m)
    g = green.astype(np.float64)
    D = g[PERM21[:20]] - _flip(g)
    E = D @ V.T
    return A, b, E


def mirror_coefficients(coeff: np.ndarray, currents: np.ndarray, A: np.ndarray, b: np.ndarray,
                        E: np.ndarray | None, chunk: int = 200_000) -> np.ndarray:
    """Mirrored PCA coefficients for a block of frames (float32 in, float32 out)."""
    out = np.empty_like(coeff, dtype=np.float32)
    for s in range(0, len(coeff), chunk):
        c = coeff[s:s + chunk].astype(np.float64) @ A.T + b
        if E is not None:
            c += currents[s:s + chunk, :20].astype(np.float64) @ E
        out[s:s + chunk] = c
    return out
