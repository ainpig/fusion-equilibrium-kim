"""Check a submission .npz against the format contract, from the local parquets.

The starter kit's ``validate_submission.py`` does the same job by streaming the whole public-test
config from the Hub to read one column, which is ~16 GB of transfer for a structure check. The
parquets are already on disk here and are the same files, so this reads ``efit_times`` from them
and is the one that runs by default. The official script stays in the project and can be run
against the same file.

Checked, per shot, in Hub listing order:

    shot_XXXX_psirz    present, shape (T, 65, 65), floating, finite (DIII-D)
    shot_XXXX_q95      present, shape (T,), floating
    shot_XXXX_betaN    present, shape (T,), floating

plus: no shot missing, no unexpected key, and no array that looks decimal-rounded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from nm1 import OUTPUTS
from nm1.fetch import listing, local_path

CONFIGS = {"diii_d_public_test": "DIII-D", "mast_public_test": "MAST",
           "diii_d_private_test": "DIII-D", "mast_private_test": "MAST"}


def _rounding_warning(arr: np.ndarray, key: str) -> str | None:
    """Flag fixed-decimal rounding, which destroys the contour-derived terms.

    float16 keeps *relative* precision and is merely lossy; ``np.round(psi, 3)`` imposes a fixed
    absolute step, and the scorer's boundary extraction reads that step as real structure.
    """
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 64:
        return None
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return None
    diffs = np.diff(np.unique(a))
    diffs = diffs[diffs > 0]
    if diffs.size < 8:
        return None
    step = float(np.min(diffs))
    if step < scale * 1e-4:
        return None
    ratio = diffs / step
    if np.allclose(ratio, np.round(ratio), atol=1e-6):
        return (f"{key}: values sit on a fixed grid of {step:.3g} ({step / scale:.1e} of the "
                f"peak) -- this looks decimal-rounded, which costs D_LCFS and Consistency")
    return None


def validate(npz_path: Path, config: str, max_shots: int = 0, check_rounding: bool = True
             ) -> int:
    machine = CONFIGS[config]
    files = listing(config)
    if max_shots:
        files = files[:max_shots]
    print(f"validating {npz_path.name} ({npz_path.stat().st_size / 1e9:.2f} GB) against "
          f"{config}: {len(files)} shots, {machine} grid 65x65", flush=True)

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    total_frames = 0

    # An .npz is a zip of .npy members, and two processes writing one file produce duplicate
    # entries: it opens cleanly, every key resolves, and the arrays behind them come from two
    # different runs. Nothing else in the checks below would notice.
    import zipfile
    with zipfile.ZipFile(npz_path) as z:
        names = z.namelist()
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        errors.append(f"{len(duplicates)} key(s) stored more than once -- this file was written "
                      f"by more than one run and its contents are mixed, e.g. "
                      f"{sorted(duplicates)[:3]}")
    with np.load(npz_path, allow_pickle=False) as sub:
        present = set(sub.files)
        for i, meta in enumerate(files):
            path = local_path(meta)
            if not path.exists():
                errors.append(f"shot {i}: local parquet missing, cannot check its frame count")
                continue
            times = pq.read_table(path, columns=["efit_times"]).column(0)[0].as_py()
            n = len(times)
            total_frames += n
            for suffix, shape in (("psirz", (n, 65, 65)), ("q95", (n,)), ("betaN", (n,))):
                key = f"shot_{i:04d}_{suffix}"
                seen.add(key)
                if key not in present:
                    errors.append(f"{key}: MISSING (expected {shape})")
                    continue
                arr = sub[key]
                if arr.shape != shape:
                    errors.append(f"{key}: shape {arr.shape}, expected {shape}")
                if not np.issubdtype(arr.dtype, np.floating):
                    errors.append(f"{key}: dtype {arr.dtype}, expected a float type")
                if not np.isfinite(arr).all():
                    errors.append(f"{key}: contains NaN or Inf in {machine}")
                if check_rounding and suffix == "psirz" and i < 5:
                    message = _rounding_warning(arr, key)
                    if message:
                        warnings.append(message)
            if (i + 1) % 200 == 0 or i + 1 == len(files):
                print(f"  {i + 1}/{len(files)} shots, {total_frames:,} frames", flush=True)
        extra = sorted(present - seen)

    if extra:
        warnings.append(f"{len(extra)} key(s) the scorer ignores, e.g. {extra[:3]}")
    for line in warnings:
        print(f"WARNING: {line}")
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for line in errors[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} shots, {total_frames:,} frames, every key present with the right "
          f"shape and dtype")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="?",
                    default=OUTPUTS / "submission" / "diii_d_public_test.npz")
    ap.add_argument("--config", default="diii_d_public_test", choices=list(CONFIGS))
    ap.add_argument("--max-shots", type=int, default=0, help="0 = every shot")
    a = ap.parse_args()
    raise SystemExit(validate(a.npz, a.config, a.max_shots))


if __name__ == "__main__":
    main()
