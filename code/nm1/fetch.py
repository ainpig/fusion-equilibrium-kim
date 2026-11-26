"""Fetch the challenge parquet splits from Hugging Face into ``data/``.

The Hub tree order is load-bearing: submissions key shots by position, so the recorded listing
for each config is written once and reused.  Downloads are atomic and resumable at file level --
a complete file of the expected size is never fetched twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
# Same rule as nm1/__init__.py, repeated rather than imported so this file also runs as a
# plain script before the package is on the path.
DATA_ROOT = Path(os.environ.get("NM1_DATA") or os.environ.get("FUSION_DATA")
                 or (Path(__file__).resolve().parents[1] / "work" / "data")).expanduser()
LISTINGS = Path(os.environ.get("NM1_LISTINGS") or (Path(os.environ.get("NM1_WORK") or (Path(__file__).resolve().parents[1] / "work")) / "listings")).expanduser()
CONFIGS = ("diii_d_train", "diii_d_public_test", "mast_public_test")
# Final-phase private folds: own dataset repository, pinned revision
PRIVATE_REPO_ID = os.environ.get("NM1_PRIVATE_REPO", "ganatma/fusion-equilibrium-challenge-final")
PRIVATE_REVISION = os.environ.get("NM1_PRIVATE_REVISION", "965040c12b814f33c8e7209f6b1799932247e793")
# The final phase swaps in blind private folds (inputs only), published under these names.
PRIVATE_CONFIGS = ("diii_d_private_test", "mast_private_test")
VAL_FRACTION = 0.12


def is_val(name: str, fraction: float = VAL_FRACTION) -> bool:
    """Shot-level train/validation split, by a hash of the file name alone.

    Byte-identical to the split the earlier caches used, so held-out shots stay held out and
    scores stay comparable across the whole project history.
    """
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"authorization": f"Bearer {token}"} if token else {}


def listing(config: str, refresh: bool = False, retries: int = 6) -> list[dict]:
    cache = LISTINGS / f"{config}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    repo, rev = (PRIVATE_REPO_ID, PRIVATE_REVISION) if config in PRIVATE_CONFIGS else (REPO_ID, "main")
    url0 = (f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/"
            f"data/{config}?recursive=false&expand=false&limit=1000")
    session = requests.Session()
    last: Exception | None = None
    for attempt in range(retries):
        files: list[dict] = []
        url = url0
        try:
            while url:
                r = session.get(url, headers=_headers(), timeout=60)
                r.raise_for_status()
                files.extend(x for x in r.json()
                             if x.get("type") == "file" and x["path"].endswith(".parquet"))
                m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("link", ""))
                url = m.group(1) if m else ""
        except Exception as exc:                       # noqa: BLE001 - retried below
            last = exc
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if not files:
            raise RuntimeError(f"no parquet files listed for {config}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(files), encoding="utf-8")
        return files
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raise RuntimeError(f"could not list {config}") from last


def local_path(meta: dict) -> Path:
    """``data/<config>/<shot>.parquet`` -- the repo's leading ``data/`` component is dropped."""
    return DATA_ROOT / Path(meta["path"]).relative_to("data")


def have(meta: dict) -> bool:
    p = local_path(meta)
    size = int(meta.get("size") or 0)
    return p.exists() and (not size or p.stat().st_size == size)


def fetch_one(meta: dict, attempts: int = 4) -> int:
    path = local_path(meta)
    if have(meta):
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{meta['path']}?download=true"
    tmp = path.with_suffix(path.suffix + f".part{os.getpid()}")
    expected = int(meta.get("size") or 0)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            tmp.unlink(missing_ok=True)
            with requests.get(url, headers=_headers(), stream=True, timeout=(20, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as out:
                    for block in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            out.write(block)
            got = tmp.stat().st_size
            if expected and got != expected:
                raise RuntimeError(f"partial download: expected {expected}, got {got}")
            tmp.replace(path)
            return got
        except Exception as exc:                       # noqa: BLE001 - retried below
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed for {meta['path']}: {last}")


def fetch(config: str, workers: int = 16, split: str = "all", limit: int = 0) -> None:
    files = listing(config)
    if split == "train":
        files = [f for f in files if not is_val(Path(f["path"]).name)]
    elif split == "val":
        files = [f for f in files if is_val(Path(f["path"]).name)]
    if limit:
        files = files[:limit]
    todo = [f for f in files if not have(f)]
    total = sum(int(f.get("size") or 0) for f in todo)
    print(f"{config}[{split}]: {len(files)} shots, {len(todo)} to fetch, "
          f"{total / 1e9:.1f} GB", flush=True)
    if not todo:
        return
    done = bytes_done = 0
    failed: list[str] = []
    started = time.perf_counter()
    # as_completed, not map: map yields in submission order, so one slow file hides the progress
    # of every file behind it. One file that will not download must also not abort the rest --
    # it is recorded and the run is repeated, which resumes on what is already on disk.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, meta): meta for meta in todo}
        for future in as_completed(futures):
            done += 1
            try:
                bytes_done += future.result()
            except Exception as exc:                   # noqa: BLE001 - reported at the end
                failed.append(f"{futures[future]['path']}: {exc}")
            if done % 25 == 0 or done == len(todo):
                el = time.perf_counter() - started
                rate = bytes_done / max(el, 1e-9)
                left = (total - bytes_done) / max(rate, 1e-9) / 60
                print(f"  {done}/{len(todo)} shots  {bytes_done / 1e9:.1f} GB  "
                      f"{rate / 1e6:.1f} MB/s  ~{left:.0f} min left", flush=True)
    if failed:
        print(f"{config}[{split}]: {len(failed)} file(s) did not download; re-run to retry",
              flush=True)
        for line in failed[:10]:
            print(f"    {line}", flush=True)
    print(f"{config}[{split}]: done in {(time.perf_counter() - started) / 60:.1f} min, "
          f"{len(todo) - len(failed)}/{len(todo)} fetched", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", default=list(CONFIGS),
                    choices=CONFIGS + PRIVATE_CONFIGS)
    ap.add_argument("--split", default="all", choices=("all", "train", "val"),
                    help="diii_d_train only: fetch just the training or just the held-out shots")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh-listing", action="store_true")
    a = ap.parse_args()
    for config in a.config:
        if a.refresh_listing:
            listing(config, refresh=True)
        fetch(config, a.workers, a.split if config == "diii_d_train" else "all", a.limit)


if __name__ == "__main__":
    sys.exit(main())
