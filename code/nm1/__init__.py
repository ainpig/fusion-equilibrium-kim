"""New v1 (nm1): one code base that rebuilds the whole submission from the official data.

Where things live is decided here and nowhere else:

    NM1_DATA       official challenge parquets, ``<NM1_DATA>/<config>/<shot>.parquet``
                   (the Hugging Face repo's ``data/`` folder; ~100 GB, read-only)
    NM1_WORK       everything this code writes: caches, member weights, outputs (default ``<repo>/work``)
    NM1_ARTIFACTS  trained weights and fitted tables (default ``<NM1_WORK>/artifacts``). A second
                   training run on a different data split uses a second directory, never the first.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT

WORK = Path(os.environ.get("NM1_WORK") or (ROOT / "work")).expanduser()
DATA = Path(os.environ.get("NM1_DATA") or os.environ.get("FUSION_DATA") or (WORK / "data")).expanduser()
CACHE_DATA = Path(os.environ.get("NM1_CACHE_DATA") or (WORK / "cache")).expanduser()
ARTIFACTS = Path(os.environ.get("NM1_ARTIFACTS") or (WORK / "artifacts")).expanduser()
OUTPUTS = Path(os.environ.get("NM1_OUTPUTS") or (WORK / "outputs")).expanduser()

__all__ = ["ROOT", "PROJECT", "WORK", "DATA", "CACHE_DATA", "ARTIFACTS", "OUTPUTS"]
