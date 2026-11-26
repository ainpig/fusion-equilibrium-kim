"""Recover the complete shots of a prediction .npz whose writer died before closing (no central directory).

``NpzStreamWriter`` writes to a seekable file, so every finished member has a local header with its true sizes
(zip64 extra); the member being written when the process died is dropped.  A shot counts only when its three
members (psirz, q95, betaN) are all complete.  Used only to resume a public-fold prediction of the SAME manifest
after an out-of-memory kill (``nm1.predict --skip``); the record of which shots came from which run is written
next to the stitched file.

    python -m nm1.salvage scan <partial.npz>
    python -m nm1.salvage stitch <out.npz> <n_shots> <part.npz> [<part.npz> ...]   (later parts win)
"""
from __future__ import annotations

import io
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

SIG = b"PK\x03\x04"
KEY = re.compile(r"shot_(\d{4})_(psirz|q95|betaN)\.npy")


def members(path: Path):
    """Yield (name, raw_compressed_bytes, method) of every complete member of a (possibly truncated) zip."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        pos = 0
        while pos + 30 <= size:
            fh.seek(pos)
            hdr = fh.read(30)
            if hdr[:4] != SIG:
                return
            (_, flags, method, _, _, crc, csize, usize, nlen, xlen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
            name = fh.read(nlen).decode("utf-8")
            extra = fh.read(xlen)
            if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
                i = 0
                while i + 4 <= len(extra):
                    tag, ln = struct.unpack("<HH", extra[i:i + 4])
                    if tag == 1:
                        vals = struct.unpack("<" + "Q" * (ln // 8), extra[i + 4:i + 4 + ln])
                        k = 0
                        if usize == 0xFFFFFFFF:
                            usize = vals[k]; k += 1
                        if csize == 0xFFFFFFFF:
                            csize = vals[k]; k += 1
                    i += 4 + ln
            start = pos + 30 + nlen + xlen
            if flags & 0x08 or csize == 0 or start + csize > size:
                return                                  # unfinished member: stop here
            fh.seek(start)
            data = fh.read(csize)
            if (zlib.crc32(zlib.decompress(data, -15) if method == 8 else data) & 0xFFFFFFFF) != crc:
                return
            yield name, data, method
            pos = start + csize


def complete_shots(path: Path) -> set[int]:
    got: dict[int, set] = {}
    for name, _, _ in members(path):
        m = KEY.fullmatch(name)
        if m:
            got.setdefault(int(m.group(1)), set()).add(m.group(2))
    return {k for k, v in got.items() if len(v) == 3}


def main() -> None:
    if sys.argv[1] == "scan":
        shots = complete_shots(Path(sys.argv[2]))
        print(json.dumps({"complete_shots": len(shots), "indices": sorted(shots)}))
        return
    out, n = Path(sys.argv[2]), int(sys.argv[3])
    parts = [Path(p) for p in sys.argv[4:]]
    source = {}
    for p in parts:                                     # later parts win
        if p.exists():
            for k in complete_shots(p):
                source[k] = p
    missing = [k for k in range(n) if k not in source]
    if missing:
        raise SystemExit(f"{len(missing)} shots missing, first {missing[:10]}")
    from nm1.predict import NpzStreamWriter
    w = NpzStreamWriter(out)
    for p in parts:
        if not p.exists():
            continue
        for name, data, method in members(p):
            m = KEY.fullmatch(name)
            if m and source.get(int(m.group(1))) == p:
                raw = zlib.decompress(data, -15) if method == 8 else data
                w.write(name[:-4], np.load(io.BytesIO(raw), allow_pickle=False))
    w.close()
    rec = {str(p): sorted(k for k, s in source.items() if s == p) for p in parts}
    out.with_name(out.stem + "_parts.json").write_text(json.dumps(rec), encoding="utf-8")
    print(json.dumps({p: len(v) for p, v in rec.items()}))


if __name__ == "__main__":
    main()
