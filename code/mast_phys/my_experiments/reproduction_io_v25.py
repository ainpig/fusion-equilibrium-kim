"""Read isolation and provenance for input-to-prediction reproduction.

The runtime is a separate directory containing only source, fixed settings and
trained weights. Existing submission/experiment files are never whitelisted.
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

POLICY = None


class NpzStreamWriter:
    """Writes one .npy member at a time into an .npz zip stream, so a whole fold never has to fit in RAM."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.zip = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                                   allowZip64=True)

    def write(self, key: str, value: np.ndarray) -> None:
        # np.lib.format.write_array makes a standard .npy member; a ZIP of such members is a valid NPZ.
        with self.zip.open(key + ".npy", "w", force_zip64=True) as stream:
            np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)

    def close(self) -> None:
        self.zip.close()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def replace_retry(src, dst, attempts=20, wait=0.5):
    """Path.replace that waits out a transient Windows sharing violation.

    On this machine a security agent briefly holds freshly written files, and a rename that
    races it fails with WinError 5 even though nothing is wrong with the file. The rename is retried; the content is never touched.
    """
    import time
    src, dst = Path(src), Path(dst)
    for attempt in range(attempts):
        try:
            return src.replace(dst)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    replace_retry(tmp, path)


class ReadPolicy:
    def __init__(self, specification):
        self.spec = specification
        self.runtime = Path(specification['runtime']).resolve()
        self.output = Path(specification['output']).resolve()
        self.forbidden_root = Path(specification['source_workspace']).resolve()
        self.assets = {str((self.runtime/p).resolve()).casefold() for p in specification['assets']}
        self.inputs = {str(Path(p).resolve()).casefold() for p in specification['inputs']}
        self.metadata = {str(Path(p).resolve()).casefold() for p in specification['metadata']}
        self.python_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
        self.reads = set()
        self.denied = []
        self.network_attempts = []

    def audit(self, event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            self.network_attempts.append(event)
            raise PermissionError('Offline reproduction forbids network access')
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        name = str(path).casefold()
        mode, flags = args[1], args[2]
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if path.is_relative_to(self.output):
            return
        if writing and (name in self.inputs or name in self.assets):
            self.denied.append(str(path))
            raise PermissionError('Input and model assets are read-only')
        if any(path.is_relative_to(p) for p in self.python_roots):
            return
        # Python probes a missing __pycache__ file before loading the manifest's
        # source. Permit the missing-file probe, not an unverified bytecode file.
        if path.is_relative_to(self.runtime) and path.suffix == '.pyc' and not path.exists():
            return
        restricted = path.is_relative_to(self.runtime) or path.is_relative_to(self.forbidden_root)
        if restricted and name not in self.assets and name not in self.inputs and name not in self.metadata:
            self.denied.append(str(path))
            raise PermissionError('Read outside the reproduction asset/input manifest: '+str(path))
        if name in self.assets or name in self.inputs:
            self.reads.add(str(path))

    def save(self):
        atomic_json(self.output/'audit'/f'process_{os.getpid()}.json', dict(
            process=os.getpid(), reads=sorted(self.reads), forbidden_reads=self.denied,
            network_attempts=self.network_attempts,
            existing_prediction_cache_allowed=False, network_allowed=False))


def install(specification):
    global POLICY
    POLICY = ReadPolicy(specification)
    sys.addaudithook(POLICY.audit)
    return POLICY


def worker_init(specification):
    import torch
    import scipy.linalg  # Load its BLAS library before per-stage thread limits.
    torch.set_num_threads(1)
    install(specification)
    from my_experiments.kinetic_equilibrium import read_inputs
    from my_experiments.released_geometry import configure
    configure(read_inputs(specification['inputs'][0]))


def worker_audit():
    from nm1.input_policy import install as input_boundary
    boundary = input_boundary()
    boundary.save()
    if boundary.report['denied'] or boundary.report['network_attempts']:
        raise PermissionError('Worker violated the v3-1 input policy')
    if POLICY is not None:
        POLICY.save()
        if POLICY.denied or POLICY.network_attempts:
            raise PermissionError('Worker attempted access outside the reproduction manifest')
