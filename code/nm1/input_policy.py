"""Offline inference: official Parquets and NPZ files created by this process only."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import sys

_POLICY = None


class InputPolicy:
    def __init__(self, output=None):
        from nm1 import DATA
        self.output = Path(output).resolve() if output else None
        self.data = DATA.resolve()
        # each root resolved on its own: a data folder may hold junctions to the per-config downloads
        self.allowed_data = tuple((self.data / name).resolve() for name in
                                  ("diii_d_train", "diii_d_public_test", "diii_d_private_test",
                                   "mast_public_test", "mast_private_test"))
        self.created = set()
        self.resume_dir = Path(os.environ["NM1_RESUME_DIR"]).resolve() if os.environ.get("NM1_RESUME_DIR") else None
        self.report = {"version": "v3-1", "old_npz_reads": 0, "created_npz_reads": 0,
                       "network_attempts": 0, "denied": [], "parquet_inputs": []}

    def deny(self, path, reason):
        self.report["denied"].append({"path": str(path), "reason": reason})
        raise PermissionError(reason + ": " + str(path))

    @staticmethod
    def is_npz(path):
        return ".npz" in path.name.lower()

    def audit(self, event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            self.report["network_attempts"] += 1
            raise PermissionError("v3-1 prediction is offline")
        if event == "os.rename":
            source, target = (Path(os.fsdecode(p)).resolve() for p in args[:2])
            if self.is_npz(source) or self.is_npz(target):
                old, new = str(source).casefold(), str(target).casefold()
                if old not in self.created or self.output is None or not target.is_relative_to(self.output):
                    self.deny(target, "Only newly generated NPZ files may be renamed within this run")
                if target.exists() and new not in self.created:
                    self.deny(target, "Refusing to replace an older NPZ")
                self.created.add(new)
            return
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                   (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR))))
        if any(p.casefold() in ("fairmast", "fair_mast", "data_probe", "mast_train") for p in path.parts):
            self.deny(path, "External or labelled MAST archive access is disabled")
        if path.suffix.lower() == ".parquet":
            if writing or not any(path.is_relative_to(root) for root in self.allowed_data):
                self.deny(path, "Only official challenge Parquets may be read")
            key = str(path)
            if key not in self.report["parquet_inputs"]:
                self.report["parquet_inputs"].append(key)
        if not self.is_npz(path):
            return
        key = str(path).casefold()
        if writing:
            if self.output is None or not path.is_relative_to(self.output):
                self.deny(path, "NPZ outputs must stay inside the fresh run directory")
            if path.exists() and key not in self.created:
                self.deny(path, "Refusing to overwrite an older NPZ")
            self.created.add(key)
        elif key in self.created:
            self.report["created_npz_reads"] += 1
        elif self.resume_dir is not None and path.is_relative_to(self.resume_dir):
            # NM1_RESUME_DIR: the intermediate files of this same run, written by an earlier process of it (the MAST
            # runtime checks run id and SHA256 of each one before use); counted separately in the report
            self.report["resume_npz_reads"] = self.report.get("resume_npz_reads", 0) + 1
        else:
            self.report["old_npz_reads"] += 1
            self.deny(path, "Pre-existing NPZ inputs, masks and predictions are forbidden")

    def save(self):
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / f"input_policy_{os.getpid()}.json").write_text(
                json.dumps(self.report, indent=2), encoding="utf-8")


def install(output=None):
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    output = output or os.environ.get("FUSION_RUN_ROOT")
    _POLICY = InputPolicy(output)
    sys.addaudithook(_POLICY.audit)
    atexit.register(_POLICY.save)
    return _POLICY
