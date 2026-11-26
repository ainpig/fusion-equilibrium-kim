"""Retry worker-process creation on Windows.

Process pools on Windows occasionally fail to spawn a worker under memory pressure ("The paging file is too
small", WinError 1455). The failure happens before any work is sent, so retrying the spawn is safe and changes
no result.
"""
from __future__ import annotations

import time

_INSTALLED = False


def install(retries: int = 10, wait: float = 3.0) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import multiprocessing.popen_spawn_win32 as pw

    original = pw.Popen.__init__

    def patched(self, process_obj):
        for attempt in range(retries):
            try:
                return original(self, process_obj)
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(wait * (attempt + 1))

    try:
        pw.Popen.__init__ = patched
    except Exception:   # noqa: BLE001 - non-Windows: nothing to patch
        return
    _INSTALLED = True
