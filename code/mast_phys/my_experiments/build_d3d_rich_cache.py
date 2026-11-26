"""The shot-level train/validation split, by a hash of the file name alone.

Only `is_val` survives here: the cache builder this module was named for belongs to the previous
project, and this one builds its own. The split itself is kept byte-identical so the priors are
fitted on exactly the shots they were fitted on before -- and on the same shots this project
trains its ensemble on.
"""
import hashlib


def is_val(name: str, fraction: float = 0.12) -> bool:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction
