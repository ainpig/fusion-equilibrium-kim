"""Schema-aware access to intermediates created within the same audited run."""
import numpy as np
from my_experiments.reproduce_from_inputs_v25 import Run


def read_arrays(run, stage, index, keys):
    assert run.exists(stage, index)
    with np.load(run.path(stage, index)) as archive:
        assert set(archive.files) == set(keys), (stage, index, archive.files)
        return {key: archive[key] for key in keys}


class BranchRun:
    """Defer assembly until branches have been combined into a final prediction."""
    def __init__(self, run, keys=('psirz', 'q95', 'betaN')):
        self.run, self.keys = run, keys

    def __getattr__(self, name):
        return getattr(self.run, name)

    def read(self, stage, index):
        return read_arrays(self.run, stage, index, self.keys)

    def assemble(self, machine, stage, paths):
        return dict(machine=machine, stage=stage, shots=len(paths), assembly_deferred=True)


class FluxRun(Run):
    """A field-only diagnostic producer has no q95 or betaN arrays to read."""
    def read(self, stage, index):
        return read_arrays(self, stage, index, ('psirz',))
