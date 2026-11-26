"""Readout of the development-phase free-boundary solver (``nm1.mast``), not part of the submitted pipeline.

The submitted MAST file comes from ``nm1.mast_phys`` / ``mast_phys/``; ``nm1.mast`` is imported there only for its
time-smoothing helper, which does not call this readout.
"""


def readout(*args, **kwargs):
    raise NotImplementedError("development-phase MAST solver readout; the submitted MAST leg is nm1.mast_phys")
