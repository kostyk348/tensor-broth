"""tensor-broth — computing without a program counter.

Two engines live here:

* :mod:`tensor_broth.von_neumann` — a classic fetch-decode-execute VM.
* :mod:`tensor_broth.broth`       — a 3D spatial dataflow engine where data
  transforms itself through local interactions (3D convolutions), with no
  instruction pointer at all.
"""

from .von_neumann import run_von_neumann_vm
from .broth import SpatialEngine, SpatialEngineNumPy, make_engine

__version__ = "0.1.0"

__all__ = [
    "run_von_neumann_vm",
    "SpatialEngine",
    "SpatialEngineNumPy",
    "make_engine",
    "__version__",
]
