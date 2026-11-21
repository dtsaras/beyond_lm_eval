# The version of the source tree actually imported (installed package metadata
# can be stale when the source is on PYTHONPATH, e.g. an exported tree).
from ._version import __version__

from .core import evaluate
from .runner import run_from_yaml
from .registry import register_task, list_tasks
from . import precision  # blme.precision.check: float32 against 16-bit next-token NLL
