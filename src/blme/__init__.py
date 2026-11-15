from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("blme")
except PackageNotFoundError:
    __version__ = "0.2.0"

from .core import evaluate
from .runner import run_from_yaml
from .registry import register_task, list_tasks
