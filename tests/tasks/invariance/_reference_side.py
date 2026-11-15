"""Reference packages that BLME's tests compare against but BLME does not depend on.

When a package is not installed it is taken from a side install under
``~/.cache/blme-sideinstalls/`` (BLME_SIDEINSTALLS overrides the root), made
without touching the venv:

* ``isoscore-2.0.1``: ``uv pip install --no-deps --target`` of IsoScore 2.0.1
  from PyPI (its N < d padding branch is what the tests pin; the GitHub
  clone p_eickhoff/isoscore@1edad0b takes (dims, points) and has none);
* ``skhubness-c36a058``: ``uv pip install --no-deps --target`` of the local
  clone of VarIr/scikit-hubness@c36a058;
* ``torchmetrics-1.9.0``: ``uv pip install --no-deps --target`` of
  torchmetrics 1.9.0 and lightning-utilities.

Skips when absent and fails under BLME_REQUIRE_REFS=1.
"""
import importlib
import os
import sys

import pytest

SIDE_INSTALLS = {"IsoScore": "isoscore-2.0.1", "skhubness": "skhubness-c36a058",
                 "torchmetrics": "torchmetrics-1.9.0"}


def require(module: str):
    """Import ``module`` (installed, else from its side install) and return it."""
    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    root = os.environ.get("BLME_SIDEINSTALLS", os.path.expanduser("~/.cache/blme-sideinstalls"))
    path = os.path.join(root, SIDE_INSTALLS[top])
    if not os.path.isdir(path):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {path}")
        pytest.skip(f"reference not available: {path}")
    if path not in sys.path:
        sys.path.append(path)
    return importlib.import_module(module)
