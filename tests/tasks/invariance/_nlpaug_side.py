"""nlpaug for the geometry_dime tests.

nlpaug is not a BLME dependency. When it is not installed it is taken from
side installs made with ``uv pip install --no-deps --target
~/.cache/blme-sideinstalls/<pkg>-<version>``: nlpaug 1.1.11 (the release's
pin) and the modules it imports, gdown 5.2.0, beautifulsoup4 4.12.3 and
soupsieve 2.6 (BLME_SIDEINSTALLS overrides the root). Skips when absent and
fails under BLME_REQUIRE_REFS=1.
"""
import importlib
import os
import sys

import pytest

SIDE_INSTALLS = ("nlpaug-1.1.11", "gdown-5.2.0", "beautifulsoup4-4.12.3", "soupsieve-2.6")
_MODULES = ("nlpaug", "nlpaug.augmenter.char", "nlpaug.augmenter.word",
            "nlpaug.augmenter.sentence", "nlpaug.flow")


def missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def require_nlpaug():
    """Import nlpaug (installed, else side-installed) and return the module."""
    try:
        for m in _MODULES:
            importlib.import_module(m)
        return sys.modules["nlpaug"]
    except ImportError:
        pass
    root = os.environ.get("BLME_SIDEINSTALLS", os.path.expanduser("~/.cache/blme-sideinstalls"))
    dirs = [os.path.join(root, d) for d in SIDE_INSTALLS]
    absent = [d for d in dirs if not os.path.isdir(d)]
    if absent:
        missing(", ".join(absent))
    # Appended, not prepended: the nlpaug wheel also ships a top-level `test` package.
    added = [d for d in dirs if d not in sys.path]
    sys.path.extend(added)
    importlib.invalidate_caches()
    try:
        for m in _MODULES:
            importlib.import_module(m)
    finally:
        for d in added:
            sys.path.remove(d)
    return sys.modules["nlpaug"]
