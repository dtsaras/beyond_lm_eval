"""The recorded version is the imported source tree's, and pyproject.toml agrees with it."""
import re
from pathlib import Path

import blme
from blme import results
from blme._version import __version__


def test_pyproject_matches_source_version():
    root = Path(blme.__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == __version__


def test_envelope_reports_the_imported_source():
    assert blme.__version__ == __version__
    assert results._get_blme_version() == __version__
