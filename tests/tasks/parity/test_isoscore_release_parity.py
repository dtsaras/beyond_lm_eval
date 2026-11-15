"""geometry_isoscore vs the released IsoScore.py, through evaluate().

The IsoScore pip package is not a BLME dependency (the live check in
test_reference_parity_formulas skips without it), so the release file
bcbi-edu/p_eickhoff_isoscore IsoScore/IsoScore.py is loaded from the clone.
evaluate() reads the post-final-norm entry with every token kept (the paper's
protocol); isoscore_effective_dims is iota (d - 1) + 1. Skips without the
clone (fails under BLME_REQUIRE_REFS=1).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

_ISO = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "p_eickhoff_isoscore" / "IsoScore" / "IsoScore.py"
D = 6


@pytest.fixture(scope="module")
def release():
    if not _ISO.is_file():
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_ISO}")
        pytest.skip(f"reference not available: {_ISO}")
    spec = importlib.util.spec_from_file_location("isoscore_release", _ISO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IsoScore


@pytest.mark.parametrize("seed", [0, 1])
def test_isoscore_keys_match_the_release(release, seed):
    g = np.random.default_rng(seed)
    blocks = {li: [g.standard_normal((12, D)) for _ in range(30)] for li in range(3)}
    scale = np.linspace(0.5, 3.0, D)
    final = [(g.standard_normal((12, D)) * scale + 0.3).astype(np.float32) for _ in range(30)]
    res = inv.run_task_on_states("geometry_isoscore", blocks, final_norm=final)
    X = np.concatenate([np.asarray(x, np.float64) for x in final])
    iota = float(release(X.T))            # the release file takes (dims, points)
    assert 0.05 < iota < 0.95
    assert res["isoscore"] == pytest.approx(iota, rel=1e-6)
    assert res["isoscore_effective_dims"] == pytest.approx(iota * (D - 1) + 1, rel=1e-6)
