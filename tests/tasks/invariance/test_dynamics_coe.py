"""dynamics_coe: release arithmetic, generation protocol and depth invariance."""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, token_batches,  # noqa: E402
                               tokenizer)

from blme.tasks.dynamics.coe import ChainOfEmbeddingTask, _coe_from_chain  # noqa: E402

_SCORE_PY = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser(
    "~/.cache/blme-refrepos")), "Chain-of-Embedding", "score.py")


def _release_score():
    if not os.path.exists(_SCORE_PY):
        pytest.skip(f"reference not available: {_SCORE_PY}")
    spec = importlib.util.spec_from_file_location("coe_release_score", _SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reference failed to load: {exc}")
    return mod


def test_no_floor_beyond_the_release():
    """DECISIONS item 1: the chain whose Z_Mag (sqrt 2) sits ~6 orders below the
    float32 resolution of its coordinates (1e13) is scored as score.py scores it,
    and the margin shows it. Pre-fix BLME's relative floor returned NaN."""
    chain = [torch.tensor(v, dtype=torch.float64)
             for v in ([1.0, 0.0], [1e13, 1e13], [0.0, 1.0])]
    out = _coe_from_chain(chain, storage_eps=float(np.finfo(np.float32).eps))
    assert math.isfinite(out["coe_r"]) and math.isfinite(out["coe_c"])
    assert out["coe_r"] > 1e12
    assert out["z_mag_margin_log10"] == pytest.approx(
        math.log10(math.sqrt(2) / (float(np.finfo(np.float32).eps) * 1e13)), rel=1e-9)
    assert out["z_mag_margin_log10"] < 0
    ref = _release_score().CoEScoreInfo([c.numpy() for c in chain])
    assert out["coe_r"] == pytest.approx(ref.compute_CoE_R(), rel=1e-9)
    assert out["coe_c"] == pytest.approx(ref.compute_CoE_C(), rel=1e-9)


def test_zero_normaliser_is_nan_as_in_the_release():
    """A chain that returns to its start: Z_Mag = 0 (score.py divides by zero)."""
    chain = [torch.tensor(v, dtype=torch.float64)
             for v in ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])]
    out = _coe_from_chain(chain)
    assert math.isnan(out["coe_r"]) and math.isnan(out["coe_c"])
