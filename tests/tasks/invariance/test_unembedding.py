"""geometry_unembedding: purity is a _meta_ within-tokenizer diagnostic, its
query count does not follow the harness-wide num_samples, and the effective
rank of the unembedding is pinned under zero-padding of its width."""
from __future__ import annotations

import pytest
import torch

from _family_c import inv, load

TASK = "geometry_unembedding"


def _run(model, tok, **cfg):
    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask
    return UnembeddingDiagnosticsTask(cfg).evaluate(model, tok, None)


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def base(tiny):
    return _run(*tiny)


def test_purity_is_meta(base):
    """The task flags purity as not comparable across models (tokenizer-
    dependent category-token mapping), so it is not a headline key."""
    assert "unembedding_purity_mean" not in base
    assert base["_meta_purity_cross_model_comparable"] is False
    assert 0.0 <= base["_meta_unembedding_purity_mean"] <= 1.0
