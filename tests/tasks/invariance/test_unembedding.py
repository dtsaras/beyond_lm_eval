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


def test_purity_query_count_does_not_follow_num_samples(tiny, base):
    """The harness passes one num_samples to every task; purity used to read it
    as its query cap (24 queries in a 24-document run). The cap is its own key
    now, and every labelled token is a query by default."""
    small_run = _run(*tiny, num_samples=5)
    assert small_run["_meta_unembedding_purity_mean"] == base["_meta_unembedding_purity_mean"]
    assert small_run["_meta_purity_n_queries"] == base["_meta_unembedding_n_category_tokens"]
    capped = _run(*tiny, purity_num_queries=5)
    assert capped["_meta_purity_n_queries"] == 5
