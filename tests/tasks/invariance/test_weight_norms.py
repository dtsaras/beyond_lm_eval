"""geometry_weight_norms: the headline stable rank is the zero-padding-invariant
raw one; sr / min(m, n) is a capacity ratio kept as _meta_."""
from __future__ import annotations

import pytest

from _family_c import inv, load

TASK = "geometry_weight_norms"


def _run(model):
    from blme.tasks.geometry.weight_norms import WeightNormProfileTask
    return WeightNormProfileTask({}).evaluate(model, None, None)


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")[0]


@pytest.fixture(scope="module")
def tiny_padded(tiny):
    return inv.zero_pad_weight_matrices(tiny, 2)


def test_headline_stable_rank_is_the_raw_one(tiny, tiny_padded):
    """Zero-padding every block matrix to (2m, 2n) leaves the operator, and the
    raw stable rank, unchanged; sr / min(m, n) halves exactly, so it is not the
    headline (normalization.py records the same /d twin as refuted)."""
    base, pad = _run(tiny), _run(tiny_padded)
    assert "mean_stable_rank_normalized" not in base
    inv.assert_invariant(base, pad, ["mean_stable_rank"], rtol=1e-12)
    assert pad["_meta_mean_stable_rank_normalized"] == pytest.approx(
        0.5 * base["_meta_mean_stable_rank_normalized"], rel=1e-12)
