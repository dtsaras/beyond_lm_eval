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


# --------------------------------------------------------------------------
# Declarations and their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.weight_norms import WeightNormProfileTask  # noqa: E402

# Known negatives under zero-padding: forms that carry min(m, n) or sqrt(mn).
_ZERO_PAD_NEGATIVES = ("_meta_mean_stable_rank_normalized", "_meta_mean_rms_weight_scale")


@pytest.mark.parametrize("key", sorted(WeightNormProfileTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny, tiny_padded):
    transforms, node = WeightNormProfileTask.SIZE_INVARIANT[key]
    assert transforms == ("zero_pad",)
    assert node.endswith(f"test_size_invariant_entry[{key}]")
    base, pad = _run(tiny), _run(tiny_padded)
    inv.assert_invariant(base, pad, [key], rtol=1e-12)
    for neg in _ZERO_PAD_NEGATIVES:
        inv.assert_moves(base, pad, neg, 0.4)


def test_declarations():
    cls = WeightNormProfileTask
    assert cls.TOKEN_POLICY == "not_token_based"
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert set(cls.SIZE_INVARIANT) <= set(cls.HEADLINE_KEYS)
    assert 0 < cls.ADAPTATION_NOTE.count(". ") + 1 <= 8
    r = _run(load("hf-internal-testing/tiny-random-gpt2")[0])
    assert set(cls.HEADLINE_KEYS) <= set(r)
