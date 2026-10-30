"""geometry_superposition_strength: only rows the tokenizer can emit enter the
packing statistics; MSO is pinned under zero-padding of the width and under
unused vocabulary rows."""
from __future__ import annotations

import pytest
import torch

from _family_c import inv, load

TASK = "geometry_superposition_strength"
KEYS = ("mean_squared_overlap", "mean_squared_overlap_centered", "superposition_ratio",
        "superposition_ratio_centered", "row_norm_cv", "frac_norm_above_half_max")


def _run(model, tok):
    from blme.tasks.geometry.superposition_strength import SuperpositionStrengthTask
    return SuperpositionStrengthTask({}).evaluate(model, tok, None)


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def base(tiny):
    return _run(*tiny)


@pytest.fixture(scope="module")
def padded_vocab(tiny):
    """64 never-emitted small random rows appended (the tokenizer is unchanged)."""
    model, tok = tiny
    return _run(inv.pad_vocabulary(model, 64, kind="small_random"), tok)


def test_unused_rows_do_not_enter(base, padded_vocab):
    """Rows past len(tokenizer) are not tokens: appending 64 of them moves no
    headline key, while the released code's every-row MSO moves."""
    inv.assert_invariant(base, padded_vocab, list(KEYS), rtol=1e-6)
    assert padded_vocab["_meta_n_unused_rows"] == 64
    assert base["_meta_n_unused_rows"] == 0
    inv.assert_moves(base, padded_vocab, "_meta_mean_squared_overlap_all_rows", 1e-5)
