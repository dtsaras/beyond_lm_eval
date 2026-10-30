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



# --------------------------------------------------------------------------
# Declarations and their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.superposition_strength import SuperpositionStrengthTask  # noqa: E402


@pytest.fixture(scope="module")
def padded_width(tiny):
    """An untied copy whose lm_head is W zero-padded to (V, 2m)."""
    import copy
    model, tok = tiny
    new = copy.deepcopy(model)
    W = model.get_output_embeddings().weight.detach()
    head = torch.nn.Linear(2 * W.shape[1], W.shape[0], bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, : W.shape[1]] = W
    new.lm_head = head
    return _run(new, tok)


@pytest.mark.parametrize("key", sorted(SuperpositionStrengthTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, base, padded_vocab, padded_width):
    """Each declared transform's direct test; known negatives: the every-row
    MSO under unused rows, the Welch ratio (kappa^2 ~ 1/m) under zero-padding."""
    transforms, node = SuperpositionStrengthTask.SIZE_INVARIANT[key]
    assert node.endswith(f"test_size_invariant_entry[{key}]")
    for t in transforms:
        other = {"V_unused_rows": padded_vocab, "zero_pad": padded_width}[t]
        inv.assert_invariant(base, other, [key], rtol=1e-6)
    inv.assert_moves(base, padded_vocab, "_meta_mean_squared_overlap_all_rows", 1e-5)
    inv.assert_moves(base, padded_width, "superposition_ratio", 0.4)


def test_declarations(base):
    cls = SuperpositionStrengthTask
    assert cls.TOKEN_POLICY == "not_token_based"
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert set(cls.SIZE_INVARIANT) <= set(cls.HEADLINE_KEYS) <= set(base)
