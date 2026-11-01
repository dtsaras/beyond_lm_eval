"""SIZE_INVARIANT of causality_layer_skip: total and maximum output change.

An inserted identity block maps x to x exactly, so skipping it changes nothing
(output change 0) and skipping a real block gives the same output as in the
original model: the per-block output changes are the original ones with zeros
inserted, and their sum and maximum are unchanged. The depth scores index the
blocks, so prepending k identity blocks adds exactly k to them and appending
blocks leaves them unchanged while their ratio to L falls by L0 / L.
"""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.causality.layer_skip import LayerSkipEffectTask  # noqa: E402

_CFG = {"num_samples": 3, "max_length": 64}


def _run(model):
    texts = [{"text": d["text"]} for d in load_default_corpus(3)]
    return LayerSkipEffectTask(_CFG).evaluate(model, fd.tokenizer(), texts)


def _model(arch):
    return fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)


def _ids():
    tok = fd.tokenizer()
    return [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_output_effects_under_identity_blocks(arch):
    model = _model(arch)
    after = [0, 1, 1]
    deep = inv.insert_identity_blocks(model, after, verify_on=_ids())
    base, ins = _run(model), _run(deep)
    assert ins["_meta_n_layers"] == base["_meta_n_layers"] + len(after)
    inv.assert_invariant(base, ins, sorted(LayerSkipEffectTask.SIZE_INVARIANT), rtol=0.0)
    layout = inv.identity_layout(base["_meta_n_layers"], after)
    expected = [base["skip_output_effect_per_layer"][i] if kind == "orig" else 0.0
                for kind, i in layout]
    assert ins["skip_output_effect_per_layer"] == expected
    # Known negative: the depth-score ratios index the blocks.
    inv.assert_moves(base, ins, "skip_depth_score_output_ratio", min_rel=1e-3)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_depth_scores_shift_by_the_prepended_blocks(arch):
    model = _model(arch)
    base = _run(model)
    front = _run(inv.insert_identity_blocks(model, [-1, -1], verify_on=_ids()))
    assert front["skip_depth_score_output"] == pytest.approx(
        base["skip_depth_score_output"] + 2, rel=1e-12)
    # The importance is the release's float32 row sum, which now adds two
    # more zeros: equal to float32 rounding.
    assert front["skip_depth_score_future"] == pytest.approx(
        base["skip_depth_score_future"] + 2, rel=1e-6)
    # Appended blocks: the output score keeps its value and its ratio falls by L0 / L.
    n0 = base["_meta_n_layers"]
    tail = _run(inv.insert_identity_blocks(model, [n0 - 1, n0 - 1], verify_on=_ids()))
    assert tail["skip_depth_score_output"] == pytest.approx(base["skip_depth_score_output"],
                                                            rel=1e-12)
    assert tail["skip_depth_score_output_ratio"] == pytest.approx(
        base["skip_depth_score_output_ratio"] * n0 / (n0 + 2), rel=1e-12)
