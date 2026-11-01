"""SIZE_INVARIANT of geometry_dime, through the real evaluate().

Identity blocks after blocks 0 and 1 duplicate two hidden states, and the same
permutations serve every layer, so the profile gains exact copies and its peak
and last entry are unchanged. Zero-padding a Llama's width 32 -> 128 leaves
the N x N Grams unchanged to float rounding. Known negatives: the peak's depth
(argmax / L) moves with inserted blocks, and the release's covariance branch
and min(log N, log D) normalisation move with the width.
"""
from __future__ import annotations

import pytest

import _family_d as fd
from _nlpaug_side import require_nlpaug

inv = fd.inv

from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry.dime import DiMETask  # noqa: E402
from invariance.model_transforms import zero_pad_width  # noqa: E402

# 48 passages give N > D = 32, the release's covariance branch.
CFG = {"num_samples": 48, "max_length": 64}


def _run(model, cfg=CFG):
    return DiMETask(cfg).evaluate(model, fd.tokenizer(), load_default_corpus(cfg["num_samples"]))


@pytest.mark.parametrize("case", ["gpt2-identity", "llama-identity", "llama-width"])
def test_headline_invariant(case):
    require_nlpaug()
    arch, transform = case.split("-")
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    keys = sorted(DiMETask.SIZE_INVARIANT)
    base = _run(model)
    assert base["_meta_n_prompts"] > base["_meta_hidden_size"]
    if transform == "identity":
        out = _run(inv.insert_identity_blocks(model, [0, 1], verify_on=ids))
        assert len(out["dime_profile"]) == len(base["dime_profile"]) + 2
        assert set(out["dime_profile"]) == set(base["dime_profile"])
        inv.assert_invariant(base, out, keys, rtol=0.0)
    else:
        out = _run(zero_pad_width(model, 4, verify_on=ids, atol=1e-5))
        assert out["_meta_hidden_size"] == 4 * base["_meta_hidden_size"]
        inv.assert_invariant(base, out, keys, rtol=1e-5)
        # The release's branch and normalisation are not width-free.
        assert (base["_meta_dime_release_branch"], out["_meta_dime_release_branch"]) == (
            "covariance", "gram")
        inv.assert_moves(base, out, "_meta_max_entropy", 0.05)
        assert max(abs(x - y) for x, y in zip(base["_meta_dime_release_profile"],
                                              out["_meta_dime_release_profile"])) > 1e-3


def test_peak_depth_moves_under_identity_insertion():
    """Known negative: argmax / L is an index summary. tiny-random-gpt2 peaks at
    hidden state 3 of 5; three blocks after block 0 put the same value at 6 of 8."""
    require_nlpaug()
    model = fd.tiny_gpt2()
    base = _run(model)
    assert 0 < base["_meta_dime_peak_layer"] < len(base["dime_profile"]) - 1
    out = _run(inv.insert_identity_blocks(model, [0, 0, 0]))
    assert out["dime_peak"] == base["dime_peak"]
    assert out["dime_peak_depth"] != base["dime_peak_depth"]
