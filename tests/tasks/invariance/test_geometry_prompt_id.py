"""SIZE_INVARIANT of geometry_prompt_id, through the real evaluate().

Each prompt's cloud is pinned at max_length tokens. Identity blocks after
blocks 0 and 1 duplicate two hidden states, so the mean profile gains exact
copies and its peak and last entry are unchanged. Zero-padding a Llama's width
32 -> 128 leaves the token distances unchanged to float rounding, and the
residual gauge (x 2, eps x 4) scales every distance of a layer by 2 exactly
(the post-final-norm state is unchanged). Known negative: the peak's depth
(peak layer / L) moves with inserted blocks.
"""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.tasks.geometry.prompt_id import PromptIntrinsicDimTask  # noqa: E402
from invariance.model_transforms import zero_pad_width  # noqa: E402

CFG = {"num_samples": 4, "max_length": 96, "split": "validation"}


def _run(model):
    return PromptIntrinsicDimTask(CFG).evaluate(model, fd.tokenizer(), None)


@pytest.mark.parametrize("case", ["gpt2-identity", "llama-identity", "llama-width", "gpt2-rescale"])
def test_headline_invariant(case):
    arch, transform = case.split("-")
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    tok = fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    keys = sorted(PromptIntrinsicDimTask.SIZE_INVARIANT)
    base = _run(model)
    assert base["_meta_n_prompts"] == CFG["num_samples"]
    if transform == "identity":
        out = _run(inv.insert_identity_blocks(model, [0, 1], verify_on=ids))
        assert len(out["prompt_id_profile"]) == len(base["prompt_id_profile"]) + 2
        assert set(out["prompt_id_profile"]) == set(base["prompt_id_profile"])
        assert set(out["prompt_log_id_surprisal_pearson"]) == set(base["prompt_log_id_surprisal_pearson"])
        inv.assert_invariant(base, out, keys, rtol=0.0)
    elif transform == "width":
        out = _run(zero_pad_width(model, 4, verify_on=ids, atol=1e-5))
        inv.assert_invariant(base, out, keys, rtol=1e-5)
    else:
        out = _run(inv.residual_scale_gauge(model, 2.0, rescale_eps=True, verify_on=ids, atol=0.0))
        inv.assert_invariant(base, out, keys + ["_meta_mean_surprisal"], rtol=0.0)
        assert out["prompt_id_profile"] == base["prompt_id_profile"]


def test_peak_depth_moves_under_identity_insertion():
    """Known negative: peak layer / L is an index summary. Three blocks after
    block 0 keep the peak value and stretch the stack (tiny-random-gpt2 peaks
    at hidden state 1 of 5, depth 0.2 -> 0.125). A peak at the last layer
    would keep depth 1; the tiny Llama's does."""
    after = [0, 0, 0]
    model = fd.tiny_gpt2()
    base = _run(model)
    assert base["_meta_prompt_id_peak_layer"] < base["_meta_n_layers"]
    out = _run(inv.insert_identity_blocks(model, after))
    p = base["_meta_prompt_id_peak_layer"]
    assert out["_meta_n_layers"] == base["_meta_n_layers"] + len(after)
    assert out["prompt_id_peak"] == base["prompt_id_peak"]
    assert out["_meta_prompt_id_peak_layer"] == (p + len(after) if p > 1 else p)
    assert out["_meta_prompt_id_peak_depth"] != base["_meta_prompt_id_peak_depth"]
