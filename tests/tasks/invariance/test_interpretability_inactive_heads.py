"""interpretability_inactive_heads: an analytic known positive, the
identity-block known negative and width padding.

Known positive: zeroing a head's value weights makes its attention output, and
so its score, exactly 0 on every sequence; under GQA every query head reading
that value head goes silent with it.

Known negative (why SIZE_INVARIANT is empty): an inserted identity block's
heads write nothing (its W_O is zero), so by the paper's definition they are
inactive, and any count of inactive heads moves with depth. The score is read
before W_O and cannot see that: it rates the inserted heads as ordinary heads.
A leading identity copy of block 0 reads the same input as block 0, so its
per-layer share is block 0's exactly and the fraction becomes the mean over
L + 1 layers.

Width: zero-padding the residual stream adds no heads and leaves every score
unchanged.
"""
from __future__ import annotations

import math

import pytest
import torch

import _family_d as fd

inv = fd.inv

from blme.tasks.common import get_layers  # noqa: E402
from blme.tasks.interpretability import inactive_heads as IH  # noqa: E402

_TEXTS = [{"text": t} for t in fd.TEXTS]


def _run(model, tau):
    return IH.InactiveHeadsTask({"num_samples": len(_TEXTS), "tau": tau}).evaluate(
        model, fd.tokenizer(), _TEXTS)


def _scores(model):
    targets, _, _ = IH.attention_targets(model, get_layers(model))
    norms, _ = IH.collect_head_norms(model, fd.tokenizer(), fd.TEXTS, targets, 512)
    return [IH.honor_scores(a) for a in norms]


def _ids():
    tok = fd.tokenizer()
    return [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]


def _model(arch):
    return fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)


def _silence_value_head(model, layer, kv_head):
    """Zero one value head's weights (and bias) in place."""
    block = get_layers(model)[layer]
    with torch.no_grad():
        if model.config.model_type == "gpt2":
            d = model.config.n_embd
            hd = d // model.config.n_head
            cols = slice(2 * d + kv_head * hd, 2 * d + (kv_head + 1) * hd)
            block.attn.c_attn.weight[:, cols] = 0.0
            block.attn.c_attn.bias[cols] = 0.0
        else:
            hd = block.self_attn.head_dim
            block.self_attn.v_proj.weight[kv_head * hd:(kv_head + 1) * hd] = 0.0


@pytest.mark.parametrize("arch,layer,kv_head,silent", [("gpt2", 2, 1, [1]),
                                                        ("llama", 1, 0, [0, 1])])
def test_a_silenced_value_head_is_exactly_inactive(arch, layer, kv_head, silent):
    model = _model(arch)
    base, base_scores = _run(model, 1e-3), _scores(model)
    _silence_value_head(model, layer, kv_head)
    out, scores = _run(model, 1e-3), _scores(model)
    n_heads, n_layers = out["_meta_n_heads"], out["_meta_n_layers_scored"]
    assert base["inactive_head_fraction"] == 0.0
    assert torch.all(scores[layer][:, silent] == 0.0)
    # The layer's other heads keep their output norms: their scores rise by
    # the layer-mean ratio, and the blocks before are untouched.
    for l in range(layer):
        assert torch.equal(scores[l], base_scores[l])
    alive = [h for h in range(n_heads) if h not in silent]
    a0 = base_scores[layer]
    expected = a0[:, alive] * a0.sum(-1, keepdim=True) / a0[:, alive].sum(-1, keepdim=True)
    assert torch.allclose(scores[layer][:, alive], expected, rtol=1e-5)
    per_layer = [len(silent) / n_heads if l == layer else 0.0 for l in range(n_layers)]
    assert out["inactive_head_fraction_per_layer"] == per_layer
    assert out["inactive_head_fraction"] == pytest.approx(len(silent) / (n_heads * n_layers),
                                                          rel=1e-12)
    assert out["_meta_cdf_thresholds"][0] == 0.0
    # The same heads, and only they, at the default tau.
    assert _run(model, 0.5)["inactive_head_fraction_per_layer"] == per_layer


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_identity_blocks_move_the_fraction(arch):
    """Known negative: the inserted heads write nothing, yet are scored."""
    model = _model(arch)
    deep = inv.insert_identity_blocks(model, [-1], verify_on=_ids())
    base, ins = _run(model, 0.9), _run(deep, 0.9)
    L = base["_meta_n_layers_scored"]
    assert ins["_meta_n_layers_scored"] == L + 1
    b = base["inactive_head_fraction_per_layer"]
    assert ins["inactive_head_fraction_per_layer"] == [b[0]] + b
    assert ins["inactive_head_fraction"] == pytest.approx(math.fsum([b[0]] + b) / (L + 1),
                                                          rel=1e-12)
    # Silent by construction, not flagged as inactive.
    assert ins["inactive_head_fraction_per_layer"][0] < 1.0
    inv.assert_moves(base, ins, "inactive_head_fraction", min_rel=1e-3)


def test_zero_padding_the_width_adds_no_heads():
    model = fd.tiny_llama(n_layers=3)
    wide = inv.model_transforms.zero_pad_width(model, k=4, verify_on=_ids(), atol=1e-5)
    assert wide.config.hidden_size == 4 * model.config.hidden_size
    for a, b in zip(_scores(model), _scores(wide)):
        assert torch.allclose(a, b, rtol=1e-5, atol=0)
    for tau in (0.5, 0.9, 1.0):
        base, out = _run(model, tau), _run(wide, tau)
        assert out["_meta_n_heads"] == base["_meta_n_heads"]
        assert out["_meta_n_layers_scored"] == base["_meta_n_layers_scored"]
        assert out["inactive_head_fraction_per_layer"] == base["inactive_head_fraction_per_layer"]
        assert out["inactive_head_threshold_5pct"] == pytest.approx(
            base["inactive_head_threshold_5pct"], rel=1e-5)
