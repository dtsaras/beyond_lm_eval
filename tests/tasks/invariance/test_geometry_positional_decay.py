"""SIZE_INVARIANT of geometry_positional_decay: the write-norm-weighted mean.

Each (layer, head, query) Spearman is weighted by ||o||, the norm of what that
head writes into the residual stream at that query. An inserted identity block
(zeroed output projections) writes nothing, so its rows get weight 0, and a
global residual rescale multiplies every weight by the same factor. The
uniform layer mean it replaced counts the identity block's attention rows.
"""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask  # noqa: E402

WEIGHTED = ("positional_decay_nosink_write_weighted",
            "_meta_positional_decay_inclusive_write_weighted")


def _run(model, n=3):
    return PositionalAttentionDecayTask({"num_samples": n}).evaluate(
        model, fd.tokenizer(), fd.dataset(n))


def _ids(n=3):
    tok = fd.tokenizer()
    return [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:n]]


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_weighted_mean_is_invariant_under_identity_blocks(arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)
    deep = inv.insert_identity_blocks(model, [0, 1, 1], verify_on=_ids())
    base, ins = _run(model), _run(deep)
    assert base["_meta_write_weights_complete"] and ins["_meta_write_weights_complete"]
    inv.assert_invariant(base, ins, list(WEIGHTED), rtol=0.0)
    # Known negative: the uniform layer means count the identity blocks' rows.
    inv.assert_moves(base, ins, "_meta_mean_positional_decay_correlation_nosink", min_rel=1e-3)
    inv.assert_moves(base, ins, "_meta_mean_positional_decay_correlation", min_rel=1e-3)
    # The same test, under the residual gauge: every write scales by c.
    for c in (4.0, 0.25):
        g = _run(inv.residual_scale_gauge(model, c, rescale_eps=True, verify_on=_ids()))
        inv.assert_invariant(base, g, list(WEIGHTED), rtol=0.0)


def test_declared_entries_are_the_weighted_headline():
    S = PositionalAttentionDecayTask.SIZE_INVARIANT
    assert set(S) == {"positional_decay_nosink_write_weighted"}
    assert S["positional_decay_nosink_write_weighted"][1].endswith(
        "test_weighted_mean_is_invariant_under_identity_blocks")
