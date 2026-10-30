"""Size dependence of topology_zigzag_persistence (SIZE_INVARIANT is empty).

Width and scale: the kNN graphs depend only on neighbour order, so the
barcode is unchanged by zero-padding, an isometric embedding and a
power-of-two rescale of every cloud. Depth: an inserted identity block adds a
cloud, which moves both headline keys -- the reason no key is declared
size-invariant.
"""
from __future__ import annotations

import numpy as np
import pytest

import _family_d as fd

inv = fd.inv
pytest.importorskip("dionysus")

from blme.tasks.topology.zigzag import (  # noqa: E402
    ZigzagPersistenceTask, _bar_scalars, _effective_bars, _paper_clouds)

HEADLINES = ZigzagPersistenceTask.HEADLINE_KEYS


def _clouds():
    clouds, _, _ = _paper_clouds(fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.TEXTS, window=512)
    return clouds


@pytest.mark.parametrize("transform", ["zero_pad", "isometry", "rescale"])
def test_barcode_is_width_and_scale_free(transform):
    clouds = _clouds()
    base = _bar_scalars(*_effective_bars(clouds, 4))
    if transform == "zero_pad":
        other = [inv.zero_pad(c, 4) for c in clouds]
    elif transform == "isometry":
        other = [inv.embed_isometric(c, 3, seed=1) for c in clouds]
    else:
        other = [inv.rescale(c, 2.0 ** -20) for c in clouds]
    inv.assert_invariant(base, _bar_scalars(*_effective_bars(other, 4)), list(base), rtol=0.0)


def test_identity_block_insertion_moves_both_headlines():
    """Known negative: the reason SIZE_INVARIANT is empty."""
    assert ZigzagPersistenceTask.SIZE_INVARIANT == {}
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [1, 1, 2], verify_on=ids)
    cfg = {"num_samples": len(fd.TEXTS)}
    base = ZigzagPersistenceTask(cfg).evaluate(model, tok, fd.dataset())
    ins = ZigzagPersistenceTask(cfg).evaluate(deep, tok, fd.dataset())
    assert ins["_meta_zz_n_clouds"] == base["_meta_zz_n_clouds"] + 3
    for k in HEADLINES:
        inv.assert_moves(base, ins, k, min_rel=1e-3)


def test_last_token_cloud_reports_sink_shares():
    res = ZigzagPersistenceTask({"num_samples": len(fd.TEXTS)}).evaluate(
        fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.dataset())
    assert 0.0 <= res["_meta_massive_token_share"] <= 1.0
    assert res["_meta_position0_share"] == 0.0          # no prompt is one token long
    assert np.isfinite(res["zz_true_adjacent_mean"])
