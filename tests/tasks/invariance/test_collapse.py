"""geometry_collapse: layer trimming, token policies, key hygiene and size
invariance, driven through the real ``evaluate()`` on injected hidden states."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load

TASK = "geometry_collapse"


def _rank_profile_states(ranks, n_docs=3, T=40, d=8, seed=0):
    """Block l holds documents whose rows span ``ranks[l]`` dimensions."""
    rng = np.random.default_rng(seed)
    blocks = {}
    for li, r in enumerate(ranks):
        basis = rng.standard_normal((r, d))
        blocks[li] = [rng.standard_normal((T, r)) @ basis for _ in range(n_docs)]
    return inv.States(blocks=blocks)


def test_hookless_collection_without_depth_config_keeps_the_profile():
    """A cache without the deepest-block capture and a config without
    num_hidden_layers / n_layer: the drop scan must trim exactly the post-norm
    entry. It used to trim to collected_layer_count(model) = 2 layers, so the
    collapse at block 4 was never seen."""
    from blme.tasks.geometry.collapse import RepresentationCollapseTask
    S = _rank_profile_states([8, 8, 8, 8, 2, 8])
    L = S.n_layers
    m = inv.StubModel(L, 8, {"num_hidden_layers": None, "n_layer": None})
    cache = inv.InjectedStateCache(S, m, inv.StubTokenizer())
    cache._last_block_raw = None          # a cache built before the hook existed
    r = RepresentationCollapseTask({"num_samples": 3}).evaluate(m, None, None, cache=cache)
    assert r["_meta_last_entry_is_block_output"] is False
    assert r["_meta_max_drop_layer"] == 4
    assert r["_meta_max_drop_n_layers"] == L - 1
