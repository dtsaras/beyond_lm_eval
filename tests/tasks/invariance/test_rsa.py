"""geometry_rsa: token policy, layer trimming, angular headline and depth
invariance, driven through the real ``evaluate()``."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load, scale_position0

TASK = "geometry_rsa"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _headline(r):
    return [k for k in r if not k.startswith("_meta_") and isinstance(r[k], float)]


def test_headline_ignores_position0_rows(tiny_states):
    """Position 0 is excluded before the evenly spaced row selection: a
    synthetic sink (position-0 rows x1e3) moves no headline key."""
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    keys = _headline(base)
    assert keys
    inv.assert_invariant(base, sunk, keys, rtol=1e-9)
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]
    assert base["_meta_rows_policy"] == "exclude_pos0"


@pytest.mark.parametrize("attrs", [{}, {"num_hidden_layers": None, "n_layer": None}])
def test_legacy_cache_trims_only_the_post_norm_entry(attrs):
    """Without the deepest-block capture exactly the post-norm entry is dropped,
    whatever the config says (a depth-less config used to leave 2 layers)."""
    from blme.tasks.geometry.rsa import RepresentationalSimilarityTask
    rng = np.random.default_rng(0)
    L = 6
    S = inv.States(blocks={li: [rng.standard_normal((40, 8)) + li for _ in range(3)]
                           for li in range(L)})
    m = inv.StubModel(L, 8, attrs)
    cache = inv.InjectedStateCache(S, m, inv.StubTokenizer())
    cache._last_block_raw = None          # a cache built before the hook existed
    r = RepresentationalSimilarityTask({"num_samples": 3}).evaluate(m, None, None, cache=cache)
    assert r["_meta_n_layers"] == L - 1
    assert r["_meta_last_entry_is_block_output"] is False
