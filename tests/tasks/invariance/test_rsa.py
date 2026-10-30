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
    assert base["_meta_rows_policy"] == "exclude_sinks"


@pytest.fixture(scope="module")
def gpt2_states():
    return inv.states_from_model(*load("gpt2"), TEXTS[:6])


def test_massive_row_at_one_middle_depth_is_dropped_at_every_layer(gpt2_states, monkeypatch):
    """A first-period-style sink at block L/2 only: row 3 of every document
    gains 1000 x its norm on one coordinate (a massive activation; a pure
    rescale is invisible to correlation distance). It moves exclude_pos0's
    adjacent minimum; exclude_sinks drops that token from every RDM, giving the
    row-deleted corpus's value bit for bit, and with no massive row it equals
    exclude_pos0 exactly."""
    from _sinks import check_state_task
    from blme.tasks.geometry.rsa import RepresentationalSimilarityTask
    check_state_task(TASK, RepresentationalSimilarityTask, gpt2_states, monkeypatch,
                     "rsa_adjacent_min_corrdist", spike_dim=0,
                     keys=list(RepresentationalSimilarityTask.HEADLINE_KEYS))


def test_headline_is_angular_not_norm_driven(tiny_states):
    """RDMs are angular (Kriegeskorte's correlation distance): rescaling every
    token's vector by its own positive factor (the same in every layer) leaves
    each RDM's ranks -- and every headline -- unchanged, while the Euclidean
    RDM, now _meta_, moves (known negative)."""
    rng = np.random.default_rng(0)
    scales = [rng.uniform(0.2, 5.0, size=(n, 1)) for n in tiny_states.lengths]
    S = tiny_states
    scaled = inv.States(
        blocks={li: [np.asarray(x, dtype=np.float64) * c for x, c in zip(S.blocks[li], scales)]
                for li in S.blocks},
        final_norm=None if S.final_norm is None else
        [np.asarray(x, dtype=np.float64) * c for x, c in zip(S.final_norm, scales)])
    import torch
    base = inv.run_task_on_states(TASK, S, store_dtype=torch.float64)
    sc = inv.run_task_on_states(TASK, scaled, store_dtype=torch.float64)
    keys = _headline(base)
    assert keys and all(k.endswith("_corrdist") for k in keys)
    inv.assert_invariant(base, sc, keys, rtol=1e-9)
    inv.assert_moves(base, sc, "_meta_rsa_adjacent_mean_euclidean", 1e-3)


@pytest.fixture(scope="module")
def tiny_identity_states(tiny):
    """tiny-random-gpt2 with exact identity blocks after blocks 1 and 3."""
    model, tok = tiny
    return inv.states_from_model(inv.insert_identity_blocks(model, [1, 3]), tok, TEXTS)


def test_headline_is_depth_invariant_under_identity_insertion(
        tiny_states, tiny_identity_states):
    """P4: identity blocks move no headline key; the adjacent mean (now
    _meta_) does (known negative)."""
    base = inv.run_task_on_states(TASK, tiny_states)
    ins = inv.run_task_on_states(TASK, tiny_identity_states)
    assert ins["_meta_n_layers"] == base["_meta_n_layers"] + 2
    keys = _headline(base)
    assert set(keys) == {"rsa_adjacent_min_corrdist", "rsa_early_late_corrdist"}
    inv.assert_invariant(base, ins, keys, rtol=1e-9)
    inv.assert_moves(base, ins, "_meta_rsa_adjacent_mean_corrdist", 1e-3)


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


# --------------------------------------------------------------------------
# SIZE_INVARIANT declarations, checked against their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.rsa import RepresentationalSimilarityTask  # noqa: E402


@pytest.mark.parametrize("key", sorted(RepresentationalSimilarityTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny_states, tiny_identity_states):
    from _family_c import check_size_invariant_entry
    check_size_invariant_entry(
        TASK, RepresentationalSimilarityTask, key, tiny_states, tiny_identity_states,
        negatives={"identity_blocks": "_meta_rsa_adjacent_mean_corrdist"}, rtol_width=1e-9)


def test_declarations():
    R = RepresentationalSimilarityTask
    assert R.TOKEN_POLICY == "exclude_sinks"
    assert set(R.SIZE_INVARIANT) <= set(R.HEADLINE_KEYS)
    assert {"paper", "code", "llm_precedent"} <= set(R.SOURCES)
    assert R.ADAPTATION_NOTE and R.ADAPTATION_NOTE.count(". ") <= 8
