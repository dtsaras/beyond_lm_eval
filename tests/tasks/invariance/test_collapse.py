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


def _qdl_reference(docs):
    """Queipo-de-Llano Eq. (1) written from the paper: uncentred T x d matrix
    of one sequence, p_j = s_j^2 / ||X||_F^2, H = -sum p log p."""
    hs = []
    for X in docs:
        X = np.asarray(X, dtype=np.float64)
        s = np.linalg.svd(X, compute_uv=False)
        p = s ** 2 / np.sum(X * X)
        p = p[p > 0]
        hs.append(-np.sum(p * np.log(p)))
    return float(np.mean(hs))


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def test_qdl_profile_is_eq1_per_sequence_uncentred(tiny_states):
    """Every profile entry is the per-sequence mean of Eq. (1) on the raw
    (uncentred, position-0-inclusive) states; the headline is its minimum."""
    r = inv.run_task_on_states(TASK, tiny_states)
    prof = r["_meta_qdl_entropy_per_layer"]
    assert len(prof) == tiny_states.n_layers
    for li, h in enumerate(prof):
        want = _qdl_reference([np.asarray(x, dtype=np.float32) for x in tiny_states.blocks[li]])
        assert h == pytest.approx(want, rel=1e-9, abs=1e-12), li
    assert r["qdl_entropy_min"] == min(prof)
    assert r["_meta_qdl_n_docs"] == tiny_states.n_docs


def test_qdl_valley_follows_a_position0_sink(tiny_states):
    """A massive position-0 row at one middle block (the BOS sink of Theorem 1)
    puts the profile minimum at that block and pushes H towards 0; centring
    would remove exactly that row's mean contribution, which Eq. (1) keeps."""
    S = tiny_states
    mid = S.n_layers // 2

    def sink(x, li):
        y = np.array(x, dtype=np.float64, copy=True)
        if li == mid:
            y[0] *= 1e3
        return y
    sunk = inv.States(blocks={li: [sink(x, li) for x in docs] for li, docs in S.blocks.items()},
                      final_norm=S.final_norm, embedding=S.embedding)
    base = inv.run_task_on_states(TASK, S)
    r = inv.run_task_on_states(TASK, sunk)
    assert r["_meta_qdl_entropy_per_layer"][mid] < 0.05
    assert r["qdl_entropy_min"] == r["_meta_qdl_entropy_per_layer"][mid]
    assert r["_meta_qdl_entropy_min_layer_frac"] == pytest.approx(mid / (S.n_layers - 1))
    assert base["_meta_qdl_entropy_per_layer"][mid] > 10 * r["_meta_qdl_entropy_per_layer"][mid]


RV_KEYS = ("max_erank", "min_erank", "collapse_ratio", "max_drop")


def _map_blocks(S, fn):
    """``fn`` on every array (blocks, final norm, embedding), in float64."""
    return S.map(lambda x: fn(np.asarray(x, dtype=np.float64)))


def test_roy_vetterli_keys_drop_position0_and_massive_rows(tiny_states):
    """Sink policy: a 1e3 position-0 row (BOS-style sink) and a 1e3 row at
    position 3 (first-delimiter-style massive row) in every block leave the
    Jing / Roy-Vetterli keys exactly where the same documents put them with
    those rows absent. Before, both rows entered the pooled cloud."""
    def sink(y):
        y = y.copy()
        y[0] *= 1e3
        y[3] *= 1e3
        return y
    sunk = inv.run_task_on_states(TASK, _map_blocks(tiny_states, sink))
    absent = inv.run_task_on_states(TASK, _map_blocks(tiny_states, lambda y: np.delete(y, 3, 0)))
    inv.assert_invariant(absent, sunk, list(RV_KEYS), rtol=1e-9)
    assert sunk["_meta_sink_n_massive_dropped"] == tiny_states.n_docs
    assert absent["_meta_sink_n_massive_dropped"] == 0
    assert sunk["_meta_rows_policy"] == "exclude_sinks"
    assert sunk["_meta_position0_share"] > 0.3 and sunk["_meta_massive_token_share"] > 0.3
    assert absent["_meta_massive_token_share"] < 0.05
    # The every-token companion still sees the sink (known negative).
    inv.assert_moves(absent, sunk, "_meta_collapse_ratio_all_tokens", 1e-2)
