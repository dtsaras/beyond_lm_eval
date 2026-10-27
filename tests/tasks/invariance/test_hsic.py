"""geometry_hsic: token policy, layer trimming, key hygiene and size
invariance, driven through the real ``evaluate()`` on injected states."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load, scale_position0

TASK = "geometry_hsic"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _headline(r):
    return [k for k in r if not k.startswith("_meta_") and isinstance(r[k], float)]


def test_headline_ignores_position0_rows(tiny_states):
    """Position 0 is excluded: scaling every document's position-0 row by 1e3
    (a synthetic sink) moves no headline key."""
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    keys = _headline(base)
    assert keys
    inv.assert_invariant(base, sunk, keys, rtol=1e-6)
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]
    inv.assert_moves(base, sunk, "_meta_incl_pos0_hsic_compression_ratio", 1e-3)
    assert base["_meta_rows_policy"] == "exclude_pos0"


@pytest.mark.parametrize("attrs", [{}, {"num_hidden_layers": None, "n_layer": None}])
def test_legacy_cache_trims_only_the_post_norm_entry(attrs):
    """A cache without the deepest-block capture ends in HF's post-norm entry,
    and exactly that entry is dropped -- whatever the model config says. A
    config without num_hidden_layers used to trim the profile to 2 layers."""
    from blme.tasks.geometry.mutual_info import HSICDependenceTask
    rng = np.random.default_rng(0)
    L = 6
    S = inv.States(blocks={li: [rng.standard_normal((40, 8)) + li for _ in range(3)]
                           for li in range(L)})
    m = inv.StubModel(L, 8, attrs)
    cache = inv.InjectedStateCache(S, m, inv.StubTokenizer())
    cache._last_block_raw = None          # a cache built before the hook existed
    r = HSICDependenceTask({"num_samples": 3}).evaluate(m, None, None, cache=cache)
    assert len(r["_meta_adjacent_hsic"]) == L - 2
    assert r["_meta_n_layers"] == L - 1
    assert r["_meta_last_entry_is_block_output"] is False


def test_headline_keys_are_bounded_similarities(tiny_states):
    """On a [X, Y, 3X] stack the manuscript's raw ratio tr(K0 K2)/tr(K0 K0)
    is exactly 9 -- an unbounded projection coefficient -- so it is _meta_ and
    every headline key is a similarity in [-1, 1]."""
    S = tiny_states
    stack = inv.States(blocks={0: S.blocks[0], 1: S.blocks[1],
                               2: [3.0 * np.asarray(x, dtype=np.float64) for x in S.blocks[0]]})
    r = inv.run_task_on_states(TASK, stack)
    assert r["_meta_hsic_compression_ratio_raw"] == pytest.approx(9.0, rel=1e-6)
    assert "hsic_compression_ratio_raw" not in r
    for k in _headline(r):
        assert -1.0 - 1e-9 <= r[k] <= 1.0 + 1e-9, (k, r[k])


@pytest.fixture(scope="module")
def tiny_identity_states(tiny):
    """tiny-random-gpt2 with exact identity blocks after blocks 1 and 3."""
    model, tok = tiny
    return inv.states_from_model(inv.insert_identity_blocks(model, [1, 3]), tok, TEXTS)


def test_headline_is_depth_invariant_under_identity_insertion(
        tiny_states, tiny_identity_states):
    """P4: identity blocks move no headline key; the uniform adjacent mean
    and the per-layer lists (now _meta_) do move (known negative)."""
    base = inv.run_task_on_states(TASK, tiny_states)
    ins = inv.run_task_on_states(TASK, tiny_identity_states)
    assert ins["_meta_n_layers"] == base["_meta_n_layers"] + 2
    keys = _headline(base)
    assert {"min_adjacent_hsic", "hsic_compression_ratio"} <= set(keys)
    inv.assert_invariant(base, ins, keys, rtol=1e-9)
    inv.assert_moves(base, ins, "_meta_avg_adjacent_hsic", 1e-3)
    assert len(ins["_meta_adjacent_hsic"]) == len(base["_meta_adjacent_hsic"]) + 2
    assert not any(isinstance(v, list) for k, v in base.items()
                   if not k.startswith("_meta_"))


def test_gram_debiased_hsic_matches_the_cka_estimator():
    """The O(N^2) Gram form equals cka._debiased_hsic (the parity-pinned
    kernel-space U-statistic) on centred Grams."""
    import torch
    from blme.tasks.geometry.cka import _debiased_hsic
    from blme.tasks.geometry.mutual_info import (_centered_linear_gram,
                                                 _debiased_hsic_from_grams)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(60, 7, generator=g, dtype=torch.float64)
    Y = X @ torch.randn(7, 5, generator=g, dtype=torch.float64) + 0.3 * torch.randn(
        60, 5, generator=g, dtype=torch.float64)
    K, L = _centered_linear_gram(X), _centered_linear_gram(Y)
    assert _debiased_hsic_from_grams(K, L) == pytest.approx(_debiased_hsic(K, L), rel=1e-10)


DEBIASED = ("hsic_compression_ratio_debiased", "min_adjacent_hsic_debiased")


def test_debiased_twin_has_no_sample_size_floor(tiny_states):
    """N variation on a row-permutation null (population nCKA = 0): the biased
    anchor-last nCKA falls with the number of documents, the debiased twins
    stay at 0 at every N."""
    from _family_c import keep_layers, permute_layer_rows
    L = tiny_states.n_layers
    null = permute_layer_rows(keep_layers(tiny_states, [0, L // 2]), 1, seed=0)
    small = [inv.run_task_on_states(TASK, inv.subsample_docs(null, 3, seed=s))
             for s in (0, 1)]
    full = inv.run_task_on_states(TASK, null)
    biased_small = np.mean([r["hsic_compression_ratio"] for r in small])
    assert biased_small - full["hsic_compression_ratio"] > 0.1      # known negative
    for r in small + [full]:
        for k in DEBIASED:
            assert abs(r[k]) < 0.03, (k, r[k])


@pytest.mark.parametrize("transform", ["zero_pad", "isometry", "rescale"])
def test_headline_is_width_and_scale_invariant(tiny_states, transform):
    St = {"zero_pad": lambda S: inv.zero_pad_docs(S, 2),
          "isometry": lambda S: inv.embed_isometric_docs(S, 2, 3),
          "rescale": lambda S: inv.rescale_docs(S, 2.0)}[transform](tiny_states)
    base = inv.run_task_on_states(TASK, tiny_states)
    inv.assert_invariant(base, inv.run_task_on_states(TASK, St), _headline(base), rtol=1e-6)
