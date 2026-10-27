"""geometry_procrustes_linearity: rank-aware headline, adjusted-R^2 twin and
the position-0 companion, driven through the real ``evaluate()``."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_procrustes_linearity"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _headline(r):
    return [k for k in r if not k.startswith("_meta_") and isinstance(r[k], float)]


def test_headline_survives_an_exact_isometric_embedding(tiny_states):
    """The least-squares map is basis-free, so an isometric embedding (x2
    width, every centred cloud now rank-deficient in the new basis) must not
    move the headline. The released unguarded-1/S rule does move: it inverts
    the ~1e-15 singular values the embedding creates (known negative)."""
    f64 = {"store_dtype": torch.float64}
    base = inv.run_task_on_states(TASK, tiny_states, **f64)
    iso = inv.run_task_on_states(TASK, inv.embed_isometric_docs(tiny_states, 2, 3), **f64)
    inv.assert_invariant(base, iso, ["procrustes_linearity_mean", "procrustes_linearity_min",
                                     "procrustes_linearity_first", "procrustes_linearity_last"],
                         rtol=1e-6)
    inv.assert_moves(base, iso, "_meta_reference_rule_procrustes_linearity_mean", 1e-3)
    assert iso["_meta_procrustes_rank_aware_gap_max"] > 1e-3


@pytest.fixture(scope="module")
def null_states(tiny_states):
    """Two blocks, the second's rows shuffled across the cloud: an independent
    target, so the population linearity is 0 and least squares on r regressors
    scores r/(N-1) by chance."""
    from _family_c import keep_layers, permute_layer_rows
    L = tiny_states.n_layers
    return permute_layer_rows(keep_layers(tiny_states, [0, L // 2], embedding=False), 1, seed=0)


def test_adjusted_twin_removes_the_chance_fit(null_states):
    """N variation on the null: the raw score falls from ~0.42 to ~0.10
    between 3 and 12 documents (r/(N-1), r = 32); the adjusted twin
    1 - (1 - sim)(N-1)/(N-1-r) stays at 0."""
    f64 = {"store_dtype": torch.float64}
    small = [inv.run_task_on_states(TASK, inv.subsample_docs(null_states, 3, seed=s), **f64)
             for s in (0, 1)]
    full = inv.run_task_on_states(TASK, null_states, **f64)
    for r in small + [full]:
        r_over = r["_meta_fit_rank_max"] / (r["_meta_n_tokens"] - 1)
        assert r["procrustes_linearity_mean"] == pytest.approx(r_over, abs=0.06)
        assert abs(r["procrustes_linearity_adjusted_mean"]) < 0.06
    drift = np.mean([r["procrustes_linearity_mean"] for r in small]) - full["procrustes_linearity_mean"]
    assert drift > 0.2                                        # known negative


def test_position0_kept_with_the_excluded_value_beside_it(tiny_states):
    """The paper keeps every position (BOS included): a synthetic sink moves
    the headline, while the _meta_ value without position 0 does not."""
    from _family_c import scale_position0
    f64 = {"store_dtype": torch.float64}
    base = inv.run_task_on_states(TASK, tiny_states, **f64)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3), **f64)
    inv.assert_moves(base, sunk, "procrustes_linearity_mean", 1e-3)
    inv.assert_invariant(base, sunk, ["_meta_excl_pos0_procrustes_linearity_mean",
                                      "_meta_excl_pos0_procrustes_linearity_adjusted_mean"],
                         rtol=1e-9)
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]


# --------------------------------------------------------------------------
# SIZE_INVARIANT declarations, checked against their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.procrustes_linearity import ProcrustesLinearityTask  # noqa: E402


@pytest.fixture(scope="module")
def tiny_identity_states(tiny):
    """tiny-random-gpt2 with exact identity blocks after blocks 1 and 3."""
    model, tok = tiny
    return inv.states_from_model(inv.insert_identity_blocks(model, [1, 3]), tok, TEXTS)


@pytest.mark.parametrize("key", sorted(ProcrustesLinearityTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny_states, tiny_identity_states, null_states):
    """Each declared transform's direct test, in float64 storage (a float32
    copy of an isometric embedding is numerically full-rank). Known
    negatives: the adjusted mean under identity blocks, the raw form under N."""
    from _family_c import check_size_invariant_entry
    negatives = {"identity_blocks": "procrustes_linearity_adjusted_mean"}
    if "_adjusted_" in key:
        negatives["N"] = key.replace("_adjusted_", "_")
    check_size_invariant_entry(TASK, ProcrustesLinearityTask, key, tiny_states,
                               tiny_identity_states, null_states=null_states,
                               negatives=negatives, rtol_width=1e-6, null_bound=0.06,
                               null_drift=0.2, store_dtype=torch.float64)


def test_declarations():
    P = ProcrustesLinearityTask
    assert P.TOKEN_POLICY.startswith("paper_protocol:")
    assert set(P.SIZE_INVARIANT) <= set(P.HEADLINE_KEYS)
    assert {"paper", "code", "llm_precedent"} <= set(P.SOURCES)
