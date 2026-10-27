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
