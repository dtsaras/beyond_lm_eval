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
