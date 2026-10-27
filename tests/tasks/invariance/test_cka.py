"""geometry_cka: token policy, key hygiene and size invariance, driven
through the real ``evaluate()`` on injected hidden states."""
from __future__ import annotations

import math

import numpy as np
import pytest

from _family_c import TEXTS, inv, load, scale_position0

TASK = "geometry_cka"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


@pytest.fixture(scope="module")
def gpt2_states():
    return inv.states_from_model(*load("gpt2"), TEXTS[:6])


def test_headline_ignores_position0_rows(tiny_states):
    """Position 0 is excluded from every headline: scaling the position-0 row
    of every document by 1e3 (a synthetic sink) must not move one of them."""
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    keys = [k for k in base if not k.startswith("_meta_")
            and isinstance(base[k], float)]
    assert keys
    inv.assert_invariant(base, sunk, keys, rtol=1e-6)


def test_position0_and_massive_shares_are_derived(tiny_states, gpt2_states):
    """The sink diagnostics are measured, not constants: gpt2's real position-0
    sink dominates the mid-layer centred SS, the synthetic one too, and the
    inclusive companion moves while the headline does not."""
    g = inv.run_task_on_states(TASK, gpt2_states)
    assert g["_meta_position0_share"] > 0.5
    assert 0.0 <= g["_meta_massive_token_share"] <= 1.0
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]
    inv.assert_moves(base, sunk, "_meta_incl_pos0_early_late_cka", 1e-3)
    assert base["_meta_rows_policy"] == "exclude_pos0"
    assert base["_meta_rows_n_rows"] == sum(tiny_states.lengths) - tiny_states.n_docs


def test_max_offdiag_duplicate_is_not_emitted(tiny_states):
    """max_offdiag_cka equalled max_adjacent_cka on 65 of 66 study models (the
    most similar pair is adjacent); the duplicate column is dropped."""
    r = inv.run_task_on_states(TASK, tiny_states)
    assert "max_offdiag_cka" not in r
    off = [k for k in r if "max_offdiag" in k]
    assert off == [], off
