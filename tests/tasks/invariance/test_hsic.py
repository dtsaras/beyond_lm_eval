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
