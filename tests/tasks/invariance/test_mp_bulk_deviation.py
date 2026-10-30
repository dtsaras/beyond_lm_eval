"""geometry_mp_bulk_deviation: sink policy, the fixed-gamma headline and its
token supply, driven through the real ``evaluate()`` on injected states."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load

TASK = "geometry_mp_bulk_deviation"


@pytest.fixture(scope="module")
def tiny_states():
    return inv.states_from_model(*load("hf-internal-testing/tiny-random-gpt2"), TEXTS)


def _map(S, fn):
    return S.map(lambda x: fn(np.asarray(x, dtype=np.float64)))


def test_position0_and_massive_rows_are_dropped(tiny_states):
    """A 1e3 position-0 row and a 1e3 row at position 3 in every entry give
    the same MP statistics as the same documents with those rows absent."""
    def sink(y):
        y = y.copy()
        y[0] *= 1e3
        y[3] *= 1e3
        return y
    sunk = inv.run_task_on_states(TASK, _map(tiny_states, sink))
    absent = inv.run_task_on_states(TASK, _map(tiny_states, lambda y: np.delete(y, 3, 0)))
    keys = [k for k in absent if k.startswith("_meta_mp_") is False and k.startswith("mp_")]
    assert keys
    inv.assert_invariant(absent, sunk, keys, rtol=1e-9)
    assert sunk["_meta_sink_n_massive_dropped"] == tiny_states.n_docs
    assert sunk["_meta_rows_policy"] == "exclude_sinks"
