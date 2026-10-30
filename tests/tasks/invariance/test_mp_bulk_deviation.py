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
    keys = [k for k in absent if k.startswith(("mp_", "_meta_mp_outlier", "_meta_mp_spike",
                                               "_meta_mp_ks"))]
    assert keys
    inv.assert_invariant(absent, sunk, keys, rtol=1e-9)
    assert sunk["_meta_sink_n_massive_dropped"] == tiny_states.n_docs
    assert sunk["_meta_rows_policy"] == "exclude_sinks"



def test_fixed_gamma_family_is_the_headline_and_is_not_capped(tiny_states):
    """The gamma-matched statistics drift with N (gamma = d/(N-1)), so the
    fixed-gamma (N = 4d+1) family is the headline. Its draw comes from the
    whole cloud: a max_tokens cap below 4d+1 used to make it NaN (every
    d >= 5000 model under the 20000 cap)."""
    r = inv.run_task_on_states(TASK, tiny_states, {"max_tokens": 100})
    d = tiny_states.d
    assert np.isfinite(r["mp_ks_distance_g25_mid"])
    assert r["_meta_mp_g25_n_tokens_needed_mid"] == 4 * d + 1
    assert r["_meta_mp_n_tokens_available_mid"] > 100
    assert "mp_ks_distance_mid" not in r and "_meta_mp_ks_distance_mid" in r



# --------------------------------------------------------------------------
# Declarations and their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.rmt_bulk import MPBulkDeviationTask  # noqa: E402

_WIDTH = {"zero_pad": lambda S: inv.zero_pad_docs(S, 2),
          "rescale": lambda S: inv.rescale_docs(S, 2.0)}


@pytest.fixture(scope="module")
def base(tiny_states):
    return inv.run_task_on_states(TASK, tiny_states)


@pytest.fixture(scope="module")
def transformed(tiny_states):
    return {t: inv.run_task_on_states(TASK, f(tiny_states)) for t, f in _WIDTH.items()}


@pytest.mark.parametrize("key", sorted(MPBulkDeviationTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, base, transformed, tiny_states):
    """Zero-padding and rescaling leave the fixed-gamma key in place; the
    gamma-matched form it replaced moves with the token count (known
    negative: half the documents)."""
    transforms, node = MPBulkDeviationTask.SIZE_INVARIANT[key]
    assert node.endswith(f"test_size_invariant_entry[{key}]")
    assert np.isfinite(base[key])
    for t in transforms:
        inv.assert_invariant(base, transformed[t], [key], rtol=1e-6)
    half = inv.run_task_on_states(TASK, inv.subsample_docs(tiny_states, 6, seed=0))
    inv.assert_moves(base, half, "_meta_mp_ks_distance_mid", 1e-3)


def test_declarations(base):
    cls = MPBulkDeviationTask
    assert cls.TOKEN_POLICY == "exclude_sinks"
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert set(cls.SIZE_INVARIANT) <= set(cls.HEADLINE_KEYS) <= set(base)
