"""geometry_isoscore: paper protocol (every token), the position-0-free value
and sink shares as _meta_, and the effective-dims twin pinned under
zero-padding, isometric embedding and rescaling."""
from __future__ import annotations

import numpy as np
import pytest

from _family_c import TEXTS, inv, load

TASK = "geometry_isoscore"


@pytest.fixture(scope="module")
def tiny_states():
    return inv.states_from_model(*load("hf-internal-testing/tiny-random-gpt2"), TEXTS)


def _sink0(S):
    def f(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[0] *= 1e3
        return y
    return S.map(f)


def test_paper_protocol_keeps_position0_and_reports_it(tiny_states):
    """Rudman et al. (and the I-STAR code) score every token, so a sink at
    position 0 moves the headline; the position-0-free value is _meta_ and
    does not move, and the share of the scored cloud's SS at position 0 is
    measured."""
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, _sink0(tiny_states))
    inv.assert_moves(base, sunk, "isoscore_effective_dims", 1e-2)
    inv.assert_invariant(base, sunk, ["_meta_isoscore_effective_dims_excl_pos0"], rtol=1e-9)
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]
    assert 0.0 <= base["_meta_massive_token_share"] <= 1.0
