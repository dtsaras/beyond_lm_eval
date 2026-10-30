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


# --------------------------------------------------------------------------
# Declarations and their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.isotropy import IsoScoreTask  # noqa: E402

_WIDTH = {"zero_pad": lambda S: inv.zero_pad_docs(S, 2),
          "isometry": lambda S: inv.embed_isometric_docs(S, 2, 3),
          "rescale": lambda S: inv.rescale_docs(S, 2.0)}


@pytest.mark.parametrize("key", sorted(IsoScoreTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny_states):
    """iso*(d-1)+1 is the participation ratio of the covariance spectrum: it
    does not move under zero-padding, a random isometric embedding (x2) or a
    rescale, while the raw IsoScore it corrects falls by (d-1)/(kd-1)."""
    transforms, node = IsoScoreTask.SIZE_INVARIANT[key]
    assert node.endswith(f"test_size_invariant_entry[{key}]")
    base = inv.run_task_on_states(TASK, tiny_states)
    d = tiny_states.d
    for t in transforms:
        other = inv.run_task_on_states(TASK, _WIDTH[t](tiny_states))
        inv.assert_invariant(base, other, [key], rtol=1e-6)
        if t in ("zero_pad", "isometry"):
            assert other["isoscore"] / base["isoscore"] == pytest.approx(
                (d - 1) / (2 * d - 1), rel=1e-5)


def test_declarations(tiny_states):
    cls = IsoScoreTask
    assert cls.TOKEN_POLICY.startswith("paper_protocol:")
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert set(cls.SIZE_INVARIANT) <= set(cls.HEADLINE_KEYS)
    r = inv.run_task_on_states(TASK, tiny_states)
    assert set(cls.HEADLINE_KEYS) <= set(r)
