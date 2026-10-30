"""SIZE_INVARIANT of topology_persistence_landscape: the /mpd^p twins.

A landscape ordinate is homogeneous of degree 1 in the metric, so an integral
is degree 2, the sup degree 1 and an L2 norm degree 1.5; dividing by the mean
pairwise distance to the same power removes a rescale, and both depend on
pairwise distances only, so zero-padding and isometric embeddings leave them.
"""
from __future__ import annotations

import pytest
import torch

import _family_d as fd

inv = fd.inv
pytest.importorskip("ripser")

from blme.tasks.topology.persistence_landscape import PersistenceLandscapeTask  # noqa: E402


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _run(S):
    return inv.run_task_on_states("topology_persistence_landscape", S,
                                  {"num_samples": S.n_docs, "subsample_size": 10,
                                   "n_subsamples": 3},
                                  force=True, store_dtype=torch.float64)


def test_mpd_twins_are_width_and_scale_free(states):
    keys = sorted(PersistenceLandscapeTask.SIZE_INVARIANT)
    assert len(keys) == 9
    base = _run(states)
    inv.assert_invariant(base, _run(inv.zero_pad_docs(states, 3)), keys, rtol=0.0)
    inv.assert_invariant(base, _run(inv.rescale_docs(states, 4.0)), keys, rtol=1e-12)
    inv.assert_invariant(base, _run(inv.embed_isometric_docs(states, 2, 1)), keys, rtol=1e-6)


def test_raw_summaries_carry_the_scale(states):
    """Known negative: the raw H1 integral scales by c^2, the sup by c."""
    base, scaled = _run(states), _run(inv.rescale_docs(states, 4.0))
    for r in ("first", "mid", "last"):
        for k, deg in (("mean_landscape_integral", 2), ("landscape_max_1", 1)):
            key = f"{r}_layer_h1_{k}"
            if base[key] == 0.0:
                continue
            assert scaled[key] == pytest.approx(base[key] * 4.0 ** deg, rel=1e-12)
            inv.assert_moves(base, scaled, key, min_rel=1.0)
