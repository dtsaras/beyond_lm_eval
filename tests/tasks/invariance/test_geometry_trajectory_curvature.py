"""SIZE_INVARIANT of geometry_trajectory_curvature: first, last, straightening.

An inserted identity block repeats the previous block's states, so the
first and last layers' curvatures (and their ratio) are untouched, while the
profile summaries it replaced (mean, quartiles, slope, middle index) move.
Angles depend on directions only, so width and scale changes leave every key.
"""
from __future__ import annotations

import pytest

import _family_d as fd

inv = fd.inv

from blme.tasks.geometry.trajectory_curvature import TrajectoryCurvatureTask  # noqa: E402

T = "geometry_trajectory_curvature"


@pytest.fixture(scope="module")
def model():
    return fd.tiny_gpt2()


@pytest.fixture(scope="module")
def states(model):
    return inv.states_from_model(model, fd.tokenizer(), fd.TEXTS)


def test_endpoints_invariant_under_identity_blocks(model, states):
    keys = sorted(TrajectoryCurvatureTask.SIZE_INVARIANT)
    tok = fd.tokenizer()
    ids = [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]
    deep = inv.insert_identity_blocks(model, [0, 2, 2], verify_on=ids)
    base = inv.run_task_on_states(T, states)
    ins = inv.run_task_on_states(T, inv.states_from_model(deep, tok, fd.TEXTS))
    assert ins["_meta_n_layers"] == base["_meta_n_layers"] + 3
    inv.assert_invariant(base, ins, keys, rtol=0.0)
    # Known negative: the profile summaries count the duplicated layers.
    for k in ("_meta_curvature_overall_mean", "_meta_curvature_slope"):
        inv.assert_moves(base, ins, k, min_rel=1e-6)


def test_every_key_ignores_width_and_scale(states):
    keys = sorted(TrajectoryCurvatureTask.SIZE_INVARIANT)
    base = inv.run_task_on_states(T, states)
    for S in (inv.zero_pad_docs(states, 3), inv.rescale_docs(states, 2.0 ** -5),
              inv.embed_isometric_docs(states, 2, 1)):
        inv.assert_invariant(base, inv.run_task_on_states(T, S), keys, rtol=1e-6)
