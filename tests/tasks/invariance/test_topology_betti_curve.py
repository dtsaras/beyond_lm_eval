"""SIZE_INVARIANT of topology_betti_curve: hop-metric keys ignore scale and width.

The kNN graph depends on neighbour order only, and the hop metric gives every
edge length one, so a rescale, zero-padding and an isometric embedding leave
every Betti number unchanged. Known negative: a Euclidean-scale summary of the
same clouds (topology_homology's mean H0 lifetime) moves with the scale.
"""
from __future__ import annotations

import re

import pytest
import torch

import _family_d as fd

inv = fd.inv
pytest.importorskip("ripser")

from blme.tasks.topology.betti_curve import BettiCurveTask  # noqa: E402


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _run(task, S):
    return inv.run_task_on_states(task, S, {"num_samples": S.n_docs, "n_neighbors": 4},
                                  force=True, store_dtype=torch.float64)


def test_hop_metric_keys_are_width_and_scale_free(states):
    keys = sorted(BettiCurveTask.SIZE_INVARIANT)
    T = "topology_betti_curve"
    base = _run(T, states)
    for S in (inv.zero_pad_docs(states, 3), inv.rescale_docs(states, 2.0 ** -7),
              inv.embed_isometric_docs(states, 2, 1)):
        inv.assert_invariant(base, _run(T, S), keys + ["betti_0_curve", "betti_1_curve"],
                             rtol=0.0)


def test_a_euclidean_summary_of_the_same_clouds_moves(states):
    base = _run("topology_homology", states)
    scaled = _run("topology_homology", inv.rescale_docs(states, 2.0 ** -7))
    inv.assert_moves(base, scaled, "mid_layer_mean_persistence_h0", min_rel=0.9)


def test_no_absolute_layer_index_is_a_feature(states):
    """max_betti_1_layer is an absolute block index; its depth fraction stays."""
    res = _run("topology_betti_curve", states)
    assert "max_betti_1_layer" not in res and "_meta_max_betti_1_layer" in res
    assert 0.0 <= res["max_betti_1_layer_frac"] <= 1.0
    assert not [k for k in res if not k.startswith("_meta_") and re.search(r"layer_\d", k)]
    assert res["_meta_knn_true_neighbours"] == res["_meta_knn_k"] - 1 == 3
