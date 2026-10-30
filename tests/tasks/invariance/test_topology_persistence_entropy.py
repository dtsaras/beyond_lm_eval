"""SIZE_INVARIANT of topology_persistence_entropy: H1 entropy and E / log2(n).

PE depends on the normalised bar lengths only, so a rescale cancels; the
barcode depends on pairwise distances only, so zero-padding and isometric
embeddings leave it. Known negative: the H1 lifetimes PE summarises (the
same barcode's mean H1 bar, topology_homology) carry the scale.
"""
from __future__ import annotations

import pytest
import torch

import _family_d as fd

inv = fd.inv
pytest.importorskip("ripser")

from blme.tasks.topology.persistence_entropy import PersistenceEntropyTask  # noqa: E402


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _run(task, S):
    return inv.run_task_on_states(task, S, {"num_samples": S.n_docs}, force=True,
                                  store_dtype=torch.float64)


def test_h1_entropy_is_width_and_scale_free(states):
    keys = sorted(PersistenceEntropyTask.SIZE_INVARIANT)
    base = _run("topology_persistence_entropy", states)
    assert any(base[f"{r}_layer_num_features_h1"] >= 2 for r in ("first", "mid", "last"))
    T = "topology_persistence_entropy"
    inv.assert_invariant(base, _run(T, inv.zero_pad_docs(states, 3)), keys, rtol=0.0)
    inv.assert_invariant(base, _run(T, inv.rescale_docs(states, 2.0 ** -7)), keys, rtol=0.0)
    inv.assert_invariant(base, _run(T, inv.embed_isometric_docs(states, 2, 1)), keys, rtol=1e-6)


def test_the_lifetimes_it_summarises_carry_the_scale(states):
    base = _run("topology_homology", states)
    scaled = _run("topology_homology", inv.rescale_docs(states, 2.0 ** -7))
    inv.assert_moves(base, scaled, "mid_layer_mean_persistence_h1", min_rel=0.9)
