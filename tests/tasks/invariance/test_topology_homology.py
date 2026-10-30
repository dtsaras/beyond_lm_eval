"""SIZE_INVARIANT of topology_homology: the mean-pairwise-distance twins.

VR lifetimes and the mean pairwise distance are both homogeneous of degree 1
in the coordinates, so their ratio is unchanged by a rescale, and both depend
only on pairwise distances, so zero-padding and an isometric embedding leave
it alone. The raw lifetimes it replaced move with the rescale.
"""
from __future__ import annotations

import pytest
import torch

import _family_d as fd

inv = fd.inv
pytest.importorskip("ripser")

from blme.tasks.topology.homology import PersistentHomologyTask  # noqa: E402

TASK = "topology_homology"


@pytest.fixture(scope="module")
def states():
    return inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _run(S):
    return inv.run_task_on_states(TASK, S, {"num_samples": S.n_docs}, force=True,
                                  store_dtype=torch.float64)


def test_mpd_twins_are_width_and_scale_free(states):
    keys = sorted(PersistentHomologyTask.SIZE_INVARIANT)
    assert keys and all(k.endswith("_norm") for k in keys)
    base = _run(states)
    inv.assert_invariant(base, _run(inv.zero_pad_docs(states, 3)), keys, rtol=0.0)
    inv.assert_invariant(base, _run(inv.rescale_docs(states, 2.0 ** -7)), keys, rtol=0.0)
    for seed in (1, 2):
        inv.assert_invariant(base, _run(inv.embed_isometric_docs(states, 2, seed)), keys,
                             rtol=1e-6)


def test_raw_lifetimes_move_with_the_scale(states):
    """Known negative: the raw lifetimes are in the representation's units."""
    base = _run(states)
    scaled = _run(inv.rescale_docs(states, 2.0 ** -7))
    for r in ("first", "mid", "last"):
        inv.assert_moves(base, scaled, f"{r}_layer_mean_persistence_h0", min_rel=0.9)
        assert scaled[f"{r}_layer_mean_persistence_h0"] == pytest.approx(
            base[f"{r}_layer_mean_persistence_h0"] * 2.0 ** -7, rel=1e-12)


def test_role_layers_report_their_depth_fraction(states):
    res = _run(states)
    L = states.n_layers
    assert res["_meta_first_layer_frac"] == 0.0 and res["_meta_last_layer_frac"] == 1.0
    assert res["_meta_mid_layer_index"] == L // 2
    assert 0.0 <= res["_meta_massive_token_share"] <= 1.0
