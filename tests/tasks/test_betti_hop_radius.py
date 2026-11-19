"""topology_betti_curve: the beta_1 read radius is a parameter, and the other
radius is always recorded.

At 2 hops (the released topnn_framework read, the default) beta_1 of LLM
last-token clouds is 0 or near 0 at N = 1000 on every checkpoint checked, so
the headline is nearly constant across models; 1 hop reads the kNN graph's
clique complex. Both come from one persistence diagram.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ripser")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invariance.driver import run_task_on_states  # noqa: E402


def test_extra_radius_equals_a_direct_read():
    from blme.tasks.topology.betti_curve import _count_betti
    X = np.random.default_rng(0).standard_normal((150, 12))
    extra = {}
    b0, b1_two = _count_betti(X, maxdim=1, extra_radii=(1.0,), extra=extra)
    _, b1_one = _count_betti(X, maxdim=1, hop_radius=1.0)
    assert extra[1.0] == b1_one
    assert (b0, b1_two) == _count_betti(X, maxdim=1)


def test_hop_radius_swaps_the_headline_and_the_recorded_variant():
    rng = np.random.default_rng(1)
    states = {l: [rng.standard_normal((3, 10)).astype(np.float32) for _ in range(120)]
              for l in range(3)}
    two = run_task_on_states("topology_betti_curve", states, {"num_samples": 120})
    one = run_task_on_states("topology_betti_curve", states, {"num_samples": 120, "hop_radius": 1.0})
    assert two["_meta_hop_radius"] == 2.0 and one["_meta_hop_radius"] == 1.0
    assert one["betti_1_curve_mean"] == two["_meta_betti_1_curve_mean_hop1"]
    assert two["betti_1_curve_mean"] == one["_meta_betti_1_curve_mean_hop2"]
