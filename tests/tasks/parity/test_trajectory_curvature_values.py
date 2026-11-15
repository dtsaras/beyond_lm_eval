"""geometry_trajectory_curvature: the headline keys on exact polygon paths.

Each injected sequence is a planar path whose direction turns by a constant
angle alpha at every token, so the Hosseini et al. (2023) curvature (the
angle between consecutive difference vectors) is alpha at every vertex and
its per-sentence mean is alpha. The first block turns by 0.5 rad and the
post-final-norm entry (the last layer, as curvature_utils reads it) by
0.1 rad, so straightening_ratio = (0.5 - 0.1) / 0.5 = 0.8.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402


def _polygon(alpha, n=9, offset=(0.0, 0.0)):
    ang = np.arange(n - 1) * alpha
    steps = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)]) + np.asarray(offset)


def test_constant_turning_paths_give_exact_curvatures():
    a_first, a_mid, a_last = 0.5, 0.3, 0.1
    blocks = {li: [_polygon(a, offset=(3.0 + i, 1.0 - i)) for i in range(5)]
              for li, a in enumerate((a_first, a_mid, 0.2))}
    final = [_polygon(a_last, offset=(3.0 + i, 1.0 - i)) for i in range(5)]
    res = inv.run_task_on_states("geometry_trajectory_curvature", blocks, final_norm=final)
    assert res["curvature_mean_first_layer"] == pytest.approx(a_first, rel=1e-6)
    assert res["curvature_mean_last_layer"] == pytest.approx(a_last, rel=1e-5)
    assert res["straightening_ratio"] == pytest.approx((a_first - a_last) / a_first, rel=1e-6)
