"""geometry_lipschitz: the depth-additive keys have closed forms.

Every injected row lives in a plane and block l maps it by h -> s_l R(t_l) h
(rotation by t_l, scale s_l); the embedding -> block 0 update is s_e R(t_e).
Then, exactly and for every row:
  e2e_log_norm_growth       = sum over block transitions of log s_l,
  total_angular_path        = sum over block transitions of t_l,
  embedding_relative_change = |s_e e^{i t_e} - 1| = sqrt(s_e^2 - 2 s_e cos t_e + 1),
  relative_change_max       = max_l sqrt(s_l^2 - 2 s_l cos t_l + 1).
evaluate() runs on the injected states with the packaged defaults.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

S_E, T_E = 1.7, 0.4                                   # embedding -> block 0
S = [1.3, 0.8, 1.1, 1.25]                             # block l -> l + 1
T = [0.3, 0.9, 0.15, 0.6]


def _rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_depth_additive_keys_have_their_closed_forms():
    g = np.random.default_rng(0)
    emb = [np.vstack([g.standard_normal((1, 2)) * 0.5,
                      g.uniform(0.8, 1.2, (7, 1)) * np.stack(
                          [np.cos(a := g.uniform(0, 2 * np.pi, 7)), np.sin(a)], 1)])
           for _ in range(6)]
    blocks = {0: [S_E * x @ _rot(T_E).T for x in emb]}
    for li, (s, t) in enumerate(zip(S, T)):
        blocks[li + 1] = [s * x @ _rot(t).T for x in blocks[li]]
    res = inv.run_task_on_states("geometry_lipschitz", blocks, embedding=emb)
    assert res["_meta_embedding_transition_available"] is True

    def rel(s, t):
        return np.sqrt(s * s - 2.0 * s * np.cos(t) + 1.0)

    assert res["e2e_log_norm_growth"] == pytest.approx(np.sum(np.log(S)), rel=1e-6)
    assert res["total_angular_path"] == pytest.approx(np.sum(T), rel=1e-6)
    assert res["embedding_relative_change"] == pytest.approx(rel(S_E, T_E), rel=1e-6)
    assert res["relative_change_max"] == pytest.approx(max(rel(s, t) for s, t in zip(S, T)),
                                                       rel=1e-6)
