"""geometry_svd: both headline keys through evaluate().

test_comprehensive_parity::test_geometry_svd importorskips the IsoScore pip
package, which is not installed, so it skips. Here evaluate() runs in the
default single-layer mode on injected states, reading HF's post-norm last
entry: effective_rank is Roy & Vetterli's exp(H(sigma / sum sigma)) of the
centred non-sink rows (scipy svdvals), avg_cosine_similarity the mean cosine
over distinct pairs of every row (Ethayarajh 2019 Eq. 4), by brute force.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import svdvals

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

N_DOCS, T, D = 12, 10, 6


def test_svd_keys_match_brute_force_definitions():
    g = np.random.default_rng(2)
    blocks = {li: [g.standard_normal((T, D)) for _ in range(N_DOCS)] for li in range(4)}
    scale = np.linspace(0.3, 2.5, D)
    final = [(g.standard_normal((T, D)) * scale + 0.8).astype(np.float32) for _ in range(N_DOCS)]
    res = inv.run_task_on_states("geometry_svd", blocks, final_norm=final)
    assert res["_meta_svd_mode"] == "single_layer" and res["_meta_sink_n_massive_dropped"] == 0

    X = np.concatenate([np.asarray(x, np.float64)[1:] for x in final])
    s = svdvals(X - X.mean(axis=0))
    p = s / s.sum()
    assert res["effective_rank"] == pytest.approx(float(np.exp(-(p * np.log(p)).sum())), rel=1e-9)

    A = np.concatenate([np.asarray(x, np.float64) for x in final])
    U = A / np.linalg.norm(A, axis=1, keepdims=True)
    C = U @ U.T
    off = C[~np.eye(len(A), dtype=bool)]
    assert res["avg_cosine_similarity"] == pytest.approx(float(off.mean()), rel=1e-9)
