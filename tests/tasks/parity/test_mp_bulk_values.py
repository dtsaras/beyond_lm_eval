"""geometry_mp_bulk_deviation: the fixed-gamma (g25) headline keys, exactly.

Each injected layer is an N x D cloud whose correlation matrix is EXACTLY the
equicorrelation matrix C = (1 - rho) I + rho 11^T, with N = 4D + 1 rows, so
gamma = D / (N - 1) = 0.25 and the fixed-gamma draw takes every row. C has
eigenvalues 1 - rho + D rho (once) and 1 - rho (D - 1 times), so the outlier
fraction and spike energy above the MP edge (1 + sqrt(gamma))^2 (1 + tol)
are closed forms, and the KS distance is scipy's kstest against the MP CDF
integrated from its closed-form density. evaluate() runs on the injected
states with the packaged defaults.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

D = 8
N = 4 * D + 1
EDGE_TOL = 0.05                       # the task default
RHO = {"first": 0.5, "mid": 0.2, "last": 0.1}


def _equicorrelated(rho, seed):
    """N x D rows with mean 0, population variance 1 and X^T X / N = C."""
    g = np.random.default_rng(seed)
    G = g.standard_normal((N, D))
    Q, _ = np.linalg.qr(G - G.mean(axis=0))          # orthonormal, orthogonal to 1
    C = (1.0 - rho) * np.eye(D) + rho * np.ones((D, D))
    w, V = np.linalg.eigh(C)
    return np.sqrt(N) * Q @ (V * np.sqrt(w)) @ V.T


def _mp_cdf_exact(x, gamma):
    from scipy.integrate import quad
    lo, hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2

    def f(t):
        return np.sqrt(max((hi - t) * (t - lo), 0.0)) / (2 * np.pi * gamma * t)
    x = np.atleast_1d(x)
    return np.array([0.0 if v <= lo else 1.0 if v >= hi else quad(f, lo, v)[0] for v in x])


def _expected(rho):
    from scipy.stats import kstest
    gamma = D / (N - 1)
    eigs = np.array([1 - rho + D * rho] + [1 - rho] * (D - 1))
    out = eigs[eigs > (1 + np.sqrt(gamma)) ** 2 * (1 + EDGE_TOL)]
    return {"outlier_frac": out.size / D, "spike_energy": out.sum() / eigs.sum(),
            "ks_distance": kstest(eigs, lambda v: _mp_cdf_exact(v, gamma)).statistic}


def _states():
    """Three blocks of 3 documents x 12 rows; row 0 (dropped by the sink
    policy) is a copy of row 1, so the kept rows are exactly the cloud."""
    blocks = {}
    for li, rho in enumerate(RHO.values()):
        X = _equicorrelated(rho, seed=li)
        docs = []
        for k in range(3):
            rows = X[11 * k:11 * (k + 1)]
            docs.append(np.vstack([rows[:1], rows]))
        blocks[li] = docs
    return blocks


def test_cloud_is_exactly_equicorrelated():
    X = _equicorrelated(0.5, 0)
    assert np.allclose(X.mean(0), 0, atol=1e-12) and np.allclose(X.std(0), 1, atol=1e-12)
    assert np.allclose(np.corrcoef(X, rowvar=False),
                       0.5 * np.eye(D) + 0.5 * np.ones((D, D)), atol=1e-12)


def test_evaluate_g25_keys_have_their_closed_forms():
    res = inv.run_task_on_states("geometry_mp_bulk_deviation", _states())
    for role in RHO:
        assert res[f"_meta_mp_g25_n_tokens_needed_{role}"] == N
        assert res[f"_meta_mp_n_tokens_available_{role}"] == N
    exp = {role: _expected(rho) for role, rho in RHO.items()}
    assert exp["first"]["outlier_frac"] == 1 / D and exp["last"]["outlier_frac"] == 0.0
    for m in ("outlier_frac", "spike_energy", "ks_distance"):
        tol = 1e-6                    # the cache stores float32
        for role in RHO:
            assert res[f"mp_{m}_g25_{role}"] == pytest.approx(exp[role][m], abs=tol), (m, role)
        assert res[f"mp_{m}_g25_mean"] == pytest.approx(
            np.mean([exp[r][m] for r in RHO]), abs=tol), m
