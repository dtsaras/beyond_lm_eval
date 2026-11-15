"""topology_homology and topology_persistence_entropy vs GUDHI, through evaluate().

Each injected document contributes its last token (TOKEN_POLICY
last_per_sequence), so the first / mid / last block clouds are known point
sets. GUDHI's Rips persistence on the same points gives the H0 and H1
lifetimes; every headline key is then a closed form of them: mean H0
lifetime, H1 bar count, the lifetimes divided by the mean pairwise distance,
persistent entropy -sum p log2 p (Atienza et al. 2020 Def. 3.1) and its
log2(m) normalisation. ripser reads float32 diagrams, hence rel 1e-6.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

gudhi = pytest.importorskip("gudhi")

N_PTS, D = 28, 3
ROLES = ("first", "mid", "last")


@pytest.fixture(scope="module")
def clouds():
    """Gaussian clouds of growing scale in R^3 (8, 5 and 6 H1 bars), stored
    as float32 as the cache stores them."""
    g = np.random.default_rng(11)
    return [(s * g.standard_normal((N_PTS, D))).astype(np.float32).astype(np.float64)
            for s in (1.0, 2.0, 3.5)]


@pytest.fixture(scope="module")
def states(clouds):
    g = np.random.default_rng(12)
    blocks = {}
    for li, X in enumerate(clouds):
        blocks[li] = [np.vstack([g.standard_normal((2, D)), X[i:i + 1]]) for i in range(N_PTS)]
    return blocks


def _gudhi_lifetimes(X):
    st = gudhi.RipsComplex(points=X, max_edge_length=np.inf).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h0 = [d - b for b, d in st.persistence_intervals_in_dimension(0) if np.isfinite(d)]
    h1 = [d - b for b, d in st.persistence_intervals_in_dimension(1) if np.isfinite(d)]
    return np.asarray(h0), np.asarray(h1)


def _entropy(lt):
    p = lt / lt.sum()
    return float(-np.sum(p * np.log2(p)))


def test_homology_evaluate_matches_gudhi(clouds, states):
    res = inv.run_task_on_states("topology_homology", states)
    for role, X in zip(ROLES, clouds):
        h0, h1 = _gudhi_lifetimes(X)
        mpd = float(pdist(X).mean())
        assert len(h1) >= 2, "the cloud must have several loops"
        assert res[f"{role}_layer_num_loops_h1"] == len(h1)
        want = {"mean_persistence_h0": h0.mean(),
                "mean_persistence_h0_norm": h0.mean() / mpd,
                "max_persistence_h0_norm": h0.max() / mpd,
                "mean_persistence_h1_norm": h1.mean() / mpd}
        for key, v in want.items():
            assert res[f"{role}_layer_{key}"] == pytest.approx(v, rel=1e-6), (role, key)


def test_persistence_entropy_evaluate_matches_gudhi(clouds, states):
    res = inv.run_task_on_states("topology_persistence_entropy", states)
    for role, X in zip(ROLES, clouds):
        _, h1 = _gudhi_lifetimes(X)
        pe = _entropy(h1)
        assert res[f"{role}_layer_pe_h1"] == pytest.approx(pe, rel=1e-6), role
        assert res[f"{role}_layer_pe_h1_norm"] == pytest.approx(pe / np.log2(len(h1)), rel=1e-6), role
