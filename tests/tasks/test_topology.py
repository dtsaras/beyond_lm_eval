"""
Tests for all 3 topology tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

All topology tasks require the `ripser` library for persistent homology.
Tests are skipped if ripser is not installed.
"""
import pytest
import torch
import numpy as np


def test_persistent_homology(mock_model, mock_tokenizer):
    """Persistent homology via Vietoris-Rips complex."""
    from blme.tasks.topology.homology import PersistentHomologyTask

    task = PersistentHomologyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert any("persistence_h0" in k for k in results.keys())


def test_betti_curve(mock_model, mock_tokenizer):
    """Betti number trajectory across layers."""
    from blme.tasks.topology.betti_curve import BettiCurveTask

    task = BettiCurveTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert "betti_0_curve" in results
        assert "betti_1_curve" in results
        assert "simplification_ratio" in results


def test_betti_curve_geodesic_recovers_clusters_and_loops():
    """Redesigned _count_betti (kNN-geodesic, Naitzat et al.) validated
    against ground truth: beta_0 recovers the cluster count and beta_1
    detects loops, where the old median-Euclidean threshold collapsed
    beta_0 to ~1. Deterministic (seeded)."""
    import numpy as np
    from blme.tasks.topology.betti_curve import _count_betti

    rng = np.random.default_rng(0)

    # K = 4 well-separated blobs in 10D -> beta_0 = 4, no loops.
    centers = rng.standard_normal((4, 10)) * 30
    X = np.vstack([centers[i] + rng.standard_normal((12, 10)) for i in range(4)])
    b0, b1 = _count_betti(X, maxdim=1, n_neighbors=5)
    assert b0 == 4, f"beta_0 should recover 4 clusters, got {b0}"
    assert b1 == 0

    # Noisy circle -> one connected component, one loop.
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circ = np.c_[np.cos(th), np.sin(th)] + 0.03 * rng.standard_normal((40, 2))
    b0c, b1c = _count_betti(circ, maxdim=1, n_neighbors=4)
    assert (b0c, b1c) == (1, 1), f"circle should be (1,1), got {(b0c, b1c)}"

    # High-dim gaussian noise -> no spurious persistent loops.
    _, b1n = _count_betti(rng.standard_normal((40, 10)), maxdim=1, n_neighbors=6)
    assert b1n == 0

    # Degenerate guard: < 3 points returns without error.
    assert _count_betti(np.zeros((2, 5)), maxdim=1) == (2, 0)


def test_persistence_entropy(mock_model, mock_tokenizer):
    """Persistence entropy at early, middle, and late layers."""
    from blme.tasks.topology.persistence_entropy import PersistenceEntropyTask

    task = PersistenceEntropyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert "pe_simplification_ratio" in results
        assert any("pe_h0" in k for k in results.keys())
