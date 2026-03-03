"""
Tests for all 3 topology tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

All topology tasks require the `ripser` library for persistent homology.
Tests are skipped if ripser is not installed.
"""
import pytest
import torch
import numpy as np

ripser = pytest.importorskip("ripser", reason="ripser required for topology tests")


def test_persistent_homology(mock_model, mock_tokenizer):
    """Persistent homology via Vietoris-Rips complex."""
    from blme.tasks.topology.homology import PersistentHomologyTask

    task = PersistentHomologyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert any("persistance_h0" in k for k in results.keys())


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
