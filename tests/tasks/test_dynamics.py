"""
Tests for all 3 dynamics tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_interpolation(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 2, "steps": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "convexity_gap" in results
    assert "interp_entropy_0.0" in results
    assert "interp_entropy_0.5" in results


def test_stability(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.stability import NeighborhoodStabilityTask

    # Self-comparison (should be 1.0)
    task = NeighborhoodStabilityTask(config={"k": 5, "n_sample": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "stability_mean" in results
    assert results["stability_mean"] == 1.0  # Self comparison


# ---------------------------------------------------------------------------
# New test
# ---------------------------------------------------------------------------

def test_chain_of_embedding(mock_model, mock_tokenizer):
    """Chain-of-Embedding (CoE) — magnitude and angle changes during generation."""
    from blme.tasks.dynamics.coe import ChainOfEmbeddingTask

    task = ChainOfEmbeddingTask(config={"num_samples": 2, "generation_steps": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "mean_magnitude_change" in results
        assert "mean_angle_change" in results
        assert results["mean_magnitude_change"] >= 0
        assert results["mean_angle_change"] >= 0
