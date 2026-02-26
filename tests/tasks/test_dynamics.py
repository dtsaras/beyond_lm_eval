import pytest
import torch

def test_trajectories(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.trajectories import MixtureTrajectoriesTask
    
    task = MixtureTrajectoriesTask(config={
        "prompts": [
            ("A", [" B"])
        ]
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "trajectory_containment_mean" in results
    assert "trajectory_smoothness_mean" in results
    assert results["trajectory_containment_mean"] >= 0

def test_interpolation(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask
    
    task = LatentInterpolationTask(config={
        "num_pairs": 2,
        "steps": 3
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "convexity_gap" in results
    assert "interp_entropy_0.0" in results
    assert "interp_entropy_0.5" in results

def test_stability(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.stability import NeighborhoodStabilityTask
    
    # Self-comparison (should be 1.0)
    task = NeighborhoodStabilityTask(config={
        "k": 5, 
        "n_sample": 10
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "stability_mean" in results
    assert results["stability_mean"] == 1.0 # Self comparison
