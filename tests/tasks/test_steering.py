import pytest
import torch

def test_concept_steering(mock_model, mock_tokenizer):
    from blme.tasks.steering.concept_vectors import ConceptSteeringTask
    
    # Needs hidden_normalized
    task = ConceptSteeringTask(config={
        "method": "hidden_normalized",
        "scales": [1.0],
        "target_prompts": ["A"],
        "positive_prompts": ["Good"],
        "negative_prompts": ["Bad"]
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "shift_scale_1.0" in results
    assert results["shift_scale_1.0"] >= 0

def test_mixture_editing(mock_model, mock_tokenizer):
    from blme.tasks.steering.editing import MixtureEditingTask
    
    task = MixtureEditingTask(config={
        "prompts": ["test"]
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "edit_drop_kl" in results
    assert "edit_swap_kl" in results
    assert "edit_boost_kl" in results
    assert results["edit_drop_kl"] >= 0
