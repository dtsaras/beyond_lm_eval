import pytest
import torch

def test_logit_lens(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.logit_lens import LogitLensTask
    
    # Mock model needs layers attribute for detection
    # The fixture mock_model has self.model.layers
    
    task = LogitLensTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    # MockModel has 2 layers
    assert "layer0_acc" in results
    assert "layer1_acc" in results
    assert "layer0_entropy" in results

def test_attribution(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.attribution import ComponentAttributionTask
    
    task = ComponentAttributionTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "component_coherence_mean" in results
    # Coherence logic involves cosine similarity mean, so it is between -1 and 1
    assert -1.0 <= results["component_coherence_mean"] <= 1.0
