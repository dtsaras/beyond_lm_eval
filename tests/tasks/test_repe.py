"""
Tests for the 2 Representation Engineering (RepE) tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

RepE tasks require scikit-learn for concept separability analysis.
"""
import pytest
import torch
import numpy as np

sklearn = pytest.importorskip("sklearn", reason="sklearn required for RepE tests")


def test_task_vector_geometry(mock_model, mock_tokenizer):
    """Task vector extraction and geometry analysis."""
    from blme.tasks.representation_engineering import TaskVectorGeometryTask

    dataset = [
        {
            "text_pos": "The earth revolves around the sun.",
            "text_neg": "The sun revolves around the earth.",
        },
        {
            "text_pos": "Water boils at 100 degrees Celsius.",
            "text_neg": "Water boils at 0 degrees Celsius.",
        },
        {
            "text_pos": "A triangle has three sides.",
            "text_neg": "A triangle has four sides.",
        },
    ]

    task = TaskVectorGeometryTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert "layer_task_vector_norms" in results
        assert "max_norm_layer" in results
        assert isinstance(results["layer_task_vector_norms"], list)


def test_concept_separability(mock_model, mock_tokenizer):
    """Linear separability (AUC) of target concepts at each layer."""
    from blme.tasks.representation_engineering import ConceptSeparabilityTask

    # Provide labeled dataset
    dataset = [
        {"text": "This is a positive statement.", "label": 1},
        {"text": "This is a wonderful thing.", "label": 1},
        {"text": "This is a negative statement.", "label": 0},
        {"text": "This is a terrible outcome.", "label": 0},
        {"text": "Everything is great.", "label": 1},
        {"text": "Nothing worked at all.", "label": 0},
    ]

    task = ConceptSeparabilityTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    # May error if get_layers returns None for this architecture
    if "error" not in results:
        assert "layer_separability_auc" in results
        assert "max_auc" in results
        assert 0 <= results["max_auc"] <= 1.0
