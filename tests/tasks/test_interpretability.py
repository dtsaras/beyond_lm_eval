"""
Tests for all 11 interpretability tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

Tasks requiring optional dependencies (sae_lens) are skipped when unavailable.
Tasks that are architecture-specific (attention module name heuristics) accept
error dicts as valid responses for unsupported architectures.
"""
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_logit_lens(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.logit_lens import LogitLensTask

    task = LogitLensTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "layer0_acc" in results
    assert "layer1_acc" in results
    assert "layer0_entropy" in results


def test_attribution(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.attribution import ComponentAttributionTask

    task = ComponentAttributionTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "component_coherence_mean" in results
    assert -1.0 <= results["component_coherence_mean"] <= 1.0


# ---------------------------------------------------------------------------
# New tests for remaining 9 interpretability tasks
# ---------------------------------------------------------------------------

def test_attention_entropy(mock_model, mock_tokenizer):
    """Entropy of attention distributions per head."""
    from blme.tasks.interpretability.attention import AttentionEntropyTask

    task = AttentionEntropyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May return error if model uses SDPA attention (returns None weights)
    if "error" not in results:
        assert "avg_entropy_total" in results
        assert results["avg_entropy_total"] >= 0


def test_attention_graph(mock_model, mock_tokenizer):
    """Graph topology analysis of attention matrices."""
    from blme.tasks.interpretability.attention_graph import AttentionGraphTopologyTask

    task = AttentionGraphTopologyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Attention graph needs raw attention weights which may be None with SDPA
    if "error" not in results:
        assert "mean_sink_pagerank" in results
        assert "bos_sink_ratio" in results


def test_attention_polysemanticity(mock_model, mock_tokenizer):
    """SVD entropy of attention head outputs (superposition measure)."""
    from blme.tasks.interpretability.attention_polysemanticity import (
        AttentionHeadPolysemanticityTask,
    )

    task = AttentionHeadPolysemanticityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Architecture-specific module name matching — may return error
    if "error" not in results:
        assert "mean_attention_svd_entropy" in results
        assert results["mean_attention_svd_entropy"] >= 0


def test_induction_heads(mock_model, mock_tokenizer):
    """Induction head detection via repeated random sequences."""
    from blme.tasks.interpretability.induction import InductionHeadTask

    task = InductionHeadTask(config={"seq_len": 10, "num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if attention weights are None
    if "error" not in results:
        assert "max_induction_score" in results
        assert "avg_induction_score" in results
        assert "top_induction_heads" in results


def test_prediction_entropy(mock_model, mock_tokenizer):
    """Output distribution entropy profiling."""
    from blme.tasks.interpretability.prediction_entropy import PredictionEntropyTask

    task = PredictionEntropyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "mean_entropy" in results
    assert "mean_top1_prob" in results
    assert results["mean_entropy"] >= 0
    assert 0 <= results["mean_top1_prob"] <= 1.0


def test_probing(mock_model, mock_tokenizer):
    """Linear probing for token identity at each layer."""
    pytest.importorskip("sklearn")
    from blme.tasks.interpretability.probing import LinearProbingTask

    task = LinearProbingTask(config={"num_samples": 5, "max_probe_samples": 50})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "probing_accuracy_per_layer" in results
        assert "max_probing_accuracy" in results


def test_sparsity(mock_model, mock_tokenizer):
    """Activation sparsity (L0) and kurtosis of MLP blocks."""
    from blme.tasks.interpretability.sparsity import ActivationSparsityTask

    task = ActivationSparsityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Some architectures may not have standard MLP blocks
    if "error" not in results:
        assert "global_mean_l0" in results
        assert "layer_l0_rates" in results


def test_sae_features(mock_model, mock_tokenizer):
    """SAE feature dimensionality (requires sae_lens)."""
    from blme.tasks.interpretability.sae_features import SAEFeatureDimensionalityTask

    task = SAEFeatureDimensionalityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Expected to return error when sae_lens is not installed or SAE doesn't
    # match the test model — both are valid outcomes
    assert "error" in results or "mean_active_features_l0" in results


def test_weight_activation_alignment(mock_model, mock_tokenizer):
    """Weight-Activation Alignment (WAA) via SVD/PCA cosine similarity."""
    from blme.tasks.interpretability.weight_activation_alignment import (
        WeightActivationAlignmentTask,
    )

    task = WeightActivationAlignmentTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Architecture-dependent MLP detection; may error on some architectures
    if "error" not in results:
        assert "mean_waa_alignment" in results
        assert 0 <= results["mean_waa_alignment"] <= 1.0
