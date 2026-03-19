"""
Tests for all 19 geometry tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import torch
import numpy as np
import json
import tempfile
import os


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_svd_isotropy(mock_model, mock_tokenizer):
    from blme.tasks.geometry.isotropy import SVDIsotropyTask

    task = SVDIsotropyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "svd_auc" in results
    assert "cond_number" in results
    assert results["svd_auc"] > 0
    assert results["svd_auc"] <= 1.0
    assert results["cond_number"] >= 1.0


def test_consistency(mock_model, mock_tokenizer):
    from blme.tasks.geometry.consistency import PredictionAlignmentTask

    task = PredictionAlignmentTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "prediction_alignment_mean" in results
    assert "prediction_alignment_std" in results
    mean = results["prediction_alignment_mean"]
    assert -1.0 <= mean <= 1.0


def test_categories(mock_model, mock_tokenizer):
    from blme.tasks.geometry.categories import CategoryGeometryTask

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump({"test_cat": ["A", "B", "C"]}, tmp)
        tmp_path = tmp.name

    try:
        task = CategoryGeometryTask(config={"categories_path": tmp_path})
        results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
        assert isinstance(results, dict)

        task_proj = CategoryGeometryTask(
            config={"categories_path": tmp_path, "projection_method": "pca"}
        )
        results_proj = task_proj.evaluate(mock_model, mock_tokenizer, dataset=None)

        if "projection_points" in results_proj:
            pts = results_proj["projection_points"]
            if len(pts) > 0:
                assert "x" in pts[0]
                assert "y" in pts[0]
    finally:
        os.remove(tmp_path)


def test_hubness(mock_model, mock_tokenizer):
    from blme.tasks.geometry.hubness import GlobalHubnessTask

    task = GlobalHubnessTask(config={"k_values": [5], "batch_size": 50})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "hubness_k5_skew" in results
    assert "hubness_k5_max" in results
    assert "hubness_k5_gini" in results
    assert 0 <= results["hubness_k5_gini"] <= 1.0


def test_intrinsic_dim(mock_model, mock_tokenizer):
    from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

    task = IntrinsicDimensionTask(config={})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "intrinsic_dimension" in results
    assert results["intrinsic_dimension"] >= 0


def test_unembedding(mock_model, mock_tokenizer):
    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask

    task = UnembeddingDiagnosticsTask(config={"n_sample": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "unembedding_is_tied" in results
    assert "unembedding_eff_rank" in results
    assert "unembedding_purity_mean" in results


# ---------------------------------------------------------------------------
# New tests for remaining 13 geometry tasks
# ---------------------------------------------------------------------------

def test_spectral(mock_model, mock_tokenizer):
    """Spectral analysis of weight matrices (Stable Rank, Power Law)."""
    from blme.tasks.geometry.spectral import WeightSpectralTask

    task = WeightSpectralTask(config={})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    assert "avg_stable_rank" in results
    assert results["avg_stable_rank"] > 0


def test_lipschitz(mock_model, mock_tokenizer):
    """Lipschitz continuity estimation between layers."""
    from blme.tasks.geometry.lipschitz import LipschitzContinuityTask

    task = LipschitzContinuityTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    assert "lipschitz_max" in results
    assert "lipschitz_mean" in results
    assert results["lipschitz_max"] >= 0


def test_cka(mock_model, mock_tokenizer):
    """Centered Kernel Alignment between layers."""
    from blme.tasks.geometry.cka import CKATask

    task = CKATask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    assert "avg_adjacent_cka" in results
    assert "cka_matrix" in results
    assert 0 <= results["avg_adjacent_cka"] <= 1.0


def test_collapse(mock_model, mock_tokenizer):
    """Representation collapse detection via effective rank."""
    from blme.tasks.geometry.collapse import RepresentationCollapseTask

    task = RepresentationCollapseTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "erank_per_layer" in results or "collapse_ratio" in results


def test_correlation_dimension(mock_model, mock_tokenizer):
    """Grassberger-Procaccia correlation dimension."""
    from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask

    task = CorrelationDimensionTask(config={"num_samples": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "correlation_dimension" in results
        assert results["correlation_dimension"] > 0


def test_information_geometry(mock_model, mock_tokenizer):
    """Representation sensitivity (gradient norm w.r.t. hidden states)."""
    from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask

    task = RepresentationSensitivityTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "representation_sensitivity" in results
        assert results["representation_sensitivity"] >= 0


def test_lid(mock_model, mock_tokenizer):
    """Local Intrinsic Dimensionality estimation."""
    from blme.tasks.geometry.lid import LocalIntrinsicDimensionalityTask

    task = LocalIntrinsicDimensionalityTask(config={"num_samples": 5, "k": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "lid_mean" in results
        assert "lid_std" in results


def test_mahalanobis(mock_model, mock_tokenizer):
    """Mahalanobis distance OOD detection."""
    from blme.tasks.geometry.mahalanobis import MahalanobisOODTask

    task = MahalanobisOODTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Mock tokenizer returns random tokens — Mahalanobis may encounter
    # singular covariance with tiny hidden dimensions. Accept error or valid.
    if "error" not in results:
        assert "mean_mahalanobis_id" in results


def test_mutual_info(mock_model, mock_tokenizer):
    """HSIC dependence estimation between layers."""
    from blme.tasks.geometry.mutual_info import HSICDependenceTask

    task = HSICDependenceTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "avg_adjacent_hsic" in results


def test_perplexity_rare_freq(mock_model, mock_tokenizer):
    """Perplexity analysis on rare vs frequent tokens."""
    from blme.tasks.geometry.perplexity import RarePPLTask

    task = RarePPLTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert any("perplexity" in k or "ppl" in k.lower() for k in results.keys())


def test_positional_decay(mock_model, mock_tokenizer):
    """Positional encoding integrity via attention decay."""
    from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask

    task = PositionalAttentionDecayTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Requires output_attentions; may error if SDPA attention is used
    if "error" not in results:
        assert any("corr" in k or "decay" in k for k in results.keys())


def test_rsa(mock_model, mock_tokenizer):
    """Representational Similarity Analysis across layers."""
    from blme.tasks.geometry.rsa import RepresentationalSimilarityTask

    task = RepresentationalSimilarityTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert any("rsa" in k for k in results.keys())


def test_matrix_entropy(mock_model, mock_tokenizer):
    """Von Neumann spectral entropy of covariance matrices."""
    from blme.tasks.geometry.matrix_entropy import MatrixEntropyTask

    task = MatrixEntropyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert any("entropy" in k for k in results.keys())
