import pytest
import torch
import numpy as np
from blme.tasks.geometry.isotropy import SVDIsotropyTask
from blme.tasks.geometry.consistency import ConsistencyTask

def test_svd_isotropy(mock_model, mock_tokenizer):
    task = SVDIsotropyTask(config={"num_samples": 5})
    
    # Run on mock model
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    # Check structure
    assert "svd_auc" in results
    assert "cond_number" in results
    
    # Check Values
    # Mock model outputs random noise -> SVD usually stable
    assert results["svd_auc"] > 0
    assert results["svd_auc"] <= 1.0 # Normalized by sum of squares
    assert results["cond_number"] >= 1.0

def test_consistency(mock_model, mock_tokenizer):
    task = ConsistencyTask(config={"num_samples": 5})
    
    # Run on mock model
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    # Check structure
    assert "cosine_consistency_mean" in results
    assert "cosine_consistency_std" in results
    
    # Check Values
    # Cosine is between -1 and 1
    mean = results["cosine_consistency_mean"] 
    assert -1.0 <= mean <= 1.0

def test_categories(mock_model, mock_tokenizer):
    from blme.tasks.geometry.categories import CategoryGeometryTask
    import tempfile
    import json
    import os
    
    # Create temp categories file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        json.dump({"test_cat": ["A", "B", "C"]}, tmp)
        tmp_path = tmp.name
        
    try:
        # Standard run
        task = CategoryGeometryTask(config={"categories_path": tmp_path})
        results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
        assert isinstance(results, dict)
        
        # Projection run
        task_proj = CategoryGeometryTask(config={
            "categories_path": tmp_path, 
            "projection_method": "pca"
        })
        results_proj = task_proj.evaluate(mock_model, mock_tokenizer, dataset=None)
        
        # Check if projection keys exist (mock model might yield empty points if filtering fails, but key should exist or error)
        # With current mock tokenizer, duplicates might be filtered out, so maybe points list is small
        if "projection_points" in results_proj:
            pts = results_proj["projection_points"]
            if len(pts) > 0:
                assert "x" in pts[0]
                assert "y" in pts[0]
            
    finally:
        os.remove(tmp_path)

def test_alignment(mock_model, mock_tokenizer):
    from blme.tasks.geometry.alignment import AlignmentResidualTask
    task = AlignmentResidualTask(config={"num_samples": 5, "k_values": [2]})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "k2_l2_mean" in results
    assert "k2_cos_mean" in results
    assert results["k2_l2_mean"] >= 0

def test_substitution(mock_model, mock_tokenizer):
    from blme.tasks.geometry.alignment import SubstitutionConsistencyTask
    task = SubstitutionConsistencyTask(config={"num_samples": 5, "k": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    
    assert "substitution_top1_agreement" in results
    assert "substitution_kl_mean" in results
    assert 0.0 <= results["substitution_top1_agreement"] <= 1.0

def test_hubness(mock_model, mock_tokenizer):
    from blme.tasks.geometry.hubness import GlobalHubnessTask
    # k must be < vocab size (100)
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
