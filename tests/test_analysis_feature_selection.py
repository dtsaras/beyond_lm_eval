"""Regression tests for analysis predictor-column selection."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_correlations as corr  # noqa: E402
import bootstrap_lasso_r2 as boot  # noqa: E402


def _analysis_frame() -> pd.DataFrame:
    n = 12
    base = np.arange(n, dtype=float)
    return pd.DataFrame({
        "model": [f"m{i}" for i in range(n)],
        "family": ["fam"] * n,
        "hf_id": ["org/model"] * n,
        "dtype": ["float16"] * n,
        "purpose": ["base"] * n,
        "d_model": base + 1000,
        "n_layers": base + 10,
        "n_heads": base + 8,
        "vocab_size": base + 32000,
        "n_params_est": base + 1_000_000,
        "n_params_M": base + 100,
        "log_n_params": base + 20,
        "composite_benchmark": base / 10,
        "benchmark_mmlu_acc": base / 12,
        "benchmark_wikitext_perplexity": 100 - base,
        "geometry_svd.effective_rank_norm": base + 0.1,
        "interpretability_attention_entropy.mean": base + 0.2,
        "causality_ablation.loss_ablate_0.1": base + 0.15,
        "dynamics_sharpness.sam_perturbed_loss": base + 0.16,
        "consistency_icl_slope.mean_nll_overall": base + 0.17,
        "geometry_cka.n_layers": base + 0.18,
        "geometry_contextualization.per_layer.n_words_tracked.mean": base + 0.19,
        "geometry_svd.effective_rank.slope": base + 0.21,
        "geometry_perplexity.ppl_overall": base + 0.3,
        "consistency_calibration.ece": base + 0.4,
        "baseline_loss": base + 0.5,
        "mean_nll": base + 0.6,
        "ppl": base + 0.7,
        "bpc": base + 0.8,
        "interpretability_logit_lens.acc.mean": base + 0.9,
        "probing_linear_probe_accuracy": base + 1.0,
        "interpretability_prediction_entropy.mean": base + 1.1,
        "geometry_tokenizer_efficiency.bytes_per_token": base + 1.2,
        "task_category_y_variable": base + 1.3,
    })


def test_analyze_and_bootstrap_use_same_conservative_predictors():
    """The two downstream scripts must agree on the primary predictor set."""
    agg = _analysis_frame()
    metadata = pd.DataFrame({
        "feature": [
            "geometry_svd.effective_rank_norm",
            "interpretability_attention_entropy.mean",
            "task_category_y_variable",
        ],
        "category": ["geometry", "interpretability", "y_variable"],
    })

    analyze_features = corr._find_feature_columns(agg, metadata)
    X, bootstrap_features = boot._feat_matrix(
        agg,
        "composite_benchmark",
        {"composite_benchmark", "benchmark_mmlu_acc", "benchmark_wikitext_perplexity"},
        metadata,
    )

    assert list(X.columns) == bootstrap_features
    assert set(analyze_features) == set(bootstrap_features)
    assert set(analyze_features) == {
        "geometry_svd.effective_rank_norm",
        "interpretability_attention_entropy.mean",
    }


def test_predictor_selection_excludes_known_leaky_patterns_without_metadata():
    """Name-based guardrails should still work if feature metadata is absent."""
    agg = _analysis_frame()
    features = corr._find_feature_columns(agg)

    excluded = {
        "d_model",
        "n_layers",
        "n_heads",
        "vocab_size",
        "n_params_est",
        "n_params_M",
        "log_n_params",
        "composite_benchmark",
        "benchmark_mmlu_acc",
        "benchmark_wikitext_perplexity",
        "geometry_perplexity.ppl_overall",
        "consistency_calibration.ece",
        "baseline_loss",
        "mean_nll",
        "ppl",
        "bpc",
        "interpretability_logit_lens.acc.mean",
        "probing_linear_probe_accuracy",
        "interpretability_prediction_entropy.mean",
        "geometry_tokenizer_efficiency.bytes_per_token",
        "causality_ablation.loss_ablate_0.1",
        "dynamics_sharpness.sam_perturbed_loss",
        "consistency_icl_slope.mean_nll_overall",
        "geometry_cka.n_layers",
        "geometry_contextualization.per_layer.n_words_tracked.mean",
        "geometry_svd.effective_rank.slope",
    }
    assert excluded.isdisjoint(features)
    assert "geometry_svd.effective_rank_norm" in features
    assert "interpretability_attention_entropy.mean" in features


def test_feature_metadata_marks_non_primary_features():
    """Aggregation metadata should flag columns excluded from primary analysis."""
    from feature_selection import is_excluded_predictor

    assert is_excluded_predictor("geometry_cka.n_layers") is True
    assert is_excluded_predictor("geometry_svd.effective_rank_norm") is False
