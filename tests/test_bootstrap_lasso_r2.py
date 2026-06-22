"""Regression tests for scripts/bootstrap_lasso_r2.py.

The bootstrap script computes out-of-bag LASSO and baseline R² CIs
over synthetic resamples. The tests pin the essential behaviour:

- ``_oob_lasso_r2`` holds out rows not in the bootstrap sample and
  returns a finite R² when ≥ 3 rows are out-of-bag.
- ``_oob_baseline_r2`` behaves the same for the 1-D log(N) baseline.
- The feature-matrix selector drops benchmark + metadata columns and
  keeps only numeric features with > 1 unique value.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import bootstrap_lasso_r2 as boot  # noqa: E402


def _synthetic_aggregated(n_models=16, n_feats=5, noise=0.2, seed=0):
    """Build a minimal aggregated-CSV-like frame for testing."""
    rng = np.random.default_rng(seed)
    families = np.array(["famA"] * (n_models // 2) + ["famB"] * (n_models - n_models // 2))
    log_N = rng.uniform(2, 4, n_models)
    # Composite is roughly linear in log_N + one informative feature
    feat0 = rng.normal(0, 1, n_models)
    y = 0.4 * log_N + 0.3 * feat0 + rng.normal(0, noise, n_models)
    data = {
        "model": [f"m{i}" for i in range(n_models)],
        "family": families,
        "log_n_params": log_N,
        "composite_benchmark": y,
        # Features
        "f.feat0": feat0,
    }
    for k in range(1, n_feats):
        data[f"f.feat{k}"] = rng.normal(0, 1, n_models)
    return pd.DataFrame(data)


def test_oob_lasso_returns_finite_r2():
    """Given an OOB split with ≥ 3 held-out rows, return a finite R²."""
    df = _synthetic_aggregated()
    X = df[[c for c in df.columns if c.startswith("f.")]].values.astype(np.float64)
    y = df["composite_benchmark"].values.astype(np.float64)
    # Deterministic in-bag: first 12 rows
    in_bag = np.arange(12)
    r2 = boot._oob_lasso_r2(X, y, in_bag)
    assert np.isfinite(r2)


def test_oob_baseline_returns_finite_r2():
    df = _synthetic_aggregated()
    y = df["composite_benchmark"].values.astype(np.float64)
    z = df["log_n_params"].values.astype(np.float64)
    in_bag = np.arange(12)
    r2 = boot._oob_baseline_r2(z, y, in_bag)
    assert np.isfinite(r2)


def test_oob_lasso_nan_when_too_few_oob():
    """If the bootstrap happens to cover every model, OOB R² is NaN."""
    df = _synthetic_aggregated()
    X = df[[c for c in df.columns if c.startswith("f.")]].values.astype(np.float64)
    y = df["composite_benchmark"].values.astype(np.float64)
    in_bag = np.arange(len(y))  # every row in-bag → 0 OOB
    r2 = boot._oob_lasso_r2(X, y, in_bag)
    assert np.isnan(r2)


def test_feat_matrix_drops_benchmarks_and_meta():
    """``_feat_matrix`` should keep only the f.* columns."""
    df = _synthetic_aggregated()
    bench_names = {"composite_benchmark"}
    X, feats = boot._feat_matrix(df, "composite_benchmark", bench_names)
    assert all(f.startswith("f.") for f in feats)
    assert "model" not in feats
    assert "family" not in feats
    assert "log_n_params" not in feats
    assert "composite_benchmark" not in feats


def test_feat_matrix_keeps_only_numeric_with_variance():
    """Constant-valued feature columns must be dropped."""
    df = _synthetic_aggregated()
    df["f.constant"] = 7.0
    df["f.string"] = "abc"
    X, feats = boot._feat_matrix(df, "composite_benchmark", {"composite_benchmark"})
    assert "f.constant" not in feats
    assert "f.string" not in feats
    assert "f.feat0" in feats


def test_oob_bootstrap_signal_stronger_than_noise():
    """With a deliberately-predictable target, LASSO should beat chance
    on average across bootstraps (median OOB R² > 0)."""
    rng = np.random.default_rng(1)
    df = _synthetic_aggregated(n_models=40, n_feats=10, noise=0.1, seed=1)
    X = df[[c for c in df.columns if c.startswith("f.")]].values.astype(np.float64)
    y = df["composite_benchmark"].values.astype(np.float64)
    r2s = []
    for _ in range(30):
        in_bag = rng.choice(len(y), size=len(y), replace=True)
        r2 = boot._oob_lasso_r2(X, y, in_bag)
        if np.isfinite(r2):
            r2s.append(r2)
    assert len(r2s) >= 20
    # Signal is informative → median OOB R² should be positive
    assert np.median(r2s) > 0.0


class _RecordingLasso(RegressorMixin, BaseEstimator):
    """Tiny estimator that verifies preprocessing happened inside the fold."""

    fit_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        _RecordingLasso.fit_calls += 1
        assert not np.isnan(X).any()
        assert np.allclose(X.mean(axis=0), 0.0, atol=1e-12)
        self.prediction_ = float(np.mean(y))
        return self

    def predict(self, X):
        assert not np.isnan(X).any()
        return np.full(X.shape[0], self.prediction_)


def _raw_matrix_with_missing_values():
    X = np.array([
        [0.0, 2.0],
        [1.0, np.nan],
        [2.0, 4.0],
        [50.0, 6.0],
        [4.0, np.nan],
        [5.0, 10.0],
        [6.0, 12.0],
        [7.0, 14.0],
        [8.0, 16.0],
        [9.0, 18.0],
        [10.0, 20.0],
        [11.0, 22.0],
    ])
    y = np.linspace(0.0, 1.0, len(X))
    return X, y


def test_lasso_loo_preprocesses_inside_each_fold(monkeypatch):
    X, y = _raw_matrix_with_missing_values()
    _RecordingLasso.fit_calls = 0
    monkeypatch.setattr(boot, "LassoCV", _RecordingLasso)

    r2 = boot._lasso_loo(X, y)

    assert np.isfinite(r2)
    assert _RecordingLasso.fit_calls == len(y)


def test_lasso_lofo_preprocesses_inside_each_family_fold(monkeypatch):
    X, y = _raw_matrix_with_missing_values()
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    _RecordingLasso.fit_calls = 0
    monkeypatch.setattr(boot, "LassoCV", _RecordingLasso)

    r2 = boot._lasso_lofo(X, y, groups)

    assert np.isfinite(r2)
    assert _RecordingLasso.fit_calls == 3


def test_oob_lasso_preprocesses_inside_in_bag_split(monkeypatch):
    X, y = _raw_matrix_with_missing_values()
    in_bag = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7])
    _RecordingLasso.fit_calls = 0
    monkeypatch.setattr(boot, "LassoCV", _RecordingLasso)

    r2 = boot._oob_lasso_r2(X, y, in_bag)

    assert np.isfinite(r2)
    assert _RecordingLasso.fit_calls == 1
