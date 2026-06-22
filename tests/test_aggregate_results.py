"""Regression tests for scripts/aggregate_results.py.

The aggregator must produce architecture-agnostic feature columns so
models with different layer counts are comparable. Per-layer results
keyed by absolute layer index create depth bias: only deep models fill
the ``layer_31`` column, so it ends up correlated with model size.

These tests pin the fix: layer-indexed dicts (``{layer_0, layer_1, ...}``
or ``{layer0, layer1, ...}``) are converted to normalised-depth
summaries (mean, std, slope, q25, q50, q75) rather than one column per
absolute index. Top-level ``layerN_metric`` keys emitted by
``interpretability_logit_lens`` are regrouped into per-metric lists and
summarised the same way.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import aggregate_results as agg  # noqa: E402


def _layer_dict(n):
    return {f"layer_{i}": float(i) for i in range(n)}


def test_layer_indexed_dict_becomes_depth_summary():
    """A dict keyed by layer_0, layer_1, ... should collapse to
    mean/std/slope/q25/q50/q75 — not one column per absolute layer."""
    task_result = {"per_layer_metric": _layer_dict(32)}
    flat = agg._flatten_dict(task_result, prefix="mytask")
    # Must NOT emit one column per absolute index.
    assert "mytask.per_layer_metric.layer_0" not in flat
    assert "mytask.per_layer_metric.layer_31" not in flat
    # Must emit normalised-depth summaries.
    assert "mytask.per_layer_metric.mean" in flat
    assert "mytask.per_layer_metric.std" in flat
    assert "mytask.per_layer_metric.slope" in flat
    assert "mytask.per_layer_metric.q25" in flat
    assert "mytask.per_layer_metric.q50" in flat
    assert "mytask.per_layer_metric.q75" in flat
    # Values should come from sorted-by-layer-index iteration.
    assert flat["mytask.per_layer_metric.mean"] == pytest.approx(15.5)
    assert flat["mytask.per_layer_metric.slope"] == pytest.approx(1.0)


def test_layer_indexed_dict_alternate_naming():
    """Accept ``layer0, layer1, ...`` without the underscore too."""
    task_result = {"per_layer": {f"layer{i}": float(i * 2) for i in range(16)}}
    flat = agg._flatten_dict(task_result, prefix="t")
    assert "t.per_layer.layer0" not in flat
    assert "t.per_layer.mean" in flat
    assert flat["t.per_layer.mean"] == pytest.approx(15.0)


def test_toplevel_layerN_metric_keys_regrouped():
    """interpretability_logit_lens emits ``layer0_acc, layer0_entropy, ...``
    at the top level. These must be regrouped per metric and summarised."""
    task_result = {
        "layer0_acc": 0.1, "layer1_acc": 0.2, "layer2_acc": 0.3,
        "layer0_entropy": 3.0, "layer1_entropy": 2.0, "layer2_entropy": 1.0,
        "best_layer_idx": 2,  # non-layer scalar should pass through
    }
    flat = agg._flatten_dict(task_result, prefix="lens")
    # Scalar non-layer key preserved.
    assert flat["lens.best_layer_idx"] == 2.0
    # Individual absolute-index keys dropped.
    assert "lens.layer0_acc" not in flat
    assert "lens.layer2_entropy" not in flat
    # Per-metric summaries emitted.
    assert "lens.acc.mean" in flat
    assert "lens.acc.slope" in flat
    assert "lens.entropy.mean" in flat
    assert "lens.entropy.slope" in flat
    # Numerically correct.
    assert flat["lens.acc.mean"] == pytest.approx(0.2)
    assert flat["lens.entropy.slope"] == pytest.approx(-1.0)


def test_layer_N_metric_topkey_regrouped():
    """causality_tracing emits ``layer_0_aie, layer_1_aie, ...`` — same
    problem, different separator. Must also be regrouped into lists."""
    task_result = {
        "layer_0_aie": 0.05, "layer_1_aie": 0.10, "layer_2_aie": 0.20,
        "layer_3_aie": 0.15, "layer_4_aie": 0.08,
        "max_aie": 0.20, "noise_std_applied": 0.13,
    }
    flat = agg._flatten_dict(task_result, prefix="trace")
    # Scalar non-layer keys preserved.
    assert flat["trace.max_aie"] == pytest.approx(0.20)
    assert flat["trace.noise_std_applied"] == pytest.approx(0.13)
    # Individual layer keys gone.
    assert "trace.layer_0_aie" not in flat
    assert "trace.layer_4_aie" not in flat
    # Regrouped summary present.
    assert "trace.aie.mean" in flat
    assert "trace.aie.q50" in flat
    assert flat["trace.aie.mean"] == pytest.approx(0.116)


def test_non_layer_dict_still_recurses():
    """Dicts that aren't per-layer must still recurse as before (e.g.
    {per_head_impacts: ..., something_else: ...} with scalar values)."""
    task_result = {
        "sub": {"foo": 1.0, "bar": 2.0},
    }
    flat = agg._flatten_dict(task_result, prefix="x")
    assert flat["x.sub.foo"] == 1.0
    assert flat["x.sub.bar"] == 2.0


def test_mixed_layer_and_nonlayer_keys_in_dict():
    """A dict with a mix of layer_N and other keys: only layer_N keys
    are regrouped; other keys pass through."""
    task_result = {
        "mixed": {
            "layer_0": 0.1, "layer_1": 0.2, "layer_2": 0.3,
            "threshold": 0.05,
        }
    }
    flat = agg._flatten_dict(task_result, prefix="t")
    assert "t.mixed.layer_0" not in flat
    assert flat["t.mixed.threshold"] == pytest.approx(0.05)
    assert "t.mixed.mean" in flat
    assert flat["t.mixed.mean"] == pytest.approx(0.2)


def test_bare_integer_keyed_dict_treated_as_layers():
    """``interpretability_waa.layer_waa_alignments`` emits a dict keyed
    by bare integer strings ('0', '1', ...). Those must be treated as
    layer indices and summarised like any other per-layer profile."""
    task_result = {
        "layer_waa_alignments": {str(i): float(i) * 0.1 for i in range(16)}
    }
    flat = agg._flatten_dict(task_result, prefix="waa")
    # Individual per-layer columns must be gone.
    assert "waa.layer_waa_alignments.0" not in flat
    assert "waa.layer_waa_alignments.15" not in flat
    # Summaries present.
    assert "waa.layer_waa_alignments.mean" in flat
    assert "waa.layer_waa_alignments.slope" in flat
    assert flat["waa.layer_waa_alignments.mean"] == pytest.approx(0.75)


def test_composite_benchmark_inverts_lower_is_better_metrics():
    """Perplexity/NLL benchmark columns must not reward worse scores."""
    bench = pd.DataFrame({
        "benchmark_arc_acc": [0.2, 0.8],
        "benchmark_wikitext_perplexity": [100.0, 10.0],
        "benchmark_lambada_nll": [5.0, 1.0],
    })

    composite = agg._compute_composite_benchmark(bench)

    assert composite.iloc[1] == pytest.approx(1.0)
    assert composite.iloc[0] == pytest.approx(0.0)


def test_composite_benchmark_drops_degenerate_columns():
    """Zero-range or single-value benchmark columns must not pollute the composite."""
    bench = pd.DataFrame({
        "benchmark_arc_acc": [0.5, 0.5, 0.8],
        "benchmark_wikitext_perplexity": [100.0, 50.0, 10.0],
        "benchmark_constant": [1.0, 1.0, 1.0],
        "benchmark_sparse": [float("nan"), float("nan"), 0.3],
    })

    composite = agg._compute_composite_benchmark(bench)

    # Best model: high acc, low perplexity; constant/sparse columns ignored.
    assert composite.iloc[2] == pytest.approx(1.0)
    assert composite.iloc[0] < composite.iloc[2]
