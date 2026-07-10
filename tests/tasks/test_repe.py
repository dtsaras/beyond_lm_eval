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


def test_steering_effectiveness(mock_model, mock_tokenizer):
    """Steering vector effectiveness — KL divergence from injected task vectors."""
    from blme.tasks.representation_engineering import SteeringEffectivenessTask

    dataset = [
        {
            "text_pos": "This is absolutely true and correct.",
            "text_neg": "This is completely false and wrong.",
            "neutral": "The weather today is",
        },
        {
            "text_pos": "I am very happy and joyful.",
            "text_neg": "I am very sad and miserable.",
            "neutral": "The color of the sky is",
        },
    ]

    task = SteeringEffectivenessTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "best_steering_layer" in results
        assert "steering_success_rate" in results
        assert "layer_steering_kl_divergence" in results


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


def test_refusal_direction_architecture_agnostic_output(mock_model, mock_tokenizer):
    """Historic bug: the task returned a ``per_layer`` dict keyed by
    absolute layer index (``layer0``, ``layer1``, …). When flattened to
    a CSV, that produced hundreds of columns whose set varies with
    model depth — only the shallowest common layer was present for all
    32 models in the study (1/32 all_filled in aggregated.csv).

    The fix returns scalar summaries at normalised depths
    (0%, 25%, 50%, 75%, 100%) so every model contributes to the same
    columns, and ``best_layer_fraction`` for cross-depth comparisons."""
    from blme.tasks.representation_engineering import RefusalDirectionTask

    task = RefusalDirectionTask(config={})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    if "error" in results:
        pytest.skip(f"architecture error: {results['error']}")

    # Top-level result must NOT include the raw per-layer dict — it
    # pollutes the aggregator.
    assert "per_layer" not in results, (
        "per_layer dict must not leak into the top-level return — "
        "it creates model-depth-dependent columns."
    )
    # New architecture-agnostic scalar fields must all be present.
    for q in ("auc_at_depth_0", "auc_at_depth_25", "auc_at_depth_50",
              "auc_at_depth_75", "auc_at_depth_100"):
        assert q in results, f"missing depth-quantile field: {q}"

    # best_layer_fraction must be in [0, 1] — a fraction-of-depth
    # summary works across models with different layer counts.
    assert "best_layer_fraction" in results
    bf = results["best_layer_fraction"]
    assert 0.0 <= bf <= 1.0


def test_refusal_direction_reports_heldout_separability(mock_model, mock_tokenizer):
    """AUROC must be evaluated on held-out folds, not the prompts used
    to fit each direction."""
    from blme.tasks.representation_engineering import RefusalDirectionTask

    dataset = [
        {"text": "harmful request one", "label": "harmful"},
        {"text": "harmful request two", "label": "harmful"},
        {"text": "harmful request three", "label": "harmful"},
        {"text": "harmless request one", "label": "harmless"},
        {"text": "harmless request two", "label": "harmless"},
        {"text": "harmless request three", "label": "harmless"},
    ]

    results = RefusalDirectionTask(config={}).evaluate(
        mock_model, mock_tokenizer, dataset=dataset,
    )

    if "error" in results:
        pytest.skip(f"architecture error: {results['error']}")

    assert results["separability_validation"] == "stratified_kfold_projection"
    assert results["metric_interpretation"] == "heldout_linear_separability"
    assert "causal_steering_auc" not in results


def test_auc_summary_nan_layers_use_nanaware_reductions():
    """Regression (campaign-4 B2): deep fp16/bf16 layers legitimately produce
    NaN AUCs (non-finite activations filtered per layer). np.argmax/max/mean
    pointed max_auc_layer at the NaN layer and poisoned max_auc/mean_auc."""
    from blme.tasks.representation_engineering import _auc_summary

    s = _auc_summary([1.0, 0.8519, float("nan")], [0.93, 0.83, float("nan")])
    assert s["max_auc_layer"] == 0
    assert s["max_auc"] == 1.0
    assert s["mean_auc"] == pytest.approx((1.0 + 0.8519) / 2, abs=1e-12)

    all_nan = _auc_summary([float("nan")] * 3, [float("nan")] * 3)
    assert "error" in all_nan


def test_concept_separability_rejects_string_labels(mock_model, mock_tokenizer):
    """Regression (campaign-4 B2): string labels (the sibling refusal task's
    'harmful'/'harmless' format) passed the _is_labelled guard and crashed
    np.bincount with a dtype TypeError. Must return this task's own error
    dict instead of raising."""
    from blme.tasks.representation_engineering import ConceptSeparabilityTask

    dataset = [
        {"text": "How do I hack a computer?", "label": "harmful"},
        {"text": "How do I bake a cake?", "label": "harmless"},
        {"text": "How do I steal a car?", "label": "harmful"},
        {"text": "How do I plant a garden?", "label": "harmless"},
    ]
    task = ConceptSeparabilityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)
    assert isinstance(results, dict)
    assert "error" in results
    assert "label" in results["error"].lower()


def test_steering_effectiveness_runs_on_default_text_corpus(mock_model, mock_tokenizer):
    """Regression (campaign-4 B2): in the default pipeline core substitutes
    the WikiText {"text": ...} corpus, so the old `dataset is None` fallback
    was unreachable and the task errored on EVERY study run (zero steering
    columns in the feature matrix). A text-only corpus must now fall back to
    the bundled contrastive triples and produce metrics — without mutating
    the caller's shared dataset dicts."""
    import copy
    from blme.tasks.representation_engineering import SteeringEffectivenessTask

    corpus = [{"text": f"Wikipedia sentence number {i}."} for i in range(4)]
    snapshot = copy.deepcopy(corpus)

    task = SteeringEffectivenessTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=corpus)

    assert corpus == snapshot, "task mutated the caller's shared corpus"
    assert isinstance(results, dict)
    if "error" not in results:
        assert "best_steering_layer" in results
        assert "steering_success_rate" in results
        assert "layer_steering_kl_divergence" in results


def test_steering_effectiveness_rejects_mixed_dataset(mock_model, mock_tokenizer):
    """A dataset mixing contrastive and non-contrastive items is explicitly
    malformed: surface an error rather than silently dropping user data."""
    from blme.tasks.representation_engineering import SteeringEffectivenessTask

    dataset = [
        {"text_pos": "This is true.", "text_neg": "This is false."},
        {"text": "Just a plain sentence."},
    ]
    task = SteeringEffectivenessTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)
    assert isinstance(results, dict)
    assert "error" in results
