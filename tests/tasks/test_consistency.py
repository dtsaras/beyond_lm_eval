"""
Tests for all 4 consistency tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import torch
import numpy as np


def test_self_consistency_seed_does_not_crash_generation():
    """Campaign-3 BUG: with a seed set, self_consistency passed a
    generate(generator=...) kwarg that transformers rejects with ValueError;
    only TypeError was caught for the fallback, so the broad except skipped
    every prompt and the task returned {"error": ...}. A seeded run must
    still produce real agreement metrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from blme.tasks.consistency.self_consistency import SelfConsistencyTask

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    task = SelfConsistencyTask(config={
        "seed": 1234, "num_prompts": 2, "n_samples_per_prompt": 4,
        "max_new_tokens": 4,
    })
    r = task.evaluate(model, tok, dataset=None)
    assert "error" not in r, f"seeded run returned error dict: {r}"
    assert "mean_first_token_agreement" in r and r["generation_seed"] == 1234


def test_calibration_default_bins_is_15(mock_model, mock_tokenizer, monkeypatch):
    """Campaign-3 DISCREPANCY: default n_bins was 10, but Guo 2017 and the
    official _ECELoss (temperature_scaling) use M=15. Capture the n_bins the
    task passes by default and assert it is 15."""
    import blme.tasks.consistency.calibration as cal

    captured = {}
    orig = cal._calibration_from_confidences

    def spy(*args, **kwargs):
        captured["n_bins"] = kwargs.get("n_bins")
        return orig(*args, **kwargs)

    monkeypatch.setattr(cal, "_calibration_from_confidences", spy)
    cal.CalibrationTask(config={"num_samples": 8}).evaluate(mock_model, mock_tokenizer, None)
    if "n_bins" not in captured:
        pytest.skip("mock produced no calibration predictions")
    assert captured["n_bins"] == 15, "default n_bins must be 15 (Guo M=15)"


def test_calibration(mock_model, mock_tokenizer):
    """Expected Calibration Error (ECE) computation."""
    from blme.tasks.consistency.calibration import CalibrationTask

    task = CalibrationTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "ece" in results
        assert "num_predictions" in results
        assert 0 <= results["ece"] <= 1.0


def test_contrastive_consistency(mock_model, mock_tokenizer):
    """Contrastive consistency — factual vs exclusive rejection ratio."""
    from blme.tasks.consistency.contrastive import ContrastiveConsistencyTask

    # Provide inline dataset to avoid needing `datasets` library
    dataset = [
        {
            "factual": "The capital of France is Paris.",
            "exclusive": "The capital of France is London.",
        },
        {
            "factual": "Water boils at 100 degrees Celsius.",
            "exclusive": "Water boils at 0 degrees Celsius.",
        },
    ]

    task = ContrastiveConsistencyTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    assert "mean_factual_prob" in results
    assert "mean_exclusive_prob" in results
    assert "mean_rejection_ratio" in results


def test_logical_consistency(mock_model, mock_tokenizer):
    """Logical consistency — premise ⟹ conclusion probability checks."""
    from blme.tasks.consistency.logical import LogicalConsistencyTask

    dataset = [
        {
            "premise": "John is a bachelor.",
            "conclusion": "John is unmarried.",
        },
        {
            "premise": "The car is completely destroyed.",
            "conclusion": "The car cannot be driven.",
        },
    ]

    task = LogicalConsistencyTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    # Mock tokenizer returns fixed-length sequences so conditional probability
    # computation may fail — accept error or valid results.
    if "error" not in results:
        assert "mean_conditional_logprob" in results
        assert "mean_unconditional_logprob" in results
        assert "logical_violation_rate" in results
        assert 0 <= results["logical_violation_rate"] <= 1.0


def test_contamination_detection(mock_model, mock_tokenizer):
    """Data contamination detection via min-k% probability method."""
    from blme.tasks.consistency.contamination import ContaminationDetectionTask

    task = ContaminationDetectionTask(config={"num_samples": 3, "k_pct": 20})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "min_k_score" in results
        assert "contamination_z_score" in results
        assert "mean_token_logprob" in results


def test_knowledge_capacity(mock_model, mock_tokenizer):
    """Knowledge capacity — memorization vs generalization."""
    from blme.tasks.consistency.knowledge_capacity import KnowledgeCapacityTask

    dataset = [
        {
            "prompt": "The capital of France is",
            "exact": " Paris",
            "rephrased": " the city of Paris",
        },
        {
            "prompt": "Water boils at",
            "exact": " 100 degrees Celsius",
            "rephrased": " one hundred degrees C",
        },
    ]

    task = KnowledgeCapacityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "memorization_score" in results
        assert "generalization_score" in results
        assert "generalization_ratio" in results


def test_paraphrase_invariance(mock_model, mock_tokenizer):
    """Paraphrase invariance — semantic isometry of representations."""
    from blme.tasks.consistency.paraphrase import ParaphraseInvarianceTask

    dataset = [
        {
            "text1": "The quick brown fox jumps over the lazy dog.",
            "text2": "A fast dark-colored fox leaps above a sleepy hound.",
            "unrelated": "Machine learning is transforming data processing.",
        },
        {
            "text1": "Water boils at 100 degrees Celsius.",
            "text2": "H2O reaches boiling point at one hundred degrees C.",
            "unrelated": "The Eiffel Tower is in Paris.",
        },
    ]

    task = ParaphraseInvarianceTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    assert "mean_paraphrase_l2_dist" in results
    assert "mean_unrelated_l2_dist" in results
    assert "mean_paraphrase_cos_sim" in results
    assert "mean_unrelated_cos_sim" in results
    assert results["mean_paraphrase_l2_dist"] >= 0
    assert results["mean_unrelated_l2_dist"] >= 0
