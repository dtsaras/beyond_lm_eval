import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


class BatchDict(dict):
    def to(self, _device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class TinyTokenizer:
    vocab_size = 32
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None, **kwargs):
        texts = list(text) if isinstance(text, (list, tuple)) else [text]
        lengths = [max(3, len(str(t).split()) + 2) for t in texts]
        if max_length is not None:
            lengths = [min(length, max_length) for length in lengths]
        width = max(lengths)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1) % self.vocab_size
            mask[row, :length] = 1
        return BatchDict({"input_ids": ids, "attention_mask": mask})

    def encode(self, text, return_tensors=None, **_kwargs):
        length = max(1, len(str(text).split()) + 1)
        ids = torch.arange(1, length + 1).unsqueeze(0)
        if return_tensors == "pt":
            return ids
        return ids[0].tolist()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **_kwargs):
        batch, width = input_ids.shape
        vocab = 32
        logits = torch.zeros((batch, width, vocab), device=input_ids.device)
        if not output_hidden_states:
            return SimpleNamespace(logits=logits)
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def test_knowledge_capacity_reports_likelihood_ratio_not_negative_logprob_ratio(monkeypatch):
    from blme.tasks.consistency.knowledge_capacity import KnowledgeCapacityTask

    def fake_completion_logprob(_model, _tokenizer, full_text, _prompt_len, _device):
        return -4.0 if "city of Paris" in full_text else -2.0

    monkeypatch.setattr(KnowledgeCapacityTask, "_completion_logprob", staticmethod(fake_completion_logprob))

    result = KnowledgeCapacityTask(config={"num_samples": 1}).evaluate(
        TinyModel(),
        TinyTokenizer(),
        [{"prompt": "The capital of France is", "exact": " Paris", "rephrased": " the city of Paris"}],
    )

    assert result["diagnostic_semantics"] == "memorization_vs_paraphrase_likelihood"
    assert result["memorization_likelihood_delta"] == pytest.approx(2.0)
    assert result["paraphrase_probability_ratio"] == pytest.approx(math.exp(-2.0))
    assert 0 < result["generalization_ratio"] < 1
    assert "capacity" in result["diagnostic_warning"].lower()


def test_membership_default_reports_separability_proxy_with_warning(monkeypatch):
    import blme.tasks.consistency.membership_inference as module
    from blme.tasks.consistency.membership_inference import MembershipInferenceTask

    def fake_nll(_model, _tokenizer, text, _device):
        if text in module._MEMBERS:
            return 1.0
        if text in module._NON_MEMBERS:
            return 2.0
        return 3.0

    monkeypatch.setattr(module, "_compute_nll", fake_nll)

    result = MembershipInferenceTask(config={}).evaluate(TinyModel(), TinyTokenizer(), dataset=None)

    assert result["score_semantics"] == "domain_frequency_separability_proxy"
    assert result["is_calibrated_membership_inference"] is False
    assert result["separability_auroc"] == result["mia_auroc"]
    assert "not membership inference" in result["diagnostic_warning"].lower()


def test_membership_can_require_comparable_member_nonmember_pairs(monkeypatch):
    import blme.tasks.consistency.membership_inference as module
    from blme.tasks.consistency.membership_inference import MembershipInferenceTask

    monkeypatch.setattr(module, "_compute_nll", lambda *_args, **_kwargs: 1.0)

    dataset = [
        {"text": "member a", "label": "member"},
        {"text": "member b", "label": "member"},
        {"text": "nonmember a", "label": "non_member"},
        {"text": "nonmember b", "label": "non_member"},
    ]
    result = MembershipInferenceTask(
        config={"require_comparable_membership_data": True}
    ).evaluate(TinyModel(), TinyTokenizer(), dataset=dataset)

    assert "error" in result
    assert "comparable" in result["error"].lower()


def test_contamination_unlabeled_results_are_score_only():
    from blme.tasks.consistency.contamination import ContaminationDetectionTask

    result = ContaminationDetectionTask(config={"num_samples": 2}).evaluate(
        TinyModel(),
        TinyTokenizer(),
        dataset=[{"text": "sample one"}, {"text": "sample two"}],
    )

    assert result["score_semantics"] == "uncalibrated_min_k_score_only"
    assert result["is_calibrated_detection"] is False
    assert "score-only" in result["diagnostic_warning"].lower()


def test_contamination_labeled_results_include_calibration_threshold():
    from blme.tasks.consistency.contamination import ContaminationDetectionTask

    result = ContaminationDetectionTask(config={"num_samples": 4}).evaluate(
        TinyModel(),
        TinyTokenizer(),
        dataset=[
            {"text": "member sample one", "label": "contaminated"},
            {"text": "member sample two", "label": 1},
            {"text": "heldout sample one", "label": "clean"},
            {"text": "heldout sample two", "label": 0},
        ],
    )

    assert result["score_semantics"] == "calibrated_min_k_detection"
    assert result["is_calibrated_detection"] is True
    assert "calibrated_threshold" in result
    assert "calibrated_auroc" in result


def test_self_consistency_uses_seeded_sampling_stability_semantics():
    from blme.tasks.consistency.self_consistency import SelfConsistencyTask

    class GeneratingModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.generators = []

        def generate(self, input_ids, **kwargs):
            self.generators.append(kwargs.get("generator"))
            new_tokens = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype)
            return torch.cat([input_ids, new_tokens], dim=1)

    model = GeneratingModel()
    result = SelfConsistencyTask(
        config={"seed": 123, "num_prompts": 1, "n_samples_per_prompt": 3, "max_new_tokens": 1}
    ).evaluate(model, TinyTokenizer(), dataset=["Two plus two equals"])

    assert result["diagnostic_semantics"] == "sampling_stability"
    assert result["generation_seed"] == 123
    assert model.generators[0] is not None
    assert "Wang" not in result["diagnostic_method"]


def test_proxy_tasks_expose_clarified_output_keys(monkeypatch):
    import blme.tasks.common as common
    from blme.tasks.consistency.format_robustness import FormatRobustnessTask
    from blme.tasks.consistency.logical import LogicalConsistencyTask
    from blme.tasks.consistency.paraphrase import ParaphraseInvarianceTask
    from blme.tasks.consistency.position_sensitivity import PositionSensitivityTask

    monkeypatch.setattr(common, "score_continuation", lambda *_args, **_kwargs: (1.0, 1, [1]))

    model = TinyModel()
    tokenizer = TinyTokenizer()

    logical = LogicalConsistencyTask(config={"num_samples": 1}).evaluate(
        model,
        tokenizer,
        [{"premise": "A implies B.", "conclusion": "B follows."}],
    )
    assert "conditional_likelihood_lift" in logical
    assert logical["logical_violation_rate"] == logical["premise_decreases_conclusion_likelihood_rate"]

    position = PositionSensitivityTask(config={"num_samples": 1, "positions": [0.0, 0.5, 1.0]}).evaluate(
        model,
        tokenizer,
        [{"passage": "one two three four five", "fact": "needle fact", "recall": " recall answer"}],
    )
    assert "lost_in_middle_nll_depth" in position
    assert position["u_curve_depth"] == position["lost_in_middle_nll_depth"]

    fmt = FormatRobustnessTask(config={"num_samples": 1}).evaluate(
        model,
        tokenizer,
        [{"question": "What is two plus two", "answer": "4"}],
    )
    assert "format_nll_sensitivity" in fmt
    assert "format_top1_disagreement_rate" in fmt

    para = ParaphraseInvarianceTask(config={"num_samples": 1}).evaluate(
        model,
        tokenizer,
        [{"text1": "short text", "text2": "short paraphrase text", "unrelated": "very unrelated longer text"}],
    )
    assert para["diagnostic_semantics"] == "last_token_representation_distance_proxy"
    assert "representation_paraphrase_l2_dist" in para
    assert "representation_distance_ratio_l2" in para


def test_contamination_in_sample_calibration_is_labeled(monkeypatch):
    from blme.tasks.consistency.contamination import ContaminationDetectionTask

    def fake_forward(_self, input_ids, **_kwargs):
        batch, width = input_ids.shape
        logits = torch.zeros((batch, width, 32), device=input_ids.device)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(TinyModel, "forward", fake_forward)

    dataset = [
        {"text": "clean sample one", "label": 0},
        {"text": "clean sample two", "label": 0},
        {"text": "dirty sample one", "label": 1},
        {"text": "dirty sample two", "label": 1},
    ]
    result = ContaminationDetectionTask(config={"num_samples": 4}).evaluate(
        TinyModel(), TinyTokenizer(), dataset,
    )
    assert result["calibration_mode"] == "in_sample"
    assert "in_sample_threshold" in result
    assert "calibration_warning" in result


def test_self_consistency_is_deterministic_with_seed():
    from blme.tasks.consistency.self_consistency import SelfConsistencyTask

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, return_tensors="pt"):
            return Batch({"input_ids": torch.tensor([[10, 11]])})

    class FakeModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def generate(self, **kwargs):
            gen = kwargs.get("generator")
            seed = gen.initial_seed() if gen is not None else 0
            torch.manual_seed(seed)
            return torch.tensor([
                [10, 11, 7, 8],
                [10, 11, 7, 9],
            ])

    model = FakeModel()
    cfg = {"num_prompts": 1, "n_samples_per_prompt": 2, "seed": 99}
    task = SelfConsistencyTask(config=cfg)
    r1 = task.evaluate(model, Tok(), dataset=["prompt"])
    r2 = task.evaluate(model, Tok(), dataset=["prompt"])
    assert r1["mean_first_token_agreement"] == r2["mean_first_token_agreement"]
    assert r1["generation_seed"] == 99
