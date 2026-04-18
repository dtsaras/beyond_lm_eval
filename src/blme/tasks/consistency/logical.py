import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")

@register_task("consistency_logical")
class LogicalConsistencyTask(DiagnosticTask):
    """
    Measures Logical Consistency (A implies B) using conditional probability.
    Evaluates whether P(conclusion | premise) > P(conclusion) — knowing the
    premise should make the conclusion more likely if entailment holds.
    A violation occurs when conditioning on the premise *decreases* the
    probability of the conclusion.

    References:
    - "Measuring and Improving Consistency in Pretrained Language Models" (Elazar et al., 2021)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Logical Consistency Analysis...")
        num_samples = self.config.get("num_samples", 5)

        device = next(model.parameters()).device

        _BUNDLED = [
            {"premise": "John is a bachelor.",
             "conclusion": "John is unmarried."},
            {"premise": "The car is completely destroyed.",
             "conclusion": "The car cannot be driven."},
            {"premise": "Paris is the capital of France.",
             "conclusion": "Paris is in France."},
            {"premise": "All mammals are warm-blooded.",
             "conclusion": "Dogs are warm-blooded."},
            {"premise": "It is raining heavily outside.",
             "conclusion": "The ground is wet."},
        ]

        # Only accept dataset entries that carry premise/conclusion.
        usable = []
        if dataset is not None and isinstance(dataset, list):
            for item in dataset[:num_samples]:
                if isinstance(item, dict) and {"premise", "conclusion"} <= set(item):
                    usable.append(item)
        if len(usable) < 1:
            usable = _BUNDLED[:num_samples]

        samples = list(usable)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 (premise, conclusion) pair"}

        from ..common import score_continuation

        conditional_logprobs = []
        unconditional_logprobs = []
        violations = 0

        with torch.no_grad():
            for s in samples:
                premise = s["premise"]
                conclusion = s["conclusion"]

                # P(conclusion | premise). Use a shared helper so the
                # premise/conclusion boundary is found by character
                # offsets — independent tokenisation produces
                # inconsistent boundaries under BPE merges.
                cond = score_continuation(model, tokenizer, premise + " ", conclusion)

                # P(conclusion) alone — same helper with empty prompt
                # (the "log-prob of the sentence itself"). We prefix a
                # single space so the tokenised "start of sentence"
                # condition is identical to how the conclusion is
                # tokenised inside the conditional prompt.
                uncond = score_continuation(model, tokenizer, " ", conclusion)

                if cond is None or uncond is None:
                    continue

                # The helper returns mean NLL (positive). Convert to
                # log-probability (negative) to match the legacy API.
                cond_lp = -cond[0]
                uncond_lp = -uncond[0]

                conditional_logprobs.append(cond_lp)
                unconditional_logprobs.append(uncond_lp)

                # Violation: premise makes conclusion LESS likely
                if cond_lp < uncond_lp:
                    violations += 1

        if not conditional_logprobs:
            return {"error": "No valid samples processed."}

        return {
            "mean_conditional_logprob": float(np.mean(conditional_logprobs)),
            "mean_unconditional_logprob": float(np.mean(unconditional_logprobs)),
            "mean_lift": float(np.mean([c - u for c, u in zip(conditional_logprobs, unconditional_logprobs)])),
            "logical_violation_rate": float(violations / len(conditional_logprobs)),
        }
