"""
Knowledge Capacity — Memorization vs Generalization
──────────────────────────────────────────────────────────────────────
Compares token-level probability of exact factual completions versus
semantically equivalent rephrasings. A model that assigns similar
probability to both has generalized the knowledge; one that strongly
prefers the exact form has memorized it.

References:
- "Do Language Models Memorize or Generalize?" (Tirumala et al., 2022)
- "Quantifying Memorization Across Neural Language Models" (Carlini et al., 2023)
"""

import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


@register_task("consistency_knowledge_capacity")
class KnowledgeCapacityTask(DiagnosticTask):
    """
    Compares token-level probability of exact factual completions vs
    semantically equivalent rephrasings to distinguish memorization
    from generalization.

    Returns memorization_score, generalization_score, and
    generalization_ratio.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Knowledge Capacity (Memorization vs Generalization)...")
        num_samples = self.config.get("num_samples", 5)

        device = next(model.parameters()).device

        if dataset is None:
            # Paired: exact factual completion + semantically equivalent rephrasing
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
                {
                    "prompt": "The speed of light is approximately",
                    "exact": " 300,000 km/s",
                    "rephrased": " three hundred thousand kilometers per second",
                },
            ] * max(1, num_samples // 3 + 1)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        required_keys = {"prompt", "exact", "rephrased"}
        if not all(required_keys.issubset(s.keys()) for s in samples if isinstance(s, dict)):
            return {"error": "Dataset must contain 'prompt', 'exact', and 'rephrased' keys."}

        exact_logprobs = []
        rephrased_logprobs = []

        with torch.no_grad():
            for s in samples:
                prompt = s["prompt"]
                exact_text = prompt + s["exact"]
                rephrased_text = prompt + s["rephrased"]

                # Tokenize prompt to find where completion starts
                prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
                prompt_len = prompt_ids.shape[1]

                # Get log probs for exact completion
                exact_lp = self._completion_logprob(
                    model, tokenizer, exact_text, prompt_len, device
                )
                rephrased_lp = self._completion_logprob(
                    model, tokenizer, rephrased_text, prompt_len, device
                )

                if exact_lp is not None and rephrased_lp is not None:
                    exact_logprobs.append(exact_lp)
                    rephrased_logprobs.append(rephrased_lp)

        if not exact_logprobs:
            return {"error": "No valid samples processed."}

        mean_exact = float(np.mean(exact_logprobs))
        mean_rephrased = float(np.mean(rephrased_logprobs))

        # Memorization score: how much more likely is the exact form
        # Higher = more memorized (exact form strongly preferred)
        memorization_score = float(mean_exact - mean_rephrased)

        # Generalization score: average of rephrased log probs
        # Higher (less negative) = better generalization
        generalization_score = mean_rephrased

        # Ratio: closer to 1.0 = better generalization
        if mean_exact != 0:
            gen_ratio = float(mean_rephrased / mean_exact)
        else:
            gen_ratio = 0.0

        return {
            "memorization_score": memorization_score,
            "generalization_score": generalization_score,
            "generalization_ratio": gen_ratio,
            "mean_exact_logprob": mean_exact,
            "mean_rephrased_logprob": mean_rephrased,
        }

    @staticmethod
    def _completion_logprob(model, tokenizer, full_text, prompt_len, device):
        """Compute mean per-token log probability of the completion portion."""
        inputs = tokenizer(full_text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] <= prompt_len:
            return None

        outputs = model(**inputs)
        logits = outputs.logits  # (1, T, V)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (1, T-1)

        # Only consider completion tokens (after prompt)
        completion_lps = token_log_probs[0, prompt_len - 1:].cpu().numpy()

        if len(completion_lps) == 0:
            return None

        return float(np.mean(completion_lps))
