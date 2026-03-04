"""
Data Contamination Detection — Min-k% Probability Method
──────────────────────────────────────────────────────────────────────
Detects whether the model has memorized specific text from its training
data by analyzing the distribution of per-token log probabilities.

The Min-k% method (Shi et al., 2023) identifies contamination by checking
if the lowest-probability tokens in a passage are still unusually high —
a signature of memorized (rather than generalized) text.

References:
- "Detecting Pretraining Data from Large Language Models"
  (Shi et al., 2023). arXiv:2310.16789
"""

import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


@register_task("consistency_contamination")
class ContaminationDetectionTask(DiagnosticTask):
    """
    Detects potential data contamination using the Min-k% probability method.

    Computes per-token log probabilities and checks whether the bottom-k%
    tokens have unusually high probabilities, indicating memorized text.
    Returns contamination_score, min_k_pct_prob, and mean_token_logprob.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Data Contamination Detection (Min-k%%)...")
        num_samples = self.config.get("num_samples", 10)
        k_pct = self.config.get("k_pct", 20)  # bottom k% of tokens

        device = next(model.parameters()).device

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog. "
                         "This sentence is commonly used in typing tests."},
                {"text": "In machine learning, overfitting occurs when a model "
                         "learns the training data too well."},
            ] * max(1, num_samples // 2)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        all_min_k_probs = []
        all_mean_logprobs = []

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(device)
                input_ids = inputs["input_ids"]

                if input_ids.shape[1] < 3:
                    continue

                outputs = model(**inputs)
                logits = outputs.logits  # (1, T, V)

                # Shift: predict token t from context [0..t-1]
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                # Per-token log probabilities
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)  # (1, T-1)

                token_lps = token_log_probs[0].cpu().numpy()

                if len(token_lps) < 2:
                    continue

                # Mean log prob
                mean_lp = float(np.mean(token_lps))
                all_mean_logprobs.append(mean_lp)

                # Min-k%: take the bottom k% of token log probs
                k_count = max(1, int(len(token_lps) * k_pct / 100))
                sorted_lps = np.sort(token_lps)
                min_k_lps = sorted_lps[:k_count]
                min_k_mean = float(np.mean(min_k_lps))
                all_min_k_probs.append(min_k_mean)

        if not all_min_k_probs:
            return {"error": "No valid samples processed."}

        # Contamination score: higher min-k% prob = more likely memorized
        # We negate and normalize so higher = more contamination
        mean_min_k = float(np.mean(all_min_k_probs))
        mean_logprob = float(np.mean(all_mean_logprobs))

        # Contamination score: ratio of min-k mean to overall mean
        # Closer to 1.0 = more uniform probs = more likely memorized
        if mean_logprob != 0:
            contamination_score = float(mean_min_k / mean_logprob)
        else:
            contamination_score = 0.0

        return {
            "contamination_score": contamination_score,
            "min_k_pct_prob": mean_min_k,
            "mean_token_logprob": mean_logprob,
            "k_pct": k_pct,
            "num_samples_analyzed": len(all_min_k_probs),
        }
