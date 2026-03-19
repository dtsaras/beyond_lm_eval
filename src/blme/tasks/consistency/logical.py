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

        if dataset is None:
            dataset = [
                {"premise": "John is a bachelor.", "conclusion": "John is unmarried."},
                {"premise": "The car is completely destroyed.", "conclusion": "The car cannot be driven."},
                {"premise": "Paris is the capital of France.", "conclusion": "Paris is in France."},
                {"premise": "All mammals are warm-blooded.", "conclusion": "Dogs are warm-blooded."},
                {"premise": "It is raining heavily outside.", "conclusion": "The ground is wet."},
            ][:num_samples]

        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
             return {"error": "Need at least 1 sample with 'premise' and 'conclusion' keys"}

        if not all(k in samples[0] for k in ["premise", "conclusion"]):
             return {"error": "Dataset must contain 'premise' and 'conclusion' keys"}

        def get_conclusion_logprob(text, prompt_len):
            """Compute mean log prob of tokens after prompt_len."""
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            if ids.shape[1] <= prompt_len or ids.shape[1] < 2:
                return None
            outputs = model(ids)
            logits = outputs.logits
            # Shifted log probs: logits[t] predicts token[t+1]
            log_probs = F.log_softmax(logits, dim=-1)
            shift_log_probs = log_probs[0, :-1, :]
            shift_labels = ids[0, 1:]
            token_log_probs = torch.gather(shift_log_probs, 1, shift_labels.unsqueeze(1)).squeeze(1)
            # Only take log probs for conclusion tokens (after prompt_len - 1 in shifted array)
            conclusion_log_probs = token_log_probs[prompt_len - 1:]
            if conclusion_log_probs.numel() == 0:
                return None
            return conclusion_log_probs.mean().item()

        conditional_logprobs = []
        unconditional_logprobs = []
        violations = 0

        with torch.no_grad():
            for s in samples:
                premise = s["premise"]
                conclusion = s["conclusion"]

                # P(conclusion | premise): tokenize premise+conclusion, score conclusion part
                premise_ids = tokenizer.encode(premise, add_special_tokens=False)
                combined_text = premise + " " + conclusion
                combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)
                # prompt_len = number of tokens for the premise + space prefix
                # Re-encode to get exact boundary
                prompt_len = len(tokenizer.encode(premise + " ", add_special_tokens=False))

                cond_lp = get_conclusion_logprob(combined_text, prompt_len)

                # P(conclusion): tokenize conclusion alone, score all tokens
                uncond_lp = get_conclusion_logprob(conclusion, 0)

                if cond_lp is None or uncond_lp is None:
                    continue

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
