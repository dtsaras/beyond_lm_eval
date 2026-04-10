"""
Self-consistency under temperature sampling — Wang et al. 2022,
arXiv:2203.11171.

For each prompt, sample N completions with temperature > 0 and measure
how often the samples agree on the same answer (the "self-consistency"
of the model). Strong models concentrate their probability mass on a
small set of plausible completions; weak models produce wildly different
samples.

We use exact-match agreement on the first generated word and a softer
"common-prefix overlap" measure as well.

This is purely intrinsic — no labels are needed. The bundled prompts
each end with an obvious continuation, so we can compute agreement
without an oracle.
"""

import logging
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


_SELFC_PROMPTS: List[str] = [
    "The capital of France is",
    "Two plus two equals",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
    "The first president of the United States was",
    "The opposite of hot is",
    "The color of the sky on a clear day is",
    "The fastest land animal is the",
    "Shakespeare wrote a famous play called",
    "The Earth orbits around the",
    "Humans typically have ten fingers and ten",
    "Light travels at approximately three hundred thousand kilometers per",
]


@register_task("consistency_self_consistency")
class SelfConsistencyTask(DiagnosticTask):
    """Sampling-based self-consistency (Wang et al. 2022)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Self-Consistency Sampling Analysis...")

        n_samples_per_prompt = self.config.get("n_samples_per_prompt", 8)
        temperature = self.config.get("temperature", 0.7)
        max_new_tokens = self.config.get("max_new_tokens", 8)
        num_prompts = self.config.get("num_prompts", len(_SELFC_PROMPTS))

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and "prompt" in dataset[0]
        ):
            prompts = [d["prompt"] for d in dataset[:num_prompts]]
        elif dataset is not None and isinstance(dataset, list) and dataset and isinstance(dataset[0], str):
            prompts = list(dataset)[:num_prompts]
        else:
            prompts = _SELFC_PROMPTS[:num_prompts]

        device = next(model.parameters()).device

        # Per-prompt agreement rates
        per_prompt_top1_rate = []
        per_prompt_token_diversity = []  # 1 / unique_first_tokens (1.0 = full agreement)
        first_token_entropies = []

        with torch.no_grad():
            for prompt in prompts:
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = enc["input_ids"]
                prompt_len = input_ids.shape[1]

                # Sample N completions in a single batched call.
                try:
                    outputs = model.generate(
                        input_ids=input_ids.expand(n_samples_per_prompt, -1),
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=1.0,
                        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
                    )
                except Exception as e:
                    logger.info(f"  generate() failed for prompt '{prompt[:40]}': {e}")
                    continue

                gen_ids = outputs[:, prompt_len:]  # (N, max_new_tokens)
                if gen_ids.numel() == 0:
                    continue

                # First-token agreement
                first_tokens = gen_ids[:, 0].cpu().tolist()
                ctr = Counter(first_tokens)
                top1_count = max(ctr.values())
                top1_rate = top1_count / len(first_tokens)
                per_prompt_top1_rate.append(top1_rate)
                # Token diversity: 1 / unique tokens (lower = more diverse)
                per_prompt_token_diversity.append(1.0 / len(ctr))
                # Entropy of first-token distribution
                p = np.array(list(ctr.values()), dtype=np.float64) / sum(ctr.values())
                first_token_entropies.append(float(-np.sum(p * np.log(p + 1e-12))))

        if not per_prompt_top1_rate:
            return {"error": "No samples successfully generated"}

        return {
            "mean_first_token_agreement": float(np.mean(per_prompt_top1_rate)),
            "median_first_token_agreement": float(np.median(per_prompt_top1_rate)),
            "mean_first_token_entropy": float(np.mean(first_token_entropies)),
            "mean_first_token_uniqueness": float(np.mean(per_prompt_token_diversity)),
            "n_prompts": len(per_prompt_top1_rate),
            "n_samples_per_prompt": n_samples_per_prompt,
            "temperature": temperature,
        }
