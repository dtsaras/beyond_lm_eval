"""
Sampling stability under temperature sampling.

For each prompt, sample N completions with temperature > 0 and measure
how often the samples agree on the first generated token. This is a
sampling-stability proxy, not reasoning-path self-consistency with
answer majority vote.

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
    """Sampling-stability proxy based on first generated token agreement."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Self-Consistency Sampling Analysis...")

        n_samples_per_prompt = self.config.get("n_samples_per_prompt", 8)
        temperature = self.config.get("temperature", 0.7)
        max_new_tokens = self.config.get("max_new_tokens", 8)
        num_prompts = self.config.get("num_prompts", len(_SELFC_PROMPTS))
        seed = self.config.get("seed", None)

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and "prompt" in dataset[0]
        ):
            prompts = [d["prompt"] for d in dataset[:num_prompts]]
        elif dataset is not None and isinstance(dataset, list) and dataset and isinstance(dataset[0], str):
            prompts = list(dataset)[:num_prompts]
        else:
            prompts = _SELFC_PROMPTS[:num_prompts]

        device = next(model.parameters()).device
        generator = None
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            try:
                generator = torch.Generator(device=device)
            except Exception:
                generator = torch.Generator()
            generator.manual_seed(seed)

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
                generate_kwargs = {
                    "input_ids": input_ids.expand(n_samples_per_prompt, -1),
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
                }
                if generator is not None:
                    generate_kwargs["generator"] = generator

                try:
                    outputs = model.generate(**generate_kwargs)
                except TypeError:
                    # Some generate implementations do not accept a
                    # generator kwarg; keep deterministic global seeding.
                    generate_kwargs.pop("generator", None)
                    if seed is not None:
                        torch.manual_seed(seed)
                    outputs = model.generate(**generate_kwargs)
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

        mean_agreement = float(np.mean(per_prompt_top1_rate))
        median_agreement = float(np.median(per_prompt_top1_rate))
        mean_entropy = float(np.mean(first_token_entropies))
        mean_uniqueness = float(np.mean(per_prompt_token_diversity))

        return {
            "diagnostic_semantics": "sampling_stability",
            "diagnostic_method": (
                "first-token agreement over sampled completions; not "
                "reasoning-path answer majority vote"
            ),
            "generation_seed": seed,
            "sampling_stability_mean_first_token_agreement": mean_agreement,
            "sampling_stability_median_first_token_agreement": median_agreement,
            "sampling_stability_mean_first_token_entropy": mean_entropy,
            "sampling_stability_mean_first_token_uniqueness": mean_uniqueness,
            # Legacy aliases retained for downstream compatibility.
            "mean_first_token_agreement": mean_agreement,
            "median_first_token_agreement": median_agreement,
            "mean_first_token_entropy": mean_entropy,
            "mean_first_token_uniqueness": mean_uniqueness,
            "n_prompts": len(per_prompt_top1_rate),
            "n_samples_per_prompt": n_samples_per_prompt,
            "temperature": temperature,
        }
