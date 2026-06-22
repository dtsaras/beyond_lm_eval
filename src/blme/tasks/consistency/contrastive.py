import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")

@register_task("consistency_contrastive")
class ContrastiveConsistencyTask(DiagnosticTask):
    """
    Measures a CounterFact-style negative-rejection proxy.

    Evaluates whether the model assigns lower probability to a mutually
    exclusive alternative than to a factual target under the same prompt.
    The fallback data use the counterfact-tracing split associated with
    Meng et al. 2022 (ROME); the metric is a BLME likelihood diagnostic,
    not a full benchmark evaluation.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Contrastive Consistency Analysis...")
        num_samples = self.config.get("num_samples", 3)
        
        device = next(model.parameters()).device
        
        def _triples_from_pair(item):
            """Normalise to a (prompt, target_true, target_false)
            triple regardless of input shape. Accepts either the new
            ``{prompt, target_true, target_false}`` form or the legacy
            ``{factual, exclusive}`` form by reconstructing the common
            prefix.
            """
            if {"prompt", "target_true", "target_false"} <= set(item):
                return item["prompt"], item["target_true"], item["target_false"]
            if {"factual", "exclusive"} <= set(item):
                a, b = item["factual"], item["exclusive"]
                i = 0
                while i < len(a) and i < len(b) and a[i] == b[i]:
                    i += 1
                return a[:i], a[i:], b[i:]
            return None

        _BUNDLED_TRIPLES = [
            {"prompt": "The capital of France is",
             "target_true": " Paris.",
             "target_false": " London."},
            {"prompt": "Water boils at",
             "target_true": " 100 degrees Celsius.",
             "target_false": " 0 degrees Celsius."},
            {"prompt": "A triangle has",
             "target_true": " three sides.",
             "target_false": " four sides."},
        ]

        usable = []
        if dataset is not None and isinstance(dataset, list):
            for item in dataset[:num_samples]:
                if not isinstance(item, dict):
                    continue
                if {"prompt", "target_true", "target_false"} <= set(item) or \
                   {"factual", "exclusive"} <= set(item):
                    usable.append(item)

        # If the input dataset doesn't carry contrastive triples (which
        # is the case for the generic cache corpus), fall back to the
        # counterfact-tracing split or the bundled examples. This keeps
        # the task useful under BLME's default pipeline — the historic
        # implementation did this too but the rewrite accidentally
        # dropped the fallback.
        if len(usable) < 1:
            try:
                from datasets import load_dataset
                dset = load_dataset(
                    "NeelNanda/counterfact-tracing", split="train",
                )
                usable = []
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    usable.append({
                        "prompt": item["prompt"],
                        "target_true": item["target_true"],
                        "target_false": item["target_false"],
                    })
            except Exception as e:
                logger.info(
                    f"Warning: counterfact-tracing unavailable ({type(e).__name__}); "
                    "using bundled triples."
                )
                usable = _BUNDLED_TRIPLES[:num_samples]

        samples = list(usable)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        from ..common import score_continuation

        factual_probs = []
        exclusive_probs = []
        contrast_ratios = []

        with torch.no_grad():
            for s in samples:
                triple = _triples_from_pair(s)
                if triple is None:
                    continue
                prompt, tgt_true, tgt_false = triple
                # Score only the target tokens given the shared prompt —
                # historic code scored the entire sequence including
                # the prompt, so the metric diluted with prompt length
                # and varied by tokeniser vocabulary.
                true_res = score_continuation(model, tokenizer, prompt, tgt_true)
                false_res = score_continuation(model, tokenizer, prompt, tgt_false)
                if true_res is None or false_res is None:
                    continue
                # score_continuation returns mean NLL (positive) →
                # convert to per-token probability via exp(-NLL).
                p_factual = float(np.exp(-true_res[0]))
                p_exclusive = float(np.exp(-false_res[0]))
                factual_probs.append(p_factual)
                exclusive_probs.append(p_exclusive)
                if p_factual > 0:
                    contrast_ratios.append(p_exclusive / p_factual)
                else:
                    contrast_ratios.append(1.0)
                    
        return {
            "mean_factual_prob": float(np.mean(factual_probs)),
            "mean_exclusive_prob": float(np.mean(exclusive_probs)),
            "mean_rejection_ratio": float(np.mean(contrast_ratios)),
        }
