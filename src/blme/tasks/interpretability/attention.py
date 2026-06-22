"""Attention entropy per head / layer.

Reference: Clark et al. 2019, "What Does BERT Look At? An Analysis of
BERT's Attention." (EMNLP BlackBoxNLP).

Caveat: attention weights don't always correlate with information flow
(Jain & Wallace 2019). High/low entropy does not imply
importance. Interpret in combination with gradient-based attribution.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _attention_entropy(att: torch.Tensor) -> torch.Tensor:
    """Per-distribution Shannon entropy (natural log) of an attention tensor.

    ``att`` is an attention-weight tensor whose LAST axis is the key
    distribution (rows sum to 1). Returns the entropy reduced over that
    last axis, i.e. shape ``att.shape[:-1]``. The 0·log0 = 0 convention is
    handled by clamping. This is the standard attention entropy of Clark
    et al. 2019; for a uniform distribution over T keys it equals log(T).
    """
    p = att.float().clamp(min=1e-12)
    return -(p * p.log()).sum(dim=-1)


@register_task("interpretability_attention_entropy")
class AttentionEntropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Entropy Analysis...")

        use_cache = self.config.get("use_cache", True)
        num_samples = self.config.get("num_samples", 100)

        # Cached-attentions fast path. ``cache.get_attentions()`` returns
        # ``{layer_idx: [tensor (H, T, T), ...]}`` — one tensor per
        # sample, already moved to CPU. Using this saves a full extra
        # forward pass (which on 8B-class models with 32 layers triples
        # peak memory when output_attentions=True is set).
        per_sample_attentions = None
        if cache is not None and cache.is_populated and use_cache:
            cached = cache.get_attentions(num_samples=num_samples)
            if cached:
                n_layers = max(cached.keys()) + 1
                max_samples = min(len(cached[0]), num_samples) if 0 in cached else 0
                if max_samples > 0:
                    per_sample_attentions = []
                    for s_i in range(max_samples):
                        layer_attns = []
                        valid = True
                        for li in range(n_layers):
                            attn = cached.get(li, [None] * (s_i + 1))[s_i]
                            if attn is None:
                                valid = False
                                break
                            layer_attns.append(attn)
                        if valid:
                            per_sample_attentions.append(layer_attns)

        if per_sample_attentions is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)

            per_sample_attentions = []
            with torch.no_grad():
                for i, sample in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = (
                        sample if isinstance(sample, str)
                        else sample.get("text", "")
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
                    if not attentions:
                        return {
                            "error": (
                                "Model does not return attention weights. "
                                "Reload with attn_implementation='eager'."
                            )
                        }
                    if any(a is None for a in attentions):
                        return {
                            "error": (
                                "Model returned None attentions — likely "
                                "SDPA / FlashAttention. Reload with "
                                "attn_implementation='eager'."
                            )
                        }
                    # Move to CPU and strip the batch dim to match the
                    # cached-attention shape ``(H, T, T)``.
                    per_sample_attentions.append(
                        [a.squeeze(0).detach().cpu() for a in attentions]
                    )

        if not per_sample_attentions:
            return {"error": "No attentions computed"}

        entropies = []    # (samples, L, H)
        seq_lengths = []
        for layer_attns in per_sample_attentions:
            if not layer_attns:
                continue
            T = layer_attns[0].shape[-1]
            seq_lengths.append(T)
            layer_entropies = []
            for layer_att in layer_attns:
                if layer_att is None:
                    layer_entropies = None
                    break
                # Per-(head, query) Shannon entropy over the key axis.
                entropy = _attention_entropy(layer_att)  # (H, T)
                # Average across query positions → (H,).
                avg_head_entropy = entropy.mean(dim=-1).numpy()
                layer_entropies.append(avg_head_entropy)
            if layer_entropies is not None:
                entropies.append(np.array(layer_entropies))

        if not entropies:
            return {"error": "No usable attention tensors after filtering"}

        # Samples may have different T and therefore different H/T array
        # widths only through their sequence length. We average per
        # sample, so stacking requires identical (L, H) shapes. All
        # attentions from a given model share (L, H), so stacking is safe.
        avg_entropies = np.mean(np.stack(entropies), axis=0)  # (L, H)

        median_T = float(np.median(seq_lengths)) if seq_lengths else 1.0
        norm_factor = float(np.log(max(2.0, median_T)))

        return {
            "avg_entropy_per_layer": np.mean(avg_entropies, axis=1).tolist(),
            "avg_entropy_total": float(np.mean(avg_entropies)),
            "min_entropy_head": float(np.min(avg_entropies)),
            "max_entropy_head": float(np.max(avg_entropies)),
            "avg_normalized_entropy_total": float(np.mean(avg_entropies) / norm_factor),
            "min_normalized_entropy_head": float(np.min(avg_entropies) / norm_factor),
            "max_normalized_entropy_head": float(np.max(avg_entropies) / norm_factor),
            "median_seq_len": median_T,
        }
