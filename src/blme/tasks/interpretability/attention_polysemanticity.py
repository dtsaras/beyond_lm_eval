"""
Attention Effective Rank Task
──────────────────────────────────────────────────────────────────────
Measures the effective rank (via SVD entropy) of the combined attention
output projection over a sequence.  Higher entropy indicates the attention
mechanism produces more diverse outputs across positions; lower entropy
(near rank-1) indicates the output is dominated by a single direction.

Note: this operates on the *combined* multi-head output (post-W_O), not
individual head outputs.  A monosemantic layer with many diverse heads
can still show high SVD entropy.

References:
- "Lorsa: Disentangling Atomic Attention Units from Attention Superposition" (2025)
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


@register_task("interpretability_attention_effective_rank")
class AttentionEffectiveRankTask(DiagnosticTask):
    """
    Measures the SVD entropy (effective rank) of combined attention output
    projections over a sequence.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Effective Rank Analysis...")
        num_samples = self.config.get("num_samples", 3)

        if dataset is None:
             try:
                 from datasets import load_dataset
                 dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                 dataset = []
                 for i in range(min(num_samples, len(dset))):
                     dataset.append({"text": dset[i]["text"]})
             except ImportError:
                 logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                 dataset = [{"text": "Attention effective rank measures the diversity of combined attention outputs."}] * num_samples
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device

        # We need the output projections of the attention modules
        # This is architecture dependent. We use a heuristic hook to catch
        # the output of the multi-head attention before the residual connection.

        target_modules = []
        for name, module in model.named_modules():
            # Standard HF naming for the attention output projection
            if "attn.c_proj" in name or "attn.out_proj" in name or "attention.output.dense" in name:
                target_modules.append(module)

        if not target_modules:
             return {"error": "Could not automatically locate attention output projections."}

        # We sample a few random layers to avoid immense computation
        import random
        if len(target_modules) > 4:
             target_modules = random.sample(target_modules, 4)

        entropies = []

        # Hook to collect the attention projection outputs
        outputs_collected = []
        def hook_fn(module, input_args, output):
             # output is typically (batch, seq_len, hidden_dim)
             val = output[0] if isinstance(output, tuple) else output
             outputs_collected.append(val.detach().cpu().float())

        handles = [m.register_forward_hook(hook_fn) for m in target_modules]

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

                # Clear hook cache
                outputs_collected.clear()

                model(**inputs)

                for out_proj in outputs_collected:
                    # out_proj shape: (batch_size, seq_len, hidden_size)
                    # We compute the SVD entropy along the sequence dimension
                    for b in range(out_proj.shape[0]):
                        mat = out_proj[b] # (seq_len, hidden_size)

                        # Singular Value Decomposition
                        # If seq_len is 1, SVD entropy is 0
                        if mat.shape[0] < 2:
                            continue

                        U, S, V = torch.svd(mat, compute_uv=False)

                        # Normalize singular values to form a probability distribution
                        S_norm = S / torch.sum(S)
                        S_norm = S_norm[S_norm > 0] # Avoid log(0)

                        # Shannon Entropy of the singular values
                        entropy = -torch.sum(S_norm * torch.log(S_norm)).item()
                        entropies.append(entropy)

        for h in handles:
            h.remove()

        if not entropies:
            return {"error": "Failed to compute effective rank entropies."}

        mean_entropy = float(np.mean(entropies))

        return {
            "mean_attention_effective_rank_entropy": mean_entropy,
            "max_effective_rank_entropy": float(np.max(entropies)),
        }
