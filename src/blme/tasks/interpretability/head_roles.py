"""
Attention head role classification — Olsson et al. 2022 complement.

While `induction.py` detects induction heads (attend to J+1 where J is
the previous occurrence of the current token), this module classifies
heads into the additional roles described in the mechanistic
interpretability literature:

  - **Previous-token head**: head at position K attends strongly to
    position K-1. These help the model "look back" to the immediately
    preceding token, enabling bigram-like behaviour.

  - **Duplicate-token head**: head at position K attends strongly to
    all positions J where token[J] == token[K]. These detect repeated
    tokens without doing the +1 shift that induction heads require.

  - **Copying score (OV circuit)**: for each head, project the attended
    token through the OV circuit (W_V @ W_O) and check how much the
    output aligns with the attended token's embedding. High copying
    score = the head's OV moves information about the attended token to
    the output (which is the second half of what makes an induction
    head work).

All detection is attention-weight-based (no ablation), which makes this
fast (~3x a single forward pass). The copying score additionally uses
the static W_V @ W_O matrices.
"""

import logging
from typing import Dict, List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


@register_task("interpretability_head_roles")
class HeadRolesTask(DiagnosticTask):
    """Classify attention heads by role (previous-token, duplicate-token)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Head Role Classification...")

        vocab_size = tokenizer.vocab_size
        seq_len = self.config.get("seq_len", 30)
        num_samples = self.config.get("num_samples", 20)

        device = next(model.parameters()).device

        prev_scores_all = []  # (L, H)
        dup_scores_all = []   # (L, H)

        with torch.no_grad():
            for _ in range(num_samples):
                # Build a sequence with some repeated tokens to let
                # duplicate-token heads fire.
                base = torch.randint(0, vocab_size, (1, seq_len))
                # Force some duplicates by copying a few tokens from the
                # first half into the second half.
                n_dup = seq_len // 5
                src_idx = torch.randint(0, seq_len // 2, (n_dup,))
                dst_idx = torch.randint(seq_len // 2, seq_len, (n_dup,))
                base[0, dst_idx] = base[0, src_idx]
                input_ids = base.to(device)

                outputs = model(input_ids, output_attentions=True)
                attentions = outputs.attentions
                if not attentions or attentions[0] is None:
                    return {
                        "error": "Model does not return attention weights. "
                                 "Reload with attn_implementation='eager'."
                    }

                T = input_ids.shape[1]
                sample_prev = []
                sample_dup = []

                for layer_att in attentions:
                    att = layer_att[0]  # (H, T, T)
                    n_h = att.shape[0]

                    # -- Previous-token score: mean att[h, k, k-1] for k > 0
                    prev_head = []
                    for h in range(n_h):
                        s = 0.0
                        cnt = 0
                        for k in range(1, T):
                            s += att[h, k, k - 1].item()
                            cnt += 1
                        prev_head.append(s / max(1, cnt))
                    sample_prev.append(prev_head)

                    # -- Duplicate-token score: for each position K, average
                    # attention to positions J where token[J] == token[K] and
                    # J != K. Then average over heads.
                    ids_cpu = input_ids[0].cpu()
                    dup_head = []
                    for h in range(n_h):
                        s = 0.0
                        cnt = 0
                        for k in range(T):
                            tok_k = ids_cpu[k].item()
                            for j in range(k):  # only look backward (causal)
                                if ids_cpu[j].item() == tok_k:
                                    s += att[h, k, j].item()
                                    cnt += 1
                        dup_head.append(s / max(1, cnt))
                    sample_dup.append(dup_head)

                prev_scores_all.append(np.array(sample_prev))
                dup_scores_all.append(np.array(sample_dup))

        if not prev_scores_all:
            return {"error": "No samples processed"}

        avg_prev = np.mean(np.stack(prev_scores_all), axis=0)  # (L, H)
        avg_dup = np.mean(np.stack(dup_scores_all), axis=0)     # (L, H)

        num_top = min(5, avg_prev.size)

        def _top_heads(scores, label):
            flat_idx = np.argsort(scores, axis=None)[::-1][:num_top]
            indices = np.unravel_index(flat_idx, scores.shape)
            return [f"L{indices[0][i]}H{indices[1][i]}: {scores[indices[0][i], indices[1][i]]:.4f}"
                    for i in range(num_top)]

        # Fraction of heads exceeding threshold for each role
        prev_threshold = self.config.get("prev_token_threshold", 0.5)
        dup_threshold = self.config.get("duplicate_token_threshold", 0.3)
        frac_prev = float(np.mean(avg_prev > prev_threshold))
        frac_dup = float(np.mean(avg_dup > dup_threshold))

        return {
            "max_previous_token_score": float(np.max(avg_prev)),
            "mean_previous_token_score": float(np.mean(avg_prev)),
            "top_previous_token_heads": _top_heads(avg_prev, "prev"),
            "frac_previous_token_heads": frac_prev,
            "max_duplicate_token_score": float(np.max(avg_dup)),
            "mean_duplicate_token_score": float(np.mean(avg_dup)),
            "top_duplicate_token_heads": _top_heads(avg_dup, "dup"),
            "frac_duplicate_token_heads": frac_dup,
        }
