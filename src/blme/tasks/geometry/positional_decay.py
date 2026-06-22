"""
Positional Attention Decay (RoPE Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates the geometric degradation of relative positional embeddings
(like RoPE) by computing the Spearman rank correlation between the
relative discrete distance of two tokens and the magnitude of their 
attention connection.

A structurally intact context window will show a strong negative correlation 
(closer tokens have higher attention in positional/local heads). If this 
correlation collapses, the model's positional geometry is degraded.

References:
- 2024-2025 Long-Context Extrapolation and RoPE literature.
"""

import torch
import numpy as np
from scipy.stats import spearmanr

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


def _row_distance_correlation(attn_row: np.ndarray) -> float | None:
    """Spearman correlation for one causal query row.

    Constant rows have no distance preference; treat them as the null
    correlation instead of letting pooled row lengths manufacture decay.
    """
    if attn_row.size < 2:
        return None
    if np.allclose(attn_row, attn_row[0]):
        return 0.0
    distances = np.arange(attn_row.size, 0, -1, dtype=np.float64)
    corr, _ = spearmanr(distances, attn_row)
    if np.isnan(corr):
        return None
    return float(corr)


@register_task("geometry_positional_decay")
class PositionalAttentionDecayTask(DiagnosticTask):
    """
    Computes the Spearman correlation between target-source token distance 
    and attention probability to measure the structural integrity of 
    positional encodings.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Positional Attention Decay Analysis...")
        num_samples = self.config.get("num_samples", 5)
        
        # We need reasonably long sequences to measure positional decay gracefully
        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)
             
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        
        # Per-layer correlations: {layer_idx: [corr_sample1, corr_sample2, ...]}
        from collections import defaultdict
        layer_correlations = defaultdict(list)

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

                seq_len = inputs.input_ids.shape[1]
                if seq_len < 4:
                   continue

                out = model(**inputs, output_attentions=True)
                if out.attentions is None or len(out.attentions) == 0:
                    return {"error": "Model did not return attentions. Cannot compute Positional Decay."}

                for li, attn_entry in enumerate(out.attentions):
                    if attn_entry is None:
                        continue
                    attn_matrix = attn_entry[0]  # (num_heads, seq_len, seq_len)
                    # Compute each query-row/head separately. Pooling
                    # triangular causal entries first confounds distance
                    # with row normalisation (uniform rows shrink as
                    # context grows) and invents decay under the null.
                    a = attn_matrix.float().cpu().numpy()
                    for head_idx in range(a.shape[0]):
                        for query_idx in range(2, seq_len):
                            corr = _row_distance_correlation(
                                a[head_idx, query_idx, :query_idx],
                            )
                            if corr is not None:
                                layer_correlations[li].append(corr)

        if not layer_correlations:
             return {"error": "Could not compute positional correlations (sequences too short or nan)."}

        # Compute per-layer means
        layer_means = {}
        for li in sorted(layer_correlations.keys()):
            layer_means[f"layer_{li}"] = float(np.mean(layer_correlations[li]))

        all_corrs = [c for corrs in layer_correlations.values() for c in corrs]

        return {
            "mean_positional_decay_correlation": float(np.mean(all_corrs)),
            "layer_positional_decay": layer_means,
        }
