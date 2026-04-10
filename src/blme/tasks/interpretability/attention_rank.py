"""
Attention rank collapse (Dong et al. 2021, arXiv:2103.03404).

Each attention head produces a (T, T) attention matrix. Dong et al. show
that as transformer depth grows, attention matrices tend to converge
toward rank-1 (token uniformity) without skip connections / MLPs. The
*effective rank* of each attention matrix — measured via Roy & Vetterli's
SVD entropy — quantifies how much the head distinguishes between
different query positions. Low effective rank = the head's attention
pattern is approximately the same for every query (collapsed). High
effective rank = the head produces structured, query-dependent attention.

We report:
  - mean / min / max effective rank across heads
  - per-layer mean effective rank
  - rank-collapse ratio: how much the effective rank decreases from early
    to late layers (Dong et al.'s main empirical finding)
"""

import logging
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _effective_rank(matrix: np.ndarray) -> float:
    """Roy & Vetterli effective rank: exp(H(p)) where p_i = sigma_i / sum sigma_j.

    matrix: (T, T) attention matrix (rows sum to 1).
    Returns a float in [1, min(T, T)] where 1 = rank-collapsed.
    """
    if matrix.size == 0:
        return float("nan")
    try:
        # SVD of attention matrix. Note: attention matrices are stochastic
        # (rows sum to 1) but not symmetric, so we can't use eigh.
        S = np.linalg.svd(matrix, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = S.sum()
    if s <= 0:
        return float("nan")
    p = S / s
    p = p[p > 1e-12]
    if p.size == 0:
        return float("nan")
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


@register_task("interpretability_attention_rank")
class AttentionRankCollapseTask(DiagnosticTask):
    """
    Per-head effective rank of attention weight matrices, aggregated
    across layers, with a layer-wise rank-collapse trend metric.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Rank Collapse Analysis...")

        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."}]

        num_samples = self.config.get("num_samples", 50)

        # rank_per_layer[layer] = list of per-sample, per-head effective ranks
        rank_per_layer: List[List[float]] = []
        n_layers_seen = 0
        normalised_per_layer: List[List[float]] = []
        # The denominator for normalisation is min(T, T) = T (the seq len),
        # since the attention matrix is T x T. We track per-sample T to
        # compute a properly-normalised collapse measure.

        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Attention rank")):
                if i >= num_samples:
                    break

                text = sample["text"] if isinstance(sample, dict) else str(sample)
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(model.device)
                if inputs["input_ids"].shape[1] < 4:
                    continue

                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions
                if not attentions or attentions[0] is None:
                    return {
                        "error": "Model does not return attention weights. "
                                 "Reload with attn_implementation='eager'."
                    }

                if not rank_per_layer:
                    n_layers_seen = len(attentions)
                    rank_per_layer = [[] for _ in range(n_layers_seen)]
                    normalised_per_layer = [[] for _ in range(n_layers_seen)]

                T = attentions[0].shape[-1]
                for li, layer_att in enumerate(attentions):
                    # (B=1, H, T, T)
                    a = layer_att[0].float().cpu().numpy()
                    for h_idx in range(a.shape[0]):
                        er = _effective_rank(a[h_idx])
                        if not np.isnan(er):
                            rank_per_layer[li].append(er)
                            normalised_per_layer[li].append(er / max(1, T))

        if n_layers_seen == 0:
            return {"error": "No samples processed"}

        # Aggregate per-layer
        layer_means = [float(np.nanmean(x)) if x else float("nan") for x in rank_per_layer]
        layer_mins = [float(np.nanmin(x)) if x else float("nan") for x in rank_per_layer]
        layer_maxs = [float(np.nanmax(x)) if x else float("nan") for x in rank_per_layer]
        layer_norm_means = [float(np.nanmean(x)) if x else float("nan") for x in normalised_per_layer]

        # Rank-collapse trend: are late layers more collapsed than early ones?
        # Use Spearman over layer index vs. mean normalised rank.
        from scipy.stats import spearmanr
        valid = [(i, v) for i, v in enumerate(layer_norm_means) if not np.isnan(v)]
        if len(valid) >= 3:
            xs = np.array([v[0] for v in valid])
            ys = np.array([v[1] for v in valid])
            try:
                corr, _ = spearmanr(xs, ys)
                rank_collapse_trend = float(corr) if corr is not None else float("nan")
            except Exception:
                rank_collapse_trend = float("nan")
        else:
            rank_collapse_trend = float("nan")

        # Flat aggregates over all (layer, head, sample) effective ranks
        all_ranks = [r for layer in rank_per_layer for r in layer]
        all_norm = [r for layer in normalised_per_layer for r in layer]

        return {
            "mean_effective_rank": float(np.mean(all_ranks)) if all_ranks else float("nan"),
            "min_effective_rank": float(np.min(all_ranks)) if all_ranks else float("nan"),
            "max_effective_rank": float(np.max(all_ranks)) if all_ranks else float("nan"),
            "mean_normalised_effective_rank": float(np.mean(all_norm)) if all_norm else float("nan"),
            "rank_collapse_trend_spearman": rank_collapse_trend,
            "layer_mean_effective_rank": layer_means,
            "layer_min_effective_rank": layer_mins,
            "layer_max_effective_rank": layer_maxs,
            "layer_mean_normalised_rank": layer_norm_means,
        }
