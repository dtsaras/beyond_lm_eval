"""
Local Intrinsic Dimensionality (LID) — per-sample local dimension estimates.

Unlike global Two-NN (which gives a single number for an entire point cloud),
LID measures the local dimensionality around each individual sample point.
This reveals whether certain inputs live on lower-dimensional sub-manifolds
(e.g., compressed representations) or higher ones (richer, more expressive encodings).

References:
- "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality"
  (Ma et al., ICLR 2018)
- "Local Intrinsic Dimensionality Estimation via Maximum Likelihood"
  (Levina & Bickel, NeurIPS 2004 — MLE-based LID)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")


def _lid_mle(distances, k):
    """Maximum Likelihood Estimator for Local Intrinsic Dimensionality.
    
    Given sorted distances from a query to its k nearest neighbors,
    estimates the local dimensionality using the MLE formula from Levina & Bickel (2004):
        LID = -k / sum_{i=1}^{k} log(d_i / d_k)
    
    Args:
        distances: sorted distances to k nearest neighbors (excluding self), shape (k,)
        k: number of neighbors
        
    Returns:
        LID estimate (float)
    """
    # Avoid log(0) by clamping
    d_k = distances[-1]
    if d_k < 1e-10:
        return 0.0
    
    ratios = distances / d_k
    ratios = np.maximum(ratios, 1e-10)
    
    log_ratios = np.log(ratios)
    sum_log = np.sum(log_ratios)
    
    if abs(sum_log) < 1e-10:
        return float("nan")

    return -k / sum_log


def _compute_lid_for_matrix(X, k, max_queries=500, seed: int = 0):
    """Compute LID statistics for a single (N, D) data matrix.

    Returns a dict of statistics or *None* if there are too few points.
    A fixed ``seed`` ensures reproducibility across reruns — the
    previous ``np.random.choice`` call used the global RNG, so reruns
    would draw different query subsets and yield different LID
    statistics (paper-grade numbers should be deterministic).
    """
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if len(X) < k + 1:
        return None

    n_queries = min(len(X), max_queries)
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(len(X), size=n_queries, replace=False)

    lid_estimates = []
    for qi in query_indices:
        diffs = X - X[qi]
        dists = np.linalg.norm(diffs, axis=1)
        sorted_dists = np.sort(dists)
        nn_dists = sorted_dists[1:k + 1]

        lid = _lid_mle(nn_dists, k)
        if not np.isnan(lid) and lid > 0:
            lid_estimates.append(lid)

    if not lid_estimates:
        return None

    lid_arr = np.array(lid_estimates)
    return {
        "lid_mean": float(np.mean(lid_arr)),
        "lid_std": float(np.std(lid_arr)),
        "lid_min": float(np.min(lid_arr)),
        "lid_max": float(np.max(lid_arr)),
        "lid_median": float(np.median(lid_arr)),
        "num_estimates": len(lid_estimates),
    }


@register_task("geometry_lid")
class LocalIntrinsicDimensionalityTask(DiagnosticTask):
    """
    Computes per-sample Local Intrinsic Dimensionality (LID) using
    the Maximum Likelihood Estimator (Levina & Bickel, 2004).

    Outputs the mean, std, min, and max LID across all samples for
    the specified layer(s).  When ``layerwise=True`` (and cache is
    available), computes LID at every layer.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Local Intrinsic Dimensionality (LID) Analysis...")
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        k = self.config.get("k", 20)
        num_samples = self.config.get("num_samples", 50)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode ---
        if layerwise and cache is not None and cache.is_populated and use_cache:
            all_layers = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
            if all_layers:
                per_layer = {}
                lid_mean_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _compute_lid_for_matrix(all_layers[li], k)
                    if m is not None:
                        per_layer[f"layer_{li}"] = m
                        lid_mean_per_layer.append(m["lid_mean"])
                    else:
                        lid_mean_per_layer.append(float("nan"))
                last_key = f"layer_{max(all_layers.keys())}"
                result = dict(per_layer.get(last_key, {}))
                result["layer_lid_mean"] = lid_mean_per_layer
                result["layer_metrics"] = per_layer
                return result

        # --- Single-layer mode (last layer) ---
        if cache is not None and cache.is_populated and use_cache:
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples)

        m = _compute_lid_for_matrix(X, k)
        if m is None:
            return {"error": f"Too few valid samples for LID with k={k}"}
        return m
