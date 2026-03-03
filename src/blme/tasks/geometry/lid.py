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
  (Levina & Bickel, NIPS 2005 — MLE-based LID)
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
    estimates the local dimensionality using the MLE formula from Levina & Bickel (2005):
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
        return 0.0
        
    return -k / sum_log


@register_task("geometry_lid")
class LocalIntrinsicDimensionalityTask(DiagnosticTask):
    """
    Computes per-sample Local Intrinsic Dimensionality (LID) using
    the Maximum Likelihood Estimator (Levina & Bickel, 2005).
    
    Outputs the mean, std, min, and max LID across all samples for
    the specified layer(s).
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Local Intrinsic Dimensionality (LID) Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        k = self.config.get("k", 20)
        num_samples = self.config.get("num_samples", 50)
        
        # Collect hidden states from the last layer
        if cache is not None and cache.is_populated:
            X = cache.get_hidden_states(layer_idx=-1)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples)
        X = X.float().numpy()
        
        # Filter non-finite rows
        finite_mask = np.all(np.isfinite(X), axis=1)
        X = X[finite_mask]
        
        if len(X) < k + 1:
            return {"error": f"Too few samples ({len(X)}) for LID with k={k}"}
        
        # Cap number of query points for speed
        max_queries = min(len(X), 500)
        query_indices = np.random.choice(len(X), size=max_queries, replace=False)
        
        # Compute pairwise distances from queries to all points
        lid_estimates = []
        for qi in query_indices:
            diffs = X - X[qi]
            dists = np.linalg.norm(diffs, axis=1)
            # Sort and take k nearest (skip self at index 0)
            sorted_dists = np.sort(dists)
            nn_dists = sorted_dists[1:k+1]  # exclude self
            
            lid = _lid_mle(nn_dists, k)
            if lid > 0:
                lid_estimates.append(lid)
                
        if not lid_estimates:
            return {"error": "Could not compute any valid LID estimates"}
            
        lid_arr = np.array(lid_estimates)
        
        return {
            "lid_mean": float(np.mean(lid_arr)),
            "lid_std": float(np.std(lid_arr)),
            "lid_min": float(np.min(lid_arr)),
            "lid_max": float(np.max(lid_arr)),
            "lid_median": float(np.median(lid_arr)),
            "num_estimates": len(lid_estimates),
        }
