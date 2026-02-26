"""
Representational Dissimilarity Analysis (RDA/RSA) — layer-wise comparison
of pairwise distance structures using Representational Dissimilarity Matrices.

While CKA measures similarity using kernel alignment, RSA/RDA directly 
compares the rank-order of pairwise distances between samples across layers.
This is the standard method in computational neuroscience for comparing
neural representations (Kriegeskorte et al., 2008).

Specifically, we compute the Representational Dissimilarity Matrix (RDM) 
at each layer (pairwise Euclidean distances between sample representations), 
then measure how well these RDMs correlate across layers using Spearman rank 
correlation. This reveals which layers share similar distance structures 
(i.e., similar "views" of the data).

References:
- "Representational Similarity Analysis — Connecting the Branches of Systems
  Neuroscience" (Kriegeskorte et al., Frontiers 2008)
- "Insights on Representational Similarity in Neural Networks with Canonical
  Correlation" (Morcos et al., NeurIPS 2018)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist


@register_task("geometry_rsa")
class RepresentationalSimilarityTask(DiagnosticTask):
    """
    Computes Representational Similarity Analysis (RSA) across layers.
    
    For each layer, constructs a Representational Dissimilarity Matrix (RDM)
    from the pairwise Euclidean distances between sample representations.
    Then computes the Spearman rank correlation between RDMs of adjacent layers
    and between early/late layers.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Representational Similarity Analysis (RSA)...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        num_samples = self.config.get("num_samples", 20)
        max_tokens_per_layer = self.config.get("max_tokens", 200)
        
        # Collect hidden states from ALL layers
        all_layers = collect_hidden_states(model, tokenizer, dataset,
                                           num_samples=num_samples,
                                           layer_idx="all")
        
        if not all_layers:
            return {"error": "No hidden states collected"}
            
        layer_indices = sorted(all_layers.keys())
        n_layers = len(layer_indices)
        
        if n_layers < 2:
            return {"error": "Need at least 2 layers for RSA"}
        
        # Subsample tokens consistently across layers
        min_tokens = min(len(all_layers[li]) for li in layer_indices)
        n_tokens = min(min_tokens, max_tokens_per_layer)
        
        if n_tokens < 3:
            return {"error": f"Too few tokens ({n_tokens}) for RSA"}
        
        # Compute RDMs for each layer
        rdms = {}
        for li in layer_indices:
            X = all_layers[li][:n_tokens].numpy()
            # pdist returns the condensed pairwise distance vector
            rdm = pdist(X, metric="euclidean")
            rdms[li] = rdm
            
        # Adjacent-layer RSA: Spearman correlation between consecutive RDMs
        adjacent_corrs = []
        for i in range(n_layers - 1):
            rdm_a = rdms[layer_indices[i]]
            rdm_b = rdms[layer_indices[i + 1]]
            rho, _ = spearmanr(rdm_a, rdm_b)
            if np.isfinite(rho):
                adjacent_corrs.append(float(rho))
            else:
                adjacent_corrs.append(0.0)
                
        # Early-Late RSA: compare first layer to last layer
        rdm_first = rdms[layer_indices[0]]
        rdm_last = rdms[layer_indices[-1]]
        early_late_rho, _ = spearmanr(rdm_first, rdm_last)
        if not np.isfinite(early_late_rho):
            early_late_rho = 0.0
            
        # First-Middle RSA
        mid_idx = n_layers // 2
        rdm_mid = rdms[layer_indices[mid_idx]]
        first_mid_rho, _ = spearmanr(rdm_first, rdm_mid)
        if not np.isfinite(first_mid_rho):
            first_mid_rho = 0.0
            
        adj_arr = np.array(adjacent_corrs)
        
        return {
            "rsa_adjacent_mean": float(np.mean(adj_arr)),
            "rsa_adjacent_min": float(np.min(adj_arr)),
            "rsa_adjacent_std": float(np.std(adj_arr)),
            "rsa_early_late": float(early_late_rho),
            "rsa_first_middle": float(first_mid_rho),
            "rsa_min_continuity_layer": int(np.argmin(adj_arr)),
        }
