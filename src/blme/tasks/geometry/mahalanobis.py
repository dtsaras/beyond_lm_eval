"""
Latent Mahalanobis OOD Distance Task
──────────────────────────────────────────────────────────────────────
Evaluates Out-Of-Distribution (OOD) detection capabilities by measuring
the Mahalanobis distance of hidden states from a reference text distribution.

A robust model encodes OOD/anomalous text geometrically far from the 
centroid of its normal distribution, normalized by the covariance 
(Mahalanobis distance).

References:
- "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (Lee et al., NeurIPS 2018)
- "Mahalanobis++: Improving OOD Detection via Feature Normalization" (ICML 2025)
"""

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import logging
logger = logging.getLogger("blme")


def _compute_mahalanobis_distances(X_train, X_test):
    """
    Computes Mahalanobis distances for X_test relative to the distribution of X_train.
    
    Args:
        X_train: (N_train, D) numpy array of reference hidden states
        X_test: (N_test, D) numpy array of OOD hidden states
        
    Returns:
        List of distances for each sample in X_test.
    """
    try:
        # Centroid
        mu = np.mean(X_train, axis=0)
        
        # Covariance matrix
        cov = np.cov(X_train, rowvar=False)
        
        # Add small ridge penalty for numerical stability in pseudo-inverse
        eps = 1e-6
        cov_reg = cov + np.eye(cov.shape[0]) * eps
        
        # Inverse covariance
        inv_cov = np.linalg.pinv(cov_reg)
        
        distances = []
        for x in X_test:
            dist = mahalanobis(x, mu, inv_cov)
            distances.append(dist)
            
        return distances
    except Exception as e:
        logger.info(f"Error computing Mahalanobis: {e}")
        return []


@register_task("geometry_mahalanobis")
class MahalanobisOODTask(DiagnosticTask):
    """
    Computes the Mahalanobis distance between an In-Distribution (ID) 
    reference dataset and an Out-Of-Distribution (OOD) dataset at the final layer.
    
    Reports the average ID distance vs OOD distance, and the separation gap.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Latent Mahalanobis OOD Detection...")
        num_samples = self.config.get("num_samples", 50)
        
        # We need two pseudo-datasets for structural evaluation if none provided:
        # 1. ID: Normal English
        # 2. OOD: Random ascii noise / structurally broken English
        if dataset is None:
             dataset_id = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(num_samples)]
             dataset_ood = [{"text": "jf8923h f82h3 f9283h f9283hf 9238h f."} for _ in range(num_samples)]
        else:
            # For standard benchmark, we split the given dataset in half and perturb the second half
            samples = list(dataset)[:num_samples*2]
            mid = max(1, len(samples) // 2)
            dataset_id = samples[:mid]
            # Create OOD by scrambling characters
            dataset_ood = [{"text": "".join(np.random.permutation(list(s["text"] if isinstance(s, dict) and "text" in s else str(s))))} for s in samples[mid:]]
            
        if len(dataset_id) < 5 or len(dataset_ood) < 5:
            return {"error": "Need at least 5 ID and 5 OOD samples to compute stable covariance."}
            
        # Collect representations from the last layer, mean-pooled over sequence
        X_id = collect_hidden_states(model, tokenizer, dataset_id, num_samples=len(dataset_id))
        X_id = X_id.float().numpy()
        
        X_ood = collect_hidden_states(model, tokenizer, dataset_ood, num_samples=len(dataset_ood))
        X_ood = X_ood.float().numpy()
        
        # Compute Mahalanobis distance of ID samples to their OWN distribution 
        # (This establishes the baseline ID structural radius)
        dists_id = _compute_mahalanobis_distances(X_id, X_id)
        
        # Compute Mahalanobis distance of OOD samples to the ID distribution
        dists_ood = _compute_mahalanobis_distances(X_id, X_ood)
        
        if not dists_id or not dists_ood:
            return {"error": "Failed to compute Mahalanobis distance (likely covariance singularity)."}
            
        mean_id = float(np.mean(dists_id))
        mean_ood = float(np.mean(dists_ood))
            
        return {
            "mean_mahalanobis_id": mean_id,
            "mean_mahalanobis_ood": mean_ood,
            "ood_separation_gap": mean_ood - mean_id,
            "ood_separation_ratio": mean_ood / max(mean_id, 1e-10)
        }
