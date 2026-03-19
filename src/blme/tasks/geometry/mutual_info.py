from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


@register_task("geometry_hsic")
class HSICDependenceTask(DiagnosticTask):
    """
    Measures statistical dependence between layer representations using
    normalized HSIC (Hilbert-Schmidt Independence Criterion) with a linear
    kernel.  This is mathematically equivalent to Linear CKA (Centered
    Kernel Alignment) from Kornblith et al., 2019.

    Ref: Kornblith et al., "Similarity of Neural Network Representations
         Revisited", ICML 2019. arXiv:1905.00414
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running HSIC Dependence Analysis...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Collect all layer activations
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
        else:
            layer_activations = collect_hidden_states(
            model, tokenizer, dataset, num_samples=num_samples, layer_idx="all"
        )

        layers = sorted(layer_activations.keys())
        n_layers = len(layers)

        if n_layers < 2:
            return {"error": "Need at least 2 layers"}

        # Subsample tokens for speed — use SAME indices across all layers
        max_tokens = self.config.get("max_hsic_tokens", 2000)
        n_tokens = layer_activations[layers[0]].shape[0]
        if n_tokens > max_tokens:
            shared_perm = torch.randperm(n_tokens)[:max_tokens]
        else:
            shared_perm = None

        # Precompute centered Gram matrices for each layer
        logger.info("  Computing Gram matrices...")
        gram_matrices = {}
        for idx in tqdm(layers, desc="Gram Matrices"):
            X = layer_activations[idx].float()
            if shared_perm is not None:
                X = X[shared_perm]

            # Linear kernel: K = X @ X^T
            K = X @ X.t()

            # Center the Gram matrix: H K H where H = I - 1/n * 11^T
            # Optimized representation:
            K_centered = K - K.mean(dim=0) - K.mean(dim=1, keepdim=True) + K.mean()

            gram_matrices[idx] = K_centered.cpu()

        # Compute HSIC between pairs of layers
        # HSIC(X, Y) = (1/(n-1)^2) * trace(K_X @ K_Y)
        # Normalized HSIC (CKA-like): HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        logger.info("  Computing pairwise HSIC...")

        # Self-HSIC for normalization
        self_hsic = {}
        for idx in layers:
            K = gram_matrices[idx]
            self_hsic[idx] = float(torch.sum(K * K))

        # Adjacent layer HSIC
        adjacent_hsic = []
        for i in range(n_layers - 1):
            K_i = gram_matrices[layers[i]]
            K_j = gram_matrices[layers[i + 1]]

            hsic_ij = float(torch.sum(K_i * K_j))
            norm = np.sqrt(self_hsic[layers[i]] * self_hsic[layers[i + 1]])
            nhsic = hsic_ij / (norm + 1e-12)
            adjacent_hsic.append(nhsic)

        # Input-to-layer HSIC (first layer vs all others)
        input_hsic = []
        K_input = gram_matrices[layers[0]]
        for i in range(n_layers):
            K_i = gram_matrices[layers[i]]
            hsic_val = float(torch.sum(K_input * K_i))
            norm = np.sqrt(self_hsic[layers[0]] * self_hsic[layers[i]])
            nhsic = hsic_val / (norm + 1e-12)
            input_hsic.append(nhsic)

        return {
            "adjacent_hsic": adjacent_hsic,
            "avg_adjacent_hsic": float(np.mean(adjacent_hsic)),
            "min_adjacent_hsic": float(np.min(adjacent_hsic)),
            "input_to_layer_hsic": input_hsic,
            "hsic_compression_ratio": float(input_hsic[-1] / (input_hsic[0] + 1e-12))
            if input_hsic
            else 0.0,
        }
