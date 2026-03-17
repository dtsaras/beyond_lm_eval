from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


@register_task("geometry_mutual_info")
class MutualInformationTask(DiagnosticTask):
    """
    Estimates Mutual Information between layer representations using
    a kernel-based estimator (HSIC as a proxy for MI).
    Ref: Shwartz-Ziv & Tishby, "Opening the Black Box of Deep Neural
         Networks via Information", 2017. arXiv:1703.00810
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Mutual Information Analysis (HSIC proxy)...")

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
        max_tokens = self.config.get("max_mi_tokens", 2000)
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

        # Adjacent layer MI proxy
        adjacent_mi = []
        for i in range(n_layers - 1):
            K_i = gram_matrices[layers[i]]
            K_j = gram_matrices[layers[i + 1]]

            hsic_ij = float(torch.sum(K_i * K_j))
            norm = np.sqrt(self_hsic[layers[i]] * self_hsic[layers[i + 1]])
            nmi = hsic_ij / (norm + 1e-12)
            adjacent_mi.append(nmi)

        # Input-to-layer MI (first layer vs all others)
        input_mi = []
        K_input = gram_matrices[layers[0]]
        for i in range(n_layers):
            K_i = gram_matrices[layers[i]]
            hsic_val = float(torch.sum(K_input * K_i))
            norm = np.sqrt(self_hsic[layers[0]] * self_hsic[layers[i]])
            nmi = hsic_val / (norm + 1e-12)
            input_mi.append(nmi)

        return {
            "adjacent_mi": adjacent_mi,
            "avg_adjacent_mi": float(np.mean(adjacent_mi)),
            "min_adjacent_mi": float(np.min(adjacent_mi)),
            "input_to_layer_mi": input_mi,
            "information_compression_ratio": float(input_mi[-1] / (input_mi[0] + 1e-12))
            if input_mi
            else 0.0,
        }
