from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _centered_linear_gram(X: torch.Tensor) -> torch.Tensor:
    """Linear kernel Gram matrix centered by H K H."""
    X = X.float()
    K = X @ X.t()
    return K - K.mean(dim=0) - K.mean(dim=1, keepdim=True) + K.mean()


def _normalized_linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Normalized linear HSIC, equivalent to linear CKA."""
    Kx = _centered_linear_gram(X)
    Ky = _centered_linear_gram(Y)
    hsic_xy = float(torch.sum(Kx * Ky))
    hsic_xx = float(torch.sum(Kx * Kx))
    hsic_yy = float(torch.sum(Ky * Ky))
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / (denom + 1e-12))


@register_task("geometry_hsic")
class HSICDependenceTask(DiagnosticTask):
    """
    Measures statistical dependence between layer representations using
    normalized HSIC (Hilbert-Schmidt Independence Criterion) with a linear
    kernel. Mathematically equivalent to Linear CKA from Kornblith 2019
    when normalised by √(HSIC(X,X)·HSIC(Y,Y)).

    References:
      * Gretton, Bousquet, Smola, Schölkopf 2005 — "Measuring Statistical
        Dependence with Hilbert-Schmidt Norms", ALT 2005. The original
        HSIC formulation used here.
      * Kornblith, Norouzi, Lee, Hinton 2019 — "Similarity of Neural
        Network Representations Revisited", ICML 2019, arXiv:1905.00414.
        HSIC → CKA normalisation in Section 3.

    ``input_to_layer_hsic.mean`` enters BLME's top-24 partial predictors
    at +0.68 partial ρ (`docs/TOP_PREDICTORS.md` §2), measuring how
    strongly per-layer activations still depend on the input embedding.
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
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.get("seed", 42)))
            shared_perm = torch.randperm(n_tokens, generator=rng)[:max_tokens]
        else:
            shared_perm = None

        # Precompute centered Gram matrices for each layer
        logger.info("  Computing Gram matrices...")
        gram_matrices = {}
        for idx in tqdm(layers, desc="Gram Matrices"):
            X = layer_activations[idx].float()
            if shared_perm is not None:
                X = X[shared_perm]

            gram_matrices[idx] = _centered_linear_gram(X).cpu()

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

        # Input-to-layer HSIC (first layer vs every *later* layer).
        # Historic bug: the loop started at ``i = 0``, including the
        # self-pair ``HSIC(input, input) = 1.0``. That self-pair had no
        # information content but pinned the list's ``max`` to 1.0 for
        # every model and skewed ``hsic_compression_ratio`` by anchoring
        # the denominator at a tautological 1.
        input_hsic = []
        K_input = gram_matrices[layers[0]]
        for i in range(1, n_layers):
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
            # Compression: last-layer HSIC-to-input divided by
            # first-transformer-block HSIC-to-input (both measured
            # after the embedding layer, not the self-pair).
            "hsic_compression_ratio": (
                float(input_hsic[-1] / (input_hsic[0] + 1e-12))
                if input_hsic else 0.0
            ),
        }
