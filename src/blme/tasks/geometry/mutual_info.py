from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .._tokens import hidden_states_for_task
from .utils import collected_layer_count
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


def _centered_linear_gram(X: torch.Tensor) -> torch.Tensor:
    """Linear Gram matrix of the column-centred cloud, (HX)(HX)^T, in float64.

    Centring X before forming the Gram (not H K H after) keeps a common offset
    from swamping the signal (tests/test_scale_invariance.py).
    """
    X = X.double()
    X = X - X.mean(dim=0, keepdim=True)
    return X @ X.t()


def _nhsic_from_grams(K_i: torch.Tensor, K_j: torch.Tensor,
                      self_i: "float | None" = None,
                      self_j: "float | None" = None) -> float:
    """Normalised HSIC tr(K_i K_j) / sqrt(tr(K_i K_i) tr(K_j K_j)) of centred
    Grams: the production kernel, parity-pinned in
    tests/tasks/test_reference_parity_formulas.py."""
    hsic_ij = float(torch.sum(K_i * K_j))
    if self_i is None:
        self_i = float(torch.sum(K_i * K_i))
    if self_j is None:
        self_j = float(torch.sum(K_j * K_j))
    # No additive epsilon (it breaks scale invariance), and sqrt(a)*sqrt(b)
    # rather than sqrt(a*b), whose product overflows float64 first.
    denom = np.sqrt(self_i) * np.sqrt(self_j)
    if not (denom > 0.0) or not np.isfinite(denom):
        return float("nan")
    return float(hsic_ij / denom)


def _normalized_linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Normalized linear HSIC, equivalent to linear CKA."""
    return _nhsic_from_grams(_centered_linear_gram(X), _centered_linear_gram(Y))


@register_task("geometry_hsic")
class HSICDependenceTask(DiagnosticTask):
    """
    Normalised linear HSIC (Gretton et al. 2005) between layer representations,
    which with the sqrt(HSIC(X,X) HSIC(Y,Y)) normalisation is linear CKA
    (Kornblith et al. 2019, arXiv:1905.00414, Sec. 3), on a seeded token
    subsample shared by every layer.

    The anchor ("input") is the first block's output: the layer set comes from
    the cache, which does not include the embedding output hs[0].
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

        layer_activations, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache)
        _last_is_block = bool(_info["last_is_block"])

        layers = sorted(layer_activations.keys())
        # Only a collection without the deepest-block hook still ends in the
        # post-norm entry; that entry is trimmed.
        if not _last_is_block:
            _keep = collected_layer_count(model)
            if _keep and len(layers) > _keep:
                layers = layers[:_keep]
        n_layers = len(layers)

        if n_layers < 2:
            return {"error": "Need at least 2 layers"}

        # Subsample tokens for speed with the SAME seeded indices in every layer.
        max_tokens = self.config.get("max_hsic_tokens", 2000)
        n_tokens = layer_activations[layers[0]].shape[0]
        if n_tokens > max_tokens:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.get("seed", 42)))
            shared_perm = torch.randperm(n_tokens, generator=rng)[:max_tokens]
        else:
            shared_perm = None

        logger.info("  Computing Gram matrices...")
        gram_matrices = {}
        for idx in tqdm(layers, desc="Gram Matrices"):
            X = layer_activations[idx].float()
            if shared_perm is not None:
                X = X[shared_perm]
            gram_matrices[idx] = _centered_linear_gram(X).cpu()

        logger.info("  Computing pairwise HSIC...")
        self_hsic = {}
        for idx in layers:
            K = gram_matrices[idx]
            self_hsic[idx] = float(torch.sum(K * K))

        adjacent_hsic = []
        for i in range(n_layers - 1):
            adjacent_hsic.append(_nhsic_from_grams(
                gram_matrices[layers[i]], gram_matrices[layers[i + 1]],
                self_hsic[layers[i]], self_hsic[layers[i + 1]],
            ))

        # Anchor against every LATER layer; the self-pair (1.0) carries nothing.
        input_hsic = []
        K_input = gram_matrices[layers[0]]
        for i in range(1, n_layers):
            input_hsic.append(_nhsic_from_grams(
                K_input, gram_matrices[layers[i]],
                self_hsic[layers[0]], self_hsic[layers[i]],
            ))

        # Raw (unnormalised) centred HSIC endpoints for the manuscript's ratio
        # tr(K_0 K_{L-1}) / tr(K_0 K_0); the 1/(n-1)^2 factor cancels.
        _K0 = gram_matrices[layers[0]]
        _KL = gram_matrices[layers[-1]]
        _raw_hsic_first_last = float(torch.sum(_K0 * _KL))
        _raw_hsic_first_self = float(torch.sum(_K0 * _K0))

        return {
            "adjacent_hsic": adjacent_hsic,
            "avg_adjacent_hsic": float(np.mean(adjacent_hsic)),
            "min_adjacent_hsic": float(np.min(adjacent_hsic)),
            "input_to_layer_hsic": input_hsic,
            # nCKA(anchor, L-1) / nCKA(anchor, 1): unbounded ([X, Y, X] gives
            # 9.40), named for what it is.
            "hsic_deep_to_shallow_ratio": (
                float(input_hsic[-1] / input_hsic[0])
                if input_hsic and input_hsic[0] != 0.0 else float("nan")
            ),
            # The manuscript's HSIC(0, L-1)/HSIC(0, 0): a projection
            # coefficient, not bounded by 1.
            "hsic_compression_ratio_raw": (
                float(_raw_hsic_first_last / _raw_hsic_first_self)
                if _raw_hsic_first_self else float("nan")),
            "hsic_compression_ratio": (
                float(input_hsic[-1]) if input_hsic else float("nan")
            ),
        }
