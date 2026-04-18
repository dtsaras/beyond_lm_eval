from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")

def _svd_metrics_for_layer(X):
    """Compute SVD-based isotropy metrics for a single (N, D) matrix."""
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    if len(X) < 10:
        return None
    X = X - np.mean(X, axis=0)

    try:
        _, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            _, S, _ = scipy_svd(X, full_matrices=False)
        except Exception:
            return None

    explained_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    # np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz for compatibility
    _trapz = getattr(np, "trapezoid", np.trapz)
    auc = _trapz(explained_variance) / max(1, len(explained_variance))

    # Canonical Roy-Vetterli effective rank on σ² (shared helper).
    from .utils import effective_rank as _effective_rank
    effective_rank = float(_effective_rank(S))

    eigenvalues = S ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    participation_ratio = float((sum_eig ** 2) / (sum_eig_sq + 1e-12))

    indices = np.random.choice(len(X), size=(min(1000, len(X)), 2), replace=True)
    vecs1 = X[indices[:, 0]]
    vecs2 = X[indices[:, 1]]
    norms1 = np.linalg.norm(vecs1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vecs2, axis=1, keepdims=True)
    vecs1 = vecs1 / (norms1 + 1e-9)
    vecs2 = vecs2 / (norms2 + 1e-9)
    cos_sims = np.sum(vecs1 * vecs2, axis=1)
    avg_cos_sim = float(np.mean(cos_sims))

    # Conditioned on the numerical rank: on models where N < D the
    # smallest singular value is float-noise at ~1e-7, so ``S[0]/S[-1]``
    # reports 1e8 dominated by the numerical floor rather than the
    # data's geometry. Cap the denominator at a relative-tolerance
    # threshold of S[0] so ``cond_number`` reflects the effective
    # conditioning of the numerical-rank subspace only.
    rel_tol = float(S[0]) * max(S.shape) * np.finfo(S.dtype).eps
    effective_rank_num = int(np.sum(S > rel_tol))
    if effective_rank_num >= 1 and S[effective_rank_num - 1] > 0:
        cond_number = float(S[0] / S[effective_rank_num - 1])
    else:
        cond_number = float("inf")

    return {
        "svd_auc": float(auc),
        "cond_number": cond_number,
        "numerical_rank": effective_rank_num,
        "avg_cosine_similarity": avg_cos_sim,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
    }


def _isoscore(X: np.ndarray) -> float:
    """IsoScore (Rudman et al. 2022, arXiv:2207.10341).

    Measures how close the empirical covariance of a point cloud is to a
    scaled identity — a more discriminative notion of isotropy than the
    Ethayarajh anisotropy baseline (which only looks at average cosine
    similarity of random pairs).

    Algorithm (sum-normalisation form):
      1. Center X and compute the eigenvalue spectrum lambda_i of cov(X)
         via SVD on centered data.
      2. Scale lambda so sum(lambda_i) = d, so the isotropic case gives
         the all-ones vector.
      3. delta^2 = ||lambda - 1||^2.
      4. delta_iso^2 = d*(d-1) — the maximum delta achieved when all the
         variance lives in a single PC (lambda = (d, 0, ..., 0)).
      5. psi = (delta_iso^2 - delta^2) / delta_iso^2  (in [0, 1]).
      6. IsoScore = ((d - 1) * psi + 1) / d  (in [1/d, 1]).

    Verified empirically: 1.00 for an isotropic Gaussian, ~0.03 when
    99.99% of the variance lives in a single dimension, ~0.64 for a
    1.5^-i power-law decay spectrum.

    Returns a scalar in [1/d, 1] where 1 = perfectly isotropic.
    """
    n, d = X.shape
    if n < 2 or d < 2:
        return float("nan")

    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    eigvals = (S ** 2) / max(1, n - 1)

    full = np.zeros(d, dtype=np.float64)
    full[: len(eigvals)] = eigvals
    s = full.sum()
    if s <= 0:
        return float("nan")
    full = full * (d / s)

    delta_sq = float(np.sum((full - 1.0) ** 2))
    delta_iso_sq = float(d * (d - 1))
    if delta_iso_sq == 0:
        return float("nan")

    psi = max(0.0, (delta_iso_sq - delta_sq) / delta_iso_sq)
    xi = ((d - 1) * psi + 1.0) / d
    return float(max(0.0, min(1.0, xi)))


@register_task("geometry_isoscore")
class IsoScoreTask(DiagnosticTask):
    """
    IsoScore (Rudman et al. 2022, arXiv:2207.10341).

    A scalar in [0, 1] measuring how close the empirical covariance of the
    final-layer hidden states is to a scaled identity matrix. Higher = more
    isotropic. This is a strictly more discriminative isotropy measure than
    Ethayarajh's average-cosine-similarity baseline (which only looks at
    pairwise directions, not the magnitude spectrum).

    Returns:
        isoscore: float in [0, 1]
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running IsoScore Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        if cache is not None and cache.is_populated and use_cache:
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples)
        X = X.float().numpy()
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            logger.info(f"  Filtered {(~finite_mask).sum()} non-finite rows out of {len(X)}")
            X = X[finite_mask]
        if len(X) < 10:
            return {"error": "Too few finite hidden states for IsoScore"}

        return {"isoscore": _isoscore(X)}


@register_task("geometry_svd")
class SVDIsotropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SVD Analysis...")
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode (all layers) ---
        if layerwise and cache is not None and cache.is_populated and use_cache:
            all_layers = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
            if all_layers:
                per_layer = {}
                erank_per_layer = []
                pr_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _svd_metrics_for_layer(all_layers[li])
                    if m is not None:
                        per_layer[f"layer_{li}"] = m
                        erank_per_layer.append(m["effective_rank"])
                        pr_per_layer.append(m["participation_ratio"])
                    else:
                        erank_per_layer.append(float("nan"))
                        pr_per_layer.append(float("nan"))
                # Last-layer metrics as top-level for backward compat
                last_key = f"layer_{max(all_layers.keys())}"
                result = dict(per_layer.get(last_key, {}))
                result["layer_effective_rank"] = erank_per_layer
                result["layer_participation_ratio"] = pr_per_layer
                result["layer_metrics"] = per_layer
                return result

        # --- Single-layer mode (last layer, original behaviour) ---
        if cache is not None and cache.is_populated and use_cache:
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples)

        m = _svd_metrics_for_layer(X)
        if m is None:
            return {"error": "Too few finite hidden states for SVD"}
        return m
