"""Vendi Score — the effective number of distinct representations in a
per-layer hidden-state cloud.

Reference:
    Friedman, D. & Dieng, A. B. (2023). "The Vendi Score: A Diversity
    Evaluation Metric for Machine Learning." Transactions on Machine
    Learning Research (TMLR), 2023. arXiv:2210.02410.
    Official repo: https://github.com/vertaix/Vendi-Score
    Official pip package: ``vendi_score`` (v0.0.3).

Definition (Friedman & Dieng 2023, Def. 1 / eqn. for q=1):
    Given ``n`` samples and a positive-semidefinite similarity matrix
    ``K`` with unit diagonal (``K_ii = 1``), let ``K/n`` have eigenvalues
    ``λ_1 … λ_n`` (each ≥ 0, summing to 1). The Vendi Score of order
    ``q = 1`` (the Shannon / exponential-entropy order used throughout the
    paper as the default) is

        VS = exp( − Σ_i λ_i log λ_i ),

    the exponential of the Shannon entropy of the (normalized) kernel
    eigenvalue spectrum. It is the *effective number of distinct
    samples*: VS = 1 when all samples are identical (K is all-ones) and
    VS = n when all samples are mutually dissimilar (K = I).

Why this is DISTINCT from effective_rank / RankMe:
    ``effective_rank`` (Roy & Vetterli 2007) and RankMe operate on the
    *raw singular values* of the (uncentered) representation matrix, i.e.
    on a **linear** Gram / covariance spectrum. The Vendi Score is
    computed on the spectrum of a **nonlinear** similarity kernel built
    from the representations. BLME's :func:`_vendi_score` deliberately
    uses a nonlinear kernel (``"cosine"`` by default, ``"rbf"`` optional)
    so that this metric captures diversity under a nonlinear notion of
    sample similarity rather than duplicating the linear-spectrum tasks.
    On the *same* kernel matrix the core reduces bit-exactly to the
    official ``vendi.score_K(K, q=1, normalize=False)``.

Numeric contract (pins parity with the official library):
    ``vendi.score_K(K, q=1, normalize=False)`` does exactly, in
    ``vendi_score/vendi.py`` (v0.0.3):
        weight_K:  K_ = K / K.shape[0]           # divide by n
        eigvals:   w  = scipy.linalg.eigvalsh(K_)  # symmetric eigensolver
        entropy_q: -(w[w>0] * log(w[w>0])).sum()   # only positive eigvals
        return     exp(entropy)
    :func:`_vendi_score` reproduces this step for step (same
    ``scipy.linalg.eigvalsh``, same positive-eigenvalue mask) so the two
    agree to machine epsilon.
"""

import logging
from typing import List

import numpy as np
import scipy.linalg
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Eigenvalues at or below this (in the K/n spectrum) are treated as the
# zero part of the PSD spectrum and dropped, exactly like the official
# ``entropy_q`` positive-mask ``p[p > 0]``. Kept at 0.0 to match the
# reference bit-for-bit; larger clamps would diverge from score_K.
_POS_EIG_THRESHOLD = 0.0


def _cosine_kernel(X: np.ndarray) -> np.ndarray:
    """n×n cosine-similarity kernel with unit diagonal (K_ii = 1).

    Rows are L2-normalized (zero rows fall back to zero vectors, giving a
    0 off-diagonal and a 0 diagonal for that sample; guarded downstream).
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    K = Xn @ Xn.T
    # Numerical hygiene: clip to [-1, 1] and force an exact unit diagonal
    # for non-degenerate rows so K_ii = 1 as the definition requires.
    np.clip(K, -1.0, 1.0, out=K)
    diag = (norms.squeeze(-1) > 0).astype(np.float64)
    np.fill_diagonal(K, diag)
    return K


def _rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """n×n RBF (Gaussian) kernel exp(-gamma * ||x_i - x_j||^2).

    Unit diagonal by construction (distance 0 -> exp(0) = 1).
    """
    X = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)  # guard tiny negatives from round-off
    K = np.exp(-gamma * d2)
    np.fill_diagonal(K, 1.0)
    return K


def _vendi_score(X, kernel: str = "cosine", gamma: float = None) -> float:
    """Vendi Score (order q=1) of a representation cloud under a
    **nonlinear** similarity kernel.

    This is the verified artifact: on the kernel matrix it builds, it
    reduces bit-exactly to the official
    ``vendi_score.vendi.score_K(K, q=1, normalize=False)``.

    Args:
        X: array-like of shape (n, d) — n representations in R^d.
        kernel: ``"cosine"`` (default) or ``"rbf"``. Both yield a PSD
                kernel with unit diagonal, so ``K/n`` has a valid
                probability spectrum.
        gamma: RBF bandwidth. If None and ``kernel == "rbf"``, defaults to
               ``1 / d`` (scikit-learn's ``"scale"``-free default of 1/n_features).

    Returns:
        The Vendi Score as a Python float in ``[1, n]``. Returns NaN for
        an empty cloud and 1.0 for a single sample.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return float("nan")
    n, d = X.shape
    if n == 1:
        return 1.0

    if kernel == "cosine":
        K = _cosine_kernel(X)
    elif kernel == "rbf":
        g = float(gamma) if gamma is not None else (1.0 / d if d > 0 else 1.0)
        K = _rbf_kernel(X, g)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}; use 'cosine' or 'rbf'.")

    # --- Exact transcription of vendi.score_K(K, q=1, normalize=False) ---
    # weight_K(K, p=None) = K / n
    Kn = K / n
    # scipy.linalg.eigvalsh: identical eigensolver the official code uses.
    w = scipy.linalg.eigvalsh(Kn)
    w = w[w > _POS_EIG_THRESHOLD]  # entropy_q: p_ = p[p > 0]
    if w.size == 0:
        return float("nan")
    shannon = -np.sum(w * np.log(w))
    return float(np.exp(shannon))


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _layer_vendi(cloud, kernel: str, gamma: float, max_points: int, seed: int) -> float:
    """Vendi Score for one layer's (N, D) token cloud.

    Subsamples to ``max_points`` rows (deterministic seed) because the
    kernel and its eigendecomposition are O(N^2)/O(N^3); the Vendi Score
    is a diversity statistic of the cloud, so a fixed random subsample is
    an unbiased estimate at controlled cost.
    """
    arr = _to_numpy(cloud)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return float("nan")
    n = arr.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        arr = arr[idx]
    return _vendi_score(arr, kernel=kernel, gamma=gamma)


@register_task("geometry_vendi_score")
class VendiScoreTask(DiagnosticTask):
    """
    Computes the Vendi Score (Friedman & Dieng, TMLR 2023,
    arXiv:2210.02410) of each layer's hidden-state cloud under a nonlinear
    similarity kernel, and summarises the effective-diversity profile
    across depth.

    The per-layer core :func:`_vendi_score` matches the official
    ``vendi_score.vendi.score_K(K, q=1, normalize=False)`` bit-exactly on
    the same kernel matrix.

    Outputs (flat floats):
        vendi_mean_first_layer / _mid_layer / _last_layer
        vendi_overall_mean       — mean of finite per-layer Vendi Scores
        vendi_normalized_mean    — per-layer VS/n_points averaged (∈(0,1])
        vendi_slope              — OLS slope of per-layer VS on normalized
                                   depth l/(n_layers-1)
        vendi_max / vendi_min    — extremes of the per-layer profile
        vendi_q25 / vendi_q50 / vendi_q75 — quartiles of per-layer VS
        _meta_ counts (n_layers, n_points_per_layer, kernel, ...)
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Vendi Score Diversity Analysis...")

        num_samples = int(self.config.get("num_samples", 100))
        kernel = str(self.config.get("kernel", "cosine"))
        gamma = self.config.get("gamma", None)
        gamma = float(gamma) if gamma is not None else None
        max_points = int(self.config.get("max_points", 512))
        seed = int(self.config.get("seed", 0))
        use_cache = self.config.get("use_cache", True)

        # --- Collect per-layer (N, D) token clouds ---------------------
        if cache is not None and cache.is_populated and use_cache:
            layer_clouds = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=False,
            )
        else:
            from ...cache import ModelOutputCache, load_default_corpus

            if dataset is None:
                dataset = load_default_corpus(num_samples)
            local_cache = ModelOutputCache(
                model, tokenizer, dataset=dataset, num_samples=num_samples,
            )
            local_cache.populate(need_hidden=True)
            layer_clouds = local_cache.get_hidden_states(
                layer_idx="all", per_sample=False,
            )

        if not layer_clouds:
            return {"error": "No hidden states available for Vendi Score"}

        layer_keys = sorted(layer_clouds.keys())
        n_layers = len(layer_keys)

        per_layer_vs: List[float] = []
        per_layer_norm: List[float] = []
        n_points_used = 0
        for k in layer_keys:
            arr = _to_numpy(layer_clouds[k])
            n_here = arr.shape[0] if arr.ndim == 2 else 0
            n_eff = min(n_here, max_points) if n_here else 0
            n_points_used = max(n_points_used, n_eff)
            vs = _layer_vendi(
                layer_clouds[k], kernel=kernel, gamma=gamma,
                max_points=max_points, seed=seed,
            )
            per_layer_vs.append(vs)
            per_layer_norm.append(vs / n_eff if (np.isfinite(vs) and n_eff > 0) else float("nan"))

        vs_arr = np.asarray(per_layer_vs, dtype=np.float64)
        norm_arr = np.asarray(per_layer_norm, dtype=np.float64)
        finite = np.isfinite(vs_arr)

        if not np.any(finite):
            return {"error": "Vendi Score undefined for all layers (degenerate clouds)"}

        v_first = vs_arr[0]
        v_last = vs_arr[-1]
        v_mid = vs_arr[n_layers // 2]

        # Slope on NORMALIZED depth l/(n_layers-1) — house convention for
        # cross-model comparability (matches trajectory_curvature).
        if n_layers >= 2 and finite.sum() >= 2:
            depth = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
            slope = float(np.polyfit(depth[finite], vs_arr[finite], 1)[0])
        else:
            slope = float("nan")

        finite_vs = vs_arr[finite]
        finite_norm = norm_arr[np.isfinite(norm_arr)]

        return {
            "vendi_mean_first_layer": float(v_first),
            "vendi_mean_mid_layer": float(v_mid),
            "vendi_mean_last_layer": float(v_last),
            "vendi_overall_mean": float(np.mean(finite_vs)),
            "vendi_normalized_mean": (
                float(np.mean(finite_norm)) if finite_norm.size else float("nan")
            ),
            "vendi_slope": slope,
            "vendi_max": float(np.max(finite_vs)),
            "vendi_min": float(np.min(finite_vs)),
            "vendi_q25": float(np.percentile(finite_vs, 25)),
            "vendi_q50": float(np.percentile(finite_vs, 50)),
            "vendi_q75": float(np.percentile(finite_vs, 75)),
            # _meta_ prefix => excluded from the analysis feature matrix so
            # these architecture/sampling counts cannot leak in as size
            # proxies (Audit-V2 convention).
            "_meta_n_layers": int(n_layers),
            "_meta_n_points_per_layer": int(n_points_used),
            "_meta_kernel": kernel,
            "_meta_q": 1,
        }
