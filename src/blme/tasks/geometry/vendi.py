"""Vendi Score — the effective number of distinct representations in a
per-layer hidden-state cloud.

Reference:
    Friedman, D. & Dieng, A. B. (2023). "The Vendi Score: A Diversity
    Evaluation Metric for Machine Learning." Transactions on Machine
    Learning Research (TMLR), 2023. arXiv:2210.02410.
    Official repo: https://github.com/vertaix/Vendi-Score
    Official pip package: ``vendi_score`` (v0.0.3).

Definition (Friedman & Dieng 2023, Definition 3.1 / eqn. for q=1):
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

Kernel: the default ``"cosine"`` kernel is the released ``score_X`` kernel, a
LINEAR Gram of the row-normalised, uncentred states, so VS(cosine) is a
spectral effective rank of those rows (exp of the Shannon entropy of the
normalised squared singular values). ``"rbf"`` is the nonlinear option. On
the same kernel matrix the core reduces bit-exactly to the official
``vendi.score_K(K, q=1, normalize=False)``.

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
from ._sequence_points import (DEFAULT_N_SEQUENCES, DEFAULT_TOKEN_DOCS, readout_for,
                               sequence_clouds)
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Eigenvalues at or below this (in the K/n spectrum) are treated as the
# zero part of the PSD spectrum and dropped, exactly like the official
# ``entropy_q`` positive-mask ``p[p > 0]``. Kept at 0.0 to match the
# reference bit-for-bit; larger clamps would diverge from score_K.
_POS_EIG_THRESHOLD = 0.0


# An isotropic-null-calibrated ratio was withdrawn: on fixed intrinsic
# structure at varying width it manufactured a width dependence the raw score
# does not have (VS <= min(n, d) = n for every model at these sizes).


def _cosine_kernel(X: np.ndarray) -> np.ndarray:
    """n×n cosine-similarity kernel with unit diagonal (K_ii = 1).

    Rows are L2-normalized; zero rows give zero off-diagonals but still carry
    K_ii = 1, as the definition requires.

    For row-normalised X this is K = Xn Xn^T, a linear Gram kernel on the
    normalised rows.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    K = Xn @ Xn.T
    # Clip to [-1, 1] and force K_ii = 1 for every row, zero rows included:
    # the definition needs a unit diagonal (test_vendi_parity.py).
    np.clip(K, -1.0, 1.0, out=K)
    np.fill_diagonal(K, 1.0)
    # Zero rows are identical to one another, so an all-zero cloud scores 1,
    # as the official score_X does.
    _zero = (norms.squeeze(-1) <= 0)
    if _zero.any():
        K[np.ix_(_zero, _zero)] = 1.0
    return K


def _rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """n×n RBF (Gaussian) kernel exp(-gamma * ||x_i - x_j||^2).

    Unit diagonal by construction (distance 0 -> exp(0) = 1).
    """
    # Direct differences, not the Gram identity: the RBF kernel is
    # translation invariant and the identity cancels for clouds far from the
    # origin. Chunked so the (chunk, n, d) temporary stays bounded.
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    d2 = np.empty((n, n), dtype=np.float64)
    _chunk = max(1, int(2 ** 22 // max(1, X.shape[1] * 8)))
    for i in range(0, n, _chunk):
        j = min(i + _chunk, n)
        diff = X[i:j, None, :] - X[None, :, :]
        d2[i:j] = np.einsum("ijk,ijk->ij", diff, diff)
    np.maximum(d2, 0.0, out=d2)  # exact zeros only; direct form cannot go negative
    K = np.exp(-gamma * d2)
    np.fill_diagonal(K, 1.0)
    return K


def _vendi_score(X, kernel: str = "cosine", gamma: float = None) -> float:
    """Vendi Score (order q=1) of a representation cloud.

    On the kernel matrix it builds, this reduces bit-exactly to the official
    ``vendi_score.vendi.score_K(K, q=1, normalize=False)``.

    Args:
        X: array-like of shape (n, d) — n representations in R^d.
        kernel: ``"cosine"`` (default, linear Gram of unit rows) or ``"rbf"``.
                Both are PSD with unit diagonal.
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
    kernel and its eigendecomposition are O(N^2)/O(N^3). NOTE: this is a
    deterministic size-CAPPED estimate, not an unbiased one — VS of a
    subsample is bounded by ``max_points``, so clouds with more than
    ``max_points`` effectively-distinct rows are under-reported (e.g.
    eye(1000) at max_points=512 gives ~512, not ~1000). The cap and the
    per-layer point counts are exposed via ``_meta_n_points_per_layer``
    and the normalized variant.
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
    arXiv:2210.02410) of each layer's hidden-state cloud and summarises the
    effective-diversity profile across depth.

    The per-layer core :func:`_vendi_score` matches the official
    ``vendi_score.vendi.score_K(K, q=1, normalize=False)`` bit-exactly on
    the same kernel matrix.

    Outputs (flat floats):
        vendi_mean_first_layer / _mid_layer / _last_layer
        vendi_max                — maximum of the per-layer profile
        _meta_vendi_overall_mean / _slope / _q25 / _q50 / _q75 — layer
                                   aggregates (move under identity blocks)
        _meta_vendi_normalized_mean — VS/n_points, kept for continuity (not a
                                   width correction).
        _meta_ counts (n_layers, n_points_per_layer, kernel, ...)
    """

    # One point per sequence (Q1): the last content token of each document.
    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": "arXiv:2210.02410",
        "code": "vertaix/Vendi-Score@ff1dfdbe6356b98a6087540f215b9a9db6db7c11",
        "llm_precedent": ("none by name; adjacent: Skean et al. 2025 arXiv:2502.02013 "
                          "(alpha=1 matrix entropy of the Gram per layer; dataset entropy "
                          "uses one vector per prompt)"),
    }
    ADAPTATION_NOTE = (
        "The Vendi Score (Friedman & Dieng 2023) is the exponential of the Shannon entropy of "
        "the eigenvalues of K/n for a unit-diagonal similarity kernel K, the effective number "
        "of distinct samples. No paper applies it to LLM hidden states, so BLME scores one "
        "embedding per sample as the paper does: the last content token of each of 1000 "
        "documents, at every block output, with the released score_X kernel (cosine "
        "similarity, a row-normalised uncentred linear Gram). With this kernel VS equals exp "
        "of the alpha=1 matrix entropy that Skean et al. 2025 apply per layer to one vector "
        "per prompt. An uncentred cosine kernel is dominated by a shared mean direction, so "
        "the score mixes anisotropy with diversity; VS grows with n up to min(n, d), so n is "
        "pinned at 1000 for every model. Layer means, slopes and quartiles move when a "
        "computation is split into more blocks and are _meta_; the token-cloud profile "
        "(position 0 and massive rows excluded) is _meta_tokens_*."
    )
    HEADLINE_KEYS = ("vendi_mean_first_layer", "vendi_mean_mid_layer", "vendi_mean_last_layer",
                     "vendi_max")
    _INV = "tests/tasks/invariance/test_geometry_vendi_score.py::"
    # vendi_mean_mid_layer is an index-selected depth and moves under
    # identity-block insertion.
    SIZE_INVARIANT = dict.fromkeys(
        ("vendi_mean_first_layer", "vendi_mean_last_layer", "vendi_max"),
        (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
         _INV + "test_endpoints_and_max_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Vendi Score Diversity Analysis...")

        num_samples = int(self.config.get("num_samples", DEFAULT_N_SEQUENCES))
        kernel = str(self.config.get("kernel", "cosine"))
        gamma = self.config.get("gamma", None)
        gamma = float(gamma) if gamma is not None else None
        max_points = int(self.config.get("max_points", 512))
        seed = int(self.config.get("seed", 0))
        use_cache = self.config.get("use_cache", True)
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        # One point per sequence (Q1), as the Vendi Score scores one embedding
        # per sample: the last content token of each of num_samples documents
        # at every block output (the deepest one the hooked raw block output).
        # Every point is scored: VS grows with N, which is pinned. A cache with
        # fewer documents is bypassed (test_vendi_protocol.py).
        sc = sequence_clouds(
            model, tokenizer, dataset, cache, n=num_samples, layer="all",
            replace_final_norm=True, use_cache=use_cache,
            token_docs=int(self.config.get("token_cloud_docs", DEFAULT_TOKEN_DOCS)),
            readout=readout_for(self.TOKEN_POLICY))
        if not sc.last:
            return {"error": "No hidden states available for Vendi Score"}

        layer_keys = sorted(sc.last)
        n_points = sc.n_points
        vs = [_vendi_score(_to_numpy(sc.last[k]), kernel=kernel, gamma=gamma) for k in layer_keys]
        out = _profile_summary(vs, [n_points] * len(vs))
        if out is None:
            return {"error": "Vendi Score undefined for all layers (degenerate clouds)"}

        # The token cloud (every token of the first token_cloud_docs documents,
        # sink policy, max_points seeded rows per layer): _meta_tokens_*.
        if sc.tokens:
            rows = [sc.token_rows(k)[0] for k in layer_keys]
            tvs = [_layer_vendi(r, kernel=kernel, gamma=gamma, max_points=max_points, seed=seed)
                   for r in rows]
            tok = _profile_summary(tvs, [min(int(r.shape[0]), max_points) for r in rows]) or {}
            out.update({f"_meta_tokens_{k}": v for k, v in tok.items() if not k.startswith("_meta_")})
            out["_meta_tokens_n_points_per_layer"] = int(max([min(int(r.shape[0]), max_points)
                                                             for r in rows] or [0]))

        out.update({
            "_meta_n_layers": int(len(layer_keys)),
            "_meta_n_points_per_layer": int(n_points),
            "_meta_kernel": kernel,
            "_meta_q": 1,
            "_meta_last_entry_is_block_output": bool(sc.info.get("last_is_block", False)),
        })
        out.update(sc.meta())
        return out


def _profile_summary(per_layer_vs, n_points_per_layer):
    """Profile summaries of per-layer Vendi Scores (None if none is finite)."""
    vs_arr = np.asarray(per_layer_vs, dtype=np.float64)
    finite = np.isfinite(vs_arr)
    if not np.any(finite):
        return None
    n_layers = len(vs_arr)
    norm_arr = np.asarray([v / n if (np.isfinite(v) and n > 0) else float("nan")
                           for v, n in zip(per_layer_vs, n_points_per_layer)], dtype=np.float64)
    # Slope on normalised depth l/(n_layers-1).
    if n_layers >= 2 and finite.sum() >= 2:
        depth = np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
        slope = float(np.polyfit(depth[finite], vs_arr[finite], 1)[0])
    else:
        slope = float("nan")
    finite_vs = vs_arr[finite]
    finite_norm = norm_arr[np.isfinite(norm_arr)]
    return {
        "vendi_mean_first_layer": float(vs_arr[0]),
        "vendi_mean_mid_layer": float(vs_arr[n_layers // 2]),
        "vendi_mean_last_layer": float(vs_arr[-1]),
        # P4: uniform means, slopes and quartiles over layers move when identity
        # blocks are inserted, so they are _meta_ profile summaries
        # (tests/tasks/invariance/test_geometry_vendi_score.py).
        "_meta_vendi_overall_mean": float(np.mean(finite_vs)),
        # VS / n_points: kept under its historic name, not a width correction.
        "_meta_vendi_normalized_mean": (
            float(np.mean(finite_norm)) if finite_norm.size else float("nan")),
        "_meta_vendi_slope": slope,
        "vendi_max": float(np.max(finite_vs)),
        # _meta_: the minimum is the last layer on most models, a
        # sometimes-duplicate of vendi_mean_last_layer.
        "_meta_vendi_min": float(np.min(finite_vs)),
        "_meta_vendi_argmin_is_last_layer": bool(
            int(np.argmin(finite_vs)) == len(finite_vs) - 1),
        "_meta_vendi_q25": float(np.percentile(finite_vs, 25)),
        "_meta_vendi_q50": float(np.percentile(finite_vs, 50)),
        "_meta_vendi_q75": float(np.percentile(finite_vs, 75)),
    }
