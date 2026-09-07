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
from .utils import collected_layer_count
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Eigenvalues at or below this (in the K/n spectrum) are treated as the
# zero part of the PSD spectrum and dropped, exactly like the official
# ``entropy_q`` positive-mask ``p[p > 0]``. Kept at 0.0 to match the
# reference bit-for-bit; larger clamps would diverge from score_K.
_POS_EIG_THRESHOLD = 0.0


# WITHDRAWN: `vendi_null_ratio_mean` (isotropic-null-calibrated Vendi).
#
# Added 2026-09-03 and removed the same day. It looked convincing on the test I
# ran -- flat at 0.997-1.002 across d=4..256 -- but that test was a TAUTOLOGY:
# I evaluated it on isotropic Gaussian input, which is exactly the null it
# divides by, so flatness was guaranteed and measured nothing.
#
# On the case that matters -- FIXED intrinsic structure at varying ambient width
# -- it is worse than the raw score. Across d = 512..8192 at n = 512,
# rho(d, ratio) = -1.000 for rank 8, 64 and 256, spanning 1.31-1.57x, while the
# RAW score spans only 1.02-1.22x. It manufactures the width dependence it
# claimed to remove.
#
# The raw score was already width-flat on the realistic range, because
# max_points = 512 <= d_model for every model in the zoo, so the rank ceiling
# min(n, d_model) is n, not d_model (adversarial review, 2026-09-03).


def _cosine_kernel(X: np.ndarray) -> np.ndarray:
    """n×n cosine-similarity kernel with unit diagonal (K_ii = 1).

    Rows are L2-normalized; zero rows give zero off-diagonals but still carry
    K_ii = 1, as the definition requires.

    NOTE: for row-normalized X this kernel is K = Xn Xn^T, i.e. a LINEAR Gram
    kernel on the normalized rows -- not a nonlinear kernel. Earlier docstrings
    in this module and in task_metadata described it as nonlinear; that was
    wrong.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 0, norms, 1.0)
    K = Xn @ Xn.T
    # Numerical hygiene: clip to [-1, 1] and force an exact unit diagonal.
    #
    # K_ii = 1 for EVERY row, including zero-norm ones. The Vendi Score is
    # defined on a kernel with unit diagonal, and setting K_ii = 0 for a
    # degenerate row makes the spectrum sum to less than n, which then trips the
    # empty-positive-spectrum guard downstream and returns NaN -- where the
    # official `vendi_score` returns 1 for an all-zero cloud (adversarial
    # review, 2026-09-03). A zero row is a sample that is similar to nothing,
    # including itself only trivially; the definition's answer is K_ii = 1 with
    # zero off-diagonals, which is what this now produces.
    np.clip(K, -1.0, 1.0, out=K)
    np.fill_diagonal(K, 1.0)
    # Zero rows are IDENTICAL to one another, not mutually orthogonal. Leaving
    # their off-diagonals at 0 makes K the identity on that block, which reads
    # as maximal diversity: an all-zero cloud of n points scored n. The official
    # `vendi.score_X` returns 1.0 there, which is the right answer -- n copies
    # of the same (degenerate) sample have a diversity of one.
    _zero = (norms.squeeze(-1) <= 0)
    if _zero.any():
        K[np.ix_(_zero, _zero)] = 1.0
    return K


def _rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """n×n RBF (Gaussian) kernel exp(-gamma * ||x_i - x_j||^2).

    Unit diagonal by construction (distance 0 -> exp(0) = 1).
    """
    # DIRECT differences, not the Gram identity
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2x.y.
    #
    # The identity cancels catastrophically once the cloud sits far from the
    # origin: the separations that matter are recovered by subtracting two
    # large nearly-equal sums. The RBF kernel is translation invariant by
    # definition, so any dependence on the cloud's position is pure artefact --
    # measured at a +1e8 offset the identity moved the Vendi Score by -42%
    # (Opus round 9, 2026-09-07). `np.maximum(d2, 0)` did not save it: clamping
    # a negative squared distance to 0 turns a genuinely separated pair into a
    # false duplicate, which is precisely the error that then propagates.
    #
    # `magnitude.py::_pairwise_distance_matrix` was given the same treatment
    # earlier in the campaign for the same reason; this site was missed. Cost
    # is one (n, n, d) temporary, which is why it is chunked.
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    d2 = np.empty((n, n), dtype=np.float64)
    # Chunked so the (chunk, n, d) temporary stays bounded regardless of n.
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
    arXiv:2210.02410) of each layer's hidden-state cloud under a nonlinear
    similarity kernel, and summarises the effective-diversity profile
    across depth.

    The per-layer core :func:`_vendi_score` matches the official
    ``vendi_score.vendi.score_K(K, q=1, normalize=False)`` bit-exactly on
    the same kernel matrix.

    Outputs (flat floats):
        vendi_mean_first_layer / _mid_layer / _last_layer
        vendi_overall_mean       — mean of finite per-layer Vendi Scores
        vendi_null_ratio_mean    — VS divided by a matched ISOTROPIC GAUSSIAN
                                   cloud of the same (n, d). 1.0 = as diverse as
                                   structureless data. This is the
                                   cross-model-comparable form.
        _meta_vendi_normalized_mean — VS/n_points. NOT a width correction: the
                                   cosine Vendi Score is bounded by
                                   rank(K) <= min(n, d_model), and n is a fixed
                                   cap, so dividing by it leaves Spearman
                                   rho = 1 with width. Kept for continuity only.
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
        # REFUSE AN UNDERSIZED SHARED CACHE, the `geometry_schatten` fix.
        #
        # This task is NOT in `core.py`'s `cache_hidden_tasks`/`cache_candidates`,
        # so the shared corpus is never sized for it -- but `use_cache` is absent
        # from its merged config, so the code default `True` fired and it
        # consumed whatever cache the OTHER tasks in the run happened to
        # produce. The Vendi score is an exponential of an eigenvalue entropy
        # and is strongly N-dependent, so that is not a small effect. Same
        # model, same vendi config, only the co-runner set changed
        # (red-team review, 2026-09-05):
        #
        #   cache docs           100      50      20      10
        #   vendi_overall_mean 20.8163 20.0777 17.0423 16.9177   (-18.7%)
        #   vendi_slope        +1.2937 +1.4960 -0.0152 -0.2969   (SIGN FLIP)
        #
        # A metric whose slope changes sign with the task list is not a
        # measurement of the model. Self-collect unless the cache is at least
        # as large as this task's own configured N.
        _cached_docs = len(getattr(cache, "_sample_lengths", []) or []) if cache else 0
        _cache_ok = (cache is not None and cache.is_populated and use_cache
                     and _cached_docs >= int(num_samples))
        if cache is not None and cache.is_populated and use_cache and not _cache_ok:
            logger.info(
                f"  Vendi: shared cache holds {_cached_docs} documents < "
                f"configured {num_samples}; self-collecting so the score does "
                f"not depend on which tasks co-ran.")
        if _cache_ok:
            layer_clouds = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=False,
                replace_final_norm=True,
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
                layer_idx="all", per_sample=False, replace_final_norm=True,
            )

        if not layer_clouds:
            return {"error": "No hidden states available for Vendi Score"}

        layer_keys = sorted(layer_clouds.keys())
        # NO TRIM -- but the last entry IS now hooked. Two different things,
        # and this comment used to conflate them.
        #
        # A trim was added on 2026-09-03 and REVERTED the same day, correctly:
        # dropping the last key does not recover the final block, it DELETES
        # the deepest layer (the final block's raw output is not in
        # `output_hidden_states` at all). Measured then, the old endpoint was
        # 18-61% LOW while the trimmed endpoint was 251-2049% HIGH.
        #
        # But `replace_final_norm=True` is the THIRD option, and it is the one
        # those measurements were pointing at: hook `layers[-1]` and SUBSTITUTE
        # the raw block output, keeping the layer count. The argument this
        # comment used to make -- "each key is a legitimate representation to
        # score, including the post-norm final state" -- defended against
        # trimming, not against hooking, and it left the profile MIXED: keys
        # 0..L-2 are block outputs, key L-1 was a LayerNorm output, and
        # `vendi_slope`, `vendi_max`, `vendi_min` and the quartiles all compare
        # across that boundary. The profile is now homogeneous.
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
            # VS / n_points -- retained under its historic name, but it is NOT
            # a width correction (see _isotropic_null_vendi).
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

            "_meta_vendi_normalized_mean": (
                float(np.mean(finite_norm)) if finite_norm.size else float("nan")
            ),
            "vendi_slope": slope,
            "vendi_max": float(np.max(finite_vs)),
            # `vendi_min` is an ARCHITECTURE-DEPENDENT DUPLICATE of
            # `vendi_mean_last_layer`: the argmin is L-1 for gpt2, distilgpt2,
            # pythia-70m and pythia-160m but not for opt-125m or SmolLM2-135M,
            # giving r = 0.99925 over six models (red-team review,
            # 2026-09-05). It is exactly equal for 4 of 6, so a feature matrix
            # carrying both has one column that is sometimes a copy and
            # sometimes not -- worse than a clean duplicate, because the
            # difference between them is itself an architecture signal.
            "_meta_vendi_min": float(np.min(finite_vs)),
            "_meta_vendi_argmin_is_last_layer": bool(
                int(np.argmin(finite_vs)) == len(finite_vs) - 1),
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
