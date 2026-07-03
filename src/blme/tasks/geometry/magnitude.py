"""Metric-space magnitude — a multi-scale measure of the effective diversity
(``|tX|`` "number of points you can tell apart") of a latent point cloud.

Reference:
    Limbeck, K., Andreeva, R., Sarkar, R. & Rieck, B. (2024). "Metric Space
    Magnitude for Evaluating the Diversity of Latent Representations."
    NeurIPS 2024, arXiv:2311.16054. Official code: aidos-lab/magnipy.

    Foundational theory:
      * Leinster, T. (2013). "The magnitude of metric spaces." Documenta
        Mathematica 18, 857-905.
      * Meckes, M. W. (2015). "Magnitude, diversity, capacities, and
        dimensions of metric spaces." Potential Analysis 42, 549-572
        (magnitude dimension = maximum slope of the magnitude-dimension
        profile).

Definition (Leinster 2013; Limbeck et al. 2024, Sec. 2):
    For a finite metric space with pairwise distance matrix ``D`` and a scale
    ``t > 0``, form the *similarity matrix*

        ζ_ij = exp(-t · D_ij)                        (ζ is symmetric, ζ_ii = 1)

    The *magnitude* at scale ``t`` is the sum of all entries of ζ's inverse,

        |tX| = 1ᵀ ζ⁻¹ 1 = Σ_ij (ζ⁻¹)_ij .

    Equivalently, solve the linear system ``ζ w = 1`` for the *weight vector*
    ``w`` and sum it: ``|tX| = Σ_i w_i``. magnipy uses exactly this weight-
    vector form (``weights_cholesky``: two triangular solves against the ones
    vector; ``weights_pinv``/``compute_magnitude_no_gpu``: ``pinv(ζ).sum()``).
    Both are algebraically the closed-form ``1ᵀ ζ⁻¹ 1``; this task uses the
    same solve.

    The *magnitude function* is ``t ↦ |tX|``. It is 1 at ``t → 0`` (the whole
    space looks like one point), non-decreasing under the usual (positive-
    definite) regime, and → n (the cardinality) as ``t → ∞`` (every point
    becomes perfectly distinguishable). We report the magnitude function
    sampled at several scales tied to the distance distribution, plus the
    *magnitude dimension* — the maximum slope of ``log|tX|`` vs ``log t`` —
    which behaves like an intrinsic-dimension estimate.

BLME notes:
    * Tier-2 task: consumes the shared cache's flattened token cloud via
      ``cache.get_hidden_states(layer_idx=..., per_sample=False)`` and forms
      the Euclidean distance matrix ``D``. A random subsample of
      ``num_samples`` points keeps the ``O(n³)`` inverse tractable; magnitude
      is a property of the finite subsample, not an estimator biased by n.
    * Scales are chosen relative to the median pairwise distance (a robust,
      scale-free anchor): ``t_med = median-heuristic scale`` and a geometric
      ladder of ``n_scales`` scales spanning ``[t_med/scale_span,
      t_med·scale_span]``. This brackets the informative "elbow" of the
      magnitude function for typical hidden-state clouds.
    * ζ becomes ill-conditioned at large ``t`` (it approaches the identity,
      but intermediate ``t`` on near-duplicate points can be singular). We
      mirror magnipy's fallback: attempt a Cholesky/solve, and on failure
      perturb ``ζ`` by ``+ε·I`` (Bunch et al. 2020) and retry; the ``pinv``
      path is used for the point-cloud entry point. All math is float64.
    * Only summary scalars are emitted (magnitude at low/median/high scale,
      the area under the magnitude function, and the magnitude dimension);
      per-scale absolute counts are hidden under ``_meta_`` so raw ``n`` /
      sampling counts cannot leak into the analysis matrix as size proxies.
"""

import logging
from typing import Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Perturbation added to ζ (Bunch et al. 2020, as used by magnipy's
# ``magnitude_weights``) when the direct solve fails on a singular matrix.
_SINGULAR_EPS = 1e-2


def _magnitude(D, t: float) -> float:
    """Metric-space magnitude ``|tX| = 1ᵀ ζ⁻¹ 1`` for distance matrix ``D``
    at scale ``t``.

    Reproduces magnipy's closed-form linear solve. The similarity matrix is
    ``ζ_ij = exp(-t · D_ij)`` and the magnitude is the sum of the magnitude
    weight vector ``w`` solving ``ζ w = 1`` — identical to summing every entry
    of ``ζ⁻¹`` (magnipy ``weights_cholesky`` / ``weights_pinv`` /
    ``compute_magnitude_no_gpu``). Computed in float64.

    Args:
        D: ``(n, n)`` symmetric distance matrix (zero diagonal). Array-like.
        t: positive scale parameter.

    Returns:
        The magnitude ``|tX|`` as a Python float. A single point (``n == 1``)
        returns exactly ``1.0``.

    On a singular ζ (near-duplicate points, or large/degenerate ``t``) the
    direct solve is retried after adding ``_SINGULAR_EPS · I`` to ζ, mirroring
    magnipy's ``perturb_singularities`` fallback; if that still fails we fall
    back to the Hermitian pseudo-inverse (magnipy ``weights_pinv``).
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if n == 0:
        return float("nan")
    if n == 1:
        # One-point property: |tX| = 1 for all t (magnipy sets w = 1/n · 1).
        return 1.0

    # ζ_ij = exp(-t · D_ij); symmetric, unit diagonal.
    Z = np.exp(-t * D)

    ones = np.ones(n, dtype=np.float64)
    try:
        # Positive-definite solve (matches magnipy's cholesky/scipy path:
        # w = ζ⁻¹ 1, magnitude = Σ w). np.linalg.solve is bit-identical to
        # the two-triangular-solve Cholesky form up to fp rounding.
        w = np.linalg.solve(Z, ones)
        return float(w.sum())
    except np.linalg.LinAlgError:
        pass

    # Fallback 1: perturb the diagonal (Bunch et al. 2020) and retry.
    try:
        w = np.linalg.solve(Z + _SINGULAR_EPS * np.eye(n), ones)
        return float(w.sum())
    except np.linalg.LinAlgError:
        pass

    # Fallback 2: Hermitian pseudo-inverse (magnipy weights_pinv).
    Zsym = 0.5 * (Z + Z.T)
    M = np.linalg.pinv(Zsym, hermitian=True)
    return float(M.sum())


def _pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix of the ``(n, d)`` point cloud."""
    X = np.asarray(X, dtype=np.float64)
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j ; clamp for fp noise.
    sq = np.sum(X * X, axis=1)
    G = X @ X.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(d2, 0.0, out=d2)
    D = np.sqrt(d2)
    np.fill_diagonal(D, 0.0)
    return D


def _median_heuristic_scale(D: np.ndarray) -> float:
    """magnipy's median-heuristic scale: ``1 / sqrt(median_offdiag / 2)``.

    (``scales.median_heuristic_from_distances``.) Returns ``nan`` if there is
    no positive off-diagonal distance.
    """
    iu = np.triu_indices(D.shape[0], k=1)
    d = D[iu]
    d = d[d > 0]
    if d.size == 0:
        return float("nan")
    median = np.median(d)
    if median <= 0:
        return float("nan")
    return float(1.0 / np.sqrt(median / 2.0))


def _magnitude_dimension(ts: np.ndarray, mags: np.ndarray) -> float:
    """Magnitude dimension = max slope of ``log|tX|`` vs ``log t``
    (Meckes 2015; magnipy ``magitude_dimension_profile_interp`` +
    ``magnitude_dimension`` = ``max`` of the secant slopes across scales).

    Uses only strictly positive magnitudes and scales.
    """
    ts = np.asarray(ts, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    valid = (ts > 0) & (mags > 0) & np.isfinite(mags)
    ts, mags = ts[valid], mags[valid]
    if ts.size < 2:
        return float("nan")
    order = np.argsort(ts)
    lt = np.log(ts[order])
    lm = np.log(mags[order])
    dlt = np.diff(lt)
    good = dlt > 0
    if not np.any(good):
        return float("nan")
    slopes = np.diff(lm)[good] / dlt[good]
    return float(np.max(slopes))


@register_task("geometry_magnitude")
class MagnitudeTask(DiagnosticTask):
    """Metric-space magnitude of the hidden-state cloud (Limbeck et al.,
    NeurIPS 2024, arXiv:2311.16054; repo aidos-lab/magnipy).

    Outputs (flat float64):
        magnitude_low_scale    — |tX| at the smallest sampled scale
        magnitude_med_scale    — |tX| at the median-heuristic scale
        magnitude_high_scale   — |tX| at the largest sampled scale
        magnitude_ratio        — |tX|_high / n  (fraction of points resolved)
        magnitude_auc          — area under log|tX| vs log t (trapezoid)
        magnitude_dimension    — max slope of log|tX| vs log t (Meckes 2015)
        magnitude_median_scale — the median-heuristic scale t_med itself
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Metric-Space Magnitude...")

        num_samples = int(self.config.get("num_samples", 100))
        n_scales = int(self.config.get("n_scales", 10))
        scale_span = float(self.config.get("scale_span", 10.0))
        layer_idx = self.config.get("layer_idx", -1)
        use_cache = self.config.get("use_cache", True)
        seed = int(self.config.get("subsample_seed", 42))

        # --- 1. Fetch a hidden-state point cloud ---------------------------
        H = None
        if cache is not None and cache.is_populated and use_cache:
            H = cache.get_hidden_states(layer_idx=layer_idx, per_sample=False)
        else:
            from .utils import collect_hidden_states

            if dataset is None:
                dataset = [
                    {"text": "The quick brown fox jumps over the lazy dog."}
                    for _ in range(max(num_samples, 50))
                ]
            H = collect_hidden_states(
                model, tokenizer, dataset,
                num_samples=max(num_samples, 50), layer_idx=layer_idx,
            )

        if H is None:
            return {"error": "No hidden states available for magnitude"}

        if isinstance(H, torch.Tensor):
            X = H.detach().float().cpu().numpy()
        else:
            X = np.asarray(H, dtype=np.float64)

        if X.ndim != 2 or X.shape[0] < 2:
            return {"error": "Need at least 2 points for magnitude"}

        # --- 2. Subsample to keep the O(n^3) inverse tractable -------------
        n_total = X.shape[0]
        if n_total > num_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_total, num_samples, replace=False)
            X = X[idx]
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # --- 3. Distance matrix + scale ladder -----------------------------
        D = _pairwise_distance_matrix(X)
        t_med = _median_heuristic_scale(D)
        if not np.isfinite(t_med) or t_med <= 0:
            return {"error": "Degenerate cloud (no positive distances)"}

        if n_scales < 2:
            n_scales = 2
        ts = np.geomspace(t_med / scale_span, t_med * scale_span, n_scales)

        # --- 4. Magnitude function -----------------------------------------
        mags = np.array([_magnitude(D, float(t)) for t in ts], dtype=np.float64)

        finite = np.isfinite(mags)
        if finite.sum() < 2:
            return {"error": "Magnitude function ill-conditioned at all scales"}

        # Index of the scale nearest t_med (median-heuristic anchor).
        med_i = int(np.argmin(np.abs(ts - t_med)))

        mag_dim = _magnitude_dimension(ts, mags)

        # Area under log|tX| vs log t (trapezoid over finite scales).
        lt = np.log(ts[finite])
        lm = np.log(np.clip(mags[finite], 1e-12, None))
        auc = float(np.trapezoid(lm, lt)) if lt.size >= 2 else float("nan")

        mag_high = float(mags[finite][-1])

        return {
            "magnitude_low_scale": float(mags[finite][0]),
            "magnitude_med_scale": float(mags[med_i])
            if np.isfinite(mags[med_i]) else float("nan"),
            "magnitude_high_scale": mag_high,
            "magnitude_ratio": float(mag_high / n),
            "magnitude_auc": auc,
            "magnitude_dimension": float(mag_dim),
            "magnitude_median_scale": float(t_med),
            # _meta_ prefix => excluded from the analysis feature matrix so raw
            # sampling counts cannot leak in as size proxies (Audit-V2).
            "_meta_n_points": int(n),
            "_meta_n_scales": int(int(np.sum(finite))),
        }
