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
      ``num_samples`` points keeps the ``O(n³)`` inverse tractable; every
      magnitude key grows with n, so n is fixed.
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
from scipy.linalg import cho_factor, solve_triangular
from scipy.spatial.distance import pdist, squareform

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

    # Duplicates are NOT removed: magnipy's compute_magnitude_until_convergence
    # perturbs the singular zeta instead (|{0,0,1}| != |{0,1}| there), and this
    # matches it (test_magnitude_parity.py::test_duplicate_cloud_magnitude_matches_magnipy).
    if n == 1:
        # One-point property: |tX| = 1 for all t (magnipy sets w = 1/n · 1).
        return 1.0

    # ζ_ij = exp(-t · D_ij); symmetric, unit diagonal.
    Z = np.exp(-t * D)

    ones = np.ones(n, dtype=np.float64)

    def _weights_cholesky(M: np.ndarray) -> np.ndarray:
        """magnipy ``weights_cholesky``, verbatim: cho_factor + 2 triangular solves."""
        c, _lower = cho_factor(M)
        x = solve_triangular(c, ones, trans=1)
        return solve_triangular(c, x.T, trans=0)

    # Failure detection is Cholesky's, as magnipy's: LU accepts a numerically
    # indefinite zeta and returns an invalid (even negative) magnitude
    # (test_magnitude_cholesky_detection.py).
    try:
        return float(_weights_cholesky(Z).sum())
    except Exception:
        pass

    # Fallback 1: one retry at zeta + 0.01 I (Bunch et al. 2020), as magnipy.
    try:
        return float(_weights_cholesky(Z + _SINGULAR_EPS * np.eye(n)).sum())
    except Exception:
        pass

    # Fallback 2: Hermitian pseudo-inverse (magnipy weights_pinv).
    Zsym = 0.5 * (Z + Z.T)
    M = np.linalg.pinv(Zsym, hermitian=True)
    return float(M.sum())


def _pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix of the ``(n, d)`` point cloud."""
    X = np.asarray(X, dtype=np.float64)
    # Direct differences (pdist), no centring: translation-exact, and the
    # centroid's ULP would otherwise become the resolution floor
    # (test_scale_invariance.py::test_magnitude_distances_are_translation_invariant).
    D = squareform(pdist(X, metric="euclidean"))
    np.fill_diagonal(D, 0.0)
    return D


def _median_heuristic_scale(D: np.ndarray) -> float:
    """magnipy's median-heuristic scale: ``1 / sqrt(median_offdiag / 2)``.

    (``scales.median_heuristic_from_distances``.) Medians ALL upper-triangle
    distances — including zero distances between duplicate points, exactly like
    magnipy — and returns ``nan`` only when that median is non-positive. The old
    code pre-filtered ``d > 0``, which shifted the scale on duplicate-heavy
    clouds away from magnipy's value.
    """
    iu = np.triu_indices(D.shape[0], k=1)
    d = D[iu]
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


def _magnitude_summary(D: np.ndarray, n_scales: int, scale_span: float) -> dict:
    """Magnitude-function summary metrics for one distance matrix ``D``.

    ``magnitude_med_scale`` is ``|tX|`` evaluated at the median-heuristic scale
    ``t_med`` EXACTLY (a direct solve), not at whichever sampled ladder point
    happened to be nearest ``t_med`` (~0.774·t_med at defaults), which is what
    the old code reported.
    """
    n = D.shape[0]
    t_med = _median_heuristic_scale(D)
    if not np.isfinite(t_med) or t_med <= 0:
        return {"error": "Degenerate cloud (no positive distances)"}

    if n_scales < 2:
        n_scales = 2
    ts = np.geomspace(t_med / scale_span, t_med * scale_span, n_scales)
    mags = np.array([_magnitude(D, float(t)) for t in ts], dtype=np.float64)
    # |tX| lies in [1, n] for a positive-definite space; anything else is a
    # failed solve.
    mags = np.where((mags >= 1.0 - 1e-6) & (mags <= n + 1e-6), mags, np.nan)
    # |tX| is non-decreasing in t, so a drop is a numerical failure too
    # (necessary, not sufficient).
    _run_max = np.fmax.accumulate(np.where(np.isnan(mags), -np.inf, mags))
    _bad = np.isfinite(mags) & (mags < _run_max - 1e-6)
    if _bad.any():
        logger.info(f"  magnitude: {int(_bad.sum())} ladder points dropped as "
                    f"non-monotone (|tX| must be non-decreasing in t)")
        mags = np.where(_bad, np.nan, mags)

    finite = np.isfinite(mags)
    if finite.sum() < 2:
        return {"error": "Magnitude function ill-conditioned at all scales"}

    mag_dim = _magnitude_dimension(ts, mags)

    # Area under log|tX| vs log t (trapezoid over finite scales).
    lt = np.log(ts[finite])
    lm = np.log(np.clip(mags[finite], 1e-12, None))
    # np.trapezoid exists only on NumPy >= 2.0, np.trapz only below.
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(_trapz(lm, lt)) if lt.size >= 2 else float("nan")

    # MagArea-like area (Limbeck et al., Def. 3.3) on a linear t axis over
    # [0, 10/median(d)]: anchored on 1/median(d), which scales as 1/s under a
    # global rescale, so the area / range is scale-free.
    _T_CUT = 10.0
    _iu = np.triu_indices(D.shape[0], k=1)
    _med_d = float(np.median(D[_iu])) if D[_iu].size else float("nan")
    t_anchor = (1.0 / _med_d) if np.isfinite(_med_d) and _med_d > 0 else t_med
    n_fine = 512
    t_fine = np.linspace(0.0, t_anchor * _T_CUT, n_fine + 1)[1:]
    m_fine = np.array([_magnitude(D, float(t)) for t in t_fine], dtype=np.float64)
    # Reject solves outside [1, n] (ill-conditioned zeta).
    m_fine = np.where((m_fine >= 1.0 - 1e-6) & (m_fine <= n + 1e-6),
                      m_fine, np.nan)
    ok = np.isfinite(m_fine)
    n_bad = int((~ok).sum())
    if n_bad:
        logger.info(f"  magnitude: {n_bad}/{n_fine} scales rejected as "
                    f"ill-conditioned (|tX| outside [1, n])")
    if ok.sum() >= 2:
        # |tX| -> 1 as t -> 0, so prepend that limit rather than dropping it.
        tt = np.concatenate([[0.0], t_fine[ok]])
        mm = np.concatenate([[1.0], m_fine[ok]])
        mag_area = float(_trapz(mm, tt))
        # Mean magnitude over the window, divided by n.
        mag_area_norm = float(mag_area / (tt[-1] * n)) if tt[-1] > 0 and n else float("nan")
    else:
        mag_area = float("nan")
        mag_area_norm = float("nan")

    mag_high = float(mags[finite][-1])
    med_scale = _magnitude(D, float(t_med))  # |tX| AT t_med, not the nearest ladder point
    if not (1.0 - 1e-6 <= med_scale <= n + 1e-6):
        med_scale = float("nan")

    return {
        "magnitude_low_scale": float(mags[finite][0]),
        "magnitude_med_scale": float(med_scale) if np.isfinite(med_scale) else float("nan"),
        # _meta_: both are saturated constants (|tX| = n at the top scale).
        "_meta_magnitude_high_scale": mag_high,
        "_meta_magnitude_ratio": float(mag_high / n),
        "magnitude_auc": auc,
        "magnitude_area": mag_area,
        "_meta_magnitude_area_anchor": float(t_anchor),
        # _meta_: dividing by n breaks the duplicate-point axiom (Table S.1).
        "_meta_magnitude_area_norm": mag_area_norm,
        "magnitude_dimension": float(mag_dim),
        "magnitude_median_scale": float(t_med),
        # Counts are _meta_ so they cannot enter as size proxies.
        "_meta_n_points": int(n),
        "_meta_n_scales": int(int(np.sum(finite))),
    }


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
        # The point count is part of the estimand, so the shared cache is used
        # only when it holds exactly num_samples documents (larger or smaller
        # caches change the cloud), and both routes keep every token.
        H = None
        _cached_docs = len(getattr(cache, "_sample_lengths", []) or []) if cache else 0
        _cache_ok = (cache is not None and cache.is_populated and use_cache
                     and _cached_docs == int(num_samples))
        if cache is not None and cache.is_populated and use_cache and not _cache_ok:
            logger.info(
                f"  magnitude: shared cache holds {_cached_docs} documents != "
                f"configured {num_samples}; self-collecting so the point count "
                f"does not depend on which tasks co-ran.")
        H = None
        if _cache_ok:
            # post-final-norm=False, DELIBERATELY: one configured layer
            # (default -1, the model's output representation).
            H = cache.get_hidden_states(layer_idx=layer_idx, per_sample=False,
                                        replace_final_norm=False)
            if H is None:
                # A populated cache without hidden states: self-collect.
                logger.info(
                    "  Magnitude: shared cache is populated but carries no "
                    "hidden states; self-collecting.")
        _cache_supplied = H is not None
        if H is None:
            from .utils import collect_hidden_states

            if dataset is None:
                dataset = [
                    {"text": "The quick brown fox jumps over the lazy dog."}
                    for _ in range(max(num_samples, 50))
                ]
            H = collect_hidden_states(
                model, tokenizer, dataset,
                num_samples=max(num_samples, 50), layer_idx=layer_idx,
                replace_final_norm=False,
                # Every token, matching the cache route.
                max_tokens_per_doc=None,
            )
        # DERIVED from what actually supplied the states, not from the plan.
        _source = "shared_cache" if (_cache_ok and _cache_supplied) else "self_collected"

        if H is None:
            return {"error": "No hidden states available for magnitude"}

        if isinstance(H, torch.Tensor):
            # float64 on both branches: float32 cannot resolve the separations
            # of a cloud far from the origin.
            X = H.detach().double().cpu().numpy()
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

        # --- 3-4. Distance matrix + magnitude summary ----------------------
        D = _pairwise_distance_matrix(X)
        _out = _magnitude_summary(D, n_scales, scale_span)
        # Provenance: route, point count and document count.
        if isinstance(_out, dict) and "error" not in _out:
            _out["_meta_magnitude_source"] = _source
            _out["_meta_magnitude_n_points"] = int(n)
            _out["_meta_magnitude_n_docs"] = int(_cached_docs) if _cache_ok else int(num_samples)
        return _out
