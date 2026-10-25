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

    # DO NOT DE-DUPLICATE. Verified against live magnipy 2026-09-07.
    #
    # Round 9 filed a RED (and an independent review confirmed it) stating that
    # magnipy de-duplicates, so |{0,1}| and |{0,0,1}| are bit-identical there
    # while BLME returned 1.454154504046096 against 1.462117157260010
    # (-0.545%). I implemented the de-duplication. It was wrong, and the
    # existing parity test caught it.
    #
    # Live `magnipy.magnitude.compute.compute_magnitude_until_convergence`
    # (method="cholesky", perturb_singularities=True) returns:
    #     |{0,1}|   = 1.4621171572600098
    #     |{0,0,1}| = 1.4541545040460961   <- NOT the same
    # printing "2-th leading minor of the array is not positive definite ...
    # perturbing matrix" on the way. magnipy PERTURBS the singular zeta; it does
    # not quotient the point set. **1.4541545040460961 is exactly what BLME
    # already returned**, so BLME was bit-matching the reference before the
    # "fix", and de-duplicating broke that agreement (2.223711313215775 against
    # the pinned 2.2096410948415732 on the 4-point duplicate cloud).
    #
    # Magnitude IS a set property in pure mathematics, and on that ground the
    # reviewers' argument is sound. It is not the ground this task stands on:
    # the campaign rule is that the released reference wins, and the reference
    # made the other choice. Kept, with the disagreement documented rather than
    # silently resolved in the maths' favour.
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

    # DETECTION MUST BE CHOLESKY, NOT LU.
    #
    # This previously ran `np.linalg.solve`, whose comment claimed it was
    # "bit-identical to the two-triangular-solve Cholesky form up to fp
    # rounding". That is true of the SOLUTION on a positive-definite ζ and
    # false of the FAILURE TEST, which is the only thing this branch is for.
    # LU factorises any nonsingular matrix, so it raises only when ζ is
    # EXACTLY singular; Cholesky raises whenever ζ is not positive definite.
    # A ζ that is numerically indefinite but LU-invertible therefore returned
    # from an invalid solve instead of reaching the perturbation below.
    #
    # Reached by a real 100x768 GPT-2 cloud with 95 distinct points (exact
    # duplicate pairs (5,6), (5,61), (6,61), (11,65), (44,59), (85,90)): at
    # t_med = 0.1713026078310342, rank(ζ) = 95 and cond(ζ) = 1.175309435246e17.
    # `np.linalg.solve` raised nothing and returned magnitude
    # -2034.8015584361294 with max|ζw - 1| = 129.0 -- a NEGATIVE magnitude,
    # which is impossible for a finite metric space at t > 0. Released
    # magnipy raises "7-th leading minor of the array is not positive
    # definite", perturbs, and returns 71.00399079602064. The public headline
    # was area 2.961573319962391 against the reference's 4.636803035166263
    # (-36.13%) with a NaN median, because 267 of 513 integration points were
    # discarded as invalid solves (codex round 12, 2026-09-08).
    #
    # magnipy catches a bare `Exception` here, so this does too: the point is
    # to reproduce the reference's failure detection, not to improve on it.
    try:
        return float(_weights_cholesky(Z).sum())
    except Exception:
        pass

    # Fallback 1: perturb the diagonal (Bunch et al. 2020) and retry, which is
    # magnipy's `perturb_singularities` branch -- ONE retry at 0.01·I, through
    # the same Cholesky routine, exactly as the reference does it.
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
    # Differences are taken DIRECTLY, not via ||x||^2+||y||^2-2x.y.
    #
    # The Gram identity catastrophically cancels once the cloud sits far from
    # the origin, and magnitude is a translation-invariant functional, so that
    # is a pure artefact. On the default_rng(343) 40x9 cloud translated by
    # norm 1e8 along a fixed unit direction, the Gram form manufactured 6
    # false zero pairs and corrupted the largest pair distance; the direct
    # form is translation-exact (adversarial review, 2026-09-04). Centering
    # first removes the leading term of the shift and costs nothing.
    # NO CENTRING. It was added 2026-09-04 "to remove the leading term of the
    # shift", and it is both redundant and destructive:
    #   * redundant -- `pdist` differences coordinates before squaring, so the
    #     shift cancels on its own. Measured on the 40x9 cloud that comment
    #     cites, centred and uncentred agree to the digit at +1e3/1e6/1e10.
    #   * destructive -- the ULP of the centroid becomes the resolution floor.
    #     For X=[0, 1, 9.999999562023526e24] the mean's ULP is 5.369e8, so
    #     d(0,1)=1 centred to 0 and magnitude read 1.9851238855228808 against
    #     executed magnipy's 2.4621171572600096 (-19.37%).
    # (gpt-6-astra round 10, 2026-09-07.)
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
    # Same [1, n] bound as the fine grid below. The guard was applied ONLY to
    # the fine grid, leaving the ladder and med_scale unguarded -- distilgpt2
    # produced `magnitude_med_scale = -661,182` (a NEGATIVE magnitude) and
    # `magnitude_auc = -18.255`, and gpt2 gave `magnitude_dimension = 102.7`
    # against 1.46 on a clean cloud (adversarial review, 2026-09-03).
    mags = np.where((mags >= 1.0 - 1e-6) & (mags <= n + 1e-6), mags, np.nan)
    # MONOTONICITY, which is strictly stronger than the [1, n] bound.
    #
    # |tX| is non-decreasing in t for a positive-definite metric space, so any
    # drop is a numerical failure.
    #
    # CORRECTION (2026-09-04): the specific counterexample described below --
    # a gpt2 cloud with one duplicate pair reading "3.0 against a true 99.0" --
    # does NOT reproduce. Re-measured, that cloud has cond 9.7e4-1.5e8 and a
    # clean monotone ladder (high_scale 99.00001 against a true 98.99999998).
    # The guard is still right and still worth having -- monotonicity is a
    # property |tX| must satisfy and the [1, n] bound cannot check it -- but the
    # number below overstates the failure it was written from. Garbage does
    # still pass: 50 NEAR-duplicate pairs give a rung of 52.500 against a true
    # saturation of 50, monotone and inside [1, n], so the guard is necessary
    # but not sufficient.
    #
    # Original (unreproduced) note: A real gpt2 cloud with ONE duplicate pair
    # produced a ladder ending 80.442, 83.002, **3.008, 3.000, 3.000, 3.000** --
    # a fall of 79.99 -- with every value inside [1, n], so the bound alone
    # flagged nothing and `magnitude_high_scale` read 3.0 against a true 99.0
    # (adversarial review, 2026-09-03).
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
    # np.trapezoid is NumPy>=2.0-only (np.trapz removed there); REPRODUCE.md
    # pins numpy<2.0, so use the same lazy lookup as isotropy/ablation/
    # persistence_landscape (eager getattr would evaluate np.trapz on 2.x).
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(_trapz(lm, lt)) if lt.size >= 2 else float("nan")

    # MagArea (Limbeck et al., Def. 3.3): the area under the magnitude function
    # itself, on a LINEAR t axis, from 0 to a cutoff:
    #
    #     MagArea = integral_0^{t_cut} |tX| dt
    #
    # `magnitude_auc` above is a DIFFERENT quantity -- log |tX| integrated over
    # log t on a per-cloud ladder anchored at the median heuristic. It is not
    # MagArea and it is not scale invariant: rescaling one fixed geometry moved
    # it by 5.0x, and on the paper's two-point space it returns 2.005593 where
    # the exact MagArea is 2*ln(10) = 4.605170, i.e. -56.4% (adversarial
    # review, 2026-09-03). Both are now reported, under names that say which is
    # which.
    #
    # t_cut is taken where the magnitude has essentially converged to n (the
    # cardinality), which is the regime Limbeck integrate over. Expressed in
    # units of t_med so the integral is invariant to a global rescaling of the
    # cloud, which MagArea on a raw t axis is not.
    # The integration window is anchored on 1/median(d), NOT on t_med.
    #
    # t_med is magnipy's `1/sqrt(median/2)`, which this module ports faithfully
    # and must keep -- but it scales as 1/sqrt(s) under a global rescaling of
    # the cloud, while magnitude is covariant as Mag_{sX}(t) = Mag_X(s*t), i.e.
    # linear in the distance. Anchoring the integral on t_med therefore leaves
    # it scale DEPENDENT (measured 0.758 vs 0.908 for the same geometry scaled
    # by 7). Anchoring on 1/median(d), which does scale as 1/s, makes the
    # normalized area exactly invariant.
    _T_CUT = 10.0
    _iu = np.triu_indices(D.shape[0], k=1)
    _med_d = float(np.median(D[_iu])) if D[_iu].size else float("nan")
    t_anchor = (1.0 / _med_d) if np.isfinite(_med_d) and _med_d > 0 else t_med
    n_fine = 512
    t_fine = np.linspace(0.0, t_anchor * _T_CUT, n_fine + 1)[1:]
    m_fine = np.array([_magnitude(D, float(t)) for t in t_fine], dtype=np.float64)
    # Reject ill-conditioned solves. zeta = exp(-t*D) becomes near-singular as
    # t -> 0 or when the cloud has duplicate points, and `np.linalg.solve` then
    # returns garbage SILENTLY: on real gpt2 hidden states with duplicates,
    # cond(zeta) reached 1.9e22 and `magnitude_area_norm` came out at
    # -7.42e+16 (adversarial review, 2026-09-03).
    #
    # Magnitude is bounded in [1, n] for a positive-definite metric space, so
    # anything outside that is a numerical failure, not a measurement.
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
        # Scale-free form: divide by the t-range, giving a MEAN magnitude over
        # the integration window, and by n so clouds of different cardinality
        # are on the same footing.
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
        # DEMOTED to _meta_ (2026-09-04). Both are SATURATED CONSTANTS, not
        # measurements: the ladder's top scale is chosen as 10/median(distance),
        # by which point magnitude has converged to the cardinality n for every
        # real representation. Measured across ALL 61 archived models and 4
        # fresh runs, `magnitude_high_scale` is 100.000000 and
        # `magnitude_ratio` is 1.000000 EVERY time -- two shipped features with
        # exactly zero variance, which a regression cannot use and a reader
        # would reasonably mistake for signal (independent Opus review,
        # 2026-09-04). Kept for provenance, out of the feature matrix.
        "_meta_magnitude_high_scale": mag_high,
        "_meta_magnitude_ratio": float(mag_high / n),
        "magnitude_auc": auc,
        "magnitude_area": mag_area,
        "_meta_magnitude_area_anchor": float(t_anchor),
        # DEMOTED to _meta_ (2026-09-04): it violates Limbeck et al.'s own
        # Table S.1 duplicate-point axiom, moving 0.930689027 -> 0.617064163
        # (-33.7%) when points are duplicated -- a transformation magnitude is
        # explicitly required to be insensitive to.
        "_meta_magnitude_area_norm": mag_area_norm,
        "magnitude_dimension": float(mag_dim),
        "magnitude_median_scale": float(t_med),
        # _meta_ prefix => excluded from the analysis feature matrix so raw
        # sampling counts cannot leak in as size proxies (Audit-V2).
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
        #
        # REFUSE AN UNDERSIZED SHARED CACHE, and MATCH THE TOKEN PROTOCOL --
        # the same two-part fix already applied to `geometry_schatten`,
        # `geometry_isoscore` and `geometry_vendi_score`, still missing here.
        #
        # `core.py` sizes the shared corpus to max(num_samples) over ALL
        # requested tasks, and magnitude is a diversity functional of a POINT
        # CLOUD, so its value depends on how many points it gets. Measured by
        # co-runner set alone (independent Opus review, 2026-09-05):
        #
        #     magnitude_area  4.894 / 5.298 / 5.689 / 5.451     (16.2% spread)
        #
        # The same review found the two routes are different PROTOCOLS: the
        # cache retains every token while `collect_hidden_states` defaults to 10
        # per document, so a cached and an uncached run of the same model on the
        # same corpus built clouds of different sizes -- and this metric is a
        # function of cloud size. Both are pinned below.
        H = None
        # EXACT match, not ">=". The guard refused an UNDERSIZED cache and
        # silently accepted an OVERSIZED one, and magnitude is a diversity
        # functional OF THE POINT COUNT, so more documents change the answer:
        #
        #   magnitude_area  5.097373 (self)  5.097373 (50, refused)
        #                   5.097373 (100)   5.689075 (200)   5.451367 (1000)
        #
        # Production `cache_num_samples` resolves to 200, so the study read
        # 5.689075 -- **+11.6%** over its own configured N -- and
        # `_meta_magnitude_n_points` reported 100 in every case, so the
        # provenance added to catch exactly this could not (independent Opus
        # review, 2026-09-06).
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
            # post-final-norm=False, DELIBERATELY: magnitude/diversity is
            # reported for ONE configured layer (default -1, the model's
            # output representation), never as a cross-layer profile.
            H = cache.get_hidden_states(layer_idx=layer_idx, per_sample=False,
                                        replace_final_norm=False)
            if H is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. A cache
                # filled for an attention-only consumer returns None here, and
                # this task then returned an error instead of self-collecting
                # -- so co-running another task deleted every magnitude
                # headline. Same defect as `geometry_vendi_score` (codex round
                # 12, 2026-09-08); this is its second site.
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
                # EVERY token, matching the cache route. The 10-per-document
                # default made the two routes different protocols.
                max_tokens_per_doc=None,
            )
        # DERIVED from what actually supplied the states, not from the plan.
        _source = "shared_cache" if (_cache_ok and _cache_supplied) else "self_collected"

        if H is None:
            return {"error": "No hidden states available for magnitude"}

        if isinstance(H, torch.Tensor):
            # float64, MATCHING the numpy branch below.
            #
            # The tensor branch narrowed to float32 while the array branch used
            # float64, so the same cloud gave different answers depending on how
            # it arrived -- and magnitude is a translation-invariant functional,
            # so a lossless shift must not move it. On a 32-point cloud the task
            # returned median 1.0002907220623911 and area 41244960994.27025, and
            # translating by exactly +1 (every pairwise distance unchanged to 0)
            # made evaluate() report "Degenerate cloud (no positive distances)":
            # at that magnitude float32 cannot resolve the separations at all.
            # `_pairwise_distance_matrix` already computes in double, so this
            # only stops the input being damaged before it gets there
            # (codex round 15, 2026-09-09).
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
        # Provenance: which route, and how many points. Magnitude is a function
        # OF THE POINT COUNT, so a comparison between two runs is only valid if
        # these agree.
        if isinstance(_out, dict) and "error" not in _out:
            _out["_meta_magnitude_source"] = _source
            _out["_meta_magnitude_n_points"] = int(n)
            # The DOCUMENT count too: `n` is the subsampled point count and was
            # 100 in every one of the five cache sizes above, so on its own it
            # could not reveal the coupling it was added to reveal.
            _out["_meta_magnitude_n_docs"] = int(_cached_docs) if _cache_ok else int(num_samples)
        return _out
