from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")

class _BF16_EPS_PROXY:
    """numpy has no bfloat16, and bf16's eps (2^-8 = 0.0078125) is 64x
    float32's. Passing this sentinel as ``source_dtype`` selects it."""
    eps = 0.0078125


def _svd_metrics_for_layer(X, source_dtype=None):
    """Compute SVD-based isotropy metrics for a single (N, D) matrix."""
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    # FLOAT64 for every reduction below.
    #
    # Removing the absolute epsilons made the formulas scale-invariant on
    # paper, but the ARITHMETIC still was not: `eigenvalues ** 2` forms S**4,
    # which underflows in float32 well before the data does. A representable
    # float32 rank-one 10x4 cloud gave the correct participation ratio
    # 1.000000 at scale 3e-12 and NaN at 1e-12 because sum_eig_sq underflowed
    # to 0, and `np.linalg.norm` squares in float32 too -- ten rows of
    # magnitude 3e-23, all nonzero, produced 0.145793468 against an exact
    # 0.288888889 (-49.54%) (adversarial review round 2, 2026-09-04).
    # THE PRECISION THAT PRODUCED THE DATA, not the container it arrives in.
    #
    # `collect_hidden_states` and the shared cache both `.float()` every tensor
    # on the way in, so sniffing the array's own dtype ALWAYS reports float32 --
    # even when the model ran in bfloat16, whose eps is 64x larger. Production
    # runs bf16, and the difference is not academic: on the same model and
    # corpus `cond_number` reads 19.811127663 under the float32 tolerance but
    # 449.643167800 under bf16's on pythia-70m (22.7x), and 24.166894336 vs
    # 629.738056415 on pythia-160m (independent Opus review, 2026-09-04). The
    # caller passes the model's real dtype; float32 remains the default for
    # direct callers with genuinely float32 data.
    # The PRIMARY tolerance always follows the CONTAINER dtype, so established
    # values never shift. `source_dtype` only feeds the disclosed variant below.
    _arr_dt = np.asarray(X).dtype
    _stored_dtype = (_arr_dt if np.issubdtype(_arr_dt, np.floating)
                     else np.dtype(np.float32))
    _stored_eps = float(np.finfo(_stored_dtype).eps)
    X = np.asarray(X, dtype=np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    if len(X) < 10:
        return None
    # Keep the RAW (uncentered) states for the Ethayarajh 2019 anisotropy
    # cosine below; SVD-based isotropy metrics use the centered version.
    X_raw = X
    X = X - np.mean(X, axis=0)

    try:
        # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
        # the singular values back to the input dtype, and that cast overflows
        # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
        # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
        # exactly. The helper factors out the max first, which only shifts
        # exponents (2026-09-07).
        from .utils import safe_svdvals
        S = safe_svdvals(X)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            _, S, _ = scipy_svd(X, full_matrices=False)
        except Exception:
            return None

    explained_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    # np.trapezoid was introduced in NumPy 2.0 and np.trapz removed in it.
    # Use a LAZY lookup: `getattr(np, "trapezoid", np.trapz)` eagerly
    # evaluates np.trapz, which raises AttributeError on NumPy 2.x.
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = _trapz(explained_variance) / max(1, len(explained_variance))

    # Roy-Vetterli effective rank on raw singular values (shared helper).
    from .utils import effective_rank as _effective_rank
    effective_rank = float(_effective_rank(S))

    # Scale S out before squaring: (sum L)^2 / sum(L^2) is invariant to any
    # positive rescale, so normalising by S[0] costs nothing and removes the
    # last route to an underflow (S**4 spans four times the dynamic range of
    # S). float64 alone is not enough at the extremes.
    _s0 = float(S[0]) if S.size else 0.0
    _Sn = (S / _s0) if _s0 > 0 else S
    eigenvalues = _Sn ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    # SCALE-INVARIANT. The participation ratio is (sum L)^2 / sum(L^2), which
    # is invariant to rescaling the matrix -- adding a fixed 1e-12 to the
    # denominator is not. Measured: on one fixed nonzero matrix the exact value
    # is 4.924101 at every scale, while the +1e-12 form gave 4.924101 at scale
    # 1, 4.812010 at 1e-4 (-2.276%) and collapsed toward 0 below that
    # (adversarial review, 2026-09-04). Guard the genuinely-zero case instead.
    participation_ratio = (float((sum_eig ** 2) / sum_eig_sq)
                           if sum_eig_sq > 0 else float("nan"))

    # Ethayarajh 2019 Eq. 4 anisotropy = E[cos(f(x), f(y))] over distinct
    # random pairs, on the RAW representations (centering would force it to ~0).
    #
    # Computed EXACTLY rather than by Monte Carlo. The mean cosine over all
    # ordered distinct pairs has a closed form in O(N*D): with u_i = x_i/||x_i||
    # and m = mean(u),
    #     E_{i != j}[u_i . u_j] = (N * ||m||^2 - 1) / (N - 1),
    # since sum_{i,j} u_i.u_j = ||sum_i u_i||^2 = N^2 ||m||^2 and the N diagonal
    # terms each equal 1.
    #
    # The previous 1000-pair sample with a HARD-CODED seed gave errors up to
    # -0.047 on a metric whose cross-model range is 0.36-0.99, and because
    # models sharing a tokenizer share N they drew the IDENTICAL index set --
    # so it was a fixed offset correlated within tokenizer family, not noise
    # that averages out (adversarial review, 2026-09-02).
    # Row-scaled before norming, for the same reason: a row of magnitude 3e-23
    # has a float64-representable norm, but squaring it does not survive in
    # float32 and barely survives in float64. Dividing each row by its own
    # largest-magnitude coordinate makes the norm exact at any scale, and the
    # unit vector it produces is identical.
    _rmax = np.max(np.abs(X_raw), axis=1)
    _safe = _rmax > 0
    _Xs = np.zeros_like(X_raw)
    _Xs[_safe] = X_raw[_safe] / _rmax[_safe, None]
    norms = np.linalg.norm(_Xs, axis=1)
    # Drop zero-norm rows rather than clamping them. A clamped row becomes the
    # ZERO vector, whose self-cosine is 0 rather than 1, which breaks the
    # "N unit diagonal terms" premise the closed form subtracts (measured
    # error 4.3e-5 with 2 zero rows in 200).
    # EXACTLY zero only. Cosine is defined for any nonzero vector however
    # short, so an absolute cutoff both breaks Eq. 4's scale invariance and
    # silently deletes real rows: for eight +e1 rows and two -1e-13*e1 rows
    # the exact ordered-distinct mean is 0.288888889, while dropping the two
    # short rows returned 1.000000000 (adversarial review, 2026-09-04).
    keep = norms > 0
    unit = _Xs[keep] / norms[keep, None]
    n_vec = len(unit)
    if n_vec > 1:
        mean_unit = unit.mean(axis=0)
        avg_cos_sim = float(
            (n_vec * float(mean_unit @ mean_unit) - 1.0) / (n_vec - 1)
        )
    else:
        avg_cos_sim = float("nan")

    # Conditioned on the numerical rank: on models where N < D the
    # smallest singular value is float-noise at ~1e-7, so ``S[0]/S[-1]``
    # reports 1e8 dominated by the numerical floor rather than the
    # data's geometry. Cap the denominator at a relative-tolerance
    # threshold of S[0] so ``cond_number`` reflects the effective
    # conditioning of the numerical-rank subspace only.
    # numpy's rank tolerance is S[0] * max(M.shape) * eps -- the larger of the
    # MATRIX dimensions. `S` is 1-D of length min(N, D), so `max(S.shape)` is
    # min(N, D), which makes the tolerance too small whenever N > D and admits
    # noise singular values into the numerical rank. Active on real gpt2:
    # rank 764 vs 759 and cond_number **7273.15 vs 621.48**, a factor of 11.7
    # (adversarial review, 2026-09-03).
    # eps of the STORED dtype, not of the float64 we promoted to. The
    # promotion is for arithmetic headroom (see above); it does not make
    # float32 quantization noise into signal. Using float64 eps after a float32
    # capture inflated the numerical rank on a fixed 100x5 float32 rank-3
    # construction (adversarial review round 2, 2026-09-04).
    rel_tol = float(S[0]) * max(X.shape) * _stored_eps
    # MODEL-PRECISION VARIANT, disclosed rather than substituted.
    #
    # The hidden states are `.float()`-cast into the cache, so the primary
    # tolerance above uses float32's eps even when the model ran in bfloat16 --
    # whose eps is 64x larger. That makes the reported numerical rank
    # PRECISION-OPTIMISTIC on a bf16 run: `cond_number` reads 19.811127663
    # under the float32 tolerance and 449.643167800 under a bf16 one on
    # pythia-70m (22.7x) (independent Opus review, 2026-09-04).
    #
    # Numpy's `max(m, n) * eps` factor is a worst-case bound calibrated for
    # eps ~ 1e-16; at bf16's 7.8e-3 it exceeds 1 and would zero the rank
    # outright, so it is not a usable tolerance there. The variant below uses
    # the Frobenius-style sqrt(max(m, n)) factor instead, and BOTH are emitted:
    # they bracket the truth, and the primary keeps its established value so no
    # published number shifts silently under a disclosure.
    if source_dtype is _BF16_EPS_PROXY:
        _src_eps = _BF16_EPS_PROXY.eps
    elif source_dtype is not None:
        _src_eps = float(np.finfo(np.dtype(source_dtype)).eps)
    else:
        _src_eps = _stored_eps
    _rel_tol_src = float(S[0]) * np.sqrt(max(X.shape)) * _src_eps
    _rank_src = int(np.sum(S > _rel_tol_src))
    _cond_src = (float(S[0] / S[_rank_src - 1]) if _rank_src >= 1
                 and S[_rank_src - 1] > 0 else float("inf"))
    effective_rank_num = int(np.sum(S > rel_tol))
    if effective_rank_num >= 1 and S[effective_rank_num - 1] > 0:
        cond_number = float(S[0] / S[effective_rank_num - 1])
    else:
        cond_number = float("inf")

    return {
        # DEMOTED to _meta_ (2026-09-04). `svd_auc` integrates the cumulative
        # explained-variance curve over its own UNPADDED length, so it is a
        # function of the ambient width as much as of the geometry: one
        # rank-one cloud scores 0.500000 / 0.750000 / 0.875000 / 0.954545 at
        # d = 2 / 4 / 8 / 22, and zero-padding a fixed cloud from d=8 to d=128
        # -- an exact isometry -- moves it. It also has no cited defining
        # source. Same bar as the normalized twins: excluded from the feature
        # matrix (adversarial review round 2, 2026-09-04).
        "_meta_svd_auc": float(auc),
        "cond_number": cond_number,
        "_meta_numerical_rank": effective_rank_num,
        # At the MODEL's precision (see the note above); on a float32 model
        # these coincide with the primary up to the sqrt vs linear factor.
        "_meta_numerical_rank_source_precision": _rank_src,
        "_meta_cond_number_source_precision": _cond_src,
        "_meta_numerical_rank_tol": float(rel_tol),
        "avg_cosine_similarity": avg_cos_sim,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
    }


def _isoscore_null_floor(n: int, d: int, seed: int = 0) -> float:
    """IsoScore of an isotropic Gaussian at the same (N, d) -- the finite-N
    floor. A measured value at or below this is not distinguishable from
    structureless data of the same shape."""
    if n < 2 or d < 2:
        return float("nan")
    # SUBSAMPLE rather than bail out. The previous `n*d > 40M -> NaN` guard was
    # dead exactly where it mattered: cached gpt2 at the current default reaches
    # N = 83,641 (64.2M), so the floor was NaN on the real runs it exists to
    # calibrate (adversarial review, 2026-09-03).
    #
    # IsoScore's finite-N bias depends on N/d, so a subsample at the SAME ratio
    # carries the same bias while fitting in memory. Cap the draw and hold the
    # ratio fixed.
    n_eff, d_eff = int(n), int(d)
    cap = 40_000_000
    if n_eff * d_eff > cap:
        n_eff = max(d_eff + 1, cap // max(d_eff, 1))
    rng = np.random.default_rng(seed)
    return _isoscore(rng.standard_normal((n_eff, d_eff)))


def _isoscore(X: np.ndarray) -> float:
    """IsoScore (Rudman et al., Findings of ACL 2022, arXiv:2108.07344).

    Measures how close the empirical covariance of a point cloud is to a
    scaled identity — a more discriminative notion of isotropy than the
    Ethayarajh anisotropy baseline (which only looks at average cosine
    similarity of random pairs).

    Algorithm (Rudman et al. Eq. 4.1):
      1. Center X and compute the eigenvalue spectrum lambda_i of cov(X)
         via SVD on centered data, padding zeros to the ambient dimension.
      2. L2-normalise the spectrum to length sqrt(d); an isotropic spectrum
         becomes the all-ones vector.
      3. Compute the isotropy defect
         delta = ||lambda_hat - 1|| / sqrt(2 * (d - sqrt(d))).
      4. IsoScore = ((d - delta^2 * (d - sqrt(d)))^2 - d) / (d * (d - 1)).

    Returns a scalar in [0, 1] where 1 = perfectly isotropic and 0 =
    effectively one-dimensional ambient-space use.
    """
    n, d = X.shape
    if n < 2 or d < 2:
        return float("nan")

    # SCALE OUT before squaring. IsoScore is exactly invariant to a positive
    # rescaling of the cloud -- Eq. 4.1 is a ratio of spectral moments -- but
    # `S ** 2` is not: a finite d=8 cloud with an exact score of 0.087912087912
    # returned 0.076750700280 at scale 3e-22 (-12.7%), 0.114285714286 at 1e-22
    # (+30.0%) and NaN at 3e-23 and at 1e19 (adversarial review round 3,
    # 2026-09-04). Dividing S by S[0] only shifts exponents and cancels in the
    # ratio, so the answer is unchanged and the arithmetic no longer over- or
    # underflows.
    # PROMOTE BEFORE THE MEAN, not after. Subtracting a float32 mean from
    # float32 data loses translation invariance that IsoScore has by
    # definition and that the released package keeps: a float32 cloud
    # translated by +1e6 gave BLME 0.001783276461 against IsoScore 2.0.1's
    # 0.995837092400, a 99.82% error (independent Opus review, 2026-09-05).
    # It is a no-op at offset 0 (0.995837039632 either way).
    Xd = np.asarray(X, dtype=np.float64)
    Xc = Xd - Xd.mean(axis=0, keepdims=True)
    try:
        # `safe_svdvals`: numpy/torch compute the SVD internally and then CAST
        # the singular values back to the input dtype, and that cast overflows
        # to inf near 3.4e38 -- so a scale-FREE consumer downstream (effective
        # rank, RankMe, IsoScore, alpha) returns NaN on data it should handle
        # exactly. The helper factors out the max first, which only shifts
        # exponents (2026-09-07).
        from .utils import safe_svdvals
        S = safe_svdvals(Xc)
    except np.linalg.LinAlgError:
        return float("nan")
    _s0 = float(S[0]) if S.size else 0.0
    if _s0 > 0 and np.isfinite(_s0):
        S = S / _s0
    eigvals = (S ** 2) / max(1, n - 1)

    full = np.zeros(d, dtype=np.float64)
    full[: len(eigvals)] = eigvals
    spectrum_norm = float(np.linalg.norm(full))
    if spectrum_norm <= 0 or not np.isfinite(spectrum_norm):
        return float("nan")

    sigma_hat = np.sqrt(d) * full / spectrum_norm
    denom = float(np.sqrt(2.0 * (d - np.sqrt(d))))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")

    defect = float(np.linalg.norm(sigma_hat - 1.0) / denom)
    score = ((d - defect ** 2 * (d - np.sqrt(d))) ** 2 - d) / (d * (d - 1))
    return float(max(0.0, min(1.0, score)))


@register_task("geometry_isoscore")
class IsoScoreTask(DiagnosticTask):
    """
    IsoScore (Rudman et al., Findings of ACL 2022, arXiv:2108.07344).

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

        # A SHARED CACHE IS USED ONLY IF IT CAN SUPPLY THE CONFIGURED
        # POPULATION. IsoScore's whole validity condition is N >> d, so
        # quietly accepting whatever the cache happens to hold made the value
        # depend on which other tasks co-ran: alone the task got its configured
        # 1000 documents, but with `geometry_svd` in the run it saw a
        # 100-document cache (adversarial review round 3, 2026-09-04). The
        # cache is NOT resized for this -- see the note in core.py -- so the
        # task self-collects instead.
        _cached_docs = len(getattr(cache, "_sample_lengths", []) or []) if cache else 0
        _cache_ok = (cache is not None and cache.is_populated and use_cache
                     and _cached_docs >= int(num_samples))
        if _cache_ok:
            # post-final-norm=False, DELIBERATELY. IsoScore/anisotropy is
            # defined on the model's FINAL CONTEXTUALIZED EMBEDDINGS -- what
            # Rudman et al. measure, and what HF's last entry is. This mode
            # reports ONE layer, so no profile mixes conventions; the
            # layerwise mode above opts IN for exactly that reason.
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples,
                                        replace_final_norm=False)
        else:
            if cache is not None and cache.is_populated and use_cache:
                logger.info(
                    f"  IsoScore: shared cache holds {_cached_docs} documents "
                    f"< configured {num_samples}; self-collecting so the N/d "
                    f"regime does not depend on which tasks co-ran.")
            # Every token, matching the cache route: the cache retains all of
            # them, and a 10-per-document cap made the two routes different
            # protocols (adversarial review round 2, 2026-09-04).
            X = collect_hidden_states(model, tokenizer, dataset,
                                      num_samples=num_samples,
                                      max_tokens_per_doc=None,
                                      replace_final_norm=False)
        _source = "shared_cache" if _cache_ok else "self_collected"
        X = X.float().numpy()
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            logger.info(f"  Filtered {(~finite_mask).sum()} non-finite rows out of {len(X)}")
            X = X[finite_mask]
        if len(X) < 10:
            return {"error": "Too few finite hidden states for IsoScore"}

        iso = _isoscore(X)
        d = int(X.shape[1])
        n_pts = int(X.shape[0])

        # Report the realized N/d ratio so the paper's sampling condition is
        # AUDITABLE. Rudman et al. require the point count to exceed the
        # dimension by orders of magnitude; a document-count config cannot
        # guarantee that, because the realized N depends on document length.
        min_ratio = float(self.config.get("min_points_per_dim", 10))
        ratio = (n_pts / d) if d else float("nan")
        regime_ok = bool(np.isfinite(ratio) and ratio >= min_ratio)

        # AMBIENT-CORRECTED IsoScore.
        #
        # IsoScore maps "the cloud occupies k of d dimensions" to
        # (k/d - 1/d) / (1 - 1/d), so it carries a structural 1/(d-1) factor:
        # a model that genuinely uses k dimensions scores LOWER simply for being
        # wider. That factor, not any property of the representations, is what
        # drives the metric's correlation with model width.
        #
        # Inverting it recovers the EFFECTIVE DIMENSION COUNT, iso*(d-1)+1.
        #
        # IDENTITY: this is EXACTLY the participation ratio of the covariance
        # spectrum, (sum lambda)^2 / sum(lambda^2), which `geometry_svd` already
        # emits as `participation_ratio` -- verified to rel 0.0 on isotropic,
        # low-rank and anisotropic clouds. The two are one quantity under two
        # names. Keeping both is deliberate (this is the interpretable form for
        # a reader of the IsoScore paper), but they must not BOTH enter the
        # feature matrix; `feature_selection.py` records the pair.
        # Measured over the zoo (adversarial review, 2026-09-03): raw IsoScore
        # has rho = -0.440 with d_model (p = 3.5e-4), while the corrected
        # quantity has rho = -0.105 (p = 0.42) -- i.e. the entire apparent
        # width dependence WAS the structural factor.
        #
        # CORRECTION (2026-09-04): an earlier version of this note added "models
        # turn out to use about 43 effective dimensions regardless of how wide
        # they are". That is FALSE and was never measured -- an independent
        # review found 1.61 on gpt2 against 43.05 on pythia-70m on the same
        # panel. The width ASSOCIATION is weak; the VALUE is not constant.
        #
        # This twin does, however, pass the mechanism test that removed all
        # eleven registered normalizers, and it is the only one in the campaign
        # that has: a d = 64 -> 4096 zero-padding isometry moves the raw score
        # 0.111111 -> 0.001709 (a factor of 65) while leaving
        # `isoscore_effective_dims` at 7.999997631, i.e. unchanged to 5.5e-14
        # (independent Opus review, 2026-09-04).
        iso_dims = (float(iso) * (d - 1) + 1.0) if d > 1 else float("nan")

        # GATE the value on the sampling regime, rather than reporting the flag
        # and emitting an ordinary number anyway. A run below the configured
        # N/d threshold is not a valid IsoScore -- the finite-N floor dominates
        # -- and emitting it as if it were lets an invalid value enter the
        # feature matrix silently (adversarial review, 2026-09-03).
        # DO NOT NaN-gate on the N/d ratio.
        #
        # A ratio gate at 50 is satisfiable only for d_model <= 465 at the
        # study's num_samples, so it returned NaN for EVERY model -- and even at
        # 1000 documents it would NaN roughly the larger half of the zoo, which
        # is precisely the half the paper's scale claims depend on.
        #
        # More importantly the gate was calibrated against nothing: the finite-N
        # bias it protects against is **0.44%** on real gpt2 states at
        # N/d = 5, and -2.0% worst case on an isotropic null at N/d = 50
        # (adversarial review, 2026-09-03). Discarding the entire zoo to avoid a
        # sub-percent bias is the wrong trade.
        #
        # The ratio is still REPORTED so a reader can judge it, and a genuinely
        # unusable regime (fewer points than dimensions, where the covariance is
        # rank-deficient by construction) still yields NaN.
        # N <= d is IN the algorithm's domain, and is reported.
        #
        # Rudman et al.'s Algorithm 1 excludes only d = 1 and singleton clouds,
        # and released IsoScore 2.0.1 computes this case: on a fresh 20x100
        # cloud the kernel returns 0.154302597356 and the package 0.1543...,
        # so NaN-gating it discarded a value both the paper and the reference
        # define (adversarial review round 3, 2026-09-04). The covariance IS
        # rank-deficient there, which is a real caveat about what the number
        # means, so it is DISCLOSED rather than used to suppress the number.
        _rank_deficient = bool(d > 0 and n_pts <= d)
        if _rank_deficient:
            logger.info(f"  IsoScore: N={n_pts} <= d={d}; the covariance is "
                        f"rank-deficient by construction. Reported with "
                        f"_meta_isoscore_rank_deficient=True.")

        return {
            "isoscore": iso,
            # Ambient-corrected: prefer this for any cross-model comparison.
            "isoscore_effective_dims": iso_dims,
            # _meta_: this is Pearson 0.99997 with the RAW isoscore and
            # therefore re-imports the exact width confound the effective-dims
            # form was created to remove. Not a feature.
            "_meta_isoscore_effective_dims_frac": (iso_dims / d) if d > 0 else float("nan"),
            "_meta_isoscore_rank_deficient": _rank_deficient,
            # Which route supplied the points, so the N/d regime is auditable.
            "_meta_isoscore_source": _source,
            "_meta_isoscore_cached_docs": int(_cached_docs),
            "_meta_d_model": d,
            "_meta_n_points": n_pts,
            # Finite-N floor: IsoScore of an ISOTROPIC GAUSSIAN at this exact
            # (N, d). Reported rather than gated on, so a reader can see how far
            # the measured value sits from what structureless data of the same
            # shape would give.
            #
            # The bias depends on the INTRINSIC dimension, which is why two
            # measurements of it disagreed: on an isotropic null (all d
            # dimensions used) it is -40.1% / -16.8% / -9.1% / -3.2% / -2.0% at
            # N/d = 1.5 / 5 / 10 / 30 / 50, but on real gpt2 states -- which use
            # about 43 effective dimensions -- it is 0.44% at N/d = 5. Real data
            # sits in the benign regime; this field lets that be checked per run
            # instead of assumed (adversarial review, 2026-09-03).
            "_meta_isoscore_null_floor": _isoscore_null_floor(n_pts, d),
            "_meta_points_per_dim": float(ratio),
            # False => the paper's sampling condition did NOT hold for this run,
            # and the raw isoscore should not be read as a calibrated value.
            "_meta_sampling_regime_ok": regime_ok,
        }


@register_task("geometry_svd")
class SVDIsotropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SVD Analysis...")
        # The model's REAL precision, for the numerical-rank tolerance. The
        # hidden states are `.float()`-cast on the way into the cache, so the
        # array itself can no longer report it (see _svd_metrics_for_layer).
        _src_dt = np.float32
        try:
            _md = next(model.parameters()).dtype
            _src_dt = {torch.bfloat16: np.dtype("float32"),
                       torch.float16: np.float16,
                       torch.float32: np.float32,
                       torch.float64: np.float64}.get(_md, np.float32)
            if _md is torch.bfloat16:
                # numpy has no bfloat16; its eps is 2^-8 = 0.0078125.
                _src_dt = _BF16_EPS_PROXY
        except Exception:
            pass
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode (all layers) ---
        if layerwise and cache is not None and cache.is_populated and use_cache:
            # post-final-norm: KEEP IT. `replace_final_norm=False`, deliberately.
            #
            # THE RELEASED REFERENCE WINS, and here it is unambiguous. Read from
            # the vendored model this task's source ships
            # (`contextual/pytorch_pretrained/modeling_gpt2.py:562-565`):
            #
            #     hidden_states = self.ln_f(hidden_states)
            #     all_hidden_states.pop()   # ignore the last layer prior to layer norm
            #     all_hidden_states.append(hidden_states.clone())
            #
            # Ethayarajh's extractor EXPLICITLY discards the pre-`ln_f` block
            # output and puts the post-`ln_f` state in its place -- its own
            # comment says so -- and `analyze.py` then scores that tuple's layer
            # 12. BLME was substituting the pre-norm output, i.e. exactly the
            # entry the reference throws away.
            #
            # This is the THIRD CLAUSE again (per-layer profiles want
            # homogeneity; a single-layer estimand may want post-norm; and the
            # released reference decides). Eighth task to land on the post-norm
            # side.
            #
            # SHIPPED NUMBERS MOVE, and by a lot: gpt2 `avg_cosine_similarity`
            # 0.834701676247 -> 0.984922922817 (+18.0%). Any archived
            # `geometry_svd` row predating 2026-09-07 used the wrong entry.
            all_layers = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
                replace_final_norm=False)
            if all_layers:
                per_layer = {}
                erank_per_layer = []
                pr_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _svd_metrics_for_layer(all_layers[li],
                                               source_dtype=_src_dt)
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
        # A SHARED CACHE IS USED ONLY IF IT CAN SUPPLY THE CONFIGURED
        # POPULATION. IsoScore's whole validity condition is N >> d, so
        # quietly accepting whatever the cache happens to hold made the value
        # depend on which other tasks co-ran: alone the task got its configured
        # 1000 documents, but with `geometry_svd` in the run it saw a
        # 100-document cache (adversarial review round 3, 2026-09-04). The
        # cache is NOT resized for this -- see the note in core.py -- so the
        # task self-collects instead.
        _cached_docs = len(getattr(cache, "_sample_lengths", []) or []) if cache else 0
        _cache_ok = (cache is not None and cache.is_populated and use_cache
                     and _cached_docs >= int(num_samples))
        if _cache_ok:
            # post-final-norm=False, DELIBERATELY -- same reading as the
            # IsoScore path above: one layer, the model's final contextualized
            # embeddings, no cross-layer profile to keep homogeneous.
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples,
                                        replace_final_norm=False)
        else:
            if cache is not None and cache.is_populated and use_cache:
                logger.info(
                    f"  IsoScore: shared cache holds {_cached_docs} documents "
                    f"< configured {num_samples}; self-collecting so the N/d "
                    f"regime does not depend on which tasks co-ran.")
            # Every token, matching the cache route: the cache retains all of
            # them, and a 10-per-document cap made the two routes different
            # protocols (adversarial review round 2, 2026-09-04).
            X = collect_hidden_states(model, tokenizer, dataset,
                                      num_samples=num_samples,
                                      max_tokens_per_doc=None,
                                      replace_final_norm=False)
        _source = "shared_cache" if _cache_ok else "self_collected"

        m = _svd_metrics_for_layer(X, source_dtype=_src_dt)
        if m is None:
            return {"error": "Too few finite hidden states for SVD"}
        return m
