
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from .utils import collect_hidden_states
import logging
logger = logging.getLogger("blme")


def _linear_cka(X: torch.Tensor, Y: torch.Tensor,
                norm_x: float = None, norm_y: float = None,
                centered: bool = False) -> float:
    """Linear CKA from Kornblith et al. 2019 on column-centered features.

    norm_x / norm_y let the caller pass ||X^T X||_F and ||Y^T Y||_F when it
    already has them. The task computes both once per layer and then called
    this for every layer PAIR, so each pair was recomputing two O(N*D^2)
    matmuls it already had — three matmuls of work for one matmul of new
    information. On a 64-layer 5120-dim model that is 2080 pairs of wasted
    GEMMs, and geometry_cka hit the 3600s per-task timeout on qwen1.5-32b.
    Passing the cached norms is arithmetically identical: the values reused
    are the very same floats this function would recompute.

    centered=True skips re-centering when the caller has already done it.
    """
    if not centered:
        # Factor out the max BEFORE centering: `mean` sums the column first, so
        # N rows of magnitude g overflow at N*g even when g itself is finite --
        # 200 rows at 3e38 sum to 6e40 and the mean becomes inf, making every
        # centered value NaN. Scaling first is exact and CKA does not care.
        def _center(M):
            M = M.float()
            g = M.abs().amax()
            if float(g) > 0:
                M = M / g
            return (M - M.mean(dim=0, keepdim=True)) * g
        X = _center(X)
        Y = _center(Y)
    # FACTOR OUT EACH MATRIX'S LARGEST MAGNITUDE FIRST.
    #
    # The numerator is ||Y^T X||_F^2 and the denominator ||X^T X||_F ||Y^T Y||_F,
    # so BOTH scale as a^2 b^2 -- and the numerator is squared again in float,
    # giving a fourth power. At a global magnitude of 1e-13 that is 1e-52, which
    # underflows float32's 1.4e-45 subnormal floor on BOTH sides, and the ratio
    # becomes 0/0 = NaN. Measured on a seeded (200, 24) centered pair
    # (independent gpt-5.6-sol review, 2026-09-06; reproduced here):
    #
    #     scale 1      0.3499494344
    #     scale 1e-6   0.3499493650
    #     scale 1e-13  NaN
    #     scale 1e-20  NaN
    #
    # Kornblith et al. Eq. 4 is EXACTLY invariant to a positive rescaling of X
    # and Y independently, so this is a value-changing violation on the paper's
    # own domain -- and the RELATIVE degeneracy test below, added for exactly
    # this class of bug, does not help because the inputs never reach it as
    # finite numbers.
    #
    # Dividing each matrix by its own max entry only shifts exponents, so it is
    # exact; a cached norm is converted rather than discarded, since
    # ||X^T X||_F carries the factor twice.
    _gx = X.abs().amax()
    _gy = Y.abs().amax()
    _tiny = torch.finfo(X.dtype).tiny
    _gxf, _gyf = float(_gx), float(_gy)
    Xs = X / _gx.clamp_min(_tiny)
    Ys = Y / _gy.clamp_min(_tiny)
    numerator = torch.norm(Ys.t() @ Xs, p="fro").item() ** 2
    # A caller-supplied norm is ALREADY in max-relative units (see the note at
    # `norms[idx]` in `evaluate`), so it is used as-is. Converting it here
    # instead would reintroduce the fourth-power underflow this is avoiding.
    if norm_x is None:
        norm_x = torch.norm(Xs.t() @ Xs, p="fro").item()
    if norm_y is None:
        norm_y = torch.norm(Ys.t() @ Ys, p="fro").item()
    denom = norm_x * norm_y
    # RELATIVE degeneracy test. CKA is scale invariant by construction --
    # CKA(aX, bY) == CKA(X, Y) for any nonzero a, b -- but an ABSOLUTE cutoff on
    # the denominator breaks that on its own valid domain: identical nonconstant
    # X returns 1.0, while the same X scaled by 1e-5 returned 0.0, because
    # ||X^T X||_F * ||Y^T Y||_F scales as a^2*b^2 and fell under 1e-12
    # (adversarial review, 2026-09-03).
    #
    # Comparing against the numerator instead is scale-free: both sides carry
    # the same power of the scale, so the test fires only for a genuinely
    # degenerate (near-constant) input, which is what it was for.
    # Genuinely undefined cases return NaN, not 0.0.
    #
    # For constant (zero-variance) features CKA is 0/0 -- undefined -- and
    # returning 0.0 asserts MAXIMAL DISSIMILARITY, which is a different and
    # false claim. The same applies when float32 under/overflow destroys the
    # denominator. NaN propagates honestly into the summaries, where a 0 would
    # silently drag min/mean (adversarial review, 2026-09-03).
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    if not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denom)


def _debiased_hsic_features(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Unbiased HSIC computed in FEATURE space, not kernel space.

    Kornblith's released notebook provides exactly this
    (`feature_space_linear_cka(..., debiased=True)`), and it matters: the
    kernel-space form materializes two (N, N) Gram matrices and costs O(N^2 D)
    plus O(N^2) memory. At the study's N = 23,272 that measured **53.0 s and
    24.9 GB RSS per layer pair** -- 68.9 min for gpt2 and **30.6 h for a
    64-layer model**, past the suite's own 3,600 s per-task timeout. The
    feature-space form is O(N D^2), gives the identical number (rel 1.09e-11),
    and takes **0.640 s** -- an 85x speedup (adversarial review, 2026-09-03).
    """
    X = X.double()
    Y = Y.double()
    n = X.shape[0]
    if n < 4:
        return float("nan")
    # Sums of squared entries of the centered Gram matrices, without forming
    # them: ||Y^T X||_F^2 etc.
    # `xtx_sq` and `yty_sq` USED TO BE COMPUTED HERE and thrown away at a
    # `_ = (xtx_sq, yty_sq)` that hid the fact. They were the two most expensive
    # operations in the function -- O(N D^2) GEMMs each -- and neither value
    # reached the return statement. Measured at the study's N = 23,272
    # (independent Opus review, 2026-09-04): 1.80x at d=768, 1.92x at d=2048,
    # 2.64x at d=5120 (3.910 s -> 1.479 s), with the reduced form agreeing to
    # relative **0.0e+00**. Deleting them takes a 64-layer 5120-wide model from
    # ~3.78 h to roughly 1.4 h, i.e. from past the study's own 3,600 s per-task
    # timeout to inside it.
    xty = Y.t() @ X
    xy_sq = float(xty.pow(2).sum())
    # Row sums of K and L, again without forming them.
    sx = X.sum(dim=0)
    sy = Y.sum(dim=0)
    kx = X @ sx                       # row sums of K = X X^T
    ly = Y @ sy
    kdiag = X.pow(2).sum(dim=1)       # diagonal of K
    ldiag = Y.pow(2).sum(dim=1)
    kt = float(kx.sum() - kdiag.sum())
    lt = float(ly.sum() - ldiag.sum())
    # (K L) trace with zero diagonals.
    term = float((kx * ly).sum()
                 - (kdiag * ly).sum() - (kx * ldiag).sum()
                 + (kdiag * ldiag).sum())
    kl = float(xy_sq
               - (kdiag * ldiag).sum())
    return float(
        kl + kt * lt / ((n - 1) * (n - 2)) - 2.0 * term / (n - 2)
    ) / (n * (n - 3))


def _debiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """Unbiased HSIC (Song et al. 2012) from explicit Gram matrices.

    Retained for the parity tests, which check small fixtures. Production uses
    the feature-space form above.
    """
    K = K.double().clone()
    L = L.double().clone()
    n = K.shape[0]
    if n < 4:
        return float("nan")
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)
    kt = K.sum()
    lt = L.sum()
    term = (K @ L).sum()
    return float(
        (K * L).sum()
        + kt * lt / ((n - 1) * (n - 2))
        - 2.0 * term / (n - 2)
    ) / (n * (n - 3))


def _linear_cka_debiased(X: torch.Tensor, Y: torch.Tensor,
                         hxx: float = None, hyy: float = None) -> float:
    """Debiased linear CKA -- the cross-model-comparable form.

    ``hxx``/``hyy`` are the per-LAYER self-HSICs. Each depends on ONE argument,
    so in a layer-pair loop they are recomputed L times each for no reason;
    passing them in lets the caller hoist them out (a further ~3x on top of the
    dead-code removal above). Omitting them keeps the standalone behaviour the
    parity tests use.
    """
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
    hxy = _debiased_hsic_features(X, Y)
    if hxx is None:
        hxx = _debiased_hsic_features(X, X)
    if hyy is None:
        hyy = _debiased_hsic_features(Y, Y)
    den = float(np.sqrt(max(hxx, 0.0) * max(hyy, 0.0)))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(hxy / den)


@register_task("geometry_cka")
class CKATask(DiagnosticTask):
    """
    Computes Centered Kernel Alignment (CKA) between all layers of the model.
    Focuses on Linear CKA which is efficient for N > D.

    References:
      * Kornblith, Norouzi, Lee, Hinton 2019 — "Similarity of Neural
        Network Representations Revisited", ICML 2019, arXiv:1905.00414.
        The linear-CKA formulation is Section 3.
      * Cortes, Mohri, Rostamizadeh 2012 — "Algorithms for Learning
        Kernels Based on Centered Alignment", arXiv:1203.0550 (the
        HSIC-based foundation of CKA).

    Per-pair min/mean/std off-diagonal CKA enter BLME's top-25 partial
    predictors (`docs/TOP_PREDICTORS.md` §2): min_offdiag_cka at +0.74
    partial ρ controlling for log(N), std_offdiag_cka at −0.74, and
    min_adjacent_cka at +0.69.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKA Layer Similarity Analysis...")
        
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)
            
        # 1. Collect Activations
        # We need synchronization: same tokens for all layers.
        # collect_hidden_states(..., layer_idx="all") does this by collecting from the same forward pass.
        
        # Limit N to avoid OOM if we keep all X in memory?
        # For CKA, we need X (N, D). If N=10k, D=4k, X is 40MB. 32 layers -> 1.2GB. Feasible.
        
        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        logger.info(f"  Collecting hidden states for {num_samples} samples...")
        # POST-FINAL-NORM. Neither route opted in, so the deepest "layer" of
        # every CKA matrix was `ln_f(b_{L-1})` rather than a block output --
        # and CKA is a matrix of CROSS-LAYER comparisons, so one non-comparable
        # entry contaminates a whole row and column. It lands on exactly the
        # columns this class's own docstring names as BLME top-25 partial
        # predictors (red-team review, 2026-09-05):
        #
        #                        gpt2                distilgpt2          pythia-70m
        #   min_offdiag_cka   0.032242->0.133067  0.028512->0.346109  0.045372->0.042223
        #                        (+313%)             (+1114%)            (-6.9%)
        #   min_adjacent_cka  0.043207->0.146178  0.035608->0.358806  0.157586->0.098780
        #   early_late_cka    0.037986->0.152255  0.039354->0.373779  0.274159->0.157115
        #   std_offdiag_cka   0.370864->0.337733  0.447573->0.295635  0.340333->0.352413
        #
        # Sign is architecture-family determined; the debiased twins move
        # identically, so it is not a common rescale that cancels in a ranking.
        _cka_status = {}
        layer_activations = None
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
                replace_final_norm=True)
            _cka_status["last_is_block"] = bool(
                getattr(cache, "_last_block_substituted", False))
            if layer_activations is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. A cache
                # filled for an attention-only consumer returns None here and the
                # dereference below raised, destroying the task rather than
                # degrading it. Found by EXECUTING this check against a real
                # attention-only cache across all 51 tasks -- the round-12 sweep
                # of this same class was done by READING and put the population
                # at 2 sites; running it puts it at 9 (codex round 13 +
                # execution sweep, 2026-09-09).
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if layer_activations is None:
            layer_activations = collect_hidden_states(
                model, tokenizer, dataset, num_samples=num_samples,
                layer_idx="all", replace_final_norm=True,
                status=_cka_status)
        
        layers = sorted(layer_activations.keys())
        n_layers = len(layers)
        
        # Pre-process: Center the columns (features) or rows?
        # Linear CKA: centered columns of X?
        # Actually, standard Linear CKA definition:
        # CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        # where X, Y are centered.
        # Centering: equivalent to subtracting mean from each column.
        
        logger.info("  Centering activations...")
        centered_acts = {}
        norms = {} # ||X^T X||_F
        
        _n_f64_centered = 0
        for idx in tqdm(layers, desc="Centering"):
            X = layer_activations[idx].float() # (N, D)
            # CENTER IN float64 WHEN THE DATA IS OFF-CENTRE, else twice in
            # float32.
            #
            # The comment that used to stand here claimed the second float32
            # pass "moved CKA from 0.7787 to 0.3069 against a float64 truth of
            # 0.7787". That is not reproducible. Measured over
            # mean in {0, 1e2, 1e4, 1e6} x sd in {1e-3, 1} on a seeded
            # (200, 24) fixture:
            #
            #   worst |single - double| over the whole grid : 1.825e-03
            #   worst |single - float64 truth|              : 4.494e-02
            #
            # and at the very case the old comment named (mean 1e4, sd 1e-3):
            # single 0.367114, double 0.368480, float64 truth 0.412052 -- the
            # second pass moves 1.37e-03 where the error is 4.49e-02, i.e. it
            # recovers 3% of it. Worse, at mean 1e6 / sd 1e-3 BOTH return
            # **NaN** against a truth of 0.400437: once |mean|/sd is large
            # enough, `x - mean` has no significant bits left in float32 and no
            # number of repeated subtractions creates any.
            #
            # So the second pass is harmless but it is not the correction it
            # was described as. The actual correction is to do the subtraction
            # in float64 -- which only matters in that regime, so it is applied
            # CONDITIONALLY (the `schatten._normalise_matrix` pattern): real
            # activations have |mean| comparable to sd and take the cheap path
            # bit-for-bit, while the ill-conditioned case gets the arithmetic
            # it actually needs.
            _mu = X.mean(dim=0, keepdim=True)
            _sd = X.std(dim=0, keepdim=True)
            _ratio = float((_mu.abs()
                            / _sd.clamp_min(torch.finfo(X.dtype).tiny)).max())
            # float32 keeps ~7 significant decimal digits; beyond ~1e4 the
            # residual has fewer than 3 left.
            #
            # A NON-FINITE ratio takes the float64 path too, and that is the
            # important half. `_ratio` is inf exactly when the float32 `std`
            # underflowed to 0 while the mean did not -- i.e. when every value
            # in the column rounded to the SAME float32, which is the most
            # degenerate case there is. Guarding with `isfinite(_ratio) and
            # ...` excluded precisely it: at mean 1e6 / sd 1e-3 the float32
            # spacing is 0.0625, all 200 values collapse to one float, and both
            # the single- and double-centered forms returned NaN against a
            # float64 truth of 0.400437.
            if (not np.isfinite(_ratio)) or _ratio > 1e4:
                Xd = layer_activations[idx].double()
                X = (Xd - Xd.mean(dim=0, keepdim=True)).float()
                _n_f64_centered += 1
            else:
                X = X - _mu
                X = X - X.mean(dim=0, keepdim=True)
            centered_acts[idx] = X
            
            # Compute denominator term
            # ||X^T X||_F
            # This can be large.
            # Optimization: ||X^T X||_F = sqrt(sum((X^T X)^2))
            
            # If D is large, X^T X is (D, D). 4096^2 floats = 64MB. Fine.
            # ||X^T X||_F IN UNITS OF X's LARGEST ENTRY, not in absolute units.
            #
            # `X.t() @ X` has entries of order g^2 and `torch.norm` squares them
            # again, so the cached norm is a FOURTH power of the data scale: at
            # g = 1e-13 that is 1e-52, which underflows float32's 1.4e-45 floor
            # to 0.0, and the pair loop then divides by zero. Hardening
            # `_linear_cka` alone did NOT fix the task -- `CKATask.evaluate`
            # still returned NaN at 6 of 7 scales (1e-30, 1e-25, 1e-13, 1e13,
            # 1e25 and 2.406e38) against 0.317322581658 at scale 1, because the
            # caller hands it these precomputed norms (gpt-5.6-sol review,
            # 2026-09-07 -- the same reviewer that found the kernel defect, on
            # the very path it warned to check).
            #
            # `_linear_cka` divides X by its own max, so a norm expressed in
            # those same units is exactly what it expects and the factor cancels.
            _gx = X.abs().amax()
            _Xn = X / _gx.clamp_min(torch.finfo(X.dtype).tiny)
            xtx = _Xn.t() @ _Xn
            norms[idx] = torch.norm(xtx, p='fro').item()
            
            # Free original X if memory tight? No need yet.
            
        logger.info("  Computing CKA Matrix...")
        cka_matrix = np.zeros((n_layers, n_layers))
        dbg_matrix = np.zeros((n_layers, n_layers))

        # HOISTED: one self-HSIC per LAYER, not one per PAIR. The debiased
        # denominator is sqrt(hxx * hyy), and each factor depends on a single
        # layer -- so inside the pair loop each was recomputed L times. L self
        # -HSICs instead of L(L+1) is a further ~3x on top of the dead-code
        # removal in `_debiased_hsic_features` (independent Opus review,
        # 2026-09-04). `centered_acts` is already column-centered, so the
        # values are unchanged. (A `_debiased_self_hsic` wrapper was added here
        # as a hoisting helper and never called -- dead code introduced by the
        # change whose headline was removing dead code. Deleted 2026-09-05.)
        _self_hsic = {}
        for _idx in layers:
            _self_hsic[_idx] = _debiased_hsic_features(
                centered_acts[_idx].float(), centered_acts[_idx].float())
        
        for i in tqdm(range(n_layers), desc="CKA Rows"):
            idx_i = layers[i]
            X = centered_acts[idx_i]
            norm_x = norms[idx_i]
            
            for j in range(i, n_layers): # Symmetric
                idx_j = layers[j]
                Y = centered_acts[idx_j]
                norm_y = norms[idx_j]
                
                # centered_acts are already column-centered above, and norms
                # holds ||X^T X||_F per layer — pass both so the pair loop
                # does one GEMM instead of three.
                cka = _linear_cka(X, Y, norm_x=norm_x, norm_y=norm_y,
                                  centered=True)
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka
                # Debiased CKA. This removes the estimator's NULL FLOOR, which
                # is a strong function of d/N: two INDEPENDENT Gaussian clouds
                # at n=256 score 0.033 / 0.110 / 0.334 / 0.668 at
                # d = 8 / 32 / 128 / 512 under the biased estimator, against
                # -0.006 to +0.003 debiased.
                #
                # It is NOT a cross-model comparability fix, and must not be
                # described as one: on 8 real models the debiased value is
                # Pearson r = 0.999987 with its biased twin and carries the
                # IDENTICAL rho = +0.946 with n_layers (adversarial review,
                # 2026-09-03). What it buys is a meaningful zero, not
                # architecture independence.
                dcka = _linear_cka_debiased(
                    X, Y, hxx=_self_hsic[idx_i], hyy=_self_hsic[idx_j])
                dbg_matrix[i, j] = dcka
                dbg_matrix[j, i] = dcka
                
        # Compute reviewer-meaningful scalar summaries and suppress
        # ``cka_matrix`` + ``layers`` from the top-level dict.
        #
        # Historic bug: the aggregator's ``_flatten_dict`` dropped
        # ``cka_matrix`` (list-of-lists, not 1-D) and then produced
        # ``cka.layers.{mean,std,min,max,slope,...}`` as descriptive
        # statistics of the layer-index vector [0, 1, …, N-1]. That
        # contaminated ~8 feature columns with pure layer-count
        # proxies — reviewer-visible if anyone inspects what those
        # columns actually encode.
        n = n_layers
        iu = np.triu_indices(n, k=1)  # off-diagonal upper triangle
        off_diag = cka_matrix[iu] if iu[0].size else np.array([], dtype=float)

        diagonal_off1 = np.array(
            [cka_matrix[i, i + 1] for i in range(n - 1)],
            dtype=float,
        ) if n > 1 else np.array([], dtype=float)

        def _stat(arr, fn, fallback=float("nan")):
            return float(fn(arr)) if arr.size else fallback

        # SYMMETRIC NaN POLICY. The biased summaries used `np.mean`/`np.min`/
        # `np.std`, which PROPAGATE a NaN, while their debiased twins used
        # `np.nanmean`/`np.nanmin`, which DROP it. One degenerate layer pair
        # therefore blanked four biased columns while the debiased ones
        # silently reported a value computed on the surviving pairs -- two
        # different answers to "what happened here?" in the same result dict
        # (independent Opus review, 2026-09-04). Both families now drop.
        results = {
            # Core adjacent-layer measure (paper's headline quantity).
            # Fallback NaN, not 0.0: 0.0 is a MEANINGFUL CKA value ("these
            # layers are unrelated"), so returning it for "there was no
            # adjacent pair to measure" is the exact false-zero the NaN policy
            # was written to eliminate -- left in the one place a 1-layer model
            # would hit it. Every sibling already returned NaN.
            "avg_adjacent_cka": _stat(diagonal_off1, np.nanmean),
            "min_adjacent_cka": _stat(diagonal_off1, np.nanmin),
            "max_adjacent_cka": _stat(diagonal_off1, np.nanmax),
            "std_adjacent_cka": _stat(diagonal_off1, np.nanstd),
            # Global off-diagonal summary: picks up "all pairs of layers".
            "mean_offdiag_cka": _stat(off_diag, np.nanmean),
            "std_offdiag_cka": _stat(off_diag, np.nanstd),
            "min_offdiag_cka": _stat(off_diag, np.nanmin),
            "max_offdiag_cka": _stat(off_diag, np.nanmax),
            # Early-vs-late comparison: layer 0 vs layer N-1.
            "early_late_cka": (
                float(cka_matrix[0, -1]) if n >= 2 else float("nan")
            ),
            # First-to-middle (captures early representation drift).
            "first_middle_cka": (
                float(cka_matrix[0, n // 2]) if n >= 3 else float("nan")
            ),
            # First-to-last normalised depth distance; lower => layers
            # diverge more across depth.
            # Debiased summaries: a meaningful ZERO (null floor removed),
            # NOT an architecture-independent quantity -- and `_meta_`.
            #
            # On real gpt2 activations at the study's N = 23,272 the debiased
            # and biased layer-pair matrices agree to Pearson **0.99999980**,
            # max |difference| 7.56e-04 (independent Opus review, 2026-09-04).
            # As feature columns they are four copies of four columns already
            # present. What they genuinely buy is a meaningful zero for a WIDE
            # model, where an unrelated pair would otherwise be floored at 0.26
            # -- a diagnostic property, which is what `_meta_` is for. The
            # values are still emitted and still computed; they simply are not
            # offered to the regression as independent structure.
            "_meta_debiased_avg_adjacent_cka": (
                float(np.nanmean([dbg_matrix[i, i + 1] for i in range(n - 1)]))
                if n > 1 else float("nan")),
            "_meta_debiased_mean_offdiag_cka": (
                float(np.nanmean(dbg_matrix[iu])) if iu[0].size else float("nan")),
            "_meta_debiased_min_offdiag_cka": (
                float(np.nanmin(dbg_matrix[iu])) if iu[0].size else float("nan")),
            "_meta_debiased_early_late_cka": (
                float(dbg_matrix[0, -1]) if n >= 2 else float("nan")),
            "_meta_n_layers": int(n),
            "_meta_n_layers_centered_in_float64": int(_n_f64_centered),
            "_meta_last_entry_is_block_output": bool(
                _cka_status.get("last_is_block", False)),
        }
        return results
