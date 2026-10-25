"""
CKNNA — mutual k-NN conditional CKA between layer representations.

Reference:
    Huh, M., Cheung, B., Wang, T. & Isola, P. (2024). "The Platonic
    Representation Hypothesis." ICML 2024, arXiv:2405.07987.
    Official code: github.com/minyoungg/platonic-rep, `metrics.py`
    function ``cknna`` (commit dcd76ba).

Definition (Huh et al. 2024, `metrics.py:cknna`):
    CKNNA is a *locality-restricted* Centered Kernel Alignment. Given two
    representations of the same N points, X (N, D_x) and Y (N, D_y), form
    the linear Gram matrices

        K = X Xᵀ,   L = Y Yᵀ.

    For each point restrict attention to its top-k neighbours *in each
    space separately* (unbiased: the diagonal self-similarity is masked
    out with -inf before the top-k, so a point never selects itself), then
    keep only the entries that survive in BOTH neighbour sets — the mutual
    k-NN mask

        mask = mask_K ⊙ mask_L,   mask_·[i, j] = 1 iff j ∈ topk(·, i).

    (Note: this is an *asymmetric-then-intersected* mask — mask_K and
    mask_L are each row-wise top-k indicator matrices, not symmetrised, and
    the "mutual" graph is their elementwise product.) CKNNA is then the CKA
    computed on the masked kernels

        CKNNA(X, Y) = HSIC(mask⊙K, mask⊙L)
                      / ( sqrt( HSIC(mask_K⊙K, mask_K⊙K)
                                · HSIC(mask_L⊙L, mask_L⊙L) ) + 1e-6 ),

    where HSIC is the unbiased Hilbert-Schmidt Independence Criterion
    (Song et al. 2012, Eq. 5) by default. Identical inputs give ~1.0
    (exactly 1.0 only with biased HSIC; the unbiased reference default
    yields ~0.9999 because of the +1e-6 denominator regularizer); it is
    invariant to orthogonal rotations of either representation (like CKA);
    independent representations give ~0.

    The reference (`measure_alignment.py:compute_score`) L2-normalises each
    row of the features before calling ``cknna`` (``F.normalize(x, p=2,
    dim=-1)``) and uses ``topk=10`` by default; BLME mirrors both.

BLME wiring:
    * Tier-2 task — reuses the same per-layer synchronised token cloud as
      ``geometry_cka`` via ``collect_hidden_states(layer_idx="all")`` (same
      forward pass, same tokens across layers). Without a populated cache it
      runs its own collection pass.
    * The verified numeric artifact is the module-level ``_cknna(X, Y,
      topk, unbiased=True)`` helper, which reproduces the reference
      ``cknna`` bit-for-bit (float32, <1e-6). See
      ``tests/tasks/parity/test_cknna_parity.py``.
    * CKNNA is computed for every layer pair; only reviewer-meaningful
      scalar summaries of the off-diagonal / adjacent-layer values are
      emitted (no full matrix, no absolute layer indices leaking in as
      size proxies — Audit-V2 house rule).
"""

import logging

import numpy as np
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .utils import collect_hidden_states
import math

logger = logging.getLogger("blme")

# Default number of mutual nearest neighbours. Reference default is 10
# (measure_alignment.py --topk). n_samples must exceed topk.
_DEFAULT_TOPK = 10


# ESTIMAND: the RELEASED platonic-rep implementation, not the paper's prose.
#
# Huh et al.'s Eqs. 16-18 read as "centre the kernels, then weight the products
# by alpha", while `platonic-rep/metrics.py::cknna` masks the RAW kernels and
# centres afterwards inside HSIC. The two are not the same operation, and every
# number the paper reports comes from the released code. BLME therefore targets
# the released code, and matches it EXACTLY: head-to-head on identical inputs
# over 16 configurations (n in {40, 64, 200}, k in {5, 10, 39}, unbiased in
# {True, False}, seeds {0, 7, 1234, 51137}) the worst relative deviation is
# 0.000e+00 -- bit-identical, including the reviewer's own seed 51137
# (adversarial review response, 2026-09-04).
#
# The argument-order dependence is likewise a property of the reference, not of
# BLME: at seed 51137 the RELEASED code gives cknna(A,B)=0.030145890 and
# cknna(B,A)=0.030152554, and BLME reproduces both to 0.000e+00. It arises
# because `mask_K * mask_L` is a mutual-neighbour indicator that is not
# symmetric, so `mask * K` is not symmetric either. Rather than silently
# deviate from the reference by symmetrising, `evaluate` DISCLOSES the size of
# the effect in `_meta_cknna_order_asymmetry_max`.


def _hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC — Song et al. (2012), Eq. 5.

    Transcribed from platonic-rep/metrics.py ``hsic_unbiased`` (lines
    230-249). Operates on Gram matrices K, L of shape (m, m).
    """
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    hsic = (
        torch.sum(K_tilde * L_tilde.T)
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )
    return hsic / (m * (m - 3))


def _hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC (original CKA) — platonic-rep/metrics.py ``hsic_biased``."""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1.0 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def _cknna(X, Y, topk: int, unbiased: bool = True, distance_agnostic: bool = False) -> float:
    """Mutual k-NN conditional CKA — bit-exact port of platonic-rep ``cknna``.

    Reproduces ``AlignmentMetrics.cknna`` (metrics.py lines 180-227) exactly:
    the same -inf diagonal masking, the same ``torch.topk`` neighbour sets,
    the same ``scatter_`` masks, their elementwise-product mutual mask, and
    the same unbiased/biased HSIC. Runs in float32 to match the reference.

    Args:
        X: (N, D_x) array/tensor of representations for N synchronised points.
        Y: (N, D_y) array/tensor for the same N points.
        topk: number of nearest neighbours (must be >= 2 and < N).
        unbiased: use unbiased HSIC and mask the self-diagonal (reference default).
        distance_agnostic: use the raw mutual-kNN mask instead of the kernel
            values. NON-FUNCTIONAL: it crashes identically in the official
            code at the pinned reference commit (dcd76ba) — kept only for
            crash-parity documentation; raises NotImplementedError.

    Returns:
        CKNNA value in [~0, 1]; identical inputs give ~1.0 (see module
        docstring for the unbiased-HSIC caveat). Returns NaN if the inputs
        are degenerate (N < topk+1, or HSIC denominator collapses).
    """
    if distance_agnostic:
        raise NotImplementedError(
            "distance_agnostic=True is non-functional: the official "
            "platonic-rep cknna crashes identically at the pinned commit "
            "(.item() on an (n,n) tensor); do not use."
        )
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(np.asarray(X))
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(np.asarray(Y))
    X = X.float()
    Y = Y.float()

    n = X.shape[0]
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    if n <= topk:
        if n <= 2:
            return float("nan")
        # The paper's k -> n limit, where CKNNA reduces to CKA exactly.
        #
        # With unbiased=False the diagonal is NOT masked, so topk = n selects
        # every column and the neighbour mask is all-ones -- which is precisely
        # what makes CKNNA collapse to CKA. Clamping to n-1 instead misses that
        # limit: the official `cknna(topk=n, unbiased=False)` returns 0.5580957
        # (= exact CKA to rel 1.3e-8) where the n-1 clamp returns 0.5430548, a
        # 2.69% error -- and it is the paper's ONLY closed-form anchor, i.e.
        # exactly the check this branch was added to enable (adversarial
        # review, 2026-09-04).
        #
        # With unbiased=True the diagonal is -inf, so k = n would select it and
        # k = n-1 IS the largest meaningful neighbourhood.
        topk = (n - 1) if unbiased else n

    device = X.device
    # RESCALE BEFORE THE GRAMS. CKNNA is a RATIO --
    # sim_kl / sqrt(sim_kk * sim_ll) -- so any positive rescale of X or Y
    # cancels exactly. The Gram does not: it squares, so at 2^-60 the entries
    # underflow and the score collapsed from 0.677187975 to 5.616e-19, to 0.0
    # at 2^-80, and to NaN at 1e20 (adversarial review round 4, 2026-09-04).
    # Dividing each view by its own largest magnitude only shifts exponents.
    # Rescale by a POWER OF TWO, but ONLY outside a band where the whole
    # pipeline is safe. Both halves of that sentence are load-bearing.
    #
    # WHY RESCALE AT ALL: the data is squared THREE times downstream -- the
    # Gram, then HSIC's elementwise product, then `mm(K_tilde, L_tilde)` -- so
    # an input at 1e-13 reaches 1e-52 and flushes to zero. End to end the score
    # went 0.677187975 -> 5.616e-19 at 2^-60 and 0.0 at 2^-80 (adversarial
    # review round 4, 2026-09-04).
    #
    # WHY ONLY OUTSIDE A BAND: the released `cknna` divides by
    # `sqrt(sim_kk*sim_ll) + 1e-6`, an ABSOLUTE epsilon. Rescaling changes the
    # sims and therefore that term's relative weight, which moves the score in
    # the 5th decimal -- so an unconditional rescale would trade a bit-exact
    # match against the reference for robustness nobody needs at ordinary
    # magnitudes. The band is set by the eighth root of the representable
    # range, since the data is squared three times: float32's tiny^(1/8) is
    # about 2e-5, so [1e-4, 1e4] is comfortably inside it.
    #
    # WHY A POWER OF TWO: dividing by 2**round(log2(max)) shifts exponents
    # only, so it introduces no rounding of its own.
    _fi2 = torch.finfo(X.dtype)
    _lo2 = float(_fi2.tiny) ** (1.0 / 8.0) * 4.0
    _hi2 = float(_fi2.max) ** (1.0 / 8.0) / 4.0

    from .utils import pow2_rescale as _p2

    def _pow2_rescale(T):
        m = float(T.abs().max()) if T.numel() else 0.0
        if not (m > 0) or not math.isfinite(m) or (_lo2 <= m <= _hi2):
            return T
        # `_p2` clamps the exponent to the dtype's normal range. The inline
        # `2.0 ** round(log2(m))` this replaces overflowed float32 to inf for
        # any m >= 2^127.5 = 2.4062e38 -- the top 29% of the range -- so the
        # divisor became inf and the data became 0 (independent Opus review,
        # 2026-09-05).
        return _p2(T, m)

    X = _pow2_rescale(X)
    Y = _pow2_rescale(Y)
    K = X @ X.T
    L = Y @ Y.T

    def similarity(K, L, topk):
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)

        mask = mask_K * mask_L

        if distance_agnostic:
            sim = mask * 1.0
        else:
            if unbiased:
                sim = _hsic_unbiased(mask * K, mask * L)
            else:
                sim = _hsic_biased(mask * K, mask * L)
        return sim

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)

    denom = (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    return float(sim_kl.item() / denom)


def _remove_outliers(X: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """Reference preprocessing: clamp features to +/- the mean per-row
    ``q``-quantile of |x|.

    `measure_alignment.py:prepare_features` calls
    `metrics.remove_outliers(feats, q=0.95)` BEFORE normalising, and BLME
    skipped it. On gpt2 the omission shifts `mean_offdiag` by 9.8%,
    `min_offdiag` by 49%, and the final adjacent pair by 2.1x (adversarial
    review, 2026-09-03).
    """
    X = X.float()
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    # Per-row 95th percentile of |x|, then the mean of those -> one scalar bound.
    #
    # float32 FIRST, double only as a fallback when float32 fails.
    #
    # Averaging N per-row quantiles each near the top of the float32 range
    # overflows the accumulator, `bound` becomes inf, the `isfinite` bail-out
    # below fires, and the clamp is **silently skipped** -- so the
    # preprocessing this function exists to perform does not happen at all.
    # Rescaling an all-finite input from 1 to 2.406159650500376e38 moved the
    # score 0.129549999 -> 0.159972884 (**+23.484%**) against a stable
    # 0.129550017 (gpt-5.6-sol round 9, 2026-09-07).
    #
    # The obvious repair -- accumulate in double always -- is WRONG HERE, and I
    # shipped it for ten minutes before testing it. Double is more accurate, so
    # it returns a different float32 bound (1.88215649128 against
    # 1.88215637207 on an 80x61 Gaussian) and the clamped output stops being
    # bit-identical to the released `metrics.remove_outliers`. This file's
    # 240/240 bit-for-bit parity with the reference is the whole basis of its
    # verdict, and the campaign rule is that the released reference wins.
    #
    # So float32 stays authoritative on every input where it works, and double
    # is consulted ONLY when float32 has already failed -- which is exactly the
    # regime where there is no reference behaviour to be faithful to, because
    # the reference overflows there too.
    bound = torch.quantile(X.abs(), q, dim=1).mean()
    if not torch.isfinite(bound):
        # Narrowing straight back after a double-precision reduction discards
        # exactly what the promotion bought. Swept as a class: the first AST query
        # searched for a trailing `.float()` and found 7 sites; this `.to(<dtype>)`
        # spelling hid 5 more, including this one (codex round 15, 2026-09-09).
        bound = torch.quantile(X.abs().double(), q, dim=1).mean()
        if not torch.isfinite(bound):
            # Genuinely unrepresentable in the working dtype: clamp in double.
            _b64 = torch.quantile(X.abs().double(), q, dim=1).mean()
            if torch.isfinite(_b64):
                return X.double().clamp(min=-_b64, max=_b64).to(X.dtype)
    if not torch.isfinite(bound):
        return X
    # bound == 0 is CLAMPED, not skipped. The released
    # `metrics.remove_outliers` applies the clamp unconditionally, so a bound
    # of exactly zero zeroes the features -- which is what an isometrically
    # zero-padded 80x161 input produces: the official pipeline then has 0
    # nonzero coordinates and scores 0.000000, while returning X unchanged did
    # not (adversarial review round 2, 2026-09-04). Only a non-finite bound is
    # a genuine bail-out.
    return X.clamp(min=-bound, max=bound)


def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation, matching the reference preprocessing
    (``measure_alignment.py:compute_score`` -> ``F.normalize(x, p=2, dim=-1)``)."""
    X = X.float()
    # SCALE-EXACT. `max(norm, eps)` matches `F.normalize`'s FORM, but the norm
    # itself still squares before summing, so the whole pipeline moved a fresh
    # score of 0.677187975 to 0.706681363 at 1e-13 (+4.36%), 5.616e-19 at
    # 2^-60 and 0.000000000 at both 2^-80 and 1e20 (adversarial review round 4,
    # 2026-09-04). Factoring out each row's largest coordinate first is exact
    # and reproduces `F.normalize` bit-for-bit on ordinary data.
    # CONDITIONAL, so released-reference parity is preserved exactly on
    # ordinary data (`torch.equal` against `F.normalize`) while extreme
    # magnitudes still normalise correctly -- the same discipline used in
    # `schatten._normalise_matrix`.
    # The band must match the SQUARING DEPTH of the pipeline this feeds, not
    # the single square a norm performs. Downstream the data is squared three
    # more times (Gram, HSIC product, mm), so a sqrt-based band let 1e-13
    # through the naive path and the end-to-end score drifted 6.40%
    # (adversarial review round 4 follow-up, 2026-09-04). Eighth root, same as
    # `_cknna`'s own band.
    # PER ROW, not on one global maximum.
    #
    # Row normalisation is independent per row, so a single ordinary row must
    # not decide the path for every other one: rows [3e-13, 4e-13] and [1, 0]
    # in the same matrix kept the tiny row on the naive norm-plus-epsilon path
    # and its norm came out 0 (codex round 5, 2026-09-05). The band is still
    # the eighth root of the representable range, for the squaring depth
    # downstream; only the DECISION is now per row.
    _fi = torch.finfo(X.dtype)
    _lo = float(_fi.tiny) ** (1.0 / 8.0) * 4.0
    _hi = float(_fi.max) ** (1.0 / 8.0) / 4.0
    _rm = X.abs().amax(dim=-1, keepdim=True)
    _unsafe = (_rm > 0) & ((_rm < _lo) | (_rm > _hi))
    if not bool(_unsafe.any()):
        return X / X.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    from .utils import safe_unit_rows_torch as _safe_unit
    _out = X / X.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(_unsafe, _safe_unit(X), _out)


@register_task("geometry_cknna")
class CKNNATask(DiagnosticTask):
    """
    Computes CKNNA (mutual k-NN conditional CKA; Huh et al., "The Platonic
    Representation Hypothesis", ICML 2024, arXiv:2405.07987) between all
    pairs of transformer layers and summarises the alignment profile.

    CKNNA is the local-neighbourhood analog of ``geometry_cka``: instead of
    the global CKA it restricts the kernel alignment to the mutual k-nearest
    neighbour graph, so it measures whether *local* representational geometry
    (not just global structure) is preserved across layers.

    Config:
        num_samples: number of documents to collect (default 100).
        topk:        mutual k-NN neighbours (default 10). num_samples' worth
                     of tokens must exceed topk.
        unbiased:    unbiased HSIC + self-diagonal masking (default True).
        use_cache:   reuse the shared forward-pass cache (default True).

    Outputs (flat floats): adjacent-layer and global off-diagonal
    mean/min/max/std of CKNNA, early-vs-late and first-vs-middle values.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKNNA (mutual k-NN conditional CKA) Analysis...")

        num_samples = self.config.get("num_samples", 100)
        topk = int(self.config.get("topk", _DEFAULT_TOPK))
        unbiased = bool(self.config.get("unbiased", True))
        use_cache = self.config.get("use_cache", True)

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        logger.info(f"  Collecting hidden states for {num_samples} samples...")
        # post-final-norm=False, DELIBERATELY: THE RELEASED REFERENCE READS THE
        # RAW TUPLE.
        #
        # `minyoungg/platonic-rep`'s `extract_features.py:82` is
        #     feats = torch.stack(llm_output["hidden_states"]).permute(1,0,2,3)
        # -- every entry of the raw HF tuple, `ln_f` included, no hook. (The
        # `pool == 'last'` branch below stacks the same tuple.)
        #
        # The 2026-09-05 sweep hooked this because CKNNA is a per-layer profile
        # and a profile must be homogeneous. But the THIRD CLAUSE outranks the
        # first: the released reference wins. This is the SEVENTH task to need
        # it -- after matrix_entropy, contextualization, coe, schatten,
        # concept_separability and task_vectors -- and every one was found by
        # review rather than by the sweep, because the sweep applied a rule
        # without reading what each reference does. Confirmed by cloning the
        # repository and reading the line (2026-09-06).
        #
        # The hook's measured impact here was only -1.0% to +2.0%: CKNNA's
        # row-L2 normalisation and outlier clamp largely absorb the LayerNorm,
        # which is exactly why it went unnoticed.
        _cknna_status = {"last_is_block": False}
        layer_activations = None
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
                replace_final_norm=False,
            )
            if layer_activations is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. This task ran c
                # orrectly with NO cache and returned an error with an attention-onl
                # y one, i.e. it refused rather than self-collecting -- the defect a
                # stra RED-flagged on geometry_vendi_score. Found by EXECUTING the c
                # heck across all 51 tasks: the class is 15 sites, not the 5 a read-
                # only sweep reported (2026-09-09).
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if layer_activations is None:
            layer_activations = collect_hidden_states(
                model, tokenizer, dataset, num_samples=num_samples, layer_idx="all",
                replace_final_norm=False,
            )

        if not layer_activations:
            return {"error": "No hidden states available for CKNNA"}

        layers = sorted(layer_activations.keys())
        n_layers = len(layers)

        # L2-normalise each layer's token cloud once (reference preprocessing).
        # N is shared across layers (same synchronised tokens), so guard once.
        n_points = int(layer_activations[layers[0]].shape[0])
        # `n_points == topk` is the paper's k -> n FULL-NEIGHBOURHOOD limit,
        # where CKNNA reduces to CKA exactly, and `_cknna` already handles it
        # (that branch is the only closed-form anchor the paper provides). The
        # guard here rejected it before the helper ran: n_points = topk = 10
        # with unbiased=False returned an error where both current and
        # reference execution give 0.605255 (adversarial review round 2,
        # 2026-09-04). Only n_points < topk, or a cloud too small to have any
        # neighbourhood at all, is genuinely unusable.
        # Unbiased HSIC divides by (m-2) and (m-3), so it is UNDEFINED for
        # N <= 3. `evaluate` accepted N=3 with topk=2, every pair came out NaN,
        # and the summary then reported avg_adjacent_cknna = 0.000000000 as if
        # it were a measurement (adversarial review round 4, 2026-09-04).
        _min_n = 4 if unbiased else 3
        if n_points < topk or n_points < _min_n:
            return {
                "error": (
                    f"CKNNA needs n_samples ({n_points}) >= topk ({topk}) "
                    f"and >= {_min_n} for "
                    f"{'unbiased' if unbiased else 'biased'} HSIC; "
                    "increase num_samples or lower topk"
                )
            }

        # Cap tokens with a seeded permutation SHARED across layers (same
        # guard as geometry_hsic; audit 2026-07-16: without it the study
        # defaults build ~1e4-5e4-point Gram matrices with O(N^3) unbiased
        # HSIC per layer pair — infeasible/OOM).
        max_tokens = int(self.config.get("max_cknna_tokens", 2000))
        shared_perm = None
        if n_points > max_tokens:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.get("seed", 42)))
            shared_perm = torch.randperm(n_points, generator=rng)[:max_tokens]
            n_points = max_tokens

        logger.info("  Normalising activations...")
        normed = {}
        for idx in tqdm(layers, desc="Normalising"):
            acts = layer_activations[idx]
            if shared_perm is not None:
                acts = acts[shared_perm]
            # Reference order: remove_outliers -> normalize.
            normed[idx] = _l2_normalize(_remove_outliers(acts.float()))

        logger.info("  Computing CKNNA matrix...")
        cknna_matrix = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
        _order_gap = 0.0
        for i in tqdm(range(n_layers), desc="CKNNA rows"):
            X = normed[layers[i]]
            for j in range(i, n_layers):
                Y = normed[layers[j]]
                val = _cknna(X, Y, topk=topk, unbiased=unbiased)
                # The reference kernel is not exactly order-symmetric (see the
                # module note). Measure the gap on the adjacent pairs rather
                # than assume it away; the headline stays the reference value.
                if j == i + 1:
                    _rev = _cknna(Y, X, topk=topk, unbiased=unbiased)
                    if np.isfinite(val) and np.isfinite(_rev):
                        _order_gap = max(_order_gap, abs(val - _rev))
                cknna_matrix[i, j] = val
                cknna_matrix[j, i] = val

        n = n_layers
        iu = np.triu_indices(n, k=1)
        off_diag = cknna_matrix[iu] if iu[0].size else np.array([], dtype=float)
        off_diag = off_diag[np.isfinite(off_diag)]

        adjacent = np.array(
            [cknna_matrix[i, i + 1] for i in range(n - 1)], dtype=float,
        ) if n > 1 else np.array([], dtype=float)
        adjacent = adjacent[np.isfinite(adjacent)]

        def _stat(arr, fn, fallback=float("nan")):
            return float(fn(arr)) if arr.size else fallback

        return {
            # Adjacent-layer local alignment (headline quantity).
            #
            # Fallback NaN, not 0.0, and `nan*` reducers throughout -- the
            # identical pair of defects was fixed in the CKA sibling on
            # 2026-09-05 and left here (independent Opus review the same day).
            # 0.0 is a MEANINGFUL CKNNA value ("these layers share no
            # neighbours"), so returning it for "there was no adjacent pair to
            # measure" is a false zero; and a summary that PROPAGATES a NaN
            # sitting beside one that DROPS it gives two different answers to
            # "what happened here?" in one result dict.
            "avg_adjacent_cknna": _stat(adjacent, np.nanmean),
            "min_adjacent_cknna": _stat(adjacent, np.nanmin),
            "max_adjacent_cknna": _stat(adjacent, np.nanmax),
            "std_adjacent_cknna": _stat(adjacent, np.nanstd),
            # Global off-diagonal summary (all layer pairs).
            "mean_offdiag_cknna": _stat(off_diag, np.nanmean),
            "std_offdiag_cknna": _stat(off_diag, np.nanstd),
            "min_offdiag_cknna": _stat(off_diag, np.nanmin),
            "max_offdiag_cknna": _stat(off_diag, np.nanmax),
            # Early-vs-late (layer 0 vs layer N-1).
            "early_late_cknna": (
                float(cknna_matrix[0, -1]) if n >= 2 else float("nan")
            ),
            # First-to-middle (early representation drift).
            "first_middle_cknna": (
                float(cknna_matrix[0, n // 2]) if n >= 3 else float("nan")
            ),
            # _meta_ prefix => excluded from the analysis feature matrix so
            # these architecture/sampling counts cannot leak in as size
            # proxies (Audit-V2).
            "_meta_n_layers": int(n),
            "_meta_last_entry_is_block_output": bool(
                _cknna_status.get("last_is_block", False)),
            "_meta_topk": int(topk),
            "_meta_n_points": int(n_points),
            # Reference-inherited argument-order dependence, measured over the
            # adjacent pairs. Disclosed, not concealed; see the module note.
            "_meta_cknna_order_asymmetry_max": float(_order_gap),
        }
