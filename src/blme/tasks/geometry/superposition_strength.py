"""Superposition strength of the unembedding — Liu, Liu & Gore, NeurIPS 2025.

"Superposition Yields Robust Neural Scaling" (arXiv:2505.10465; official code
https://github.com/liuyz0/SuperpositionScaling) shows that the degree of
representation superposition — more features than dimensions, packed as
almost-orthogonal directions — drives neural scaling, and that open-weight
LLMs operate in the STRONG-superposition regime: the mean squared overlap of
their (row-normalized) LM-head vectors follows the ~1/m scaling of the Welch
bound, the tightest possible packing of n >> m unit vectors in m dimensions.

Measured quantities (weights-only, Tier 1):

  * ``mean_squared_overlap``: E_{i != j} [ cos^2(W_i, W_j) ] over row
    pairs of the RAW unembedding matrix W in R^{V x m} (row-normalized,
    no centering) — the official-repo protocol (overlap-0.py). A
    mean-centered variant is reported as ``mean_squared_overlap_centered``
    (a disclosed BLME auxiliary that removes the shared anisotropy
    direction; see the protocol note in evaluate()).
  * ``welch_bound``: kappa^2 = (V - m) / (m * (V - 1)) — the minimum
    achievable mean squared overlap for V unit vectors in R^m (~1/m for
    V >> m).
  * ``superposition_ratio``: mean_squared_overlap / welch_bound (raw
    protocol; on real LLMs the shared mean direction makes this >> 1 —
    gpt2: ~58). ``superposition_ratio_centered`` divides the centered
    variant instead (~1-2 on real LLMs: near-optimal packing net of the
    common mean; random Gaussian rows give E[cos^2] = 1/m, i.e. ratio
    ~ 1/(1 - m/V), slightly above 1).
  * ``row_norm_cv`` and ``frac_norm_above_half_max``: dispersion of the
    row-norm distribution (the paper's toy-model phi thresholds are
    norm-scale dependent; we report scale-free analogues).

Analytic anchors (used by the parity tests): orthonormal rows (V = m) give
mean_squared_overlap = 0 = Welch bound; a simplex ETF of V = m + 1 vectors
achieves the Welch bound exactly (cos^2 = 1/m^2 for all pairs); iid Gaussian
rows give E[cos^2] = 1/m.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_embeddings, get_lm_head
from .utils import safe_unit_rows_torch

logger = logging.getLogger("blme")


def _mean_squared_overlap(W: torch.Tensor, max_pairs: int = 200_000,
                          seed: int = 42) -> float:
    """Mean squared cosine overlap over distinct row pairs of ``W``.

    ALWAYS exact over all V(V-1) ordered distinct pairs, via the Gram identity
    (see below). ``max_pairs`` and ``seed`` are accepted for call-site
    compatibility and are unused -- there is no longer any sampling.
    """
    W = W.float()
    # BARE norm over nonzero rows -- matching the official overlap-0.py, which
    # uses no epsilon.
    #
    # A `W / (||W|| + 1e-9)` variant was added on 2026-09-03 and REVERTED the
    # same day. It was recommended on the grounds that the official code uses
    # that epsilon; the official file was then fetched and it does NOT (the only
    # 1e-8 in that repo is the toy model's AdamW). The epsilon also breaks the
    # property the metric depends on: cosine similarity is scale invariant, but
    # with the epsilon a global rescaling of W to 1e-9 moved the score by -60%
    # on gpt2 and -97% on pythia-70m, and it left V_eff inconsistent so
    # `superposition_ratio` could fall BELOW the Welch bound (0.615 measured
    # against a correct 1.094 at 25% zero rows), which is provably impossible.
    #
    # On real unembedding matrices the two agree to rel 5.4e-8 over all 1.26e9
    # gpt2 pairs, so this only ever mattered for degenerate rows -- which are
    # excluded, as the official does.
    # ZERO-ROW FILTER via max-abs, not via the norm. A row has zero L2 norm
    # iff its largest absolute entry is zero -- that predicate is EXACT at
    # every scale, whereas `W.norm(dim=1)` squares first and so reports 0.0 for
    # rows that are merely small: at a global scale of 1e-25 all 3000 rows of a
    # 64-dim Gaussian were dropped as "zero" (sum of squares 6.4e-49 underflows
    # float32's 1.4e-45 subnormal floor) and the task returned NaN. Same class
    # as `safe_norm`, one level up.
    rmax = W.abs().amax(dim=1)
    n_zero_rows = int((rmax <= 0).sum())
    keep = rmax > 0
    W = W[keep]
    V = W.shape[0]
    if V < 2:
        return float("nan")

    # EXACT over all V(V-1) ordered distinct pairs, in closed form.
    #
    #   sum_{i,j} (u_i . u_j)^2 = ||U U^T||_F^2 = ||U^T U||_F^2
    #
    # and the V diagonal terms are each 1, so
    #
    #   E_{i != j}[(u_i . u_j)^2] = (||U^T U||_F^2 - V) / (V (V - 1)).
    #
    # `U^T U` is only (m, m), so this costs O(V m^2) instead of O(V^2 m) and
    # needs no pair sampling at all. It removes the 200,000-pair cap, which was
    # the last deviation from the official overlap-0.py -- Appendix C.1 and the
    # released code evaluate ALL V(V-1) pairs (adversarial review, 2026-09-03).
    #
    # For gpt2 that is 2.5e9 pairs, computed here from a 768x768 Gram matrix.
    V_f = float(V)
    if V < 2:
        return float("nan")
    # float64 CHUNKED accumulation, not a single float32 matmul. Two separate
    # problems, both measured (red-team review, 2026-09-05):
    #
    #  * SCALE. The float32 row normalisation above returned NaN for W scaled
    #    to <= 1e-25 (`norm` underflows to 0 while the rows are still finite)
    #    and, at >= 1e19, overflowed to give a NEGATIVE overlap of -2.5004e-4
    #    = -1/(V-1) -- i.e. a `superposition_ratio` BELOW the Welch bound,
    #    which is the exact outcome this file's own comment calls "provably
    #    impossible" when it reverted the +1e-9 epsilon. Same impossible
    #    state, different route. `safe_unit_rows_torch` factors out the
    #    largest per-row magnitude first, which only shifts exponents.
    #
    #  * CANCELLATION. `total_sq - V` assumes each diagonal Gram entry is
    #    EXACTLY 1; in float32 they are not, and the difference is the whole
    #    answer when overlaps are small. On an orthonormal 256x256 matrix
    #    (true MSO = 0) the old form gave 9.35e-10 against an exact-float64
    #    2.68e-10, +249% rel. Subtracting the ACTUAL sum of squared row norms
    #    instead of the integer V removes that mismatch at its source.
    #
    # Cost is bounded: the accumulator is (m, m) and only one (B, m) chunk is
    # promoted at a time, so a 256k-vocab 4608-dim head needs ~300 MB, not the
    # 9.4 GB a wholesale `.double()` would take.
    Wn = safe_unit_rows_torch(W)
    m_dim = Wn.shape[1]
    gram = torch.zeros((m_dim, m_dim), dtype=torch.float64, device=Wn.device)
    diag_sq = 0.0
    for i in range(0, V, 8192):
        c = Wn[i:i + 8192].double()
        gram += c.t() @ c
        diag_sq += float((c.pow(2).sum(dim=1) ** 2).sum().item())
    total_sq = float((gram ** 2).sum().item())
    return float((total_sq - diag_sq) / (V_f * (V_f - 1.0)))


def _welch_bound(V: int, m: int) -> float:
    """Minimum mean squared overlap of V unit vectors in R^m (Welch 1974)."""
    if V <= m:
        return 0.0
    return float((V - m) / (m * (V - 1)))


@register_task("geometry_superposition_strength")
class SuperpositionStrengthTask(DiagnosticTask):
    """Welch-bound-normalized packing of the unembedding rows (weights-only)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Superposition Strength (Liu-Liu-Gore 2025)...")
        max_pairs = int(self.config.get("max_pairs", 200_000))
        seed = int(self.config.get("seed", 42))

        head = get_lm_head(model)
        if head is not None and hasattr(head, "weight"):
            W = head.weight.detach()
        else:
            W = get_embeddings(model)
            if W is None:
                return {"error": "Could not find lm_head or embeddings"}

        W = W.float().cpu()
        V, m = W.shape
        # THE SAME PREDICATE AS THE KERNEL, which is the point of this line and
        # was not what it did.
        #
        # `_mean_squared_overlap` drops a row when its MAX-ABS is zero -- an
        # exact test at every scale. This line tested `W.norm(dim=1) > 0`, which
        # squares first, so at a global scale of 1e-23 every row's norm
        # underflowed to 0.0, `V_eff` became 0, the Welch bound became 0, and
        # `superposition_ratio` came back **NaN** while `mean_squared_overlap`
        # was correct to 1.3e-9. That is exactly the "V_eff inconsistent"
        # failure the comment on the kernel says was removed -- removed from the
        # kernel, left here (independent Opus review, 2026-09-05).
        V_eff = int((W.abs().amax(dim=1) > 0).sum().item())

        # Headline statistic follows the OFFICIAL protocol (repo
        # liuyz0/SuperpositionScaling, LLMs/overlap-0.py): RAW rows,
        # normalized without mean-centering. The paper's ~1/m claim is a
        # log-log scaling TREND of this raw quantity across hidden sizes,
        # not a coefficient-1 statement.
        # BLME ALSO reports a mean-centered variant (BLME auxiliary, NOT
        # part of the official protocol): LLM unembeddings carry a shared
        # anisotropy direction that dominates raw cosines (gpt2: raw MSO
        # 0.075 = 58x Welch vs centered 0.0025 = 1.9x Welch), so the
        # centered variant isolates packing efficiency net of the common
        # mean. Audit 2026-07-16 verified the official code never centers.
        mso = _mean_squared_overlap(W, max_pairs=max_pairs, seed=seed)
        W_centered = W - W.mean(dim=0, keepdim=True)
        mso_centered = _mean_squared_overlap(W_centered, max_pairs=max_pairs,
                                             seed=seed)
        welch = _welch_bound(V_eff, m)

        # Row norms via the same factor-out-the-max trick: `norm_cv` is a
        # COEFFICIENT OF VARIATION and `frac_norm_above_half_max` a fraction of
        # a maximum, so both are scale-free by definition and neither may be
        # computed with an arithmetic that is not.
        # In units of the GLOBAL max, never in absolute units. Both statistics
        # are invariant to a common factor -- `norm_cv` is std/mean and
        # `frac_norm_above_half_max` compares each norm to the largest -- so
        # dividing every row by one global constant changes neither, and it
        # keeps every intermediate inside [0, sqrt(m)].
        #
        # Materialising the ABSOLUTE norms does not: at a global scale of
        # 3.4e38 a row norm is up to sqrt(m) times that, which overflows float32
        # to inf and gave `row_norm_cv = nan` and
        # `frac_norm_above_half_max = 0.0` -- a scale-free quantity destroyed by
        # the arithmetic used to compute it, which is this campaign's most
        # repeated numeric defect in its purest form.
        _g = W.abs().amax()
        _rmax = W.abs().amax(dim=1)
        _nz = _rmax > 0
        norms = torch.zeros_like(_rmax)
        if bool(_nz.any()) and float(_g) > 0:
            norms[_nz] = (W[_nz] / _g).norm(dim=1)
        norms = norms[_nz]
        norm_cv = float((norms.std() / norms.mean()).item()) if len(norms) else float("nan")
        frac_half = float((norms > 0.5 * norms.max()).float().mean().item()) if len(norms) else float("nan")

        return {
            "mean_squared_overlap": mso,
            "mean_squared_overlap_centered": mso_centered,
            # `_meta_`: a CLOSED FORM IN (V, m) and nothing else. Reproduced
            # from the two integers to exactly 0.0 on all 66 archived models,
            # pearson 0.99982 with 1/d_model. In a study whose claim is
            # "structure predicts capability BEYOND scale", a deterministic
            # function of vocab and width is not a structural predictor; it
            # was passing the exclusion gate because the blocklist named
            # `d_model`/`hidden_size` but neither `welch_bound` nor
            # `hidden_dim` (red-team review, 2026-09-05).
            "_meta_welch_bound": welch,
            # ratios vs the packing optimum; guard V<=m (welch=0).
            "superposition_ratio": (mso / welch) if welch > 0 else float("nan"),
            "superposition_ratio_centered": (mso_centered / welch) if welch > 0 else float("nan"),
            # ratios vs the random-Gaussian baseline E[cos^2] = 1/m.
            #
            # `_meta_`: ALGEBRAICALLY the same degree of freedom as
            # `superposition_ratio`, scaled by a pure architecture constant.
            #   superposition_ratio = mso / welch = mso * m * (V-1)/(V-m)
            #   overlap_vs_random   = mso * m
            # so their ratio is exactly (V-1)/(V-m) -- 1.0153 on gpt2 -- and a
            # sweep over 5 models measures pearson 0.999994 / 0.999994 for the
            # raw and centered pairs (duplicate-column sweep, 2026-09-05). The
            # Welch-normalized form is the paper's headline and stays the
            # feature; this one is the same number carrying an extra (V, m)
            # term, which is precisely the architecture signal the study must
            # not admit as structure.
            "_meta_overlap_vs_random": mso * m,
            "_meta_overlap_vs_random_centered": mso_centered * m,
            "row_norm_cv": norm_cv,
            "frac_norm_above_half_max": frac_half,
            "_meta_vocab_size": int(V),
            # `_meta_`: this IS d_model (spearman +0.9503 with log-params),
            # spelled with a name the blocklist did not have.
            "_meta_hidden_dim": int(m),
        }
