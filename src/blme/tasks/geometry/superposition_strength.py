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
    # Bare row normalisation, no epsilon, as the official overlap-0.py (an
    # epsilon breaks scale invariance); exactly-zero rows are excluded, by a
    # max-abs test that is exact at every scale.
    rmax = W.abs().amax(dim=1)
    n_zero_rows = int((rmax <= 0).sum())
    keep = rmax > 0
    W = W[keep]
    V = W.shape[0]
    if V < 2:
        return float("nan")

    # Exact over all V(V-1) ordered distinct pairs, in closed form:
    #   E_{i != j}[(u_i . u_j)^2] = (||U^T U||_F^2 - sum_i ||u_i||^4) / (V (V - 1)),
    # from an (m, m) Gram, as the released code evaluates every pair.
    V_f = float(V)
    if V < 2:
        return float("nan")
    # float64 chunked accumulation: max-factored unit rows (scale-safe), and
    # the diagonal subtracted as the actual sum of squared row norms rather
    # than the integer V (float32 cancellation when overlaps are small).
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

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2505.10465 (Liu, Liu & Gore 2025, Sec. 3.2-3.3)",
        "code": "liuyz0/SuperpositionScaling@771670329ae948550ccfa0160edeb614f31d092b "
                "(LLMs/overlap-0.py:68-130)",
        "llm_precedent": "arXiv:2505.10465 (mean squared overlap of open LLMs' LM-head rows)",
    }
    HEADLINE_KEYS = ("mean_squared_overlap", "mean_squared_overlap_centered",
                     "superposition_ratio", "superposition_ratio_centered",
                     "row_norm_cv", "frac_norm_above_half_max")
    # Width (zero-padding W's hidden dimension keeps every cosine and norm) and
    # vocabulary (unused rows past len(tokenizer)). The Welch ratios carry
    # kappa^2 ~ 1/m, a capacity ratio, so only the vocabulary test applies.
    SIZE_INVARIANT = {k: ((("V_unused_rows",) if k.startswith("superposition_ratio")
                           else ("zero_pad", "V_unused_rows")),
                          "tests/tasks/invariance/test_superposition_strength.py::"
                          f"test_size_invariant_entry[{k}]")
                      for k in HEADLINE_KEYS}

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
        V_all, m = W.shape
        # Rows the tokenizer can emit (ids < len(tokenizer)): matrices are
        # padded past the tokenizer (pythia 27 rows, Qwen3 267) and those rows
        # are not tokens; on Qwen3-4B they move the centred keys by 7%. The
        # released overlap-0.py's every-row MSO stays as _meta_.
        from .categories import _n_emittable
        n_emit = _n_emittable(tokenizer, V_all) if tokenizer is not None else V_all
        mso_all_rows = (_mean_squared_overlap(W, max_pairs=max_pairs, seed=seed)
                        if n_emit < V_all else None)
        W = W[:n_emit]
        V = n_emit
        # The kernel's zero-row predicate (max-abs, exact at every scale).
        V_eff = int((W.abs().amax(dim=1) > 0).sum().item())

        # Headline: the official protocol (overlap-0.py), raw rows normalised
        # without centring. The mean-centred variant is a BLME auxiliary that
        # removes the shared anisotropy direction.
        mso = _mean_squared_overlap(W, max_pairs=max_pairs, seed=seed)
        W_centered = W - W.mean(dim=0, keepdim=True)
        mso_centered = _mean_squared_overlap(W_centered, max_pairs=max_pairs,
                                             seed=seed)
        welch = _welch_bound(V_eff, m)

        # Row norms in units of the global max: both statistics are scale-free,
        # and absolute norms can overflow float32 near its maximum.
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
            # A closed form in (V, m): an architecture constant.
            "_meta_welch_bound": welch,
            # ratios vs the packing optimum; guard V<=m (welch=0).
            "superposition_ratio": (mso / welch) if welch > 0 else float("nan"),
            "superposition_ratio_centered": (mso_centered / welch) if welch > 0 else float("nan"),
            # Versus the random baseline 1/m: the Welch ratio times (V-m)/(V-1).
            "_meta_overlap_vs_random": mso * m,
            "_meta_overlap_vs_random_centered": mso_centered * m,
            "row_norm_cv": norm_cv,
            "frac_norm_above_half_max": frac_half,
            "_meta_vocab_size": int(V_all),
            "_meta_n_emittable_rows": int(n_emit),
            "_meta_n_unused_rows": int(V_all - n_emit),
            "_meta_mean_squared_overlap_all_rows": (
                mso if mso_all_rows is None else mso_all_rows),
            "_meta_hidden_dim": int(m),
        }
