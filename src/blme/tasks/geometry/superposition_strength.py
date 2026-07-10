"""Superposition strength of the unembedding — Liu, Liu & Gore, NeurIPS 2025.

"Superposition Yields Robust Neural Scaling" (arXiv:2505.10465; official code
https://github.com/liuyz0/SuperpositionScaling) shows that the degree of
representation superposition — more features than dimensions, packed as
almost-orthogonal directions — drives neural scaling, and that open-weight
LLMs operate in the STRONG-superposition regime: the mean squared overlap of
their (row-normalized) LM-head vectors follows the ~1/m scaling of the Welch
bound, the tightest possible packing of n >> m unit vectors in m dimensions.

Measured quantities (weights-only, Tier 1):

  * ``mean_squared_overlap``: E_{i != j} [ cos^2(W_i, W_j) ] over sampled
    row pairs of the MEAN-CENTERED unembedding matrix W in R^{V x m}
    (row-normalized after centering; the raw-row variant is reported as
    ``mean_squared_overlap_raw`` — see the protocol note in evaluate()).
  * ``welch_bound``: kappa^2 = (V - m) / (m * (V - 1)) — the minimum
    achievable mean squared overlap for V unit vectors in R^m (~1/m for
    V >> m).
  * ``superposition_ratio``: mean_squared_overlap / welch_bound. ~1 means
    the vocabulary is packed at the theoretical optimum (strong
    superposition); >> 1 means far-from-optimal packing; random Gaussian
    rows give E[cos^2] = 1/m, i.e. ratio ~ 1/(1 - m/V) (slightly above 1).
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

logger = logging.getLogger("blme")


def _mean_squared_overlap(W: torch.Tensor, max_pairs: int = 200_000,
                          seed: int = 42) -> float:
    """Mean squared cosine overlap over distinct row pairs of ``W``.

    Exact (all pairs) when V*(V-1)/2 <= max_pairs, otherwise a seeded
    uniform pair sample (self-pairs rejected).
    """
    W = W.float()
    norms = W.norm(dim=1, keepdim=True)
    keep = norms.squeeze(1) > 0
    W = W[keep]
    V = W.shape[0]
    if V < 2:
        return float("nan")
    Wn = W / W.norm(dim=1, keepdim=True)

    n_pairs = V * (V - 1) // 2
    if n_pairs <= max_pairs:
        G = Wn @ Wn.T
        iu = torch.triu_indices(V, V, offset=1)
        return float((G[iu[0], iu[1]] ** 2).mean().item())

    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, V, (max_pairs,), generator=gen)
    b = torch.randint(0, V, (max_pairs,), generator=gen)
    mask = a != b
    a, b = a[mask], b[mask]
    cos = (Wn[a] * Wn[b]).sum(dim=1)
    return float((cos ** 2).mean().item())


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

        # Headline statistic uses MEAN-CENTERED rows: LLM unembeddings carry
        # a large shared mean direction (the anisotropy cone), which inflates
        # raw cosine overlaps by orders of magnitude (gpt2: raw MSO 0.075 vs
        # centered 0.0025 ~ 1.9/m). The paper's ~1/m Welch-scaling finding on
        # open LLMs only reproduces after removing the mean component, and
        # centering matches this library's unembedding-erank convention. The
        # raw variant is reported alongside.
        W_centered = W - W.mean(dim=0, keepdim=True)
        mso = _mean_squared_overlap(W_centered, max_pairs=max_pairs, seed=seed)
        mso_raw = _mean_squared_overlap(W, max_pairs=max_pairs, seed=seed)
        welch = _welch_bound(V, m)

        norms = W.norm(dim=1)
        norms = norms[norms > 0]
        norm_cv = float((norms.std() / norms.mean()).item()) if len(norms) else float("nan")
        frac_half = float((norms > 0.5 * norms.max()).float().mean().item()) if len(norms) else float("nan")

        return {
            "mean_squared_overlap": mso,
            "mean_squared_overlap_raw": mso_raw,
            "welch_bound": welch,
            # ratio vs the packing optimum; guard V<=m (welch=0).
            "superposition_ratio": (mso / welch) if welch > 0 else float("nan"),
            # ratio vs the random-Gaussian baseline E[cos^2] = 1/m.
            "overlap_vs_random": mso * m,
            "row_norm_cv": norm_cv,
            "frac_norm_above_half_max": frac_half,
            "vocab_size": int(V),
            "hidden_dim": int(m),
        }
