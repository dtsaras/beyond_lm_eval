"""Schatten-p norms + Matrix Nuclear-Norm + RankMe (2024-2025).

Three related spectral measurements of the hidden-state matrix at each
layer:

1. **Schatten-p norms** ``‖X‖_{S_p} = (Σ σᵢ^p)^{1/p}`` for
   ``p ∈ {1, 2, 4, ∞}`` (classical; named after R. Schatten) — used as
   reference-free text-quality proxies by Yusupov et al. 2025 ("From
   Internal Representations to Text Quality: A Geometric Approach to LLM
   Evaluation", arXiv:2509.25359). Schatten-1 = nuclear norm,
   Schatten-2 = Frobenius, Schatten-∞ = spectral (largest singular
   value). NB: that paper's own conclusion is that Schatten/MOM scores
   largely track *output length* once it is controlled — see AUDIT_V2 §5.

2. **Matrix Nuclear-Norm (MNN)** — Li et al. 2024
   (arXiv:2410.10672) — an L1,2-norm-based approximation to the true
   nuclear norm that runs 8-24× faster than SVD-based matrix entropy.
   Reference implementation: the ``matrix_nuclear_norm`` helper at
   https://github.com/MLGroupJLU/MatrixNuclearNorm sorts per-column
   L2-norms in descending order and sums the top ``D = min(N, d)``.
   We replicate that here.

3. **RankMe** — Garrido et al. 2023 (arXiv:2210.02885) — effective
   rank via ``exp(H(p))`` with ``p_i = σ_i / Σ σ_j`` (i.e.,
   normalising the *raw* singular values). Distinct from the
   Roy-Vetterli effective rank in ``geometry.utils.effective_rank``
   which normalises ``σ_i² / Σ σ_j²``; the two are not equivalent on
   non-uniform spectra and both are cited in the recent literature
   (Tracing Representation Geometry, Li et al. 2025,
   arXiv:2509.23024). We expose both so the paper can discuss the
   difference.

Reports the per-layer profile and a "last-layer" headline (Yusupov et al.
convention for the MLP-output representation) for each metric, as well
as mean/std/slope/q25/q50/q75 summaries via the aggregator's layer
collapse.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# ── Math helpers ────────────────────────────────────────────────────


def _schatten_p_norm(S, p) -> float:
    """Schatten-p norm of a matrix with singular values ``S``.

    p = 1     → nuclear norm (Σ σ)
    p = 2     → Frobenius (sqrt Σ σ²)
    p = ∞     → spectral (max σ)
    otherwise → (Σ σ^p)^{1/p}
    """
    arr = np.asarray(S, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0:
        return float("nan")
    if math.isinf(p):
        return float(arr.max())
    if p == 1:
        return float(arr.sum())
    if p == 2:
        return float(np.sqrt(np.sum(arr * arr)))
    return float(np.sum(arr ** p) ** (1.0 / p))


def _matrix_nuclear_norm_fast(X: torch.Tensor, D: Optional[int] = None) -> float:
    """Li et al. 2024 fast approximation to the nuclear norm via the
    L1,2 column-norm upper bound. Re-creates the reference impl at
    https://github.com/MLGroupJLU/MatrixNuclearNorm.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    X = X.float()
    if X.dim() != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return float("nan")
    if D is None:
        D = int(min(X.shape[0], X.shape[1]))
    # Column L2 norms, sorted descending, summed top-D.
    l2_norms = torch.sqrt((X * X).sum(dim=0))
    sorted_norms, _ = torch.sort(l2_norms, descending=True)
    return float(sorted_norms[:D].sum().item())


def _rankme(S, eps: float = 1e-12) -> float:
    """RankMe (Garrido et al. 2023): ``exp(H(p))`` with
    ``p_i = σ_i / (Σ σ_j + ε)``.

    Bounded in ``[1, rank(X)]``; reduces to ``1`` for a pure rank-1
    matrix and to ``N`` for a uniform spectrum of length ``N``.
    """
    arr = np.asarray(S, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0:
        return float("nan")
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    p = arr / (total + eps)
    # Clip to avoid log(0); contributes 0 by convention.
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    H = float(-np.sum(p * np.log(p)))
    return float(np.exp(H))


# ── Per-sample helpers (match matrix_entropy's cache path) ──────────


def _normalise_matrix(X: torch.Tensor) -> torch.Tensor:
    """Replicate the Li et al. 2024 preprocessing: center columns, then
    L2-normalise each row.
    """
    if X.dim() != 2 or X.shape[0] < 2:
        return X
    X = X - X.mean(dim=0, keepdim=True)
    row_norms = X.norm(p=2, dim=1, keepdim=True)
    keep = row_norms.squeeze(-1) > 1e-12
    if int(keep.sum().item()) < 2:
        return X
    return X[keep] / row_norms[keep]


def _per_sentence_measurements(Z: torch.Tensor) -> Dict[str, float]:
    """Schatten norms + MNN + RankMe for a single sentence's hidden
    states ``Z ∈ R^{T × D}``."""
    if Z.numel() == 0 or Z.shape[0] < 2 or Z.shape[1] < 2:
        return {}
    Z = Z.float()
    if not Z.is_cuda and torch.cuda.is_available():
        try:
            Z = Z.cuda(non_blocking=True)
        except Exception:
            pass

    # Preprocess like the MNN reference (centre + row L2) so the two
    # paths are directly comparable.
    Z_norm = _normalise_matrix(Z)

    try:
        S = torch.linalg.svdvals(Z_norm).detach().cpu().numpy()
    except Exception:
        return {}

    mnn = _matrix_nuclear_norm_fast(Z_norm)
    # Normalise the MNN / Schatten norms by sqrt(D) — scale-invariance
    # across d_model. Wei et al. 2025 reports ``‖X‖_{S_p} / d^{1/p}``
    # as the comparable quantity across models of different widths.
    d = float(max(Z_norm.shape[1], 2))
    return {
        "schatten_1": _schatten_p_norm(S, p=1) / d,
        "schatten_2": _schatten_p_norm(S, p=2) / math.sqrt(d),
        "schatten_4": _schatten_p_norm(S, p=4) / (d ** 0.25),
        "schatten_inf": _schatten_p_norm(S, p=float("inf")),
        "matrix_nuclear_norm": mnn / d,
        "rankme": _rankme(S),
    }


def _collect_per_layer(cache, num_samples, use_cache):
    """Match matrix_entropy's cache path, returning per-layer per-
    sentence hidden states."""
    if cache is None or not cache.is_populated or not use_cache:
        return None
    hs = cache.get_hidden_states(
        layer_idx="all", num_samples=num_samples, per_sample=True,
    )
    if not hs:
        return None
    return {li: chunks for li, chunks in hs.items() if chunks}


def _collect_per_layer_fresh(model, tokenizer, samples, max_length):
    """Fallback path (no cache) — run a forward per sample, asking HF
    for ``hidden_states``. Mirrors matrix_entropy's fresh-collection
    helper so the two share semantics."""
    device = next(model.parameters()).device
    per_layer: Dict[int, List[torch.Tensor]] = {}
    with torch.no_grad():
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            )
            # BatchEncoding objects have .to(); plain dicts need
            # per-tensor movement.
            if hasattr(inputs, "to") and callable(inputs.to):
                inputs = inputs.to(device)
            else:
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            if not hs:
                continue
            for li, h in enumerate(hs[1:]):  # skip embedding
                per_layer.setdefault(li, []).append(h[0].detach().float())
    return per_layer


@register_task("geometry_schatten")
class SchattenNormTask(DiagnosticTask):
    """Per-layer Schatten-p norms + Matrix Nuclear-Norm + RankMe.

    Reports:
      - ``schatten_1_last``, ``schatten_2_last``, ``schatten_4_last``,
        ``schatten_inf_last`` — last-layer values of each norm,
        normalised by ``d^{1/p}`` so they're comparable across
        d_model (Wei et al. 2025 convention).
      - ``matrix_nuclear_norm_last`` — fast L1,2 approximation, also
        normalised by ``d``.
      - ``rankme_last`` — raw-singular-value effective rank (Garrido
        2023), complement to our Roy-Vetterli ``effective_rank`` in
        ``geometry_svd``.
      - Per-layer lists (``*_per_layer``) so the aggregator can emit
        mean/std/slope/q25/q50/q75 cross-model summaries.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Schatten-p norms + MNN + RankMe...")
        num_samples = int(self.config.get("num_samples", 10))
        use_cache = self.config.get("use_cache", True)
        max_length = int(self.config.get("max_length", 128))

        per_layer = _collect_per_layer(cache, num_samples, use_cache)
        if per_layer is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)
            samples = list(dataset)[:num_samples]
            if not samples:
                return {"error": "Need at least 1 sample"}
            per_layer = _collect_per_layer_fresh(
                model, tokenizer, samples, max_length,
            )

        if not per_layer:
            return {"error": "Could not collect hidden states"}

        per_layer_profiles: Dict[str, List[float]] = {
            "schatten_1": [],
            "schatten_2": [],
            "schatten_4": [],
            "schatten_inf": [],
            "matrix_nuclear_norm": [],
            "rankme": [],
        }

        for li in sorted(per_layer.keys()):
            per_sentence = {k: [] for k in per_layer_profiles}
            for Z in per_layer[li]:
                res = _per_sentence_measurements(Z)
                for k in per_layer_profiles:
                    v = res.get(k)
                    if v is not None and np.isfinite(v):
                        per_sentence[k].append(v)
            for k in per_layer_profiles:
                per_layer_profiles[k].append(
                    float(np.mean(per_sentence[k])) if per_sentence[k] else float("nan")
                )

        headline_idx = max(per_layer.keys())
        result = {}
        for k, profile in per_layer_profiles.items():
            finite = [v for v in profile if np.isfinite(v)]
            if not finite:
                result[f"{k}_last"] = float("nan")
                result[f"{k}_per_layer"] = profile
                continue
            # Headline = last layer (Wei et al. 2025 convention).
            last_val = profile[sorted(per_layer.keys()).index(headline_idx)]
            result[f"{k}_last"] = float(last_val) if np.isfinite(last_val) else float("nan")
            result[f"{k}_per_layer"] = profile

        result["n_layers"] = len(per_layer_profiles["schatten_1"])
        return result
