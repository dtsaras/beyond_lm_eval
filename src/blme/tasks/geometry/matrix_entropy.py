"""Matrix Entropy (Wei et al. 2024, arXiv:2401.17139).

Computes the **per-sentence** von Neumann entropy of the centered-and-
row-normalised token covariance, then averages over sentences and
normalises by ``log d`` (paper Def. 4.3).

Rewrite history:
  * 2026-04-17: added per-sentence aggregation, **row-wise L2
    normalisation after centering**, and ``/ log d`` normalisation to
    match Def. 4.1 and Def. 4.3 of Wei et al. The original BLME
    implementation pooled every token from every sample into a single
    covariance matrix per layer and skipped both the row normalisation
    and the ``log d`` divisor — so the reported headline number was
    architecture-dependent and dominated by whichever tokens happened to
    have large activation norm (a known issue for LLM hidden states).

References:
  * Wei, Tan, Li, Wang, Huang, "Large Language Model Evaluation via
    Matrix Entropy", arXiv:2401.17139 (2024).
  * Official reference code: https://github.com/waltonfuture/Matrix-Entropy
    (now `Diff-eRank`).
"""

import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _matrix_entropy(Z: torch.Tensor) -> dict:
    """Compute Wei et al. matrix entropy for a single sentence's hidden
    states ``Z ∈ R^{N_tokens × d}``.

    Steps, matching Def. 4.1–4.3 of the paper:
      1. Center: ``Z ← Z − mean(Z, dim=0)``.
      2. Row L2 normalise: ``Z ← Z / ‖Z_row‖₂``.
      3. Covariance: ``A = Zᵀ Z / N`` (d × d).
      4. Unit trace: ``A ← A / tr(A)``.
      5. Entropy: ``H = − Σ λ log λ`` over eigenvalues of ``A``.
      6. Normalised entropy: ``H / log d`` (Def. 4.3).

    Returns a dict with both the raw and normalised entropies so the
    caller can decide which to report. NaN is returned when the input
    is degenerate (< 2 tokens, zero rows after normalisation, zero
    trace, etc.).
    """
    if Z.numel() == 0 or Z.shape[0] < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    d = int(Z.shape[1])
    if d < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}

    Z = Z.float()
    # If the caller handed us a CPU tensor and CUDA is available,
    # move to GPU for the SVD — at D = 8192 (Qwen-9B) a CPU SVD of
    # (128, D) takes ~1 s while the GPU one takes <10 ms; this was
    # the dominant cost when the task timed out on the v2 run.
    if not Z.is_cuda and torch.cuda.is_available():
        try:
            Z = Z.cuda(non_blocking=True)
        except Exception:
            pass
    Z = Z - Z.mean(dim=0, keepdim=True)

    # Row L2 normalisation — Def. 4.1.
    row_norms = Z.norm(p=2, dim=1, keepdim=True)
    keep = row_norms.squeeze(-1) > 1e-12
    if int(keep.sum().item()) < 2:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    Z = Z[keep] / row_norms[keep]

    # A = Zᵀ Z / N. Use SVD on Z to avoid materialising a d×d matrix
    # when N < d. Squared singular values / N are the eigenvalues of A.
    S = torch.linalg.svdvals(Z)
    lam = (S ** 2) / float(Z.shape[0])
    trace = float(lam.sum().item())
    if not math.isfinite(trace) or trace <= 0:
        return {"entropy": float("nan"), "entropy_normalized": float("nan")}
    lam = lam / trace

    # − Σ λ log λ with the 0·log 0 = 0 convention.
    lam = lam.clamp(min=1e-30)
    H = float(-torch.sum(lam * torch.log(lam)).item())

    return {
        "entropy": H,
        "entropy_normalized": H / math.log(max(d, 2)),
    }


def _collect_per_sample_layer_tokens(cache, num_samples, use_cache):
    """Return ``{layer_idx: List[Tensor(T_i, D)]}`` — one tensor per
    sentence per layer — from the shared cache if available."""
    if cache is None or not cache.is_populated or not use_cache:
        return None
    hs = cache.get_hidden_states(
        layer_idx="all", num_samples=num_samples, per_sample=True,
    )
    if not hs:
        return None
    return {li: chunks for li, chunks in hs.items() if chunks}


def _collect_per_sample_layer_tokens_fresh(model, tokenizer, samples, max_length):
    """Fallback path: run our own forward pass sample-by-sample and
    return ``{layer_idx: List[Tensor(T_i, D)]}``.

    Keeps the collected tensors on the model's device so the downstream
    per-sentence SVD runs on GPU — GPU SVD of a ``(128, D)`` matrix is
    100-1000× faster than CPU SVD at D=8192+, which was the dominant
    cost when this task timed out on the v2 run.
    """
    device = next(model.parameters()).device
    per_layer: dict[int, list[torch.Tensor]] = {}
    with torch.no_grad():
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            ).to(device)
            out = model(**inputs, output_hidden_states=True)
            if not getattr(out, "hidden_states", None):
                continue
            for li, h in enumerate(out.hidden_states[1:]):  # skip embedding
                per_layer.setdefault(li, []).append(
                    h[0].detach().float()
                )
    return per_layer


@register_task("geometry_matrix_entropy")
class MatrixEntropyTask(DiagnosticTask):
    """Per-sentence-averaged Wei et al. 2024 matrix entropy per layer.

    Canonical "Wei et al. number" is the entropy of the **last** hidden
    layer; we report every layer too as a convenient extension but flag
    the last-layer value as the headline metric.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Matrix Entropy (Wei et al. 2024)...")
        num_samples = int(self.config.get("num_samples", 10))
        use_cache = self.config.get("use_cache", True)
        max_length = int(self.config.get("max_length", 128))

        per_layer_tokens = _collect_per_sample_layer_tokens(
            cache, num_samples, use_cache,
        )

        if per_layer_tokens is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)
            samples = list(dataset)[:num_samples]
            if not samples:
                return {"error": "Need at least 1 sample."}
            per_layer_tokens = _collect_per_sample_layer_tokens_fresh(
                model, tokenizer, samples, max_length,
            )

        if not per_layer_tokens:
            return {"error": "Could not collect hidden states."}

        layer_entropies: dict[str, float] = {}
        layer_entropies_norm: dict[str, float] = {}
        for li in sorted(per_layer_tokens.keys()):
            per_sentence = []
            per_sentence_norm = []
            for Z in per_layer_tokens[li]:
                res = _matrix_entropy(Z)
                if np.isfinite(res["entropy"]):
                    per_sentence.append(res["entropy"])
                if np.isfinite(res["entropy_normalized"]):
                    per_sentence_norm.append(res["entropy_normalized"])
            if per_sentence:
                layer_entropies[f"layer_{li}"] = float(np.mean(per_sentence))
            if per_sentence_norm:
                layer_entropies_norm[f"layer_{li}"] = float(np.mean(per_sentence_norm))

        finite_raw = [v for v in layer_entropies.values() if np.isfinite(v)]
        finite_norm = [v for v in layer_entropies_norm.values() if np.isfinite(v)]
        if not finite_raw:
            return {"error": "All layers produced non-finite entropy."}

        # Headline = last-layer entropy, matching Wei et al.'s reported
        # number. Aggregate mean is also exposed as an extension.
        last_key = max(layer_entropies.keys(), key=lambda k: int(k.split("_")[1]))
        headline = layer_entropies[last_key]
        headline_norm = layer_entropies_norm.get(last_key, float("nan"))

        return {
            # Headline "Wei et al." number for the model.
            "matrix_entropy": headline,
            "matrix_entropy_normalized": headline_norm,
            # Per-layer extension (same metric applied to every block).
            "layer_matrix_entropies": layer_entropies,
            "layer_matrix_entropies_normalized": layer_entropies_norm,
            # Backwards-compatible aggregate across layers.
            "mean_matrix_entropy": float(np.mean(finite_raw)),
            "mean_matrix_entropy_normalized": (
                float(np.mean(finite_norm)) if finite_norm else float("nan")
            ),
            "interpretation": (
                "Lower last-layer matrix entropy indicates a stronger "
                "information bottleneck (Wei et al., 2024)."
            ),
        }
