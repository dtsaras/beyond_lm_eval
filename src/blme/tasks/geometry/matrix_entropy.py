"""Matrix Entropy (Wei et al. 2024, arXiv:2401.17139 v1).

Computes the **per-sentence** von Neumann entropy of the centered-and-
row-normalised token covariance, then averages over sentences and
normalises by ``log d`` (paper Def. 4.1 and Def. 4.3).

References:
  * Wei, Tan, Li, Wang, Huang, "Large Language Model Evaluation via
    Matrix Entropy", arXiv:2401.17139 v1 (2024); v2 is "Diff-eRank".
  * Official reference code: https://github.com/waltonfuture/Matrix-Entropy
    (now `Diff-eRank`); parity pinned by
    tests/tasks/parity/test_matrix_entropy_parity.py.
"""

import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import hidden_states_for_task
from .utils import pow2_rescale as _pow2_rescale

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

    # Centre in DOUBLE first (Wei et al. centre every token, so a common
    # offset must not move the entropy; float32 would quantise it away), then
    # power-of-two rescale into float32 range (exact), then cast for the SVD.
    # A GPU, when present, takes the per-sentence SVD.
    Z = Z.double()
    if not Z.is_cuda and torch.cuda.is_available():
        try:
            Z = Z.cuda(non_blocking=True)
        except Exception:
            pass
    Z = Z - Z.mean(dim=0, keepdim=True)

    _fi0 = torch.finfo(torch.float32)
    _gmax0 = Z.abs().max()
    _lo0 = float(_fi0.tiny) ** 0.5 * 4.0
    _hi0 = (float(_fi0.max) / max(1, Z.shape[1])) ** 0.5 / 4.0
    if float(_gmax0) > 0 and (float(_gmax0) < _lo0 or float(_gmax0) > _hi0):
        Z = _pow2_rescale(Z, float(_gmax0))
    # Back to float32 for the SVD, now that the data is centred and in range.
    Z = Z.float()

    # Row L2 normalisation (Def. 4.1), each row first scaled by its own max
    # (exact) so the squared norm cannot underflow; only an exactly zero row
    # is dropped (tests/test_scale_invariance.py).
    _rmax = Z.abs().amax(dim=1, keepdim=True)
    Z = Z / _rmax.clamp_min(torch.finfo(Z.dtype).tiny)
    Z = torch.where(_rmax > 0, Z, torch.zeros_like(Z))
    row_norms = Z.norm(p=2, dim=1, keepdim=True)
    keep = row_norms.squeeze(-1) > 0
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


def _collect_per_sample_layer_tokens(cache, num_samples, use_cache,
                                     replace_final_norm=True, max_length=None):
    """``{layer_idx: List[Tensor(T_i, D)]}`` from the shared cache, or None.

    This task reads the cache twice: the hooked profile and the untouched
    post-norm headline (see the call site). A cache tokenised at another
    window than ``max_length`` is refused.
    """
    hs, _ = hidden_states_for_task(
        None, None, None, cache, num_samples=num_samples, layer_idx="all",
        use_cache=use_cache, per_sample=True, max_length=max_length,
        replace_final_norm=replace_final_norm, fallback="none")
    if not hs:
        return None
    return {li: chunks for li, chunks in hs.items() if chunks}


def _collect_per_sample_layer_tokens_fresh(model, tokenizer, samples, max_length):
    """Fallback path: run our own forward pass sample-by-sample and
    return ``{layer_idx: List[Tensor(T_i, D)]}``.

    Tensors stay on the model's device so the per-sentence SVD runs there.
    Returns the hooked profile (deepest entry = raw last block) and HF's
    untouched post-norm last entry, which is the headline.
    """
    device = next(model.parameters()).device
    per_layer: dict[int, list[torch.Tensor]] = {}
    post_norm_last: list[torch.Tensor] = []
    _sink = {}
    _hook = None
    try:
        from ..common import get_layers
        _blocks = get_layers(model)
        if _blocks is not None and len(_blocks) > 0:
            _hook = _blocks[-1].register_forward_hook(
                lambda mod, inp, outp: _sink.__setitem__(
                    "h", (outp[0] if isinstance(outp, tuple) else outp).detach()))
    except Exception as e:  # pragma: no cover - defensive
        logger.info(f"matrix_entropy: could not hook last block ({e})")
    # try/finally: the hook must not outlive an exception.
    try:
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_length,
                ).to(device)
                _sink.pop("h", None)
                out = model(**inputs, output_hidden_states=True)
                if not getattr(out, "hidden_states", None):
                    continue
                hs = list(out.hidden_states)
                post_norm_last.append(hs[-1][0].detach().float())
                _h_raw = _sink.get("h")
                if (_h_raw is not None and len(hs) >= 2
                        and _h_raw.shape == hs[-1].shape):
                    hs[-1] = _h_raw
                for li, h in enumerate(hs[1:]):  # skip embedding
                    per_layer.setdefault(li, []).append(
                        h[0].detach().float()
                    )
    finally:
        if _hook is not None:
            _hook.remove()
    return per_layer, post_norm_last


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
        # 512 = the shared cache's window (Wei et al. use 2048; study
        # documents are ~100 tokens, so the window does not bind).
        max_length = int(self.config.get("max_length", 512))

        # Two tensors: the per-layer PROFILE is hooked (every entry a block
        # output), while the HEADLINE is Wei et al.'s released recipe,
        # AutoModel(...)[0] = last_hidden_state = HF's post-norm last entry.
        per_layer_tokens = _collect_per_sample_layer_tokens(
            cache, num_samples, use_cache, replace_final_norm=True,
            max_length=max_length,
        )
        _postnorm_tokens = None
        _source = "shared_cache" if per_layer_tokens is not None else "self_collected"

        if per_layer_tokens is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)
            samples = list(dataset)[:num_samples]
            if not samples:
                return {"error": "Need at least 1 sample."}
            per_layer_tokens, _postnorm_tokens = (
                _collect_per_sample_layer_tokens_fresh(
                    model, tokenizer, samples, max_length))
        else:
            # Same cache, same documents, the untouched deepest entry.
            _pn = _collect_per_sample_layer_tokens(
                cache, num_samples, use_cache, replace_final_norm=False,
                max_length=max_length)
            if _pn:
                _postnorm_tokens = _pn[max(_pn.keys())]

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

        # Headline = the POST-NORM last-layer entropy (see collection).
        last_key = max(layer_entropies.keys(), key=lambda k: int(k.split("_")[1]))
        _hl, _hl_norm = [], []
        for Z in (_postnorm_tokens or []):
            res = _matrix_entropy(Z)
            if np.isfinite(res["entropy"]):
                _hl.append(res["entropy"])
            if np.isfinite(res["entropy_normalized"]):
                _hl_norm.append(res["entropy_normalized"])
        if _hl:
            headline = float(np.mean(_hl))
            headline_norm = float(np.mean(_hl_norm)) if _hl_norm else float("nan")
            _headline_is_postnorm = True
        else:
            # No post-norm tensor: fall back to the profile's deepest entry.
            headline = layer_entropies[last_key]
            headline_norm = layer_entropies_norm.get(last_key, float("nan"))
            _headline_is_postnorm = False

        return {
            # Headline = H / log d, Wei et al. Def. 4.3 (their tables);
            # provenance flags derived, never asserted.
            "_meta_headline_is_post_final_norm": bool(_headline_is_postnorm),
            "_meta_last_entry_is_block_output": bool(
                per_layer_tokens is not None and _postnorm_tokens is not None),
            "_meta_matrix_entropy_source": _source,
            "_meta_matrix_entropy_max_length": int(max_length),
            "matrix_entropy": headline_norm,
            # Raw H: unchanged by zero-padding / isometric embedding, while the
            # 1/log d of the headline is a width factor.
            "matrix_entropy_raw": headline,
            # Per-layer extension (same metric applied to every block).
            "layer_matrix_entropies": layer_entropies,
            "layer_matrix_entropies_normalized": layer_entropies_norm,
            "mean_matrix_entropy": (
                float(np.mean(finite_norm)) if finite_norm else float("nan")
            ),
            "mean_matrix_entropy_raw": float(np.mean(finite_raw)),
            "interpretation": (
                "Lower last-layer matrix entropy indicates a stronger "
                "information bottleneck (Wei et al., 2024)."
            ),
        }
