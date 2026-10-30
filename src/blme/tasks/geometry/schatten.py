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
   normalising the *raw* singular values). This matches the shared
   Roy-Vetterli effective rank convention in
   ``geometry.utils.effective_rank``.

Reports the per-layer profile and a "last-layer" headline for each metric, as
well as mean/std/slope/q25/q50/q75 summaries via the aggregator's layer
collapse. The last-layer state is the deepest BLOCK's raw output, recovered by
a forward hook (HF's final `hidden_states` entry is `ln_f(block L-1)`);
`_meta_last_entry_is_block_output` records whether that substitution actually
happened.

``schatten_1`` / ``schatten_4`` follow Yusupov et al. (Sec. 4.1, Eq. 5): the
Schatten norm of each text's RAW MLP-internal activation matrix, averaged over
all layers. The raw residual-stream norms are ``_meta_schatten_*``, and the
centred, row-L2-normalised family carries the ``row_normalized_`` prefix. Dividing by
d^(1/p) (``_meta_row_normalized_schatten_1/4``) manufactures a width
dependence: zero-padding 768 -> 3072 leaves the raw value unchanged and
divides the divided form by 4.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import strip_trailing_special
from .utils import safe_svdvals
from .utils import pow2_rescale as _pow2_rescale

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


def _rankme(S, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al. 2023, Eq. 2): ``exp(H(p))`` with

        p_i = σ_i / Σ σ_j + ε          (epsilon ADDED to the ratio)

    NOT ``σ_i / (Σ σ_j + ε)``. The distinction is not cosmetic: the ratio form
    is scale-invariant, the denominator form is not. On ``1e-13 * I_2`` the
    denominator form returned 1.513085749 against the correct 1.999999877
    (-24.35%), and on ``1e-20 * I_2`` it collapsed to 1.000000368 -- reporting
    a rank-one spectrum for a perfectly isotropic one (adversarial review
    round 2, 2026-09-04).

    The default epsilon is 1e-7, matching the RELEASED implementation. BLME
    shipped 1e-12, which is not the reference value and is not what the task
    metadata claimed either ("O(1e-5)"): the two differ by 1.52e-4 at T = 256
    (independent Opus review, 2026-09-04).

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
    p = arr / total + eps
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
    # Float32-FIRST, as the released MNN.py (bit-exact parity pinned by
    # tests/test_reference_parity_guard.py). Only outside a safe magnitude band
    # (where float32 would over/underflow, and the reference too) is the matrix
    # promoted, centred in double and power-of-two rescaled.
    _fi0 = torch.finfo(torch.float32)
    _D = max(1, X.shape[1])
    _gmax = X.abs().max()
    _lo0 = float(_fi0.tiny) ** 0.5 * 4.0
    _hi0 = (float(_fi0.max) / _D) ** 0.5 / 4.0
    if float(_gmax) > 0 and (float(_gmax) < _lo0 or float(_gmax) > _hi0):
        X = X.double()
        X = X - X.mean(dim=0, keepdim=True)
        _gmax = X.abs().max()
        X = _pow2_rescale(X, float(_gmax))
    else:
        # Inside the safe band: the ORIGINAL float32 centring, untouched, so
        # ordinary data stays bit-identical to the released reference.
        X = X - X.mean(dim=0, keepdim=True)
    # Row max-scaling before the norm only outside the safe band, so ordinary
    # data stays bit-identical to the reference.
    _rmax = X.abs().amax(dim=1, keepdim=True)
    _fi = torch.finfo(X.dtype)
    _lo = float(_fi.tiny) ** 0.5 * 4.0
    _hi = (float(_fi.max) / max(1, X.shape[1])) ** 0.5 / 4.0
    if bool(((_rmax > 0) & ((_rmax < _lo) | (_rmax > _hi))).any()):
        X = X / _rmax.clamp_min(_fi.tiny)
        X = torch.where(_rmax > 0, X, torch.zeros_like(X))
    row_norms = X.norm(p=2, dim=1, keepdim=True)
    keep = row_norms.squeeze(-1) > 0
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
        S = safe_svdvals(Z_norm).detach().cpu().numpy()
        # RAW spectrum (Yusupov Eq. 5 and RankMe are on the raw matrix), with
        # the same magnitude guard; Schatten norms are homogeneous of degree
        # 1, so a power-of-two factor multiplies straight back (in float64).
        _fi_r = torch.finfo(Z.dtype)
        _rmax_r = float(Z.abs().max())
        _lo_r = float(_fi_r.tiny) ** 0.5 * 4.0
        _hi_r = (float(_fi_r.max) / max(1, Z.shape[1])) ** 0.5 / 4.0
        _sc_r = 1.0
        _Zr = Z
        if _rmax_r > 0 and (_rmax_r < _lo_r or _rmax_r > _hi_r):
            import math as _m
            _e_r = int(max(-126, min(127, round(_m.log2(_rmax_r)))))
            _Zr = torch.ldexp(Z, torch.tensor(-_e_r, dtype=torch.int32))
            _sc_r = float(2.0) ** _e_r
        S_raw = (torch.linalg.svdvals(_Zr).detach().cpu().numpy()
                 .astype(np.float64) * float(_sc_r))
    except Exception:
        logger.warning("schatten: per-sentence measurement failed",
                       exc_info=True)
        return {}

    mnn = _matrix_nuclear_norm_fast(Z_norm)
    d = float(max(Z_norm.shape[1], 2))
    # MNN / T, the row count (MNN.py: mnn / len(token_ids[0])).
    T = float(max(Z_norm.shape[0], 1))
    # Schatten-2 is omitted: ||X||_F = sqrt(T) for T unit rows.
    return {
        # RAW matrix, no divisor.
        "schatten_1": _schatten_p_norm(S_raw, p=1),
        "schatten_4": _schatten_p_norm(S_raw, p=4),
        "rankme": _rankme(S_raw),
        # BLME's row-normalized variants, on the MNN-preprocessed matrix.
        "row_normalized_schatten_1": _schatten_p_norm(S, p=1) / d,
        "row_normalized_schatten_4": _schatten_p_norm(S, p=4) / (d ** 0.25),
        "row_normalized_schatten_inf": _schatten_p_norm(S, p=float("inf")),
        "row_normalized_rankme": _rankme(S),
        # MNN keeps the Li et al. preprocessing, which is its definition.
        "row_normalized_matrix_nuclear_norm": mnn / T,
    }


def _mlp_down_projections(model):
    """(block index, down-projection) for every block whose MLP is found.

    The down-projection's INPUT is the MLP-internal activation (after the
    activation function, before the projection back to the residual stream).
    """
    from ..common import get_layers
    from ..interpretability.sparsity import _find_down_proj
    out = []
    for i, layer in enumerate(get_layers(model) or []):
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        dp = _find_down_proj(mlp) if mlp is not None else None
        if dp is not None:
            out.append((i, dp))
    return out


def _collect_per_layer_fresh(model, tokenizer, samples, max_length, flag=None):
    """One forward per text, full text up to ``max_length`` (None: the model's
    context). Returns the residual profile (deepest entry the hooked raw
    block) and, in ``flag``: HF's post-norm last entry, the per-layer Schatten
    norms of the MLP-internal activations (reduced on the fly, so the (T,
    d_ff) matrices are never stored), and the provenance counters."""
    device = next(model.parameters()).device
    per_layer: Dict[int, List[torch.Tensor]] = {}
    post_norm_last: List[torch.Tensor] = []
    mlp: Dict[int, Dict[str, List[float]]] = {}
    from ..common import get_layers as _gl
    _blocks = _gl(model)
    _sink, _act = {}, {}
    _hooks = []
    _n_fwd = _n_sub = _n_trunc = _n_strip = 0
    if _blocks is not None and len(_blocks) > 0:
        def _cap_last(module, inputs, output):
            _sink["h"] = (output[0] if isinstance(output, tuple) else output).detach()
        _hooks.append(_blocks[-1].register_forward_hook(_cap_last))
    _down = _mlp_down_projections(model)
    for _i, _dp in _down:
        def _cap_mlp(module, args, _i=_i):
            if args and isinstance(args[0], torch.Tensor):
                _act[_i] = args[0].detach()
        _hooks.append(_dp.register_forward_pre_hook(_cap_mlp))
    _cfg = getattr(model, "config", None)
    _cap = max_length if max_length is not None else getattr(_cfg, "max_position_embeddings", None)
    try:
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                if _cap is not None:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=int(_cap))
                    _full = tokenizer(text, return_tensors="pt")["input_ids"].shape[-1]
                    _n_trunc += int(_full > int(_cap))
                else:
                    inputs = tokenizer(text, return_tensors="pt")
                # Drop a tokenizer-appended EOS/SEP, as the shared cache does.
                inputs, _k = strip_trailing_special(inputs, tokenizer)
                _n_strip += _k
                inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                # Clear both sinks before every forward, so a block that skips
                # one sample cannot leave a stale tensor behind.
                _sink.pop("h", None)
                _act.clear()
                out = model(**inputs, output_hidden_states=True)
                hs = getattr(out, "hidden_states", None)
                if not hs:
                    continue
                hs = list(hs)
                post_norm_last.append(hs[-1][0].detach().float())
                _n_fwd += 1
                _h = _sink.get("h")
                if _h is not None and tuple(_h.shape) == tuple(hs[-1].shape):
                    hs[-1] = _h
                    _n_sub += 1
                for li, h in enumerate(hs[1:]):  # skip embedding
                    per_layer.setdefault(li, []).append(h[0].detach().float())
                for li, A in _act.items():
                    A = A.reshape(-1, A.shape[-1]).float()
                    if A.shape[0] < 2:
                        continue
                    S = safe_svdvals(A).detach().cpu().numpy().astype(np.float64)
                    acc = mlp.setdefault(li, {"schatten_1": [], "schatten_4": []})
                    acc["schatten_1"].append(_schatten_p_norm(S, p=1))
                    acc["schatten_4"].append(_schatten_p_norm(S, p=4))
    finally:
        for _h in _hooks:
            _h.remove()
    if flag is not None:
        flag["last_is_block"] = bool(_n_fwd > 0 and _n_sub == _n_fwd)
        flag["post_norm_last"] = post_norm_last
        flag["mlp"] = mlp
        flag["n_mlp_layers_found"] = len(_down)
        flag["max_length"] = _cap
        flag["n_truncated"] = _n_trunc
        flag["n_stripped"] = _n_strip
    return per_layer


@register_task("geometry_schatten")
class SchattenNormTask(DiagnosticTask):
    """Schatten-p norms (Yusupov et al. 2025), Matrix Nuclear-Norm (Li et al.
    2024) and RankMe (Garrido et al. 2023) of per-text token matrices.

    Reports:
      - ``schatten_1``, ``schatten_4`` — Yusupov et al.'s quantity: the
        Schatten-p norm of each text's RAW MLP-internal activation matrix
        (the down-projection input), averaged over all L layers, then over
        texts. The residual-stream last-layer versions are
        ``_meta_schatten_{1,4}_last`` / ``_meta_schatten_{1,4}_per_layer``.
      - ``row_normalized_matrix_nuclear_norm_last`` — MNN / T on HF's
        post-norm last entry, as the released MNN.py.
      - ``rankme_last``, ``row_normalized_rankme_last``,
        ``row_normalized_schatten_inf_last`` — residual post-norm last layer.
      - Per-layer residual lists (``*_per_layer``); Schatten-2 is omitted
        (content-free under row-L2 normalisation).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Schatten-p norms + MNN + RankMe...")
        num_samples = int(self.config.get("num_samples", 10))
        _ml = self.config.get("max_length", 128)
        max_length = int(_ml) if _ml is not None else None

        # The MLP-internal activations are not in the shared cache, so this
        # task always runs its own forward (one per text) and reads the
        # residual states from the same forward.
        _prov = {"last_is_block": False}
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample"}
        per_layer = _collect_per_layer_fresh(model, tokenizer, samples, max_length,
                                             flag=_prov)
        if not per_layer:
            return {"error": "Could not collect hidden states"}

        per_layer_profiles: Dict[str, List[float]] = {
            k: [] for k in ("schatten_1", "schatten_4", "row_normalized_schatten_1",
                            "row_normalized_schatten_4", "row_normalized_schatten_inf",
                            "row_normalized_matrix_nuclear_norm", "rankme",
                            "row_normalized_rankme")}
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
                    float(np.mean(per_sentence[k])) if per_sentence[k] else float("nan"))

        headline_idx = max(per_layer.keys())
        # `*_last` follows the released MNN.py (MNN.py:184, last_hidden_state =
        # HF's post-norm last entry); the per-layer profile stays hooked.
        _pn_tokens = _prov.get("post_norm_last") or []
        _pn_profile = {}
        if _pn_tokens:
            _acc = {k: [] for k in per_layer_profiles}
            for Z in _pn_tokens:
                res = _per_sentence_measurements(Z)
                for k in _acc:
                    v = res.get(k)
                    if v is not None and np.isfinite(v):
                        _acc[k].append(v)
            _pn_profile = {k: (float(np.mean(v)) if v else float("nan"))
                           for k, v in _acc.items()}

        result = {}
        # _meta_: the d^(1/p)-divided keys (the divisor manufactures a width
        # dependence) and the residual-stream Schatten norms (Yusupov's
        # quantity is on the MLP activations, below).
        _META = {"row_normalized_schatten_1", "row_normalized_schatten_4",
                 "schatten_1", "schatten_4"}

        for k, profile in per_layer_profiles.items():
            _k = f"{'_meta_' if k in _META else ''}{k}"
            finite = [v for v in profile if np.isfinite(v)]
            if not finite:
                result[f"{_k}_last"] = float("nan")
                result[f"{_k}_per_layer"] = profile
                continue
            if k in _pn_profile and np.isfinite(_pn_profile[k]):
                last_val = _pn_profile[k]
            else:
                last_val = profile[sorted(per_layer.keys()).index(headline_idx)]
            result[f"{_k}_last"] = float(last_val) if np.isfinite(last_val) else float("nan")
            result[f"{_k}_per_layer"] = profile

        # Yusupov et al. 2025 Sec. 4.1: per text, the mean over all L layers
        # of the Schatten-p norm of the MLP-internal activations; then the
        # mean over texts.
        mlp = _prov.get("mlp") or {}
        for key in ("schatten_1", "schatten_4"):
            layers_ok = sorted(li for li in mlp if mlp[li][key])
            n_docs = min((len(mlp[li][key]) for li in layers_ok), default=0)
            per_text = [float(np.mean([mlp[li][key][j] for li in layers_ok]))
                        for j in range(n_docs)]
            result[key] = float(np.mean(per_text)) if per_text else float("nan")
            result[f"_meta_{key}_mlp_per_layer"] = [
                float(np.mean(mlp[li][key])) if li in mlp and mlp[li][key] else float("nan")
                for li in sorted(per_layer.keys())]

        result["_meta_last_entry_is_block_output"] = bool(_prov.get("last_is_block"))
        result["_meta_schatten_headline_is_post_final_norm"] = bool(_pn_profile)
        result["_meta_schatten_max_length"] = (int(_prov["max_length"])
                                               if _prov.get("max_length") is not None
                                               else -1)
        result["_meta_schatten_n_docs_truncated"] = int(_prov.get("n_truncated", 0))
        result["_meta_schatten_mlp_layers_found"] = int(_prov.get("n_mlp_layers_found", 0))
        result["_meta_n_trailing_special_stripped"] = int(_prov.get("n_stripped", 0))
        result["_meta_schatten_source"] = "self_collected"
        result["preprocessing"] = "center_columns_then_row_l2_normalize"
        result["schatten_2_excluded_reason"] = (
            "content_free_under_row_l2_normalization"
        )
        return result
