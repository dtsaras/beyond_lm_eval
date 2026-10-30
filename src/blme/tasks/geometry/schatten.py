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

NOT a reproduction of Yusupov et al. Eq. 5, which is the Schatten norm of the
RAW MLP-block activation matrix averaged over layers; ``schatten_*`` here are
raw (uncentred, un-normalised) residual-stream norms, and the centred,
row-L2-normalised family carries the ``row_normalized_`` prefix. Dividing by
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


def _collect_per_layer(cache, num_samples, use_cache, flag=None):
    """Match matrix_entropy's cache path, returning per-layer per-
    sentence hidden states."""
    if cache is None or not cache.is_populated or not use_cache:
        return None
    # BOTH, because this task needs both. The per-layer PROFILE must be
    # homogeneous, so it is hooked; the `*_last` HEADLINE is what the released
    # MNN reference measures, and that reference reads the POST-NORM state --
    # see the note at `headline_idx`.
    hs = cache.get_hidden_states(
        layer_idx="all", num_samples=num_samples, per_sample=True,
        replace_final_norm=True,
    )
    if not hs:
        return None
    if flag is not None:
        # READ THE SUBSTITUTION FLAG BEFORE the post-norm call, which resets it.
        # `get_hidden_states(replace_final_norm=False)` clears
        # `_last_block_substituted`, so querying it afterwards made
        # `_meta_last_entry_is_block_output` structurally always False on this
        # route -- the split fix defeated its own provenance (independent Opus
        # review, 2026-09-06).
        flag["last_is_block"] = bool(
            getattr(cache, "_last_block_substituted", False))
        _pn = cache.get_hidden_states(
            layer_idx="all", num_samples=num_samples, per_sample=True,
            replace_final_norm=False)
        if _pn:
            flag["post_norm_last"] = _pn[max(_pn.keys())]
        # (The stale re-read that used to sit here is gone: it ran AFTER the
        # post-norm call and therefore always saw False.)
        # Set ONLY on the path that actually returns cached states, so the
        # caller's `_meta_schatten_source` cannot claim a cache it did not use.
        flag["from_cache"] = True
    return {li: chunks for li, chunks in hs.items() if chunks}


def _collect_per_layer_fresh(model, tokenizer, samples, max_length, flag=None):
    """Fallback path (no cache) — run a forward per sample, asking HF
    for ``hidden_states``. Mirrors matrix_entropy's fresh-collection
    helper so the two share semantics."""
    device = next(model.parameters()).device
    per_layer: Dict[int, List[torch.Tensor]] = {}
    # post-final-norm: BOTH. The per-layer PROFILE is hooked (homogeneous);
    # `post_norm_last` keeps the untouched `ln_f` entry, which is what the
    # released MNN reference measures -- see the note at `headline_idx`.
    post_norm_last: List[torch.Tensor] = []
    from ..common import get_layers as _gl
    _blocks = _gl(model)
    _sink = {}
    _hook = None
    _n_fwd = 0
    _n_sub = 0
    if _blocks is not None and len(_blocks) > 0:
        def _cap_last(module, inputs, output):
            _sink["h"] = (output[0] if isinstance(output, tuple)
                          else output).detach()
        _hook = _blocks[-1].register_forward_hook(_cap_last)
    try:
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
                # CLEAR the sink before every forward, and require the swap on
                # EVERY forward. A conditional final block that skips one sample
                # left a stale, same-shaped tensor behind, which was then
                # substituted into the NEXT sample while `substituted` stayed true
                # (adversarial review round 2, 2026-09-04).
                _sink.pop("h", None)
                out = model(**inputs, output_hidden_states=True)
                # post-final-norm: SUBSTITUTED -- this route feeds a per-layer
                # profile, and the cache route above already opts in, so both must.
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
    finally:
        if _hook is not None:
            _hook.remove()
    if flag is not None:
        flag["last_is_block"] = bool(_n_fwd > 0 and _n_sub == _n_fwd)
        flag["post_norm_last"] = post_norm_last
    return per_layer


@register_task("geometry_schatten")
class SchattenNormTask(DiagnosticTask):
    """Per-layer Schatten-p norms + Matrix Nuclear-Norm + RankMe.

    Reports:
      - ``row_normalized_schatten_1_last``, ``row_normalized_schatten_4_last``,
        ``row_normalized_schatten_inf_last`` — last-layer values of each norm,
        normalised by ``d^{1/p}`` so they're comparable across
        d_model (BLME width-comparability convention — Yusupov et al. use the PLAIN norm on MLP-internal activations with no dimension normalization; attribution fixed, audit 2026-07-16). Schatten-2 is omitted
        because it is content-free under row-L2 preprocessing
        (``‖X‖_F = sqrt(T)`` for ``T`` unit-norm rows).
      - ``row_normalized_matrix_nuclear_norm_last`` — fast L1,2 approximation,
        also normalised by ``d``.
      - ``rankme_last`` — raw-singular-value effective rank (Garrido
        2023), matching our Roy-Vetterli ``effective_rank`` convention
        in ``geometry_svd``.
      - Per-layer lists (``*_per_layer``) so the aggregator can emit
        mean/std/slope/q25/q50/q75 cross-model summaries.
      - ``preprocessing`` metadata describing centre + row-L2 normalisation.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Schatten-p norms + MNN + RankMe...")
        num_samples = int(self.config.get("num_samples", 10))
        use_cache = self.config.get("use_cache", True)
        max_length = int(self.config.get("max_length", 128))

        _prov = {"last_is_block": False}
        # THE SHARED CACHE IS REFUSED UNLESS ITS WINDOW MATCHES OURS.
        #
        # `defaults.yaml` sets `max_length: 128` for this task, but the shared
        # cache tokenises at a HARD-CODED 512 (`cache.py:422`) and `core.py`
        # hands the same cache to every task. `use_cache` is absent from this
        # task's config, so the code default True fired and the declared 128
        # was DEAD whenever any other task populated a hidden cache -- which is
        # every run in the archive. The effect is not marginal:
        #
        #   gpt2      rankme_last                      54.762 -> 141.895  (+159.1%)
        #   pythia-70m rankme_last                     30.394 ->  83.781  (+175.7%)
        #   gpt2      schatten_1_last                20119.27 -> 46961.73 (+133.4%)
        #   gpt2      row_normalized_matrix_nuclear...  0.578345 -> 0.712491 (+23.2%)
        #
        # and `results/blme_verified_a/*.json` shows
        # `cache_num_samples_resolved = 200` with seven hidden-cached tasks, so
        # EVERY ARCHIVED SCHATTEN NUMBER WAS COMPUTED AT 512 TOKENS, NOT 128 --
        # with nothing in the output to say so (red-team review, 2026-09-05).
        #
        # Same discipline as `geometry_isoscore`: a task whose value depends on
        # its window must not silently accept someone else's. The window
        # actually used is emitted below.
        # Read from the cache module rather than duplicated here: two
        # hard-coded 512s coupled by nothing but a comment is how the original
        # mismatch survived (independent Opus review, 2026-09-05).
        from ...cache import CACHE_TOKENIZER_MAX_LENGTH as _CACHE_WINDOW
        _cache_window_ok = (max_length == _CACHE_WINDOW)
        if use_cache and cache is not None and getattr(cache, "is_populated", False) \
                and not _cache_window_ok:
            logger.info(
                f"  schatten: shared cache tokenises at {_CACHE_WINDOW} but this "
                f"task is configured for max_length={max_length}; self-collecting "
                f"so the window is the one declared.")
        per_layer = _collect_per_layer(
            cache, num_samples, use_cache and _cache_window_ok, flag=_prov)
        if per_layer is None:
            if dataset is None:
                from ...cache import load_default_corpus
                dataset = load_default_corpus(num_samples)
            samples = list(dataset)[:num_samples]
            if not samples:
                return {"error": "Need at least 1 sample"}
            per_layer = _collect_per_layer_fresh(
                model, tokenizer, samples, max_length, flag=_prov,
            )

        if not per_layer:
            return {"error": "Could not collect hidden states"}

        per_layer_profiles: Dict[str, List[float]] = {
            # The undivided PAPER quantities. They were computed in
            # `_per_sentence_measurements` but not listed here, so the loop
            # discarded them and the fix never reached the output at all
            # (adversarial review, 2026-09-03).
            "schatten_1": [],
            "schatten_4": [],
            "row_normalized_schatten_1": [],
            "row_normalized_schatten_4": [],
            "row_normalized_schatten_inf": [],
            "row_normalized_matrix_nuclear_norm": [],
            "rankme": [],
            # THE SAME OMISSION, TWICE. This dict is the ONLY thing the
            # emission loop iterates, so a key computed in
            # `_per_sentence_measurements` and missing here is discarded
            # silently. That is exactly the note above -- and
            # `row_normalized_rankme` was then added to the measurement
            # function on 2026-09-05 without being added here, so it never
            # reached the output and the ledger recorded it as fixed when it
            # was not (independent Opus review, 2026-09-05).
            "row_normalized_rankme": [],
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

        # THE HEADLINE FOLLOWS THE RELEASED REFERENCE, WHICH READS POST-NORM.
        #
        # `MLGroupJLU/MatrixNuclearNorm`'s `code/MNN.py:184` is
        #     R = model(token_ids)[0][0, :, :]
        # with `model = AutoModel.from_pretrained(...)` (lines 116/118), i.e.
        # `last_hidden_state` -- which is `hidden_states[-1] = ln_f(b_{L-1})`.
        # That is the IDENTICAL recipe to Wei et al.'s Diff-eRank, and the same
        # third clause applies: a per-layer profile must be homogeneous, a
        # single-layer estimand may want post-norm, and THE RELEASED REFERENCE
        # WINS. This is the fourth task to need it, after `matrix_entropy`,
        # `contextualization` and `coe`.
        #
        # Flagged by an independent review on 2026-09-05 and confirmed here by
        # cloning the repository and reading the line, rather than taken on
        # report -- the previous three cost a RED and two YELLOWs because a
        # blanket rule was applied without checking what each reference does.
        #
        # So the profile stays hooked and `*_last` is recomputed on the
        # untouched post-norm entry. `_meta_schatten_headline_is_post_final_norm`
        # records whether that actually happened.
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
        # The d-DIVIDED keys go to _meta_: the divisor INTRODUCES the width
        # dependence it was meant to remove.
        #
        # A Schatten norm is isometry-invariant, so embedding the same cloud
        # into higher d must not change it -- and the raw value is
        # bit-identical (120.403159) across d = 16..4096, while
        # `raw / d` falls by exactly x256 over the same range. On the 61-model
        # archive that shows up as row_normalized_schatten_1_last vs d_model
        # Spearman **-0.950** (p = 1.6e-31), against +0.289 for the undivided
        # schatten_1 and +0.072 for schatten_4. The undivided keys are both the
        # paper's quantity and the width-neutral one (adversarial review,
        # 2026-09-04).
        _D_DIVIDED = {"row_normalized_schatten_1", "row_normalized_schatten_4"}

        def _pfx(name):
            return "_meta_" if name in _D_DIVIDED else ""

        for k, profile in per_layer_profiles.items():
            _k = f"{_pfx(k)}{k}"
            finite = [v for v in profile if np.isfinite(v)]
            if not finite:
                result[f"{_k}_last"] = float("nan")
                result[f"{_k}_per_layer"] = profile
                continue
            # Headline = last layer, on the POST-NORM entry (see above). The
            # per-layer profile beside it is the hooked, homogeneous one; the
            # two are different tensors on purpose.
            if k in _pn_profile and np.isfinite(_pn_profile[k]):
                last_val = _pn_profile[k]
            else:
                last_val = profile[sorted(per_layer.keys()).index(headline_idx)]
            result[f"{_k}_last"] = float(last_val) if np.isfinite(last_val) else float("nan")
            result[f"{_k}_per_layer"] = profile

        # DERIVED, not asserted. Hard-coding True made this flag lie on any
        # architecture where get_layers() returns None: the hook never
        # attaches, `*_last` silently reverts to the post-ln_f value (gpt2
        # 35.19 vs 62.32, -43.5%; pythia-70m +18.7%, sign-flipped by family)
        # while the flag still claimed a block output.
        result["_meta_last_entry_is_block_output"] = bool(_prov.get("last_is_block"))
        # THE WINDOW ACTUALLY USED, and where the states came from. Neither
        # was emitted, which is why an entire archived sweep ran at 512 tokens
        # under a config that says 128 with nothing to reveal it.
        result["_meta_schatten_headline_is_post_final_norm"] = bool(_pn_profile)
        result["_meta_schatten_max_length"] = int(max_length)
        # DERIVED from what `_collect_per_layer` actually returned, not from
        # the request. With a LOGITS-ONLY cache (`need_hidden=False`) the task
        # self-collects but the request-shaped test still said "shared_cache"
        # (independent Opus review, 2026-09-05). Same discipline as
        # `_meta_last_entry_is_block_output`.
        result["_meta_schatten_source"] = (
            "shared_cache" if _prov.get("from_cache") else "self_collected")
        result["preprocessing"] = "center_columns_then_row_l2_normalize"
        result["schatten_2_excluded_reason"] = (
            "content_free_under_row_l2_normalization"
        )
        return result
