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
    * Points are texts: each layer of each text is its masked token mean
      (reference pool='avg'), from the shared cache when it can supply them,
      else from a pooled forward pass (``_pooled_layer_features``).
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
from ...cache import CACHE_TOKENIZER_MAX_LENGTH
from .._tokens import (hidden_states_for_task, massive_token_share,
                       position0_share, select_rows)
import math

logger = logging.getLogger("blme")

# Default number of mutual nearest neighbours. Reference default is 10
# (measure_alignment.py --topk). n_samples must exceed topk.
_DEFAULT_TOPK = 10


# Estimand: the released platonic-rep ``metrics.cknna`` (masks the raw kernels,
# centres inside HSIC), matched bit for bit in tests/tasks/parity/
# test_cknna_parity.py. Its mutual-kNN mask is not order-symmetric; evaluate()
# discloses the gap in _meta_cknna_order_asymmetry_max instead of symmetrising.


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
        # The k -> n limit, where CKNNA reduces to CKA (the paper's closed-form
        # anchor): biased keeps the diagonal so k = n selects every column;
        # unbiased masks it, so k = n - 1 is the full neighbourhood.
        topk = (n - 1) if unbiased else n

    device = X.device
    # Rescale by a power of two (exact) only OUTSIDE [tiny^(1/8), max^(1/8)]:
    # the data is squared three times downstream, and inside the band the
    # reference's absolute +1e-6 would make an unconditional rescale non-bit-exact.
    _fi2 = torch.finfo(X.dtype)
    _lo2 = float(_fi2.tiny) ** (1.0 / 8.0) * 4.0
    _hi2 = float(_fi2.max) ** (1.0 / 8.0) / 4.0

    from .utils import pow2_rescale as _p2

    def _pow2_rescale(T):
        m = float(T.abs().max()) if T.numel() else 0.0
        if not (m > 0) or not math.isfinite(m) or (_lo2 <= m <= _hi2):
            return T
        # `_p2` clamps the exponent, so m >= 2^127.5 cannot overflow the divisor.
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
    """Reference preprocessing (`metrics.remove_outliers`, exact=False): clamp
    to +/- the mean over rows of each row's ``q``-quantile of |x|.

    float32 is authoritative wherever it works (bit parity with the reference);
    double is used only when the float32 bound is already non-finite.
    """
    X = X.float()
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    bound = torch.quantile(X.abs(), q, dim=1).mean()
    if not torch.isfinite(bound):
        bound = torch.quantile(X.abs().double(), q, dim=1).mean()
        if not torch.isfinite(bound):
            # Genuinely unrepresentable in the working dtype: clamp in double.
            _b64 = torch.quantile(X.abs().double(), q, dim=1).mean()
            if torch.isfinite(_b64):
                return X.double().clamp(min=-_b64, max=_b64).to(X.dtype)
    if not torch.isfinite(bound):
        return X
    # A zero bound is applied, as the reference does (it zeroes the features).
    return X.clamp(min=-bound, max=bound)


def _remove_outliers_joint(F, layers, q: float = 0.95):
    """``metrics.remove_outliers(exact=False)`` on the stacked (N, L, D)
    features, as ``measure_alignment.py:103`` applies it: each text's
    ``q``-quantile of |x| over ALL its layers and dims, averaged over texts,
    gives ONE bound that clamps every layer.

    Quantiles are taken row by row (the same per-row values as the batched
    call, without materialising N x L x D), float32 first and double only if
    the float32 bound is non-finite, as ``_remove_outliers``. Returns
    ``({layer: clamped (N, D)}, bound)``.
    """
    n = int(F[layers[0]].shape[0])

    def _bound(dtype):
        qs = [torch.quantile(torch.cat([F[li][i].to(dtype) for li in layers]).abs(), q)
              for i in range(n)]
        return torch.stack(qs).mean()

    bound = _bound(torch.float32)
    if not torch.isfinite(bound):
        bound = _bound(torch.float64)
        if not torch.isfinite(bound):
            return {li: F[li].float() for li in layers}, float("nan")
    out = {li: F[li].float().clamp(min=-bound, max=bound) for li in layers}
    return out, float(bound)


def _l2_normalize(X: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation, matching the reference preprocessing
    (``measure_alignment.py:compute_score`` -> ``F.normalize(x, p=2, dim=-1)``)."""
    X = X.float()
    # Per row, rows outside the eighth-root band take a scale-exact unit path;
    # ordinary rows reproduce F.normalize bit for bit (test_cknna_parity.py).
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


_SHARE_DOCS = 200   # documents of the middle layer kept for the sink diagnostics


def _pooled_layer_features(model, tokenizer, dataset, cache, *, num_samples,
                           use_cache, pool, include_embedding=False,
                           min_cached_docs=None):
    """One vector per text and layer: the masked mean over its tokens.

    Mirrors platonic-rep ``extract_features.py:81-84`` (pool='avg'): every
    entry of the raw HF ``hidden_states`` tuple (post-norm last entry kept),
    averaged over the attention mask of each text. The shared cache supplies
    the token rows when it can; otherwise one forward pass per text pools on
    the fly, so no all-layer token cloud is held. Both routes average the same
    float rows in float64 and agree bit for bit.

    Returns ``(feats, feats_excl_pos0, mid_docs, info)``: ``{layer: (N, D)
    float64}`` under ``pool`` and under ``mean_excl_pos0``, the middle layer's
    token rows of the first ``_SHARE_DOCS`` texts, and the gate info.
    """
    if include_embedding and getattr(cache, "_embedding_states", None) is None:
        use_cache = False          # a cache without hs[0] cannot supply the layer set
    per_doc, info = hidden_states_for_task(
        model, tokenizer, dataset, cache, num_samples=num_samples,
        # post-final-norm=False, DELIBERATELY: the released reference stacks
        # the raw HF tuple, ln_f entry included (extract_features.py:82).
        replace_final_norm=False, layer_idx="all", use_cache=use_cache,
        per_sample=True, include_embedding=include_embedding,
        min_cached_docs=min_cached_docs, fallback="none")
    if per_doc:
        layers = sorted(per_doc)
        feats = {li: select_rows(per_doc[li], pool)[0] for li in layers}
        excl = {li: select_rows(per_doc[li], "mean_excl_pos0")[0] for li in layers}
        mid = list(per_doc[layers[len(layers) // 2]])[:_SHARE_DOCS]
        return feats, excl, mid, info

    if dataset is None:
        raise ValueError("geometry_cknna: no usable cache and no dataset")
    device = next(model.parameters()).device
    sums, sums_excl, mid_rows = {}, {}, []
    first = 0 if include_embedding else 1
    n_done = 0
    with torch.no_grad():
        for sample in dataset:
            if n_done >= num_samples:
                break
            text = sample["text"] if isinstance(sample, dict) else str(sample)
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=CACHE_TOKENIZER_MAX_LENGTH).to(device)
            if inputs["input_ids"].shape[1] < 1:
                continue
            out = model(**inputs, output_hidden_states=True)
            # post-final-norm=False, DELIBERATELY: the raw tuple, as the reference.
            hs = out.hidden_states[first:]
            for li, h in enumerate(hs):
                rows = h[0].detach().to("cpu")
                if not rows.is_floating_point():
                    rows = rows.float()
                docs = [rows]
                sums.setdefault(li, []).append(select_rows(docs, pool)[0])
                if rows.shape[0] > 1:
                    sums_excl.setdefault(li, []).append(
                        select_rows(docs, "mean_excl_pos0")[0])
                if li == len(hs) // 2 and n_done < _SHARE_DOCS:
                    mid_rows.append(rows.float())
            n_done += 1
    feats = {li: torch.cat(v, 0) for li, v in sums.items()}
    excl = {li: torch.cat(v, 0) for li, v in sums_excl.items()}
    info = dict(info, source="self_collected", fallback="pooled_forward")
    return feats, excl, mid_rows, info


@register_task("geometry_cknna")
class CKNNATask(DiagnosticTask):
    """
    Computes CKNNA (mutual k-NN conditional CKA; Huh et al., "The Platonic
    Representation Hypothesis", ICML 2024, arXiv:2405.07987) between all
    pairs of transformer layers and summarises the alignment profile.

    Points are TEXTS, not tokens: each layer of each text is its masked mean
    over all positions, position 0 included, as the reference's pool='avg'
    (the paper rejects last-token pooling, App. C.2). The within-model layer
    profile is a BLME use of a cross-model metric.

    Config:
        num_samples: number of texts (default 1024, the reference size).
        topk:        mutual k-NN neighbours (default 10); needs num_samples >= topk.
        unbiased:    unbiased HSIC + self-diagonal masking (default True).
        use_cache:   reuse the shared forward-pass cache (default True).
    """
    TOKEN_POLICY = "paper_protocol:Huh2024 per-text masked mean, position 0 included"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKNNA (mutual k-NN conditional CKA) Analysis...")

        # 1024 texts at k=10, the reference sample size (extract_features.py:178).
        num_samples = int(self.config.get("num_samples", 1024))
        topk = int(self.config.get("topk", _DEFAULT_TOPK))
        unbiased = bool(self.config.get("unbiased", True))
        use_cache = self.config.get("use_cache", True)

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        pool = "mean_excl_pos0" if self.TOKEN_POLICY == "exclude_pos0" else "mean"
        logger.info(f"  Pooling hidden states over {num_samples} texts...")
        # Every hidden_states entry, the embedding output included, as
        # extract_features.py:82 stacks them.
        feats, feats_excl, mid_docs, _info = _pooled_layer_features(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            use_cache=use_cache, pool=pool, include_embedding=True,
            # The cache serves at most cache_num_samples documents; one that
            # holds fewer than requested is not allowed to shrink N silently.
            min_cached_docs=num_samples)

        if not feats:
            return {"error": "No hidden states available for CKNNA"}
        if int(next(iter(feats.values())).shape[0]) < num_samples:
            logger.warning(f"  geometry_cknna: {next(iter(feats.values())).shape[0]} "
                           f"texts available, {num_samples} requested")

        layers = sorted(feats.keys())
        n_layers = len(layers)

        n_points = int(feats[layers[0]].shape[0])
        # n_points == topk is the k -> n limit `_cknna` handles; unbiased HSIC
        # divides by (m-2)(m-3), so it needs at least 4 points.
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

        def _prepare(F):
            # Reference order: one joint remove_outliers bound -> per-layer
            # row normalisation (measure_alignment.py:103, compute_score).
            ls = sorted(F)
            clamped, bound = _remove_outliers_joint(F, ls)
            return {li: _l2_normalize(clamped[li]) for li in ls}, bound

        logger.info("  Normalising activations...")
        normed, _clamp_bound = _prepare(feats)

        logger.info("  Computing CKNNA matrix...")
        cknna_matrix = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
        _order_gap = 0.0
        for i in tqdm(range(n_layers), desc="CKNNA rows"):
            X = normed[layers[i]]
            for j in range(i, n_layers):
                Y = normed[layers[j]]
                val = _cknna(X, Y, topk=topk, unbiased=unbiased)
                # Order gap measured on adjacent pairs; headline = reference.
                if j == i + 1:
                    _rev = _cknna(Y, X, topk=topk, unbiased=unbiased)
                    if np.isfinite(val) and np.isfinite(_rev):
                        _order_gap = max(_order_gap, abs(val - _rev))
                cknna_matrix[i, j] = val
                cknna_matrix[j, i] = val

        # Position-0-excluded pooling (_meta_), endpoint pair only.
        _excl_el = float("nan")
        _ends = (layers[0], layers[-1])
        if (set(feats_excl) == set(layers) and n_layers >= 2
                and all(feats_excl[li].shape[0] == n_points for li in layers)):
            _ne, _ = _prepare(feats_excl)
            _excl_el = _cknna(_ne[_ends[0]], _ne[_ends[1]], topk=topk, unbiased=unbiased)

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
            # nan* reducers and NaN fallbacks: 0.0 is a meaningful CKNNA.
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
            "_meta_n_layers": int(n),
            "_meta_includes_embedding_layer": True,
            "_meta_last_entry_is_block_output": bool(_info["last_is_block"]),
            "_meta_topk": int(topk),
            "_meta_n_points": int(n_points),
            "_meta_n_texts_requested": int(num_samples),
            "_meta_pooling": pool,
            "_meta_outlier_clamp": "joint_over_layers",
            "_meta_outlier_clamp_bound": float(_clamp_bound),
            "_meta_features_source": str(_info.get("source")),
            "_meta_cknna_order_asymmetry_max": float(_order_gap),
            "_meta_excl_pos0_early_late_cknna": float(_excl_el),
            # Sink diagnostics on the middle layer's token rows (first texts).
            "_meta_position0_share": (position0_share(mid_docs) if mid_docs
                                      else float("nan")),
            "_meta_massive_token_share": (massive_token_share(mid_docs) if mid_docs
                                          else float("nan")),
        }
