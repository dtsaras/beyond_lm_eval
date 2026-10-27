"""Linear CKA between every pair of layers (Kornblith et al. 2019).

The biased estimator is Kornblith's Eq. 3-4; the debiased one is the
feature-space U-statistic of the official Demo.ipynb (``debiased=True``,
Song et al. 2007). Both are scale-free ratios, so every guard below only
shifts exponents or promotes precision where float32 has already failed.
"""
import logging

import numpy as np
import torch
from tqdm import tqdm

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import (hidden_states_for_task, massive_token_share,
                       position0_share, select_rows)

logger = logging.getLogger("blme")


def _linear_cka(X: torch.Tensor, Y: torch.Tensor,
                norm_x: float = None, norm_y: float = None,
                centered: bool = False) -> float:
    """Linear CKA (Kornblith et al. 2019, Eq. 4) on column-centred features.

    ``norm_x`` / ``norm_y`` are ||X^T X||_F and ||Y^T Y||_F in units of each
    matrix's largest entry (as ``evaluate`` caches them per layer), so a pair
    costs one GEMM. ``centered=True`` skips re-centring.
    """
    if not centered:
        # Factor out the max before the column mean: the sum overflows at
        # N * max even when max is finite (tests/test_scale_invariance.py).
        def _center(M):
            M = M.float()
            g = M.abs().amax()
            if float(g) > 0:
                M = M / g
            return (M - M.mean(dim=0, keepdim=True)) * g
        X = _center(X)
        Y = _center(Y)
    # Numerator and denominator are fourth powers of the data scale; dividing
    # each matrix by its own max keeps them inside float32 (CKA is invariant to
    # a positive rescale of either view; tests/test_scale_invariance.py).
    _gx = X.abs().amax()
    _gy = Y.abs().amax()
    _tiny = torch.finfo(X.dtype).tiny
    Xs = X / _gx.clamp_min(_tiny)
    Ys = Y / _gy.clamp_min(_tiny)
    numerator = torch.norm(Ys.t() @ Xs, p="fro").item() ** 2
    if norm_x is None:
        norm_x = torch.norm(Xs.t() @ Xs, p="fro").item()
    if norm_y is None:
        norm_y = torch.norm(Ys.t() @ Ys, p="fro").item()
    denom = norm_x * norm_y
    # Constant features make CKA 0/0: NaN, never 0.0 (which would claim
    # maximal dissimilarity). The test is relative, so it is scale-free.
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    if not np.isfinite(numerator):
        return float("nan")
    return float(numerator / denom)


def _debiased_hsic_features(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Unbiased HSIC (Song et al. 2007) in feature space, O(N D^2).

    Identical to the Gram-space U-statistic (rel 1e-11) without forming two
    (N, N) matrices; kernel-space form kept in ``_debiased_hsic`` for parity.
    """
    X = X.double()
    Y = Y.double()
    n = X.shape[0]
    if n < 4:
        return float("nan")
    xty = Y.t() @ X
    xy_sq = float(xty.pow(2).sum())
    sx = X.sum(dim=0)
    sy = Y.sum(dim=0)
    kx = X @ sx                       # row sums of K = X X^T
    ly = Y @ sy
    kdiag = X.pow(2).sum(dim=1)       # diagonal of K
    ldiag = Y.pow(2).sum(dim=1)
    kt = float(kx.sum() - kdiag.sum())
    lt = float(ly.sum() - ldiag.sum())
    # (K L) trace with zero diagonals.
    term = float((kx * ly).sum()
                 - (kdiag * ly).sum() - (kx * ldiag).sum()
                 + (kdiag * ldiag).sum())
    kl = float(xy_sq
               - (kdiag * ldiag).sum())
    return float(
        kl + kt * lt / ((n - 1) * (n - 2)) - 2.0 * term / (n - 2)
    ) / (n * (n - 3))


def _debiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """Unbiased HSIC (Song et al. 2012) from explicit Gram matrices.

    Retained for the parity tests, which check small fixtures. Production uses
    the feature-space form above.
    """
    K = K.double().clone()
    L = L.double().clone()
    n = K.shape[0]
    if n < 4:
        return float("nan")
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)
    kt = K.sum()
    lt = L.sum()
    term = (K @ L).sum()
    return float(
        (K * L).sum()
        + kt * lt / ((n - 1) * (n - 2))
        - 2.0 * term / (n - 2)
    ) / (n * (n - 3))


def _linear_cka_debiased(X: torch.Tensor, Y: torch.Tensor,
                         hxx: float = None, hyy: float = None) -> float:
    """Debiased linear CKA: HSIC_u(X,Y) / sqrt(HSIC_u(X,X) HSIC_u(Y,Y)).

    ``hxx``/``hyy`` are the per-layer self-HSICs, which a layer-pair loop
    hoists; omitting them keeps the standalone form the parity tests use.
    """
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
    hxy = _debiased_hsic_features(X, Y)
    if hxx is None:
        hxx = _debiased_hsic_features(X, X)
    if hyy is None:
        hyy = _debiased_hsic_features(Y, Y)
    den = float(np.sqrt(max(hxx, 0.0) * max(hyy, 0.0)))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(hxy / den)


def _center_layer(X: torch.Tensor):
    """Column-centre one layer's (N, D) cloud; float64 only when float32 fails.

    Returns ``(centred_float32, used_float64)``. The float64 branch fires when
    |mean|/sd exceeds 1e4 or the float32 std underflowed (the column collapsed
    onto one float); tests/test_scale_invariance.py pins both branches.
    """
    Xf = X.float()
    _mu = Xf.mean(dim=0, keepdim=True)
    _sd = Xf.std(dim=0, keepdim=True)
    _ratio = float((_mu.abs() / _sd.clamp_min(torch.finfo(Xf.dtype).tiny)).max())
    if (not np.isfinite(_ratio)) or _ratio > 1e4:
        Xd = X.double()
        return (Xd - Xd.mean(dim=0, keepdim=True)).float(), True
    Xc = Xf - _mu
    return Xc - Xc.mean(dim=0, keepdim=True), False


def _max_relative_gram_norm(X: torch.Tensor) -> float:
    """||X^T X||_F in units of X's largest entry (what ``_linear_cka`` expects)."""
    _Xn = X / X.abs().amax().clamp_min(torch.finfo(X.dtype).tiny)
    return torch.norm(_Xn.t() @ _Xn, p="fro").item()


@register_task("geometry_cka")
class CKATask(DiagnosticTask):
    """Linear CKA (Kornblith et al. 2019, arXiv:1905.00414) between all layers.

    Every block output is compared with every other on the same token rows
    (one forward pass); the deepest entry is the raw block output, not
    ``ln_f(b_{L-1})``. Biased CKA is the paper's Eq. 3-4; the debiased form
    (official Demo.ipynb) removes the d_eff/N null floor.

    Rows follow ``TOKEN_POLICY``: position 0 is dropped, as Kornblith's own
    Transformer experiment drops special tokens (App. F.1) -- the sink rows
    otherwise carry most of the centred SS and pin mid-layer CKA near 1
    (tests/tasks/invariance/test_cka.py).
    """
    TOKEN_POLICY = "exclude_pos0"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKA Layer Similarity Analysis...")

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        logger.info(f"  Collecting hidden states for {num_samples} samples...")
        # Block outputs only: a post-norm deepest entry would contaminate a
        # whole row and column of the cross-layer matrix.
        per_doc, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            per_sample=True)

        layers = sorted(per_doc.keys())
        n_layers = len(layers)

        logger.info("  Centering activations...")
        centered_acts = {}
        norms = {}  # ||X^T X||_F, max-relative
        _n_f64_centered = 0
        _rows_meta = None
        for idx in tqdm(layers, desc="Centering"):
            X, _rows_meta = select_rows(per_doc[idx], self.TOKEN_POLICY)
            X, _f64 = _center_layer(X)
            _n_f64_centered += int(_f64)
            centered_acts[idx] = X
            norms[idx] = _max_relative_gram_norm(X)

        logger.info("  Computing CKA Matrix...")
        cka_matrix = np.zeros((n_layers, n_layers))
        dbg_matrix = np.zeros((n_layers, n_layers))

        # One self-HSIC per layer, hoisted out of the pair loop.
        _self_hsic = {}
        for _idx in layers:
            _self_hsic[_idx] = _debiased_hsic_features(
                centered_acts[_idx].float(), centered_acts[_idx].float())

        for i in tqdm(range(n_layers), desc="CKA Rows"):
            idx_i = layers[i]
            X = centered_acts[idx_i]
            norm_x = norms[idx_i]
            for j in range(i, n_layers):
                idx_j = layers[j]
                Y = centered_acts[idx_j]
                norm_y = norms[idx_j]
                cka = _linear_cka(X, Y, norm_x=norm_x, norm_y=norm_y,
                                  centered=True)
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka
                dcka = _linear_cka_debiased(
                    X, Y, hxx=_self_hsic[idx_i], hyy=_self_hsic[idx_j])
                dbg_matrix[i, j] = dcka
                dbg_matrix[j, i] = dcka

        # Position-0-inclusive companions (_meta_): adjacent pairs and the
        # endpoints only, one layer at a time, to bound the extra cost.
        def _incl(idx):
            Xi, _ = _center_layer(select_rows(per_doc[idx], "all")[0])
            return Xi, _max_relative_gram_norm(Xi)

        _incl_adj, _incl_el = [], float("nan")
        if n_layers >= 2:
            _first = _incl(layers[0])
            _prev = _first
            for k in range(1, n_layers):
                _cur = _incl(layers[k])
                _incl_adj.append(_linear_cka(_prev[0], _cur[0], norm_x=_prev[1],
                                             norm_y=_cur[1], centered=True))
                if k == n_layers - 1:
                    _incl_el = _linear_cka(_first[0], _cur[0], norm_x=_first[1],
                                           norm_y=_cur[1], centered=True)
                _prev = _cur
            del _first, _prev, _cur
        _incl_adj = np.asarray(_incl_adj, dtype=float)
        _mid = per_doc[layers[n_layers // 2]] if n_layers else []

        # Scalar summaries only; the matrix and layer indices are not emitted
        # (a flattened index vector would be a pure layer-count proxy).
        n = n_layers
        iu = np.triu_indices(n, k=1)
        off_diag = cka_matrix[iu] if iu[0].size else np.array([], dtype=float)

        diagonal_off1 = np.array(
            [cka_matrix[i, i + 1] for i in range(n - 1)],
            dtype=float,
        ) if n > 1 else np.array([], dtype=float)

        def _stat(arr, fn, fallback=float("nan")):
            return float(fn(arr)) if arr.size else fallback

        # NaN policy: a degenerate pair is dropped by every summary (nan*
        # reducers), and "no pair to measure" is NaN, never 0.0.
        # Headline keys are the ones an identity block cannot move (P4): the
        # endpoint pair and the minima. Uniform means, spreads, the maximum
        # (an identity pair reads 1.0) and the index-based first-middle pair
        # are _meta_ profile summaries (test_cka.py identity-insertion test).
        results = {
            "min_adjacent_cka": _stat(diagonal_off1, np.nanmin),
            "min_offdiag_cka": _stat(off_diag, np.nanmin),
            # max_offdiag_cka is not emitted: the most similar pair is an
            # adjacent one, so it duplicated max_adjacent_cka (65/66 models).
            "_meta_avg_adjacent_cka": _stat(diagonal_off1, np.nanmean),
            "_meta_max_adjacent_cka": _stat(diagonal_off1, np.nanmax),
            "_meta_std_adjacent_cka": _stat(diagonal_off1, np.nanstd),
            "_meta_mean_offdiag_cka": _stat(off_diag, np.nanmean),
            "_meta_std_offdiag_cka": _stat(off_diag, np.nanstd),
            "early_late_cka": (
                float(cka_matrix[0, -1]) if n >= 2 else float("nan")
            ),
            "_meta_first_middle_cka": (
                float(cka_matrix[0, n // 2]) if n >= 3 else float("nan")
            ),
            "_meta_debiased_avg_adjacent_cka": (
                float(np.nanmean([dbg_matrix[i, i + 1] for i in range(n - 1)]))
                if n > 1 else float("nan")),
            "_meta_debiased_mean_offdiag_cka": (
                float(np.nanmean(dbg_matrix[iu])) if iu[0].size else float("nan")),
            # Debiased twins of the headline keys (Demo.ipynb debiased=True):
            # the U-statistic has no d_eff/N floor, so they do not drift with N
            # (test_cka.py permutation-null N test).
            "early_late_cka_debiased": (
                float(dbg_matrix[0, -1]) if n >= 2 else float("nan")),
            "min_offdiag_cka_debiased": (
                float(np.nanmin(dbg_matrix[iu])) if iu[0].size else float("nan")),
            "min_adjacent_cka_debiased": (
                float(np.nanmin([dbg_matrix[i, i + 1] for i in range(n - 1)]))
                if n > 1 else float("nan")),
            "_meta_n_layers": int(n),
            "_meta_n_layers_centered_in_float64": int(_n_f64_centered),
            "_meta_last_entry_is_block_output": bool(_info["last_is_block"]),
            # Sink diagnostics on the full (inclusive) middle-layer cloud.
            "_meta_position0_share": position0_share(_mid) if n_layers else float("nan"),
            "_meta_massive_token_share": (
                massive_token_share(_mid) if n_layers else float("nan")),
            "_meta_incl_pos0_avg_adjacent_cka": _stat(_incl_adj, np.nanmean),
            "_meta_incl_pos0_min_adjacent_cka": _stat(_incl_adj, np.nanmin),
            "_meta_incl_pos0_early_late_cka": float(_incl_el),
        }
        if _rows_meta is not None:
            results.update(_rows_meta.as_meta())
        return results
