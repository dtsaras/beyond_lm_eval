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
    * Tier-2 task — reuses the same per-layer synchronised token cloud as
      ``geometry_cka`` via ``collect_hidden_states(layer_idx="all")`` (same
      forward pass, same tokens across layers). Without a populated cache it
      runs its own collection pass.
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
from .._tokens import hidden_states_for_task
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


@register_task("geometry_cknna")
class CKNNATask(DiagnosticTask):
    """
    Computes CKNNA (mutual k-NN conditional CKA; Huh et al., "The Platonic
    Representation Hypothesis", ICML 2024, arXiv:2405.07987) between all
    pairs of transformer layers and summarises the alignment profile.

    CKNNA is the local-neighbourhood analog of ``geometry_cka``: instead of
    the global CKA it restricts the kernel alignment to the mutual k-nearest
    neighbour graph, so it measures whether *local* representational geometry
    (not just global structure) is preserved across layers.

    Config:
        num_samples: number of documents to collect (default 100).
        topk:        mutual k-NN neighbours (default 10). num_samples' worth
                     of tokens must exceed topk.
        unbiased:    unbiased HSIC + self-diagonal masking (default True).
        use_cache:   reuse the shared forward-pass cache (default True).

    Outputs (flat floats): adjacent-layer and global off-diagonal
    mean/min/max/std of CKNNA, early-vs-late and first-vs-middle values.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKNNA (mutual k-NN conditional CKA) Analysis...")

        num_samples = self.config.get("num_samples", 100)
        topk = int(self.config.get("topk", _DEFAULT_TOPK))
        unbiased = bool(self.config.get("unbiased", True))
        use_cache = self.config.get("use_cache", True)

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        logger.info(f"  Collecting hidden states for {num_samples} samples...")
        # post-final-norm=False, DELIBERATELY: the released reference stacks
        # the raw HF tuple, ln_f entry included (extract_features.py:82).
        layer_activations, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=False, layer_idx="all", use_cache=use_cache)

        if not layer_activations:
            return {"error": "No hidden states available for CKNNA"}

        layers = sorted(layer_activations.keys())
        n_layers = len(layers)

        n_points = int(layer_activations[layers[0]].shape[0])
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

        # Seeded token cap shared by every layer (unbiased HSIC is O(N^3)).
        max_tokens = int(self.config.get("max_cknna_tokens", 2000))
        shared_perm = None
        if n_points > max_tokens:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(self.config.get("seed", 42)))
            shared_perm = torch.randperm(n_points, generator=rng)[:max_tokens]
            n_points = max_tokens

        logger.info("  Normalising activations...")
        normed = {}
        for idx in tqdm(layers, desc="Normalising"):
            acts = layer_activations[idx]
            if shared_perm is not None:
                acts = acts[shared_perm]
            # Reference order: remove_outliers -> normalize.
            normed[idx] = _l2_normalize(_remove_outliers(acts.float()))

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
            "_meta_last_entry_is_block_output": bool(_info["last_is_block"]),
            "_meta_topk": int(topk),
            "_meta_n_points": int(n_points),
            "_meta_cknna_order_asymmetry_max": float(_order_gap),
        }
