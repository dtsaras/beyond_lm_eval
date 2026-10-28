from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .._tokens import hidden_states_for_task
from .utils import effective_rank as _effective_rank, safe_svdvals
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")

class _BF16_EPS_PROXY:
    """numpy has no bfloat16, and bf16's eps (2^-8 = 0.0078125) is 64x
    float32's. Passing this sentinel as ``source_dtype`` selects it."""
    eps = 0.0078125


def _svd_metrics_for_layer(X, source_dtype=None):
    """Compute SVD-based isotropy metrics for a single (N, D) matrix.

    Every reduction runs in float64; the rank tolerance uses the eps of the
    dtype the data ARRIVED in (captured before promotion), and ``source_dtype``
    (the model's precision) only feeds the disclosed ``*_source_precision``
    variant (tests/tasks/test_isotropy_stored_dtype.py).
    """
    if isinstance(X, torch.Tensor):
        _stored_eps = (float(torch.finfo(X.dtype).eps)
                       if X.dtype.is_floating_point
                       else float(np.finfo(np.float32).eps))
        _stored_dtype = X.dtype
        X = X.detach().cpu().double().numpy()
    else:
        _arr_dt = np.asarray(X).dtype
        _stored_dtype = (_arr_dt if np.issubdtype(_arr_dt, np.floating)
                         else np.dtype(np.float32))
        _stored_eps = float(np.finfo(_stored_dtype).eps)
    X = np.asarray(X, dtype=np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    if len(X) < 10:
        return None
    # Keep the RAW (uncentered) states for the Ethayarajh 2019 anisotropy
    # cosine below; SVD-based isotropy metrics use the centered version.
    X_raw = X
    X = X - np.mean(X, axis=0)

    try:
        # Max-factored SVD (exact near the top of the float32 range).
        S = safe_svdvals(X)
    # safe_svdvals raises torch's LinAlgError, which is not numpy's.
    except (np.linalg.LinAlgError, torch._C._LinAlgError):
        try:
            from scipy.linalg import svd as scipy_svd
            _, S, _ = scipy_svd(X, full_matrices=False)
        except Exception:
            return None

    explained_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    # np.trapezoid is NumPy >= 2.0; look np.trapz up lazily.
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = _trapz(explained_variance) / max(1, len(explained_variance))

    # Roy-Vetterli effective rank on raw singular values (shared helper).
    effective_rank = float(_effective_rank(S))

    # (sum L)^2 / sum(L^2) is scale-invariant: divide by S[0] before squaring
    # so S**4 cannot underflow (tests/test_scale_invariance.py).
    _s0 = float(S[0]) if S.size else 0.0
    _Sn = (S / _s0) if _s0 > 0 else S
    eigenvalues = _Sn ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    participation_ratio = (float((sum_eig ** 2) / sum_eig_sq)
                           if sum_eig_sq > 0 else float("nan"))

    # Ethayarajh 2019 Eq. 4 anisotropy E[cos(f(x), f(y))] over distinct pairs
    # of RAW states, computed exactly: with u_i = x_i/||x_i|| and m = mean(u),
    #     E_{i != j}[u_i . u_j] = (N * ||m||^2 - 1) / (N - 1).
    # Rows are max-scaled before norming (exact at any scale); only exactly
    # zero rows are dropped (they would break the unit-diagonal premise).
    _rmax = np.max(np.abs(X_raw), axis=1)
    _safe = _rmax > 0
    _Xs = np.zeros_like(X_raw)
    _Xs[_safe] = X_raw[_safe] / _rmax[_safe, None]
    norms = np.linalg.norm(_Xs, axis=1)
    keep = norms > 0
    unit = _Xs[keep] / norms[keep, None]
    n_vec = len(unit)
    if n_vec > 1:
        mean_unit = unit.mean(axis=0)
        avg_cos_sim = float(
            (n_vec * float(mean_unit @ mean_unit) - 1.0) / (n_vec - 1)
        )
    else:
        avg_cos_sim = float("nan")

    # cond_number on the numerical-rank subspace, with numpy's tolerance
    # S[0] * max(N, D) * eps(stored dtype); where that factor reaches 1 (bf16
    # eps at width >= 128) the sqrt(max(N, D)) factor is used instead, which
    # the disclosed source-precision variant always uses.
    _fac = max(X.shape) * _stored_eps
    if _fac >= 1.0:
        _fac = float(np.sqrt(max(X.shape))) * _stored_eps
    rel_tol = float(S[0]) * _fac
    if source_dtype is _BF16_EPS_PROXY:
        _src_eps = _BF16_EPS_PROXY.eps
    elif source_dtype is not None:
        _src_eps = float(np.finfo(np.dtype(source_dtype)).eps)
    else:
        _src_eps = _stored_eps
    _rel_tol_src = float(S[0]) * np.sqrt(max(X.shape)) * _src_eps
    _rank_src = int(np.sum(S > _rel_tol_src))
    _cond_src = (float(S[0] / S[_rank_src - 1]) if _rank_src >= 1
                 and S[_rank_src - 1] > 0 else float("inf"))
    effective_rank_num = int(np.sum(S > rel_tol))
    if effective_rank_num >= 1 and S[effective_rank_num - 1] > 0:
        cond_number = float(S[0] / S[effective_rank_num - 1])
    else:
        cond_number = float("inf")

    return {
        # _meta_: the explained-variance AUC is integrated over its own
        # unpadded length, so zero-padding moves it; it has no defining source.
        "_meta_svd_auc": float(auc),
        "cond_number": cond_number,
        "_meta_numerical_rank": effective_rank_num,
        # At the model's precision (sqrt(max(N, D)) factor).
        "_meta_numerical_rank_source_precision": _rank_src,
        "_meta_cond_number_source_precision": _cond_src,
        "_meta_numerical_rank_tol": float(rel_tol),
        "avg_cosine_similarity": avg_cos_sim,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
    }


def _isoscore_null_floor(n: int, d: int, seed: int = 0) -> float:
    """IsoScore of an isotropic Gaussian at the same (N, d) -- the finite-N
    floor. A measured value at or below this is not distinguishable from
    structureless data of the same shape."""
    if n < 2 or d < 2:
        return float("nan")
    # The finite-N bias depends on N/d: subsample at the same ratio when the
    # draw would not fit in memory.
    n_eff, d_eff = int(n), int(d)
    cap = 40_000_000
    if n_eff * d_eff > cap:
        n_eff = max(d_eff + 1, cap // max(d_eff, 1))
    rng = np.random.default_rng(seed)
    return _isoscore(rng.standard_normal((n_eff, d_eff)))


def _isoscore(X: np.ndarray) -> float:
    """IsoScore (Rudman et al., Findings of ACL 2022, arXiv:2108.07344).

    Measures how close the empirical covariance of a point cloud is to a
    scaled identity — a more discriminative notion of isotropy than the
    Ethayarajh anisotropy baseline (which only looks at average cosine
    similarity of random pairs).

    Algorithm (Rudman et al. Eq. 4.1):
      1. Center X and compute the eigenvalue spectrum lambda_i of cov(X)
         via SVD on centered data, padding zeros to the ambient dimension.
      2. L2-normalise the spectrum to length sqrt(d); an isotropic spectrum
         becomes the all-ones vector.
      3. Compute the isotropy defect
         delta = ||lambda_hat - 1|| / sqrt(2 * (d - sqrt(d))).
      4. IsoScore = ((d - delta^2 * (d - sqrt(d)))^2 - d) / (d * (d - 1)).

    Returns a scalar in [0, 1] where 1 = perfectly isotropic and 0 =
    effectively one-dimensional ambient-space use.
    """
    n, d = X.shape
    if n < 2 or d < 2:
        return float("nan")

    # Promote BEFORE the mean (translation invariance, as IsoScore 2.0.1),
    # and divide S by S[0] before squaring (scale invariance in arithmetic).
    Xd = np.asarray(X, dtype=np.float64)
    Xc = Xd - Xd.mean(axis=0, keepdims=True)
    try:
        S = safe_svdvals(Xc)
    except (np.linalg.LinAlgError, torch._C._LinAlgError):
        return float("nan")
    _s0 = float(S[0]) if S.size else 0.0
    if _s0 > 0 and np.isfinite(_s0):
        S = S / _s0
    eigvals = (S ** 2) / max(1, n - 1)

    full = np.zeros(d, dtype=np.float64)
    full[: len(eigvals)] = eigvals
    spectrum_norm = float(np.linalg.norm(full))
    if spectrum_norm <= 0 or not np.isfinite(spectrum_norm):
        return float("nan")

    sigma_hat = np.sqrt(d) * full / spectrum_norm
    denom = float(np.sqrt(2.0 * (d - np.sqrt(d))))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")

    defect = float(np.linalg.norm(sigma_hat - 1.0) / denom)
    score = ((d - defect ** 2 * (d - np.sqrt(d))) ** 2 - d) / (d * (d - 1))
    return float(max(0.0, min(1.0, score)))


@register_task("geometry_isoscore")
class IsoScoreTask(DiagnosticTask):
    """
    IsoScore (Rudman et al., Findings of ACL 2022, arXiv:2108.07344).

    A scalar in [0, 1] measuring how close the empirical covariance of the
    final-layer hidden states is to a scaled identity matrix. Higher = more
    isotropic. This is a strictly more discriminative isotropy measure than
    Ethayarajh's average-cosine-similarity baseline (which only looks at
    pairwise directions, not the magnitude spectrum).

    Returns:
        isoscore: float in [0, 1]
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running IsoScore Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # The shared cache only if it holds the configured population (the
        # N/d regime must not depend on which tasks co-ran); HF's post-norm
        # last entry, every token.
        X, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            layer_idx=-1, use_cache=use_cache, replace_final_norm=False,
            min_cached_docs=int(num_samples), max_tokens_per_doc=None)
        _source = _info["source"]
        _cached_docs = _info["cached_docs"]
        X = X.float().numpy()
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            logger.info(f"  Filtered {(~finite_mask).sum()} non-finite rows out of {len(X)}")
            X = X[finite_mask]
        if len(X) < 10:
            return {"error": "Too few finite hidden states for IsoScore"}

        iso = _isoscore(X)
        d = int(X.shape[1])
        n_pts = int(X.shape[0])

        # The realised N/d, so the paper's N >> d condition is auditable.
        min_ratio = float(self.config.get("min_points_per_dim", 10))
        ratio = (n_pts / d) if d else float("nan")
        regime_ok = bool(np.isfinite(ratio) and ratio >= min_ratio)

        # IsoScore = (PR - 1) / (d - 1) for the participation ratio PR of the
        # covariance spectrum, so iso * (d - 1) + 1 = PR, the effective
        # dimension count: zero-padding moves the raw score by (d-1)/(kd-1)
        # and leaves this one (tests/test_invariance_harness.py).
        iso_dims = (float(iso) * (d - 1) + 1.0) if d > 1 else float("nan")

        # Not NaN-gated on N/d (the finite-N bias is sub-percent on real
        # states); N <= d is in Algorithm 1's domain and is flagged.
        _rank_deficient = bool(d > 0 and n_pts <= d)
        if _rank_deficient:
            logger.info(f"  IsoScore: N={n_pts} <= d={d}; the covariance is "
                        f"rank-deficient by construction. Reported with "
                        f"_meta_isoscore_rank_deficient=True.")

        return {
            "isoscore": iso,
            "isoscore_effective_dims": iso_dims,
            # /d re-imports the width factor (Pearson 0.99997 with isoscore).
            "_meta_isoscore_effective_dims_frac": (iso_dims / d) if d > 0 else float("nan"),
            "_meta_isoscore_rank_deficient": _rank_deficient,
            "_meta_isoscore_source": _source,
            "_meta_isoscore_cached_docs": int(_cached_docs),
            "_meta_d_model": d,
            "_meta_n_points": n_pts,
            # IsoScore of an isotropic Gaussian at this (N, d): the finite-N
            # floor, reported rather than gated on.
            "_meta_isoscore_null_floor": _isoscore_null_floor(n_pts, d),
            "_meta_points_per_dim": float(ratio),
            "_meta_sampling_regime_ok": regime_ok,
        }


@register_task("geometry_svd")
class SVDIsotropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SVD Analysis...")
        # The model's precision, for the disclosed source-precision rank
        # variant (the cache stores float32 whatever the model ran in).
        _src_dt = np.float32
        try:
            _md = next(model.parameters()).dtype
            _src_dt = {torch.bfloat16: np.dtype("float32"),
                       torch.float16: np.float16,
                       torch.float32: np.float32,
                       torch.float64: np.float64}.get(_md, np.float32)
            if _md is torch.bfloat16:
                # numpy has no bfloat16; its eps is 2^-8 = 0.0078125.
                _src_dt = _BF16_EPS_PROXY
        except Exception:
            pass
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode (all layers) ---
        # HF's post-norm last entry is kept (replace_final_norm=False): it is
        # what Ethayarajh's extractor scores as the last layer
        # (contextual/pytorch_pretrained/modeling_gpt2.py:562-565).
        if layerwise:
            all_layers, _ = hidden_states_for_task(
                model, tokenizer, dataset, cache, num_samples=num_samples,
                layer_idx="all", use_cache=use_cache, replace_final_norm=False,
                fallback="none")
            if all_layers:
                per_layer = {}
                erank_per_layer = []
                pr_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _svd_metrics_for_layer(all_layers[li],
                                               source_dtype=_src_dt)
                    if m is not None:
                        per_layer[f"layer_{li}"] = m
                        erank_per_layer.append(m["effective_rank"])
                        pr_per_layer.append(m["participation_ratio"])
                    else:
                        erank_per_layer.append(float("nan"))
                        pr_per_layer.append(float("nan"))
                # Last-layer metrics as top-level for backward compat
                last_key = f"layer_{max(all_layers.keys())}"
                result = dict(per_layer.get(last_key, {}))
                result["layer_effective_rank"] = erank_per_layer
                result["layer_participation_ratio"] = pr_per_layer
                result["layer_metrics"] = per_layer
                return result

        # --- Single-layer mode (last layer, original behaviour) ---
        # Same gate as geometry_isoscore: the cache only if it holds N docs.
        X, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            layer_idx=-1, use_cache=use_cache, replace_final_norm=False,
            min_cached_docs=int(num_samples), max_tokens_per_doc=None)

        m = _svd_metrics_for_layer(X, source_dtype=_src_dt)
        if m is None:
            return {"error": "Too few finite hidden states for SVD"}
        return m
