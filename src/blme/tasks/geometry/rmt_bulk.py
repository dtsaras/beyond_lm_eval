"""
Marchenko-Pastur bulk deviation — random-matrix-theory spectral diagnostics.

References:
    * Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of
      eigenvalues for some sets of random matrices." Mat. Sb. (N.S.)
      72(114):4, 507-536; English transl. Math. USSR-Sbornik 1(4),
      457-483. — the MP law for the bulk spectrum of sample
      covariance/correlation matrices.
    * Baik, J., Ben Arous, G. & Péché, S. (2005). "Phase transition of
      the largest eigenvalue for nonnull complex sample covariance
      matrices." Annals of Probability 33(5), 1643-1697,
      arXiv:math/0403022. — the BBP transition: sufficiently strong
      population spikes detach from the MP bulk and become observable
      outlier eigenvalues.

Method:
    Take the flattened token cloud X (N tokens x D dims) at a layer
    (``per_sample=False`` is deliberate — RMT statements are about an
    unordered population of rows, not trajectories). Z-score each
    dimension across tokens (zero-variance dimensions are dropped and
    counted), form the sample correlation matrix C = Z^T Z / N and its
    eigenvalues via ``np.linalg.eigvalsh`` in float64. Under the iid
    null the empirical spectral distribution converges to the MP law
    with ratio gamma = D/N and bulk support
    [(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2]
    (plus a point mass 1 - 1/gamma at 0 when gamma > 1).

Metrics per selected layer (first/mid/last by default):
    * ``mp_outlier_frac``  — fraction of eigenvalues above the upper
      bulk edge inflated by ``edge_tol`` (default 0.05). The buffer
      absorbs finite-size Tracy-Widom fluctuations of the largest bulk
      eigenvalue, which are O(N^{-2/3}) around the edge.
    * ``mp_spike_energy``  — sum of outlier eigenvalues / trace(C):
      the fraction of total variance carried by structured (BBP
      supercritical) spikes.
    * ``mp_ks_distance``   — Kolmogorov-Smirnov distance between the
      empirical eigenvalue CDF and the MP CDF at the SAME gamma
      (numerically integrated MP density on a fine grid).

Cross-model comparability:
    gamma = D/N varies across models, but the MP reference is
    gamma-matched per layer so the gamma-matched comparison is fair by
    construction. We ALSO emit fixed-gamma variants (``*_g25``)
    computed on a seeded (``np.random.default_rng(0)``) token
    subsample of size N' = ceil(D / 0.25), so every model is compared
    at the same aspect ratio gamma = 0.25; they are NaN when too few
    tokens exist. ``mp_gamma_*`` and ``mp_n_tokens_*`` record the
    actual ratio and token count used.
"""

import logging
import math
from typing import Optional

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._tokens import hidden_states_for_task, select_rows

logger = logging.getLogger("blme")

# Exactly zero variance only: an absolute cutoff breaks the correlation
# matrix's invariance to a global rescale (tests/test_scale_invariance.py).
_ZERO_VAR_EPS = 0.0


def _mp_bulk_edge(gamma: float) -> float:
    """Upper edge (1 + sqrt(gamma))^2 of the Marchenko-Pastur bulk."""
    return (1.0 + math.sqrt(gamma)) ** 2


def _mp_lower_edge(gamma: float) -> float:
    """Lower edge (1 - sqrt(gamma))^2 of the Marchenko-Pastur bulk."""
    return (1.0 - math.sqrt(gamma)) ** 2


def _mp_cdf(x, gamma: float, n_grid: int = 4096) -> np.ndarray:
    """Marchenko-Pastur CDF (sigma^2 = 1) evaluated at points ``x``.

    Computed by numerically integrating the MP density

        f(t) = sqrt((t_+ - t)(t - t_-)) / (2 * pi * gamma * t)

    on a fine grid over the bulk support [t_-, t_+]. For gamma > 1 the
    point mass 1 - 1/gamma at 0 is included (the correlation matrix is
    rank-deficient when D > N).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lo, hi = _mp_lower_edge(gamma), _mp_bulk_edge(gamma)

    # Continuous part carries mass 1 (gamma <= 1) or 1/gamma (gamma > 1).
    cont_mass = 1.0 if gamma <= 1.0 else 1.0 / gamma
    point_mass = 0.0 if gamma <= 1.0 else 1.0 - 1.0 / gamma

    # Trigonometric substitution t = (lo+hi)/2 - (hi-lo)/2 cos(theta),
    # theta in [0, pi]. Then sqrt((hi-t)(t-lo)) = (hi-lo)/2 |sin theta| and
    # dt = (hi-lo)/2 sin(theta) dtheta, so the integrand
    #   g(theta) = f(t) dt/dtheta = (hi-lo)^2/4 * sin^2(theta) / (2 pi gamma t)
    # is BOUNDED across the whole interval, including t -> 0 at gamma == 1
    # (where the old uniform-t grid produced a huge spurious spike). The
    # limit as theta -> 0 with lo == 0 is (hi-lo)/(2 pi gamma).
    theta = np.linspace(0.0, np.pi, n_grid)
    t = 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = ((hi - lo) ** 2 / 4.0) * np.sin(theta) ** 2 / (2.0 * np.pi * gamma * t)
    g = np.where(np.isfinite(g), g, (hi - lo) / (2.0 * np.pi * gamma))
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(theta))]
    )
    # Normalize the numeric integral to the exact continuous mass.
    if cum[-1] > 0:
        cum *= cont_mass / cum[-1]

    cos_arg = np.clip((lo + hi - 2.0 * x) / (hi - lo), -1.0, 1.0)
    theta_x = np.arccos(cos_arg)
    cdf = np.interp(theta_x, theta, cum)
    cdf = np.where(x <= lo, 0.0, cdf)
    cdf = np.where(x >= hi, cont_mass, cdf)
    cdf = cdf + point_mass * (x >= 0.0)
    return np.clip(cdf, 0.0, 1.0)


def _ks_distance(eigs: np.ndarray, gamma: float) -> float:
    """Kolmogorov-Smirnov distance between the empirical eigenvalue CDF
    and the MP CDF at the same gamma."""
    eigs = np.sort(np.asarray(eigs, dtype=np.float64))
    n = eigs.size
    if n == 0:
        return float("nan")
    lo = _mp_lower_edge(gamma)
    point_mass = 0.0 if gamma <= 1.0 else 1.0 - 1.0 / gamma
    # For gamma > 1 the spectrum has ~(D-N) rank-deficiency eigenvalues at ~0.
    # Snap those (below the lower bulk edge) onto the atom location 0 so they
    # group as ties; otherwise each is compared against F(0) = point mass and
    # the KS statistic collapses to ~1-1/gamma regardless of fit.
    if gamma > 1.0:
        eigs = np.where(eigs < lo, 0.0, eigs)
    uniq, counts = np.unique(eigs, return_counts=True)
    csum = np.cumsum(counts)
    after = csum / n                     # empirical CDF at / after each value
    before = (csum - counts) / n         # empirical CDF just before each value
    F = _mp_cdf(uniq, gamma)             # theoretical CDF (includes atom at 0)
    # Theoretical LEFT limit: the atom at 0 hasn't been crossed yet.
    F_left = F.copy()
    F_left[uniq <= 0.0] = F[uniq <= 0.0] - point_mass
    d = np.maximum(np.abs(after - F), np.abs(before - F_left))
    return float(np.max(d))



def _estimate_sigma2(eigs: np.ndarray, gamma: float, edge_tol: float,
                     n_iter: int = 12) -> float:
    """Estimate the MP scale sigma^2 by QUANTILE MATCHING.

        sigma^2_hat = median(lambda) / median(MP_gamma)

    A median is a 50%-breakdown statistic, so a minority of spiked directions
    does not move it. It recovers the bulk's MEDIAN scale (the anchor for the
    MP edge), not the mean population variance; on isotropic bulks the two
    coincide (within 2.4% over the gamma range). ``n_iter`` and ``edge_tol``
    are unused and kept for signature stability.
    """
    eigs = np.asarray(eigs, dtype=np.float64)
    if eigs.size == 0:
        return float("nan")
    lo, hi = _mp_lower_edge(gamma), _mp_bulk_edge(gamma)
    grid = np.linspace(lo, hi, 4096)
    cdf = _mp_cdf(grid, gamma)
    # Median of the standard MP law (the atom at 0 for gamma > 1 is already in
    # the CDF, so this is the median of the FULL spectrum either way).
    j = int(np.searchsorted(cdf, 0.5))
    if j >= grid.size:
        mp_median = float(grid[-1])
    elif j <= 0:
        mp_median = float(grid[0]) if gamma <= 1.0 else 0.0
    else:
        mp_median = float(grid[j])
    emp_median = float(np.median(eigs))
    if not np.isfinite(mp_median) or mp_median <= 1e-12:
        # gamma > 1: the zero atom holds more than half the eigenvalues, so
        # quantile-match the NONZERO part (relative zero test, scale-free).
        top = float(np.max(eigs)) if eigs.size else 0.0
        nz = eigs[eigs > top * 1e-12] if top > 0 else eigs[:0]
        if nz.size == 0:
            return float("nan")
        grid_nz = np.linspace(lo, hi, 4096)
        cdf_nz = _mp_cdf(grid_nz, gamma)
        atom = max(0.0, 1.0 - 1.0 / gamma)
        # Median OF THE CONTINUOUS PART = the point where the full CDF reaches
        # atom + (1 - atom)/2.
        target = atom + 0.5 * (1.0 - atom)
        j = int(np.searchsorted(cdf_nz, target))
        mp_med_nz = float(grid_nz[min(j, grid_nz.size - 1)])
        if mp_med_nz <= 1e-12:
            return float(np.median(nz) / max(1e-12, hi * 0.5))
        return float(np.median(nz) / mp_med_nz)
    if not np.isfinite(emp_median) or emp_median <= 0:
        return float("nan")
    return float(emp_median / mp_median)


def _permutation_spike_metrics(X: np.ndarray, n_perm: int = 12,
                              seed: int = 0, q: float = 99.0) -> dict:
    """Spike content against a COLUMN-PERMUTATION null.

    Shuffling each column independently keeps every marginal and destroys all
    cross-dimension structure, so unlike MP it does not assume an isotropic
    population. Returns the fraction of eigenvalues above the null's upper
    envelope and the share of total variance they carry.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 8 or d < 2:
        return {"perm_outlier_frac": float("nan"),
                "perm_spike_energy": float("nan"),
                "perm_top_ratio": float("nan")}
    Xc = X - X.mean(axis=0)
    eigs = np.clip(np.linalg.eigvalsh((Xc.T @ Xc) / n), 0.0, None)

    rng = np.random.default_rng(seed)
    null_top = []
    for _ in range(n_perm):
        P = np.empty_like(Xc)
        for j in range(d):
            P[:, j] = Xc[rng.permutation(n), j]
        Pc = P - P.mean(axis=0)
        null_top.append(np.clip(np.linalg.eigvalsh((Pc.T @ Pc) / n), 0.0, None))
    null_top = np.asarray(null_top)
    # Upper envelope of the null spectrum, per rank order.
    envelope = np.percentile(null_top, q, axis=0)
    thresh = float(envelope.max())

    total = float(eigs.sum())
    out = eigs[eigs > thresh]
    return {
        "perm_outlier_frac": float(out.size / eigs.size),
        "perm_spike_energy": float(out.sum() / total) if total > 0 else float("nan"),
        "perm_top_ratio": float(eigs.max() / thresh) if thresh > 0 else float("nan"),
    }


def _mp_layer_metrics(
    X,
    edge_tol: float = 0.05,
    max_tokens: Optional[int] = None,
    seed: int = 0,
    fixed_gamma: Optional[float] = 0.25,
) -> dict:
    """Pure compute core: MP bulk-deviation metrics for one (N, D)
    activation matrix. Designed for direct testing on synthetic data.

    Returns a flat dict with keys ``outlier_frac``, ``spike_energy``,
    ``ks_distance``, ``gamma``, ``n_tokens``, ``n_dims_dropped``, plus
    fixed-gamma variants (``outlier_frac_g25`` etc.) that are NaN when
    fewer than ceil(D / fixed_gamma) tokens are available.
    """
    if isinstance(X, torch.Tensor):
        # Straight to float64 (narrowing first could only lose data).
        X = X.detach().double().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4 or X.shape[1] < 2:
        return {"error": f"Activation matrix too small for MP analysis: {X.shape}"}

    rng = np.random.default_rng(seed)

    # Global rescale before any second moment (exact; every statistic here is
    # invariant to it, the arithmetic is not).
    _gmax = float(np.max(np.abs(X))) if X.size else 0.0
    if _gmax > 0 and np.isfinite(_gmax):
        X = X / _gmax

    # Drop non-finite rows, then cap tokens with a seeded subsample.
    X = X[np.all(np.isfinite(X), axis=1)]
    if max_tokens is not None and X.shape[0] > max_tokens:
        idx = rng.choice(X.shape[0], size=max_tokens, replace=False)
        X = X[idx]

    core = _mp_metrics_core(X, edge_tol)
    if "error" in core:
        return core

    out = {
        "outlier_frac": core["outlier_frac"],
        "spike_energy": core["spike_energy"],
        "lower_outlier_frac": core["lower_outlier_frac"],
        "lower_energy": core["lower_energy"],
        "ks_distance": core["ks_distance"],
        "gamma": core["gamma"],
        "n_tokens": core["n_tokens"],
        "n_dims_dropped": core["n_dims_dropped"],
    }

    # Covariance and permutation-null variants (see _key_prefix for why they
    # are _meta_).
    out.update(_permutation_spike_metrics(X, seed=seed))

    cov = _mp_metrics_core(X, edge_tol, standardize=False)
    for key in ("outlier_frac", "spike_energy", "lower_outlier_frac",
                "lower_energy", "ks_distance"):
        out[f"{key}_cov"] = cov.get(key, float("nan"))
    out["sigma2_cov"] = cov.get("sigma2", float("nan"))

    # Fixed-gamma variant: same metrics at a common aspect ratio so the
    # MP reference is identical across models.
    if fixed_gamma:
        suffix = f"g{int(round(fixed_gamma * 100))}"
        # gamma from the dimensions the core KEEPS (constant columns are
        # dropped), re-sized against the subsample's own constants.
        _n_const = int(np.count_nonzero(np.ptp(X, axis=0) == 0))
        _d_eff = X.shape[1] - _n_const
        n_fixed = int(math.ceil(_d_eff / fixed_gamma)) + 1
        if X.shape[0] >= n_fixed:
            idx = rng.choice(X.shape[0], size=n_fixed, replace=False)
            for _ in range(4):
                _d_sub = int(np.count_nonzero(np.ptp(X[idx], axis=0) > 0))
                if _d_sub == _d_eff:
                    break
                _d_eff = _d_sub
                n_fixed = int(math.ceil(_d_eff / fixed_gamma)) + 1
                if X.shape[0] < n_fixed:
                    break
                idx = rng.choice(X.shape[0], size=n_fixed, replace=False)
            fixed = _mp_metrics_core(X[idx], edge_tol)
            fixed_cov = _mp_metrics_core(X[idx], edge_tol, standardize=False)
            out[f"outlier_frac_{suffix}"] = fixed.get("outlier_frac", float("nan"))
            out[f"spike_energy_{suffix}"] = fixed.get("spike_energy", float("nan"))
            out[f"ks_distance_{suffix}"] = fixed.get("ks_distance", float("nan"))
            out[f"outlier_frac_cov_{suffix}"] = fixed_cov.get("outlier_frac", float("nan"))
            out[f"spike_energy_cov_{suffix}"] = fixed_cov.get("spike_energy", float("nan"))
            out[f"ks_distance_cov_{suffix}"] = fixed_cov.get("ks_distance", float("nan"))
        else:
            for _k in ("outlier_frac", "spike_energy", "ks_distance"):
                out[f"{_k}_{suffix}"] = float("nan")
                out[f"{_k}_cov_{suffix}"] = float("nan")
    return out


def _mp_metrics_core(X: np.ndarray, edge_tol: float,
                     standardize: bool = True) -> dict:
    """Eigendecompose the activation second-moment matrix and compare it
    against the gamma-matched Marchenko-Pastur law.

    ``standardize=True`` z-scores each dimension, giving the CORRELATION
    matrix, for which sigma^2 = 1 exactly.

    ``standardize=False`` only centers, giving the COVARIANCE matrix, and
    estimates sigma^2 from the bulk.

    Both are reported: the correlation matrix is exactly invariant to
    per-dimension rescaling and therefore blind to rogue dimensions, which the
    covariance variant sees.
    """
    n_tokens = X.shape[0]
    # Global rescale before any second moment (exact).
    _gmax = float(np.max(np.abs(X))) if X.size else 0.0
    if _gmax > 0 and np.isfinite(_gmax):
        X = X / _gmax
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Exact constancy is max == min (a computed std can be a tiny nonzero).
    keep = np.ptp(X, axis=0) > 0
    n_dropped = int(np.size(keep) - np.count_nonzero(keep))
    if np.count_nonzero(keep) < 2:
        return {"error": "Fewer than 2 non-constant dimensions"}

    # Per-column rescale before centring (exact; restored for the covariance).
    _Xk = X[:, keep]
    _cmax = np.max(np.abs(_Xk), axis=0)
    _cmax = np.where(_cmax > 0, _cmax, 1.0)
    _Xs = _Xk / _cmax
    _mean_s = _Xs.mean(axis=0)
    _std_s = _Xs.std(axis=0)
    if standardize:
        # The correlation matrix is EXACTLY invariant to per-dimension scaling,
        # so the factor simply cancels.
        Z = (_Xs - _mean_s) / np.where(_std_s > 0, _std_s, 1.0)
    else:
        # The covariance branch is NOT scale-free, so put the factor back.
        Z = (_Xs - _mean_s) * _cmax
    Xc = Z if not standardize else (_Xs - _mean_s) * _cmax
    D = Z.shape[1]
    # gamma = D / (N - 1): centring removes one degree of freedom.
    gamma = D / max(1, n_tokens - 1)

    C = (Z.T @ Z) / n_tokens
    eigs = np.linalg.eigvalsh(C)
    eigs = np.clip(eigs, 0.0, None)  # clip tiny negative round-off

    if standardize:
        sigma2 = 1.0
    else:
        sigma2 = _estimate_sigma2(eigs, gamma, edge_tol)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return {"error": "Could not estimate MP scale sigma^2"}
    # Work in sigma^2 units so every edge/CDF below is the standard MP law.
    eigs = eigs / sigma2

    threshold = _mp_bulk_edge(gamma) * (1.0 + edge_tol)
    outliers = eigs[eigs > threshold]
    trace = float(np.sum(eigs))

    # Lower spectral edge, defined only for gamma < 1 (the MP lower edge is
    # 0 / an atom otherwise).
    if gamma < 1.0:
        lo_threshold = _mp_lower_edge(gamma) * (1.0 - edge_tol)
        lower_outliers = eigs[eigs < lo_threshold]
        lower_outlier_frac = float(lower_outliers.size / eigs.size)
        lower_energy = (float(np.sum(lower_outliers) / trace)
                        if trace > 0 else float("nan"))
    else:
        lower_outlier_frac = float("nan")
        lower_energy = float("nan")

    return {
        "outlier_frac": float(outliers.size / eigs.size),
        "spike_energy": float(np.sum(outliers) / trace) if trace > 0 else float("nan"),
        "lower_outlier_frac": lower_outlier_frac,
        "lower_energy": lower_energy,
        "ks_distance": _ks_distance(eigs, gamma),
        "gamma": float(gamma),
        "n_tokens": int(n_tokens),
        "n_dims_dropped": n_dropped,
        "sigma2": float(sigma2),
    }


def _key_prefix(name: str) -> str:
    """`mp_` for the correlation family, `_meta_mp_` for the others.

    `*_cov` measures deviation from an ISOTROPIC MP null and scores
    heteroscedasticity as spikes; `perm_*` collapses to the 1/D floor under a
    single massive-activation dimension; `lower_*` is collinear with the KS
    distance. All stay emitted for audit.
    """
    return "_meta_mp_" if (name.endswith("_cov") or name.startswith("perm_")
                           or "_cov_" in name
                           or name.startswith("lower_")) else "mp_"


@register_task("geometry_mp_bulk_deviation")
class MPBulkDeviationTask(DiagnosticTask):
    """
    Measures how strongly per-layer activation spectra deviate from the
    Marchenko-Pastur (1967) iid null, and how much variance sits in
    BBP-supercritical spikes (Baik, Ben Arous & Péché 2005).

    Outputs flat scalars per layer tag (first/mid/last) plus
    cross-layer means: ``mp_outlier_frac_*``, ``mp_spike_energy_*``,
    ``mp_ks_distance_*`` (and ``*_g25`` fixed-gamma variants), with
    ``mp_gamma_*`` / ``mp_n_tokens_*`` / ``mp_dropped_dims_*`` metadata.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Marchenko-Pastur Bulk Deviation Analysis...")

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        max_tokens = self.config.get("max_tokens", 20000)
        edge_tol = self.config.get("edge_tol", 0.05)
        fixed_gamma = self.config.get("fixed_gamma", 0.25)
        layer_select = self.config.get("layers", "first_mid_last")

        # Block outputs (deepest entry = the hooked raw block), per document,
        # pooled into one unordered cloud per layer (RMT is about rows, not
        # trajectories).
        per_doc, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            layer_idx="all", use_cache=use_cache, replace_final_norm=True,
            per_sample=True, fallback="none")
        if per_doc is None:
            from ...cache import load_default_corpus
            if dataset is None:
                dataset = load_default_corpus(num_samples)
            per_doc, _info = hidden_states_for_task(
                model, tokenizer, dataset, None, num_samples=num_samples,
                layer_idx="all", use_cache=False, replace_final_norm=True,
                per_sample=True)
        layer_activations = ({li: select_rows(per_doc[li], "all")[0] for li in per_doc}
                             if per_doc else None)

        if not layer_activations:
            return {"error": "No hidden states available for MP analysis"}

        keys = sorted(layer_activations.keys())
        if layer_select == "first_mid_last":
            tagged = {"first": keys[0], "mid": keys[len(keys) // 2], "last": keys[-1]}
        else:
            tagged = {f"layer{li}": li for li in layer_select if li in keys}
        if not tagged:
            return {"error": f"No layers selected for MP analysis: {layer_select}"}

        suffix = f"g{int(round(fixed_gamma * 100))}" if fixed_gamma else None
        # Correlation family (headline), covariance and permutation-null
        # families (_meta_, see _key_prefix), lower-edge statistics (_meta_).
        _BASE = ("outlier_frac", "spike_energy", "ks_distance")
        _COV = _BASE
        _PERM = ("perm_outlier_frac", "perm_spike_energy", "perm_top_ratio")
        metric_names = (list(_BASE) + [f"{m}_cov" for m in _COV] + list(_PERM)
                        + ["lower_outlier_frac", "lower_energy",
                           "lower_outlier_frac_cov", "lower_energy_cov"])
        if suffix:
            metric_names += [f"{m}_{suffix}" for m in _BASE]
            metric_names += [f"{m}_cov_{suffix}" for m in _COV]

        results: dict = {}
        collected: dict = {m: [] for m in metric_names}
        seen_layers = set()
        for tag, li in tagged.items():
            if li in seen_layers:
                # Tiny models: first == mid == last; compute once but
                # still emit every tag for schema stability.
                pass
            seen_layers.add(li)
            m = _mp_layer_metrics(
                layer_activations[li],
                edge_tol=edge_tol,
                max_tokens=max_tokens,
                seed=0,
                fixed_gamma=fixed_gamma,
            )
            if "error" in m:
                logger.info(f"  MP analysis skipped for layer {li}: {m['error']}")
                for name in metric_names:
                    results[f"{_key_prefix(name)}{name}_{tag}"] = float("nan")
                continue
            for name in metric_names:
                val = float(m.get(name, float("nan")))
                results[f"{_key_prefix(name)}{name}_{tag}"] = val
                collected[name].append(val)
            # gamma, token count and dropped dimensions are sampling diagnostics.
            results[f"_meta_mp_gamma_{tag}"] = float(m["gamma"])
            results[f"_meta_mp_n_tokens_{tag}"] = int(m["n_tokens"])
            results[f"_meta_mp_dropped_dims_{tag}"] = int(m["n_dims_dropped"])

        for name in metric_names:
            vals = np.asarray(collected[name], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            results[f"{_key_prefix(name)}{name}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
        results["_meta_mp_edge_tol"] = float(edge_tol)
        return results
