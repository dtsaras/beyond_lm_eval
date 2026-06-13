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

logger = logging.getLogger("blme")

_ZERO_VAR_EPS = 1e-12


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

    # Integrable inverse-sqrt singularities can sit at both edges
    # (gamma = 1 puts t_- at 0); nudge the grid inward.
    eps = max((hi - lo) * 1e-9, 1e-12)
    grid = np.linspace(lo + eps, hi - eps, n_grid)
    dens = np.sqrt(np.maximum((hi - grid) * (grid - lo), 0.0)) / (
        2.0 * np.pi * gamma * grid
    )
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))]
    )
    # Normalize the numeric integral to the exact continuous mass to
    # kill trapezoid truncation error at the edge singularities.
    if cum[-1] > 0:
        cum *= cont_mass / cum[-1]

    cdf = np.interp(x, grid, cum, left=0.0, right=cont_mass)
    cdf = cdf + point_mass * (x >= 0.0)
    return np.clip(cdf, 0.0, 1.0)


def _ks_distance(eigs: np.ndarray, gamma: float) -> float:
    """Kolmogorov-Smirnov distance between the empirical eigenvalue CDF
    and the MP CDF at the same gamma."""
    eigs = np.sort(np.asarray(eigs, dtype=np.float64))
    n = eigs.size
    if n == 0:
        return float("nan")
    F = _mp_cdf(eigs, gamma)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(np.abs(F - i / n), np.abs(F - (i - 1) / n))))


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
        X = X.detach().float().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 4 or X.shape[1] < 2:
        return {"error": f"Activation matrix too small for MP analysis: {X.shape}"}

    rng = np.random.default_rng(seed)

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
        "ks_distance": core["ks_distance"],
        "gamma": core["gamma"],
        "n_tokens": core["n_tokens"],
        "n_dims_dropped": core["n_dims_dropped"],
    }

    # Fixed-gamma variant: same metrics at a common aspect ratio so the
    # MP reference is identical across models.
    if fixed_gamma:
        suffix = f"g{int(round(fixed_gamma * 100))}"
        n_fixed = int(math.ceil(X.shape[1] / fixed_gamma))
        if X.shape[0] >= n_fixed:
            idx = rng.choice(X.shape[0], size=n_fixed, replace=False)
            fixed = _mp_metrics_core(X[idx], edge_tol)
            out[f"outlier_frac_{suffix}"] = fixed.get("outlier_frac", float("nan"))
            out[f"spike_energy_{suffix}"] = fixed.get("spike_energy", float("nan"))
            out[f"ks_distance_{suffix}"] = fixed.get("ks_distance", float("nan"))
        else:
            out[f"outlier_frac_{suffix}"] = float("nan")
            out[f"spike_energy_{suffix}"] = float("nan")
            out[f"ks_distance_{suffix}"] = float("nan")
    return out


def _mp_metrics_core(X: np.ndarray, edge_tol: float) -> dict:
    """Z-score columns, drop zero-variance dims, eigendecompose the
    correlation matrix, and compare against the gamma-matched MP law."""
    n_tokens = X.shape[0]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    keep = std > _ZERO_VAR_EPS
    n_dropped = int(np.size(keep) - np.count_nonzero(keep))
    if np.count_nonzero(keep) < 2:
        return {"error": "Fewer than 2 non-constant dimensions"}

    Z = (X[:, keep] - mean[keep]) / std[keep]
    D = Z.shape[1]
    gamma = D / n_tokens

    C = (Z.T @ Z) / n_tokens
    eigs = np.linalg.eigvalsh(C)
    eigs = np.clip(eigs, 0.0, None)  # clip tiny negative round-off

    threshold = _mp_bulk_edge(gamma) * (1.0 + edge_tol)
    outliers = eigs[eigs > threshold]
    trace = float(np.sum(eigs))

    return {
        "outlier_frac": float(outliers.size / eigs.size),
        "spike_energy": float(np.sum(outliers) / trace) if trace > 0 else float("nan"),
        "ks_distance": _ks_distance(eigs, gamma),
        "gamma": float(gamma),
        "n_tokens": int(n_tokens),
        "n_dims_dropped": n_dropped,
    }


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

        # The unordered flattened token cloud is exactly what RMT wants
        # (per_sample=False) — row order is irrelevant to the spectrum.
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
            )
        else:
            from .utils import collect_hidden_states
            from ...cache import load_default_corpus

            if dataset is None:
                dataset = load_default_corpus(num_samples)
            layer_activations = collect_hidden_states(
                model, tokenizer, dataset, num_samples=num_samples,
                layer_idx="all",
            )

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
        metric_names = ["outlier_frac", "spike_energy", "ks_distance"]
        if suffix:
            metric_names += [f"{m}_{suffix}" for m in
                             ("outlier_frac", "spike_energy", "ks_distance")]

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
                    results[f"mp_{name}_{tag}"] = float("nan")
                continue
            for name in metric_names:
                val = float(m.get(name, float("nan")))
                results[f"mp_{name}_{tag}"] = val
                collected[name].append(val)
            # _meta_ prefix => excluded from the analysis feature matrix. gamma=D/N
            # is a d_model proxy and n_tokens/dropped_dims are sampling diagnostics,
            # none of which are intrinsic representational properties (Audit-V2).
            results[f"_meta_mp_gamma_{tag}"] = float(m["gamma"])
            results[f"_meta_mp_n_tokens_{tag}"] = int(m["n_tokens"])
            results[f"_meta_mp_dropped_dims_{tag}"] = int(m["n_dims_dropped"])

        for name in metric_names:
            vals = np.asarray(collected[name], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            results[f"mp_{name}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
        results["mp_edge_tol"] = float(edge_tol)
        return results
