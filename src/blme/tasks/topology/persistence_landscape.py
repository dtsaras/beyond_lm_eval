"""
Persistence landscapes — Bubenik 2015 (arXiv:1501.00179); Chazal et al. 2015.

A persistence landscape is a functional summary of a persistence diagram
that is richer than scalar statistics (lifespans, entropy) and amenable
to statistical operations (mean, variance, hypothesis tests).

For each birth-death pair (b, d) in a persistence diagram, define the
*tent function* f(t) = max(0, min(t - b, d - t)). This is a triangle
peaked at the midpoint (b+d)/2 with height (d-b)/2.

The k-th persistence landscape L_k(t) is the k-th largest tent function
value at each t: sort all tent function values at t in decreasing order,
then L_k = the k-th value (1-indexed). L_1 captures the most prominent
topological feature, L_2 the second, etc.

Reported metrics (per layer, per homology dimension H0/H1):
  - **landscape_integral_k**: integral of L_k over t, for k = 1..K
  - **landscape_max_k**: max of L_k over t
  - **landscape_norm_k**: L2 norm of L_k
  - **landscape_mean_integral**: mean integral across k = 1..K

These complement the scalar persistence entropy and Betti curve tasks.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _compute_landscape(dgm: np.ndarray, n_landscapes: int = 5,
                       n_points: int = 200) -> np.ndarray:
    """Compute the first `n_landscapes` persistence landscapes from a
    persistence diagram.

    Args:
        dgm: (N, 2) array of (birth, death) pairs (finite only).
        n_landscapes: number of landscape functions to compute.
        n_points: resolution of the discretised t-axis.

    Returns:
        (n_landscapes, n_points) array. L[k, i] = L_{k+1}(t_i).
    """
    # Filter out infinite-death and degenerate pairs.
    finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else dgm
    finite = finite[finite[:, 1] > finite[:, 0]]
    if finite.size == 0:
        return np.zeros((n_landscapes, n_points), dtype=np.float64)

    births = finite[:, 0]
    deaths = finite[:, 1]
    t_min = float(births.min())
    t_max = float(deaths.max())
    if t_max <= t_min:
        return np.zeros((n_landscapes, n_points), dtype=np.float64)

    ts = np.linspace(t_min, t_max, n_points)
    # Evaluate all tent functions at all t-points: (N_pairs, n_points)
    # f_i(t) = max(0, min(t - b_i, d_i - t))
    left = ts[np.newaxis, :] - births[:, np.newaxis]   # (N, n_points)
    right = deaths[:, np.newaxis] - ts[np.newaxis, :]   # (N, n_points)
    tents = np.maximum(0.0, np.minimum(left, right))     # (N, n_points)

    # At each t-point, sort tent values descending and take the top K.
    # Transpose to (n_points, N), sort along axis=1 descending, take :K.
    sorted_vals = np.sort(tents, axis=0)[::-1]  # (N, n_points) descending
    K = min(n_landscapes, sorted_vals.shape[0])
    landscapes = np.zeros((n_landscapes, n_points), dtype=np.float64)
    landscapes[:K, :] = sorted_vals[:K, :]

    return landscapes


def _landscape_stats(landscapes: np.ndarray, dt: float
                     ) -> Dict[str, float]:
    """Summary statistics from a (K, n_points) landscape array."""
    # np.trapezoid (NumPy 2.0) replaces the deprecated np.trapz; fall
    # back on older NumPy.
    _trapz = getattr(np, "trapezoid", np.trapz)
    K = landscapes.shape[0]
    integrals = []
    maxes = []
    norms = []
    for k in range(K):
        Lk = landscapes[k]
        integrals.append(float(_trapz(Lk, dx=dt)))
        maxes.append(float(Lk.max()))
        norms.append(float(np.sqrt(_trapz(Lk ** 2, dx=dt))))
    return {
        "landscape_integrals": integrals,
        "landscape_maxes": maxes,
        "landscape_norms": norms,
        "mean_landscape_integral": float(np.mean(integrals)),
    }


@register_task("topology_persistence_landscape")
class PersistenceLandscapeTask(DiagnosticTask):
    """Persistence landscapes (Bubenik 2015) at early / mid / late layers."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistence Landscape Analysis...")

        if not HAS_RIPSER:
            return {"error": "ripser not installed. Install with: pip install ripser"}

        num_samples = self.config.get("num_samples", 20)
        n_landscapes = self.config.get("n_landscapes", 5)
        n_points = self.config.get("n_points", 200)

        if dataset is None:
            dataset = [
                {"text": f"Sample {i} for topological landscape analysis."}
                for i in range(num_samples)
            ]

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)
        target_layers = [0, n_layers // 2, n_layers - 1]

        # Collect mean-pooled per-sentence representations at target layers.
        layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in target_layers}

        with torch.no_grad():
            for i, s in enumerate(dataset):
                if i >= num_samples:
                    break
                text = s["text"] if isinstance(s, dict) else str(s)
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128).to(device)
                out = model(**enc, output_hidden_states=True)
                for l_idx in target_layers:
                    h = out.hidden_states[l_idx + 1][0]  # (T, D)
                    layer_reps[l_idx].append(h.float().mean(dim=0).cpu().numpy())

        results: Dict[str, object] = {}

        for l_idx in target_layers:
            if len(layer_reps[l_idx]) < 3:
                continue
            X = np.stack(layer_reps[l_idx])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(X, maxdim=1)["dgms"]

            for dim_idx, dim_label in enumerate(["h0", "h1"]):
                if dim_idx >= len(dgms):
                    continue
                dgm = dgms[dim_idx]
                landscapes = _compute_landscape(dgm, n_landscapes, n_points)
                # Compute dt for integration.
                finite = dgm[np.isfinite(dgm[:, 1])]
                finite = finite[finite[:, 1] > finite[:, 0]] if finite.size else finite
                if finite.size > 0:
                    dt = (finite[:, 1].max() - finite[:, 0].min()) / max(1, n_points - 1)
                else:
                    dt = 1.0
                stats = _landscape_stats(landscapes, dt)
                prefix = f"layer_{l_idx}_{dim_label}"
                results[f"{prefix}_mean_landscape_integral"] = stats["mean_landscape_integral"]
                results[f"{prefix}_landscape_max_1"] = stats["landscape_maxes"][0] if stats["landscape_maxes"] else 0.0
                results[f"{prefix}_landscape_norm_1"] = stats["landscape_norms"][0] if stats["landscape_norms"] else 0.0
                results[f"{prefix}_landscape_integrals"] = stats["landscape_integrals"]

        if not results:
            return {"error": "No layers produced landscape features"}

        return results
