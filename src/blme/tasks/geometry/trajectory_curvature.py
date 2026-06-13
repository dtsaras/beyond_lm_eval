"""
Trajectory curvature — do deeper layers straighten sentence trajectories?

Reference:
    Hosseini, E. A. & Fedorenko, E. (2023). "Large language models
    implicitly learn to straighten neural sentence trajectories to
    construct a predictive representation of natural language."
    NeurIPS 2023, arXiv:2311.04930.

Definition (Hosseini & Fedorenko 2023, Methods):
    For the hidden states x_1 .. x_T of a single sample at a given layer,
    form the difference ("velocity") vectors

        v_t = x_{t+1} - x_t,

    and measure the discrete curvature at each interior point as the
    angle between consecutive difference vectors:

        c_t = arccos( <v_t, v_{t+1}> / (||v_t|| · ||v_{t+1}||) ).

    The per-layer curvature is the mean of c_t over interior positions t
    and over samples (radians; 0 = perfectly straight, pi = full
    reversal). Their headline finding: trained LMs *straighten*
    trajectories with depth (curvature decreases relative to the
    earliest layer), and models with better next-word prediction
    straighten more.

Implementation notes:
    * Tier 2 — consumes the shared cache via
      ``cache.get_hidden_states(layer_idx="all", per_sample=True)``.
      Trajectory order matters, so the per-sample ``(T_i, D)`` chunks
      are required; the flattened token cloud would silently mix
      documents. Without a cache the task runs its own forward pass
      through a private :class:`~blme.cache.ModelOutputCache`.
    * The shared cache stores only post-block hidden states (it drops
      the embedding output ``hs[0]``), so "first layer" here is the
      output of the first transformer block — Hosseini & Fedorenko
      compare against the input embedding, which is one step earlier.
      The straightening trend is unaffected; only the baseline differs.
    * The BOS token is skipped (``skip_first_tokens``, default 1)
      because the BOS hidden state is an extreme outlier in most
      architectures and the first difference vector through it would
      dominate the angle statistics.
    * Angles are computed in float64 with the cosine clamped to
      [-1, 1]; zero-norm difference vectors (repeated identical hidden
      states) are skipped rather than propagated as NaN.
    * Only summary scalars are emitted (no per-layer list) so the
      aggregator never sees absolute layer indices; the slope is
      regressed on the *normalized* depth l/(n_layers - 1) for
      cross-model comparability.
"""

import logging
from typing import List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Difference vectors with a norm below this are treated as degenerate
# (repeated identical hidden states) and the angles touching them are
# skipped instead of producing NaN/garbage arccos values.
_ZERO_NORM_EPS = 1e-12


def _trajectory_angles(X) -> np.ndarray:
    """Angles (radians) between consecutive difference vectors of one
    trajectory.

    Args:
        X: array-like of shape (T, D) — ordered hidden states of a
           single sample at one layer.

    Returns:
        1-D float64 array of valid angles (may be empty). An angle at
        interior position t is emitted only when both v_t and v_{t+1}
        have non-negligible norm.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        return np.empty(0, dtype=np.float64)

    V = np.diff(X, axis=0)                       # (T-1, D)
    norms = np.linalg.norm(V, axis=1)            # (T-1,)

    dots = np.einsum("ij,ij->i", V[:-1], V[1:])  # (T-2,)
    denom = norms[:-1] * norms[1:]
    valid = (norms[:-1] > _ZERO_NORM_EPS) & (norms[1:] > _ZERO_NORM_EPS)
    if not np.any(valid):
        return np.empty(0, dtype=np.float64)

    cosines = np.clip(dots[valid] / denom[valid], -1.0, 1.0)
    return np.arccos(cosines)


def _mean_curvature(X) -> float:
    """Mean discrete curvature (radians) of one (T, D) trajectory.

    Returns NaN when the trajectory has no valid angle (T < 3 or all
    difference vectors degenerate).
    """
    angles = _trajectory_angles(X)
    return float(np.mean(angles)) if angles.size else float("nan")


def _pooled_layer_curvature(chunks: List, skip_first: int = 1) -> float:
    """Mean curvature pooled over t and over samples for one layer.

    Args:
        chunks: list of (T_i, D) tensors/arrays — per-sample hidden
                states in trajectory order.
        skip_first: number of leading tokens to drop per sample (BOS).

    Returns:
        Pooled mean angle in radians, or NaN if no sample yields a
        valid angle. Samples with fewer than ``skip_first + 3`` tokens
        are skipped (two difference vectors are needed for one angle).
    """
    all_angles = []
    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().float().cpu().numpy()
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] < skip_first + 3:
            continue
        angles = _trajectory_angles(chunk[skip_first:])
        if angles.size:
            all_angles.append(angles)
    if not all_angles:
        return float("nan")
    return float(np.mean(np.concatenate(all_angles)))


@register_task("geometry_trajectory_curvature")
class TrajectoryCurvatureTask(DiagnosticTask):
    """
    Computes the Hosseini & Fedorenko (NeurIPS 2023, arXiv:2311.04930)
    discrete trajectory curvature per layer and summarises the
    straightening profile across depth.

    Outputs (flat floats, radians):
        curvature_mean_first_layer / _mid_layer / _last_layer
        curvature_overall_mean   — mean of per-layer means
        straightening_ratio      — (c_first - c_last) / c_first
        curvature_slope          — OLS slope of per-layer curvature on
                                   normalized depth l/(n_layers-1)
        curvature_q25/q50/q75    — quartiles of per-layer means
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Trajectory Curvature Analysis...")

        num_samples = self.config.get("num_samples", 100)
        skip_first = int(self.config.get("skip_first_tokens", 1))
        use_cache = self.config.get("use_cache", True)

        if cache is not None and cache.is_populated and use_cache:
            per_layer_chunks = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=True,
            )
        else:
            # Fallback: run our own forward pass through a private cache
            # so the per-sample (T_i, D) split logic stays in one place.
            from ...cache import ModelOutputCache, load_default_corpus

            if dataset is None:
                dataset = load_default_corpus(num_samples)
            local_cache = ModelOutputCache(
                model, tokenizer, dataset=dataset, num_samples=num_samples,
            )
            local_cache.populate(need_hidden=True)
            per_layer_chunks = local_cache.get_hidden_states(
                layer_idx="all", per_sample=True,
            )

        if not per_layer_chunks:
            return {"error": "No hidden states available for trajectory curvature"}

        layer_keys = sorted(per_layer_chunks.keys())
        n_layers = len(layer_keys)
        n_samples_used = max(
            (len(per_layer_chunks[k]) for k in layer_keys), default=0,
        )

        per_layer_curvature = [
            _pooled_layer_curvature(per_layer_chunks[k], skip_first=skip_first)
            for k in layer_keys
        ]
        curv = np.asarray(per_layer_curvature, dtype=np.float64)

        if not np.any(np.isfinite(curv)):
            return {"error": "No sample long enough for trajectory curvature"}

        c_first = curv[0]
        c_last = curv[-1]
        c_mid = curv[n_layers // 2]

        # Straightening ratio: positive => deeper layers are straighter.
        if np.isfinite(c_first) and np.isfinite(c_last) and abs(c_first) > 0:
            straightening_ratio = float((c_first - c_last) / c_first)
        else:
            straightening_ratio = float("nan")

        # Slope on NORMALIZED depth l/(n_layers-1) — house convention
        # for cross-model comparability (AUDIT_REPORT fix #45).
        finite = np.isfinite(curv)
        if n_layers >= 2 and finite.sum() >= 2:
            depth = (
                np.arange(n_layers, dtype=np.float64) / (n_layers - 1)
                if n_layers > 1 else np.zeros(n_layers)
            )
            slope = float(np.polyfit(depth[finite], curv[finite], 1)[0])
        else:
            slope = float("nan")

        finite_curv = curv[finite]
        return {
            "curvature_mean_first_layer": float(c_first),
            "curvature_mean_mid_layer": float(c_mid),
            "curvature_mean_last_layer": float(c_last),
            "curvature_overall_mean": float(np.mean(finite_curv)),
            "straightening_ratio": straightening_ratio,
            "curvature_slope": slope,
            "curvature_q25": float(np.percentile(finite_curv, 25)),
            "curvature_q50": float(np.percentile(finite_curv, 50)),
            "curvature_q75": float(np.percentile(finite_curv, 75)),
            # _meta_ prefix => excluded from the analysis feature matrix so these
            # architecture/sampling counts cannot leak in as size proxies (Audit-V2).
            "_meta_n_layers": int(n_layers),
            "_meta_n_samples_used": int(n_samples_used),
        }
