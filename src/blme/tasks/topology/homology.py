"""Persistent Homology on hidden-state point clouds.

Standard Topological Data Analysis pipeline: take a set of hidden
representations, build the Vietoris-Rips filtration, compute
persistence diagrams with ripser, and report the H₀ / H₁ lifespans.

References:
  * Zomorodian, Carlsson 2005 — "Computing Persistent Homology",
    *Discrete and Computational Geometry* 33(2).
  * Edelsbrunner, Harer 2008 — "Persistent Homology: a Survey".
  * Naitzat, Zhitnikov, Lim 2020 — "Topology of Deep Neural
    Networks", JMLR 21(184):1-40 (arXiv:2004.06093; motivation for
    applying TDA to learned representations).
"""

import torch
import numpy as np
import warnings

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ._points import (fallback_dataset, last_token_points, point_meta, rips_diagrams,
                      role_layers, rows_policy)
import logging
logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _scale_stratified_summary(dgms, scale: float, frac: float = 0.5) -> dict:
    """Scale-stratified persistence summaries (a BLME construct).

    Bars are stratified by the scale at which they DIE relative to the
    point cloud's mean pairwise distance: fine = death < frac * scale,
    coarse = death >= frac * scale. Reported per homology dimension:
    fine/coarse counts and the fine fraction of total persistence mass.
    """
    out = {}
    if scale <= 0:
        return out
    thresh = frac * scale
    for k, name in ((0, "h0"), (1, "h1")):
        if len(dgms) <= k:
            continue
        bars = [(b, d) for b, d in dgms[k] if np.isfinite(d)]
        fine = [(b, d) for b, d in bars if d < thresh]
        coarse = [(b, d) for b, d in bars if d >= thresh]
        out[f"{name}_fine_count"] = len(fine)
        out[f"{name}_coarse_count"] = len(coarse)
        total_mass = sum(d - b for b, d in bars)
        fine_mass = sum(d - b for b, d in fine)
        out[f"{name}_fine_persistence_frac"] = (
            fine_mass / total_mass if total_mass > 0 else float("nan")
        )
    return out


def _lifespan_summary(dgms) -> dict:
    """Summarize ripser/GUDHI persistence diagrams into finite-bar lifespan
    statistics (lifespan = death - birth; the essential infinite-death H0
    bar is dropped). Matches the textbook persistence summary and is exact
    against an analytically-known diagram (e.g. unit square -> H0 mean/max=1,
    one H1 loop of persistence sqrt(2)-1).
    """
    h0 = [d - b for b, d in dgms[0] if d != np.inf]
    h1 = [d - b for b, d in dgms[1] if d != np.inf] if len(dgms) > 1 else []
    return {
        "mean_persistence_h0": float(np.mean(h0)) if h0 else 0.0,
        "max_persistence_h0": float(np.max(h0)) if h0 else 0.0,
        "mean_persistence_h1": float(np.mean(h1)) if h1 else 0.0,
        "num_loops_h1": len(h1),
    }


@register_task("topology_homology")
class PersistentHomologyTask(DiagnosticTask):
    """
    Implements a Topological Data Analysis (TDA) task.
    Takes a set of hidden representations and computes their persistent homology
    using the Vietoris-Rips complex, specifically extracting Betti-0 and Betti-1
    persistent features (holes/clusters in the manifold).

    See module docstring for paper references (Zomorodian-Carlsson
    2005, Edelsbrunner-Harer 2008, Naitzat-Zhitnikov-Lim 2020).
    """
    TOKEN_POLICY = "last_per_sequence"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistent Homology (TDA)...")
        # Q1: one point per sequence (its last token), N = 1000 for every model.
        num_samples = int(self.config.get("num_samples", 1000))
        use_cache = self.config.get("use_cache", True)

        if not HAS_RIPSER:
            msg = "Ripser library not installed. Skipping TDA module. Install with: pip install ripser"
            logger.info(msg)
            return {"error": msg}

        if dataset is None:
            dataset = fallback_dataset(num_samples, "Homology")
        pts, pinfo = last_token_points(
            model, tokenizer, dataset, cache, num_samples=num_samples, layers=role_layers,
            policy=rows_policy(self.TOKEN_POLICY), use_cache=use_cache)
        layer_representations = {l: X for l, X in pts.items()}
        if not layer_representations or min(len(X) for X in pts.values()) < 3:
            return {"error": "Need at least 3 samples for meaningful topological features"}

        results = point_meta(pinfo, pts[sorted(pts)[len(pts) // 2]])
        # Depth-role names (first/mid/last); absolute-index names alias
        # different depths across models and are _meta_.
        _role = {}
        _sorted_targets = sorted(layer_representations.keys())
        if len(_sorted_targets) == 3:
            _role = dict(zip(_sorted_targets, ("first", "mid", "last")))
        elif len(_sorted_targets) == 2:
            _role = dict(zip(_sorted_targets, ("first", "last")))
        elif len(_sorted_targets) == 1:
            _role = {_sorted_targets[0]: "first"}

        for l_idx, data_points in layer_representations.items():
            _r = _role.get(l_idx)
            data_matrix = np.array(data_points)  # Shape: (num_samples, hidden_dim)

            # Duplicate points are quotiented (a repeated point is a zero-length
            # H0 bar); ripser reads float64 pairwise distances. `data_matrix`
            # itself is kept for the mpd below.
            dgms, _scale = rips_diagrams(
                np.unique(np.asarray(data_matrix, dtype=np.float64), axis=0))
            results[f"_meta_layer_{l_idx}_rips_scale"] = _scale

            summary = _lifespan_summary(dgms)
            for _k in ("mean_persistence_h0", "max_persistence_h0",
                       "mean_persistence_h1", "num_loops_h1"):
                results[f"_meta_layer_{l_idx}_{_k}"] = summary[_k]
                if _r:
                    results[f"{_r}_layer_{_k}"] = summary[_k]

            # Lifetimes divided by the cloud's mean pairwise distance (mpd):
            # VR lifetimes and the mpd are both homogeneous of degree 1.
            from scipy.spatial.distance import pdist
            pairwise = pdist(data_matrix)
            mpd = float(pairwise.mean()) if pairwise.size else 0.0
            if mpd > 0:
                for key in ("mean_persistence_h0", "max_persistence_h0",
                            "mean_persistence_h1"):
                    results[f"_meta_layer_{l_idx}_{key}_norm"] = summary[key] / mpd
                    if _r:
                        results[f"{_r}_layer_{key}_norm"] = summary[key] / mpd
                results[f"_meta_layer_{l_idx}_mean_pairwise_distance"] = mpd
                if _r:
                    results[f"_meta_{_r}_layer_mean_pairwise_distance"] = mpd
                for k, v in _scale_stratified_summary(dgms, mpd).items():
                    results[f"_meta_layer_{l_idx}_{k}"] = v
                    if _r:
                        # h1_fine_* was constant (0.0) on every model.
                        _pfx = "_meta_" if k.startswith("h1_fine") else ""
                        results[f"{_pfx}{_r}_layer_{k}"] = v

        # H1 summaries do not separate models below a few hundred points.
        results["_meta_h1_resolves_between_models"] = bool(
            len(layer_representations[_sorted_targets[0]]) >= 200)
        return results
