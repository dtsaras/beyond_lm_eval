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
                      role_layers, role_meta, role_names, rows_policy)
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


_ROLES = ("first", "mid", "last")
_TWINS = ("mean_persistence_h0", "max_persistence_h0", "mean_persistence_h1")
_HEADLINES = tuple(
    [f"{r}_layer_{k}" for r in _ROLES for k in ("mean_persistence_h0", "num_loops_h1")]
    + [f"{r}_layer_{k}_norm" for r in _ROLES for k in _TWINS])
_INV = "tests/tasks/invariance/test_topology_homology.py::"
# The mean-pairwise-distance twins; the raw lifetimes move with any rescale.
_SIZE_INVARIANT = {
    f"{r}_layer_{k}_norm": (("embed_isometric", "zero_pad", "rescale"),
                            _INV + "test_mpd_twins_are_width_and_scale_free")
    for r in _ROLES for k in _TWINS}


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
    SOURCES = {
        "paper": "doi:10.1007/s00454-004-1146-y (Zomorodian & Carlsson 2005)",
        "code": "scikit-tda/ripser.py (installed package)",
        "llm_precedent": ("arXiv:2505.20435 (Fay et al. 2025: Ripser VR barcodes of last-token "
                          "clouds; mean 0-bar death and 1-bar count as headline summaries)"),
    }
    ADAPTATION_NOTE = (
        "Vietoris-Rips persistence (ripser, dimensions 0 and 1) is computed on one point per "
        "sequence, its last token, at the first, middle and last block, as Fay et al. (2025) do "
        "for LLMs, with N = 1000 sequences for every model. ripser reads float64 pairwise "
        "distances. H0 / H1 lifetimes are summarised by their mean and maximum and the H1 bar "
        "count. Raw lifetimes are in the representation's units, so each is also divided by the "
        "cloud's mean pairwise distance (the _norm twins), which removes a global rescale and a "
        "change of embedding width exactly. The twins still drift with N (about -4.5% per "
        "doubling for mean H0), so models are compared at the same N. The fine/coarse split at "
        "half the mean pairwise distance is a BLME construct."
    )
    HEADLINE_KEYS = _HEADLINES
    SIZE_INVARIANT = _SIZE_INVARIANT

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
        _sorted_targets = sorted(layer_representations.keys())
        _role = role_names(_sorted_targets)
        results.update(role_meta(_role, pinfo["n_layers"]))

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
