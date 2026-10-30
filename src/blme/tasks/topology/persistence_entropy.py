"""
Persistence Entropy — Shannon entropy of the persistence diagram lifespans.

Quantifies the "complexity" or "disorder" of the topological features found
in the representation space. High entropy indicates many features with similar
lifetimes (complex, disordered topology), while low entropy indicates a few
dominant persistent features (simple, ordered topology).

References:
- "Characterisation of the Idiotypic Immune Network Through Persistent
  Entropy" (Rucco, Castiglione, Merelli, Pettini — Proceedings of ECCS 2014,
  Springer Proceedings in Complexity, 2016; persistent entropy introduced)
- "An Entropy-based Persistence Barcode" (Chintakunta, Gentimis,
  Gonzalez-Diaz, Jimenez, Krim — Pattern Recognition 48(2):391-401, 2015)
- "On the Stability of Persistent Entropy and New Summary Functions for
  Topological Data Analysis" (Atienza, Gonzalez-Diaz, Soriano-Trigueros —
  Pattern Recognition 107:107509, 2020; arXiv:1803.08304; Def. 3.1 is the
  log2 formulation implemented here)
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


def _persistence_entropy(lifespans):
    """Compute Shannon entropy of normalized persistence lifespans.

    PE = -Σ p_i · log2(p_i)   where p_i = lifespan_i / Σ lifespans

    Uses base-2 logarithm to match the canonical definition (Atienza et al.
    Def. 3.1, which states "log will refer to the log-base-2 function") and
    the reference implementation giotto-tda (scipy.stats.entropy base=2).

    Args:
        lifespans: list of positive floats (birth-death intervals)

    Returns:
        float: persistence entropy (0 = single dominant feature, high = uniform)
    """
    if not lifespans or len(lifespans) < 2:
        return 0.0

    # Scale invariant: PE depends on p_i = l_i / sum(l) only, so both
    # degeneracy tests below are relative, not absolute cutoffs.
    arr = np.asarray(lifespans, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return 0.0
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        return 0.0

    p = arr / total
    p = p[p > 1e-15]
    if p.size < 2:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


_H0_FAMILY = ("pe_h0", "pe_h0_full", "pe_h0_norm", "num_features_h0")
_ROLES = ("first", "mid", "last")
_HEADLINES = tuple(f"{r}_layer_{k}" for r in _ROLES for k in ("pe_h1", "pe_h1_norm"))
# PE is a function of the normalised bar lengths, and E / log2(n) of the bar
# count, so both ignore a rescale and depend on pairwise distances only.
_SIZE_INVARIANT = {
    k: (("embed_isometric", "zero_pad", "rescale"),
        "tests/tasks/invariance/test_topology_persistence_entropy.py::"
        "test_h1_entropy_is_width_and_scale_free")
    for k in _HEADLINES}


@register_task("topology_persistence_entropy")
class PersistenceEntropyTask(DiagnosticTask):
    """
    Computes Persistence Entropy at early, middle, and late layers.

    For each layer, constructs a persistence diagram via Vietoris-Rips
    and computes the Shannon entropy of the H0 and H1 lifespan distributions.
    """
    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": "arXiv:1803.08304",
        "code": "giotto-ai/giotto-tda@6f896531b7c59216fd292018bf73728dea40b7e6",
        "llm_precedent": ("arXiv:2505.20435 (Fay et al. 2025: persistent entropy among the "
                          "barcode summaries of last-token clouds)"),
    }
    ADAPTATION_NOTE = (
        "Persistent entropy (Atienza et al. 2020 Def. 3.1, log base 2) is computed per homology "
        "dimension on the Vietoris-Rips barcode of one point per sequence, its last token, at the "
        "first, middle and last block (the LLM precedent's point cloud, Fay et al. 2025), with "
        "N = 1000 sequences. The essential H0 bar is dropped, as giotto-tda's reduced homology "
        "does; pe_h0_full keeps it through Atienza's scale-invariant mu_2 projection. The H0 "
        "family is _meta_: raw H0 entropy of N points is log2(N - 1) to three digits, and its "
        "E / log2(n) form, though N-flat, does not separate models. The features are the H1 "
        "entropy and E / log2(n), the maximum-entropy normalisation Atienza's Sec. 3.3 uses; "
        "giotto-tda's normalize=True (E / log2 of total persistence) depends on the units and "
        "is not used."
    )
    HEADLINE_KEYS = _HEADLINES
    SIZE_INVARIANT = _SIZE_INVARIANT

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistence Entropy Analysis...")
        # Q1: one point per sequence (its last token), N = 1000 for every model.
        num_samples = int(self.config.get("num_samples", 1000))
        use_cache = self.config.get("use_cache", True)

        if not HAS_RIPSER:
            msg = "Ripser library not installed. Install with: pip install ripser"
            logger.info(msg)
            return {"error": msg}

        if dataset is None:
            dataset = fallback_dataset(num_samples, "Persistence entropy")
        pts, pinfo = last_token_points(
            model, tokenizer, dataset, cache, num_samples=num_samples, layers=role_layers,
            policy=rows_policy(self.TOKEN_POLICY), use_cache=use_cache)
        if not pts or min(len(X) for X in pts.values()) < 3:
            return {"error": "Need at least 3 samples for persistence entropy"}
        target_layers = sorted(pts)
        layer_reps = {l: list(pts[l]) for l in target_layers}

        results = point_meta(pinfo, pts[target_layers[len(target_layers) // 2]])
        # Depth-role names (first/mid/last); absolute-index names alias
        # different depths across models and are _meta_.
        roles = role_names(target_layers)
        results.update(role_meta(roles, pinfo["n_layers"]))
        for l_idx, points in layer_reps.items():
            # ripser reads float64 pairwise distances (PE is scale-free, so the
            # exact power-of-two shift rips_diagrams may apply cancels).
            dgms, _scale = rips_diagrams(np.stack(points))
            results[f"_meta_layer_{l_idx}_rips_scale"] = _scale

            # H0 lifespans (excluding infinite-lifespan component)
            lifespans_h0 = [d - b for b, d in dgms[0] if d != np.inf]
            # H1 lifespans
            lifespans_h1 = [d - b for b, d in dgms[1] if d != np.inf]

            pe_h0 = _persistence_entropy(lifespans_h0)
            pe_h1 = _persistence_entropy(lifespans_h1)

            # pe_h0_full: the essential bar replaced by 2x the longest finite
            # lifetime (Atienza Def. 3.15, mu_lambda with lambda = 2).
            finite_h0 = [d - b for b, d in dgms[0] if d != np.inf]
            inf_births = [b for b, d in dgms[0] if d == np.inf]
            if finite_h0 and inf_births:
                max_finite = max(finite_h0)
                approx_inf = max_finite * 2.0
                lifespans_h0_full = finite_h0 + [approx_inf] * len(inf_births)
                pe_h0_full = _persistence_entropy(lifespans_h0_full)
            else:
                pe_h0_full = pe_h0

            vals = {
                "pe_h0": pe_h0,
                "pe_h0_full": pe_h0_full,
                "pe_h1": pe_h1,
                "num_features_h0": len(lifespans_h0),
                "num_features_h1": len(lifespans_h1),
                # PE / log2(m), in [0, 1]: log2(m) is PE's maximum over m bars. A
                # BLME normalisation; giotto-tda's normalize=True divides by
                # log2(total persistence) instead, which depends on the units.
                "pe_h0_norm": (pe_h0 / float(np.log2(len(lifespans_h0)))
                               if len(lifespans_h0) >= 2 else 0.0),
                "pe_h1_norm": (pe_h1 / float(np.log2(len(lifespans_h1)))
                               if len(lifespans_h1) >= 2 else 0.0),
            }
            for k, v in vals.items():
                results[f"_meta_layer_{l_idx}_{k}"] = v
                if l_idx in roles:
                    # The H0 family is _meta_: raw H0 entropy of N points is
                    # log2(N - 1) to three digits, and E / log2(n) passes every
                    # direct test but does not separate models (between/within
                    # SD 0.5-1.3). H1 entropy and its E / log2(n) twin remain.
                    pfx = "_meta_" if k in _H0_FAMILY else ""
                    results[f"{pfx}{roles[l_idx]}_layer_{k}"] = v

        # Summary: how much does H0 entropy change across layers? (_meta_ with
        # the rest of the H0 family.)
        pe_values = [results[f"_meta_layer_{l}_pe_h0"] for l in target_layers]
        if pe_values[0] > 0:
            results["_meta_pe_simplification_ratio"] = float(pe_values[-1] / pe_values[0])
        else:
            results["_meta_pe_simplification_ratio"] = 1.0

        return results
