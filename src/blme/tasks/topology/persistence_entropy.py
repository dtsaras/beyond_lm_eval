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
from ._points import fallback_dataset, last_token_points, point_meta, role_layers, rows_policy
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


@register_task("topology_persistence_entropy")
class PersistenceEntropyTask(DiagnosticTask):
    """
    Computes Persistence Entropy at early, middle, and late layers.

    For each layer, constructs a persistence diagram via Vietoris-Rips
    and computes the Shannon entropy of the H0 and H1 lifespan distributions.
    """
    TOKEN_POLICY = "last_per_sequence"

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
        for l_idx, points in layer_reps.items():
            # Centre (VR is translation invariant), then a power-of-two rescale
            # (exact) when the magnitude is out of float32's comfortable range,
            # because ripser computes in float32.
            if points:
                _P = np.stack(points).astype(np.float64)
                _P = _P - _P.mean(axis=0, keepdims=True)
                _pm = float(np.max(np.abs(_P))) if _P.size else 0.0
                if _pm > 0 and np.isfinite(_pm) and not (1e-6 < _pm < 1e6):
                    import math as _math
                    _e = int(max(-126, min(127, round(_math.log2(_pm)))))
                    points = list(np.ldexp(_P, -_e))
                else:
                    points = list(_P)
            data = np.array(points)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(data, maxdim=1)['dgms']

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

            results[f"layer_{l_idx}_pe_h0"] = pe_h0
            results[f"layer_{l_idx}_pe_h0_full"] = pe_h0_full
            results[f"layer_{l_idx}_pe_h1"] = pe_h1
            results[f"layer_{l_idx}_num_features_h0"] = len(lifespans_h0)
            results[f"layer_{l_idx}_num_features_h1"] = len(lifespans_h1)
            # PE / log2(m), in [0, 1]: log2(m) is PE's maximum over m bars. A
            # BLME normalisation; giotto-tda's normalize=True divides by
            # log2(total persistence) instead, which depends on the units.
            results[f"layer_{l_idx}_pe_h0_norm"] = (
                pe_h0 / float(np.log2(len(lifespans_h0)))
                if len(lifespans_h0) >= 2 else 0.0
            )
            results[f"layer_{l_idx}_pe_h1_norm"] = (
                pe_h1 / float(np.log2(len(lifespans_h1)))
                if len(lifespans_h1) >= 2 else 0.0
            )

        # Summary: how much does topological entropy change across layers?
        pe_values = [results[f"layer_{l}_pe_h0"] for l in target_layers]
        if pe_values[0] > 0:
            results["pe_simplification_ratio"] = float(pe_values[-1] / pe_values[0])
        else:
            results["pe_simplification_ratio"] = 1.0

        return results
