"""
Local Intrinsic Dimensionality (LID) — per-sample local dimension estimates.

Unlike global Two-NN (which gives a single number for an entire point cloud),
LID measures the local dimensionality around each individual sample point.
This reveals whether certain inputs live on lower-dimensional sub-manifolds
(e.g., compressed representations) or higher ones (richer, more expressive encodings).

References:
- "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality"
  (Ma et al., ICLR 2018)
- "Maximum Likelihood Estimation of Intrinsic Dimension"
  (Levina & Bickel, NeurIPS 2004 — MLE-based LID)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")


def _lid_mle(distances, k):
    """Maximum Likelihood Estimator for Local Intrinsic Dimensionality.
    
    Given sorted distances from a query to its k nearest neighbors,
    estimates the local dimensionality using the -k MLE variant of Ma et al. 2018 (Eq. 4), derived from Levina & Bickel (2004):
        LID = -k / sum_{i=1}^{k} log(d_i / d_k)
    
    Args:
        distances: sorted distances to k nearest neighbors (excluding self), shape (k,)
        k: number of neighbors
        
    Returns:
        LID estimate (float)
    """
    # Eq. (4) is scale-invariant (distances enter only as d_i/d_k), so only an
    # exactly zero radius is degenerate (test_scale_invariance.py).
    d_k = distances[-1]
    if not (d_k > 0.0):
        return 0.0
    
    ratios = distances / d_k
    # Exact-duplicate neighbour (d_i = 0): the released mle_batch gives
    # -k/(-inf) = 0.0 (test_b8_new_metrics_parity.py).
    if np.any(ratios <= 0):
        return 0.0
    log_ratios = np.log(ratios)
    sum_log = np.sum(log_ratios)

    # sum_log <= 0, with equality only when every neighbour sits on the k-th
    # shell; tested exactly, a magnitude threshold would drop large LIDs.
    if sum_log == 0.0:
        return float("nan")

    return -k / sum_log


def _compute_lid_for_matrix(X, k, max_queries=500, seed: int = 0):
    """Compute LID statistics for a single (N, D) data matrix.

    Returns a dict of statistics or *None* if there are too few points.
    A fixed ``seed`` makes the query draw reproducible
    (test_round4_fixes.py::test_lid_is_deterministic_across_reruns).
    """
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if len(X) < k + 1:
        return None

    n_queries = min(len(X), max_queries)
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(len(X), size=n_queries, replace=False)

    # Candidates are SELECTED with the float64 Gram identity (one GEMM per
    # query block) and their distances recomputed exactly; each row's set is
    # verified against a Gram lower bound and escalated to full search when
    # the check fails, which also handles translated clouds (no centring: the
    # centroid's ULP would become the resolution floor).
    X64 = X.astype(np.float64, copy=False)
    sq_norms = np.einsum("ij,ij->i", X64, X64)

    lid_estimates = []
    lid_positive = []
    _n_escalated = [0]
    block = 128  # bounds the (block, N) distance matrix
    for start in range(0, n_queries, block):
        q_idx = query_indices[start:start + block]
        Q = X64[q_idx]
        d2 = sq_norms[None, :] - 2.0 * (Q @ X64.T)
        d2 += np.einsum("ij,ij->i", Q, Q)[:, None]

        # The exact k-th distance within the candidates must not exceed the
        # smallest excluded candidate's Gram lower bound; otherwise escalate.
        margin = 8
        take = min(k + 1 + margin, d2.shape[1])
        part = np.argpartition(d2, take - 1, axis=1)[:, :take]

        nearest = np.empty((len(q_idx), k + 1), dtype=np.float64)
        for r in range(len(q_idx)):
            cand = part[r]
            exact = np.linalg.norm(X64[cand] - X64[q_idx[r]], axis=1)
            exact.sort()
            if take < d2.shape[1]:
                # Rounding envelope on the Gram bound: a length-D dot product
                # carries O(D eps) relative error, 4*D*eps is a majorant.
                _excl = np.partition(d2[r], take)[take]
                _scale = float(sq_norms.max() + sq_norms[q_idx[r]])
                _err = 4.0 * X64.shape[1] * np.finfo(np.float64).eps * _scale
                _bound = float(np.sqrt(max(_excl - _err, 0.0)))
                if exact[k] > _bound:
                    exact = np.linalg.norm(X64 - X64[q_idx[r]], axis=1)
                    exact.sort()
                    _n_escalated[0] += 1
            nearest[r] = exact[:k + 1]

        for row in nearest[:, 1:k + 1]:
            lid = _lid_mle(row, k)
            if np.isnan(lid):
                continue
            # Zero estimates (exact-duplicate queries) are kept in the mean, as
            # the released mle_batch does; the positive-only mean is _meta_.
            lid_estimates.append(lid)
            if lid > 0:
                lid_positive.append(lid)

    if not lid_estimates:
        return None

    lid_arr = np.array(lid_estimates)
    return {
        "lid_mean": float(np.mean(lid_arr)),
        "lid_std": float(np.std(lid_arr)),
        "lid_min": float(np.min(lid_arr)),
        "lid_max": float(np.max(lid_arr)),
        "lid_median": float(np.median(lid_arr)),
        "_meta_num_estimates": len(lid_estimates),
        "_meta_num_zero_estimates": len(lid_estimates) - len(lid_positive),
        "_meta_num_escalated_rows": int(_n_escalated[0]),
        "_meta_lid_mean_positive_only": (float(np.mean(lid_positive))
                                         if lid_positive else float("nan")),
    }


@register_task("geometry_lid")
class LocalIntrinsicDimensionalityTask(DiagnosticTask):
    """
    Computes per-sample Local Intrinsic Dimensionality (LID) using the
    **Ma et al. 2018 Eq. (4)** MLE — the ``-k`` (biased) variant,

        LID = -[ (1/k) * sum_i log(r_i / r_k) ]^-1

    NOT the Levina & Bickel (2004) estimator, which normalises by (k-1) and
    which this docstring previously named. The two differ by exactly the factor
    k/(k-1): at the shipped k=20 this estimator reads **+5.26%** above the
    Levina-Bickel (k-1) MLE and **+11.1%** above the (k-2) bias-corrected form.
    Levina & Bickel remains the origin of the maximum-likelihood derivation and
    is cited as such, but the implemented formula is Ma et al.'s.

    Outputs the mean, std, min, and max LID across all samples for
    the specified layer(s).  When ``layerwise=True`` (and cache is
    available), computes LID at every layer.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Local Intrinsic Dimensionality (LID) Analysis...")
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        k = self.config.get("k", 20)
        num_samples = self.config.get("num_samples", 50)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode ---
        if layerwise and cache is not None and cache.is_populated and use_cache:
            all_layers = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
                replace_final_norm=True)
            if all_layers:
                per_layer = {}
                lid_mean_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _compute_lid_for_matrix(all_layers[li], k)
                    if m is not None:
                        per_layer[f"layer_{li}"] = m
                        lid_mean_per_layer.append(m["lid_mean"])
                    else:
                        lid_mean_per_layer.append(float("nan"))
                last_key = f"layer_{max(all_layers.keys())}"
                result = dict(per_layer.get(last_key, {}))
                result["layer_lid_mean"] = lid_mean_per_layer
                # _meta_: one column per absolute layer index.
                result["_meta_layer_metrics"] = per_layer
                # lid_mean already is the deepest layer, so no lid_last_layer.
                _finite = [v for v in lid_mean_per_layer if np.isfinite(v)]
                if _finite:
                    _n = len(lid_mean_per_layer)
                    result["lid_first_layer"] = float(lid_mean_per_layer[0])
                    result["lid_mid_layer"] = float(lid_mean_per_layer[_n // 2])
                # The two modes are different estimands under one key set:
                # _meta_lid_mode says which one ran.
                result["_meta_lid_mode"] = "layerwise"
                if _finite:
                    _med = float(np.median(_finite))
                    result["_meta_lid_max_over_median"] = (
                        float(max(_finite) / _med) if _med > 0 else float("nan"))
                return result

        # --- Single-layer mode (last layer) ---
        X = None
        if cache is not None and cache.is_populated and use_cache:
            # post-final-norm=False, DELIBERATELY: the single-layer mode reports
            # the model's output representation; the layerwise mode opts in.
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples,
                                        replace_final_norm=False)
            if X is None:
                # A populated cache without hidden states: self-collect.
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if X is None:
            # Every token, matching the cache route: the point count is part of
            # the estimand.
            X = collect_hidden_states(model, tokenizer, dataset,
                                      num_samples=num_samples,
                                      max_tokens_per_doc=None,
                                      replace_final_norm=False)

        m = _compute_lid_for_matrix(X, k)
        if m is None:
            return {"error": f"Too few valid samples for LID with k={k}"}
        m["_meta_lid_mode"] = "single_layer"
        return m
