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
    # Eq. (4) is exactly scale-invariant: every distance enters only through
    # the ratio d_i/d_k. An ABSOLUTE floor therefore destroys that invariance.
    # A five-distance anchor gave 1.53388451231901 at scale 1 but 0.0 at
    # 1e-11, and a 100-point cloud fell out of the estimator entirely where
    # official mle_batch still returned 6.99074513291813 (adversarial review,
    # 2026-09-04). Only an exactly zero radius is degenerate.
    d_k = distances[-1]
    if not (d_k > 0.0):
        return 0.0
    
    ratios = distances / d_k
    # Exact-duplicate neighbor (d_i = 0): official mle_batch yields
    # -k / (-inf) = 0.0; the old 1e-10 clamp returned ~k/23 instead, which
    # then passed the lid > 0 filter and dragged lid_min/lid_mean on
    # duplicate-heavy clouds (audit 2026-07-16). Return 0.0 to match the
    # reference (0.0 estimates are excluded downstream).
    if np.any(ratios <= 0):
        return 0.0
    log_ratios = np.log(ratios)
    sum_log = np.sum(log_ratios)

    # sum_log is itself scale-free (a sum of log-ratios) and is <= 0, with
    # equality only when every neighbour sits exactly on the k-th shell. Test
    # that exactly; a magnitude threshold would discard genuine large-LID
    # estimates instead.
    if sum_log == 0.0:
        return float("nan")

    return -k / sum_log


def _compute_lid_for_matrix(X, k, max_queries=500, seed: int = 0):
    """Compute LID statistics for a single (N, D) data matrix.

    Returns a dict of statistics or *None* if there are too few points.
    A fixed ``seed`` ensures reproducibility across reruns — the
    previous ``np.random.choice`` call used the global RNG, so reruns
    would draw different query subsets and yield different LID
    statistics (paper-grade numbers should be deterministic).
    """
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]

    if len(X) < k + 1:
        return None

    n_queries = min(len(X), max_queries)
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(len(X), size=n_queries, replace=False)

    # Neighbour distances via the float64 Gram identity
    #     ||x - q||^2 = ||x||^2 - 2 q.x + ||q||^2
    # evaluated as one BLAS GEMM per query block, with the k+1 smallest picked
    # by argpartition (O(N)) instead of sorting all N distances.
    #
    # This replaced a per-query Python loop of
    #     np.linalg.norm(X - X[qi], axis=1)  ->  np.sort(...)
    # on 2026-08-19. That loop allocated a full (N, D) temporary per query and
    # was the single most expensive thing in the whole suite: 79% of total
    # runtime, 2846s on gpt2-774m alone, and an estimated 5.6h on a 64-layer
    # 32B checkpoint — past any workable per-task timeout.
    #
    # The rewrite is both faster and MORE accurate, because the old loop
    # accumulated in float32. Against an exact float64 reference on a
    # (4000, 768) cloud, max relative error on the returned LID estimates:
    #     old fp32 loop   1.8e-05
    #     this kernel     7.0e-14
    # Consequence for provenance: agreement with the official mle_batch (which
    # is itself a float32 implementation) loosens from ~1.2e-05 to ~3e-05,
    # while agreement with the true value improves by nine orders of
    # magnitude. VERIFICATION_STATUS.md records both figures.
    X64 = X.astype(np.float64, copy=False)
    # CENTRE FIRST. Distances are translation-invariant, but the Gram identity
    # ||x-q||^2 = ||x||^2 - 2q.x + ||q||^2 is not: once the cloud sits far from
    # the origin, ||x||^2 dwarfs the separation and the identity's rounding
    # floor swamps the ordering the selector depends on. On a cloud whose
    # float32 pairwise differences are EXACTLY unchanged by translation,
    # shifting by +10000 made 31 of 100 neighbourhood rows differ from full
    # search and moved lid_mean 77.372506032887 -> 76.176419631200 (-1.546%),
    # with -11.49% at +16000 (adversarial review, 2026-09-04). Centring costs
    # one pass and removes the leading term of any such shift.
    # NO CENTRING -- removed 2026-09-07, verified on both axes.
    #
    # It was added 2026-09-04 because a +10000 shift moved lid_mean -1.546%.
    # That is no longer true: this function now VERIFIES each row's candidate
    # set against a Gram lower bound and escalates when the check fails, so
    # translation is handled by the escalation, not by the centring. Measured
    # with and without, at +10000: **12.058202777345784 both ways, identical**.
    #
    # And the centring actively destroys wide-dynamic-range clouds, because the
    # ULP of the centroid becomes the resolution floor. On the lattices
    # {j*2^-80} U {1+j/32} it collapsed 21 distinct points onto one, producing
    # 21 spurious ZERO estimates that halve the headline -- BLME 0.5081480494776388
    # against the executed released `mle_batch` 1.0162960989552776, exactly 50%
    # low (gpt-6-astra round 10, 2026-09-07). Without it: 0 zero estimates, the
    # 21 rows escalate to exact search, and `_meta_lid_mean_positive_only` --
    # which was already correct at 1.6185015223582724 both ways -- stops being
    # the only trustworthy column.
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

        # The Gram identity is used ONLY to SELECT candidate neighbours; the
        # distances that reach the estimator are recomputed exactly by direct
        # subtraction (adversarial review, 2026-09-02).
        #
        # Why: ||x-q||^2 = ||x||^2 - 2q.x + ||q||^2 cancels catastrophically
        # for close pairs. For a genuinely distinct neighbour one float32 ULP
        # away, ||x||^2 ~ 1e2 while the true d^2 ~ 1e-16 — below the identity's
        # own rounding floor, so it can return a negative number. Clamping
        # turns that into a false duplicate and silently discards a valid
        # estimate; the reference implementation returns a finite LID there.
        # Selection only needs the ORDER to be right, which survives the error,
        # so we take a few extra candidates and then measure them properly.
        # Cost is (k+margin)*D per query instead of N*D — negligible.
        # VERIFIED selection, not a fixed margin. Centring removes the bulk of
        # the Gram error, but nothing bounds it in general, so each row's
        # candidate set is CHECKED and the row is escalated (to a wider take,
        # ultimately to full exact search) whenever the check fails. The check
        # is exact: if the k-th exact distance within the candidate set is no
        # larger than the smallest EXCLUDED candidate's Gram lower bound, no
        # true neighbour can have been missed.
        margin = 8
        take = min(k + 1 + margin, d2.shape[1])
        part = np.argpartition(d2, take - 1, axis=1)[:, :take]

        nearest = np.empty((len(q_idx), k + 1), dtype=np.float64)
        for r in range(len(q_idx)):
            cand = part[r]
            exact = np.linalg.norm(X64[cand] - X64[q_idx[r]], axis=1)
            exact.sort()
            # Smallest Gram distance among the candidates NOT taken. Every
            # excluded point has d2 at least this large up to the identity's
            # error, so if our k-th exact distance is below it the set is safe.
            if take < d2.shape[1]:
                # ROUNDING ENVELOPE on the Gram bound. `d2` comes from the
                # cancellation-prone identity, so the smallest excluded value
                # is itself uncertain; subtracting a bound on that uncertainty
                # makes the check conservative instead of merely plausible.
                # Without it, two separated 512-dimensional clusters with
                # one-or-two-ULP perturbations defeated the check (adversarial
                # review round 2, 2026-09-04). The identity's absolute error is
                # O(eps * (||x||^2 + ||q||^2 + 2|q.x|)); 8*eps*max_term is a
                # generous envelope over that.
                _excl = np.partition(d2[r], take)[take]
                _scale = float(sq_norms.max() + sq_norms[q_idx[r]])
                # The envelope must scale with the DOT-PRODUCT LENGTH. A
                # constant 8*eps is not a bound on a BLAS dot over D terms:
                # on 100 float32 points in 2048 dimensions only 50 of 100 rows
                # were correctly verified (adversarial review round 3,
                # 2026-09-04). The standard bound for a length-D dot product
                # is D*eps/(1-D*eps) times the sum of |terms|; 4*D*eps is a
                # generous majorant of that at these sizes.
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
            # RETAIN zero estimates. An exact-duplicate query gives
            # -k/(-inf) = 0.0 under the released mle_batch, and the released
            # code keeps it in the mean. Dropping it made BLME report
            # 4.608033135762 where the reference gives 4.580384936947 on the
            # same GPT-2 population, +0.604% with 3 of 500 rows affected
            # (adversarial review round 2, 2026-09-04). The positive-only
            # figure is still emitted, under its own name.
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
                # `_meta_`: flattening this nested dict produces one column per
                # ABSOLUTE layer index, so the same name means a different
                # depth on models of different sizes. Same defect and same fix
                # as `topology_homology` and `geometry_intrinsic_dim`.
                result["_meta_layer_metrics"] = per_layer
                # NO `lid_last_layer`. `result` is already
                # `per_layer[last_key]`, so `lid_mean` IS the deepest layer's
                # value -- adding `lid_last_layer` created an exact duplicate,
                # bit-for-bit on 10/10 models, with both feature-eligible
                # (independent Opus review, 2026-09-05). A fix for a duplicate
                # column that introduces a duplicate column.
                _finite = [v for v in lid_mean_per_layer if np.isfinite(v)]
                if _finite:
                    _n = len(lid_mean_per_layer)
                    result["lid_first_layer"] = float(lid_mean_per_layer[0])
                    result["lid_mid_layer"] = float(lid_mean_per_layer[_n // 2])
                # The layerwise and single-layer modes return DIFFERENT
                # ESTIMANDS under the same key names, and the gap is not small:
                # on pythia-160m `lid_mean` is 23,714.2 layerwise against 18.66
                # single-layer on the same weights -- a factor of 1,271 coming
                # from layer 11 alone, with layers 0-10 all inside [14.9, 17.7].
                # A caller cannot tell the two apart from the output, so the
                # mode is now on the record and a blown-up layer is flagged.
                result["_meta_lid_mode"] = "layerwise"
                if _finite:
                    _med = float(np.median(_finite))
                    result["_meta_lid_max_over_median"] = (
                        float(max(_finite) / _med) if _med > 0 else float("nan"))
                return result

        # --- Single-layer mode (last layer) ---
        X = None
        if cache is not None and cache.is_populated and use_cache:
            # post-final-norm=False, DELIBERATELY: the single-layer mode
            # reports the LID of the model's OUTPUT representation. The
            # layerwise mode above opts IN, because that one is a profile.
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples,
                                        replace_final_norm=False)
            if X is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. A cache
                # filled for an attention-only consumer returns None here and the
                # dereference below raised, destroying the task rather than
                # degrading it. Found by EXECUTING this check against a real
                # attention-only cache across all 51 tasks -- the round-12 sweep
                # of this same class was done by READING and put the population
                # at 2 sites; running it puts it at 9 (codex round 13 +
                # execution sweep, 2026-09-09).
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if X is None:
            # EVERY token, matching the cache route (see the note in
            # collect_hidden_states). LID is a local-density estimator, so the
            # point count is part of the estimand, not an implementation
            # detail.
            X = collect_hidden_states(model, tokenizer, dataset,
                                      num_samples=num_samples,
                                      max_tokens_per_doc=None,
                                      replace_final_norm=False)

        m = _compute_lid_for_matrix(X, k)
        if m is None:
            return {"error": f"Too few valid samples for LID with k={k}"}
        # See the note in the layerwise branch: the two modes ship different
        # estimands under the same key names, so which one produced a number
        # must be readable from the output.
        m["_meta_lid_mode"] = "single_layer"
        return m
