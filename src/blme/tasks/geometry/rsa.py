"""
Representational Dissimilarity Analysis (RDA/RSA) — layer-wise comparison
of pairwise distance structures using Representational Dissimilarity Matrices.

While CKA measures similarity using kernel alignment, RSA/RDA directly 
compares the rank-order of pairwise distances between samples across layers.
This is the standard method in computational neuroscience for comparing
neural representations (Kriegeskorte et al., 2008).

Specifically, we compute the Representational Dissimilarity Matrix (RDM) 
at each layer (pairwise Euclidean distances between sample representations), 
then measure how well these RDMs correlate across layers using Spearman rank 
correlation. This reveals which layers share similar distance structures 
(i.e., similar "views" of the data).

References:
- "Representational Similarity Analysis — Connecting the Branches of Systems
  Neuroscience" (Kriegeskorte et al., Frontiers 2008)
- "Insights on Representational Similarity in Neural Networks with Canonical
  Correlation" (Morcos et al., NeurIPS 2018)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import as_numpy, collect_hidden_states, collected_layer_count
import numpy as np
import torch
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import logging
logger = logging.getLogger("blme")


@register_task("geometry_rsa")
class RepresentationalSimilarityTask(DiagnosticTask):
    """
    Computes Representational Similarity Analysis (RSA) across layers.
    
    For each layer, constructs a Representational Dissimilarity Matrix (RDM)
    from the pairwise Euclidean distances between sample representations.
    Then computes the Spearman rank correlation between RDMs of adjacent layers
    and between early/late layers.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representational Similarity Analysis (RSA)...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        num_samples = self.config.get("num_samples", 20)
        max_tokens_per_layer = self.config.get("max_tokens", 200)
        use_cache = self.config.get("use_cache", True)
        
        # Collect hidden states from ALL layers
        all_layers = None
        if cache is not None and cache.is_populated and use_cache:
            # The branch production TAKES (geometry_rsa is in core.py's
            # cache_hidden_tasks). Requesting the hook here is necessary but
            # NOT sufficient: the trim below would then delete the substituted
            # entry, so _last_is_block must be set too. Both changes are
            # required for the fix to have any effect end-to-end.
            all_layers = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, replace_final_norm=True)
            # Derived from whether the swap ACTUALLY fired, not from the
            # mere presence of a hooked tensor (a shape mismatch is
            # rejected inside get_hidden_states).
            _last_is_block = bool(getattr(cache, "_last_block_substituted", False))
            if all_layers is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. This task ran c
                # orrectly with NO cache and returned an error with an attention-onl
                # y one, i.e. it refused rather than self-collecting -- the defect a
                # stra RED-flagged on geometry_vendi_score. Found by EXECUTING the c
                # heck across all 51 tasks: the class is 15 sites, not the 5 a read-
                # only sweep reported (2026-09-09).
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if all_layers is None:
            _st = {}
            all_layers = collect_hidden_states(model, tokenizer, dataset,
                                           num_samples=num_samples,
                                           layer_idx="all",
                                           replace_final_norm=True, status=_st)
            # Derived, never assumed: the hook fires only when `get_layers`
            # locates the blocks and the captured shape matches the slot.
            _last_is_block = bool(_st.get("last_is_block", False))
        
        if not all_layers:
            return {"error": "No hidden states collected"}
            
        layer_indices = sorted(all_layers.keys())
        # The deepest layer is now the block's RAW output, recovered by a
        # forward hook (verified: ln_f(hooked) == HF hidden_states[-1] to
        # 0.0e+00 on gpt2/distilgpt2/pythia-70m). Previously this entry was
        # ln_f(b_{L-1}), which made rsa_adjacent_min the final NORM's own
        # transition in 16/16 gpt2 windows (0.5042 reported vs 0.7255 for the
        # true least-similar block pair, -30%; -42% under correlation
        # distance) and rsa_early_late off -21%/-30%. Deleting the entry
        # instead -- the previous fix -- discarded the deepest block, so the
        # hook is what actually resolves it: every RDM is now a block output
        # AND no block is lost.
        #
        # The cache serves the hooked entry too (it stores _last_block_raw and
        # accepts replace_final_norm), so this trim is dead on any cache built
        # by the current code; it survives only for one built before that.
        if not _last_is_block:
            _keep = collected_layer_count(model)
            if _keep and len(layer_indices) > _keep:
                layer_indices = layer_indices[:_keep]
        n_layers = len(layer_indices)
        
        if n_layers < 2:
            return {"error": "Need at least 2 layers for RSA"}
        
        # Subsample tokens consistently across layers
        min_tokens = min(len(all_layers[li]) for li in layer_indices)
        n_tokens = min(min_tokens, max_tokens_per_layer)

        # STRATIFIED subsample, not the first n_tokens.
        #
        # collect_hidden_states concatenates documents in order, so taking the
        # leading slice meant the RDM was built from document 0 plus part of
        # document 1 -- 18 of the 20 requested documents were silently
        # discarded. Window resampling on gpt2 then produced noise that EQUALS
        # OR EXCEEDS the entire 5-model between-model range for
        # rsa_adjacent_min and rsa_early_late_corrdist, i.e. the metric's
        # cross-model differences were not distinguishable from which window
        # happened to be taken (adversarial review, 2026-09-03).
        #
        # Evenly spaced indices over the whole collection keep the same token
        # count while spanning every document, and are deterministic.
        _avail = min_tokens
        if _avail > n_tokens:
            sel = np.linspace(0, _avail - 1, n_tokens).astype(int)
        else:
            sel = np.arange(_avail)
        
        if n_tokens < 3:
            return {"error": f"Too few tokens ({n_tokens}) for RSA"}
        
        # Compute RDMs for each layer, under BOTH dissimilarity measures.
        #
        # Kriegeskorte, Mur & Bandettini (2008) — the paper that defines RSA —
        # use CORRELATION distance (1 - Pearson r) as the RDM measure, and it
        # is the default in every RSA toolbox (rsatoolbox, PyRSA). Euclidean
        # distance was the only measure here until 2026-09-03.
        #
        # The difference is not cosmetic for transformer activations. Euclidean
        # RDM entries are dominated by per-token NORM, and the residual stream's
        # norm grows steeply and unevenly with depth (attention sinks and rogue
        # dimensions give a handful of tokens norms orders of magnitude above
        # the rest). Those tokens' rows then dominate the RDM at every layer,
        # which makes consecutive RDMs agree for a reason that has nothing to do
        # with representational geometry. Correlation distance centers and
        # scales each token vector first, so it measures the PATTERN across
        # dimensions -- which is what "representational similarity" means.
        #
        # Both are reported. Euclidean keeps the historic column names;
        # correlation distance is the `*_corrdist` family.
        # Counts rows the correlation-distance degeneracy guard had to jitter.
        # The guard never fired on any real model tested, so it is latent --
        # emitting the count is how a future regression becomes visible instead
        # of silently altering distances.
        _n_degenerate = [0]

        def _rdms(metric):
            out = {}
            for li in layer_indices:
                X = np.asarray(as_numpy(all_layers[li])[sel], dtype=np.float64)
                # (see _n_degenerate below)
                if metric == "correlation":
                    # Guard the degenerate rows correlation distance cannot
                    # handle: a token whose activation is constant across all
                    # dimensions has zero variance and yields NaN.
                    # EXACTLY constant rows only, and NO jitter.
                    #
                    # Correlation distance is exactly invariant to positive
                    # rescaling, so both an ABSOLUTE sd threshold and a FIXED
                    # jitter magnitude break it. Measured across a scale sweep:
                    # the reference stays 0.072408640645 throughout, while
                    # `sd <= 1e-12` + `1e-9` jitter marked 46 of 82 layer-rows
                    # at scale 1e-12 and erred by 0.0198, and at 1e-13 the
                    # jitter dominated the data entirely -- returning
                    # 0.999999673539 against a true 0.072408640645
                    # (adversarial review, 2026-09-04).
                    #
                    # A row with ANY variation has a well-defined correlation
                    # distance no matter how small that variation is, so the
                    # only genuine degeneracy is an exactly constant row.
                    # Those are rescaled to a canonical unit row instead of
                    # being jittered: it is deterministic, and it keeps them
                    # mutually identical, which is what a constant row IS.
                    _bad = np.ptp(X, axis=1) == 0.0
                    if np.any(_bad):
                        # PROPAGATE, do not invent. Pearson correlation
                        # distance is UNDEFINED for a constant row: scipy's
                        # `pdist(..., "correlation")` returns NaN there.
                        #
                        # CORRECTION (2026-09-04): an earlier note here claimed
                        # rsatoolbox also propagates NaN. It does NOT -- it
                        # MASKS the degenerate rows and returns a number
                        # (0.5428571428571429 on the fixture where BLME returns
                        # NaN). BLME follows scipy, which is the stricter and
                        # more honest convention for a quantity the definition
                        # leaves undefined, and the divergence from rsatoolbox
                        # is recorded rather than asserted away (independent
                        # Opus review, 2026-09-04).
                        #
                        # Mapping such rows to the arbitrary unit vector e_0
                        # manufactured an answer:
                        # a 2-layer, 4-token fixture where the reference is NaN
                        # returned correlation RSA 1.000000 (adversarial review
                        # round 2, 2026-09-04). Matching the reference means
                        # emitting NaN, and counting how many rows caused it.
                        X = X.copy().astype(np.float64)
                        X[_bad] = np.nan
                        _n_degenerate[0] += int(_bad.sum())
                out[li] = pdist(X, metric=metric)
            return out

        def _summarize(rdms):
            adjacent_corrs = []
            for i in range(n_layers - 1):
                rho, _ = spearmanr(rdms[layer_indices[i]],
                                   rdms[layer_indices[i + 1]])
                # PROPAGATE non-finite, do not map it to 0.0.
                #
                # 0.0 is a LEGAL RSA value -- it means "unrelated geometries" --
                # so substituting it for "undefined" fabricates a measurement
                # and then averages it in. It also defeated the round-3 repair
                # that makes a constant row emit NaN as scipy/rsatoolbox do:
                # with one shared constant row across two 5-token layers, 6 of
                # 10 pair cells stayed finite yet the summary reported a
                # concrete number (adversarial review round 3, 2026-09-04).
                adjacent_corrs.append(float(rho))
            rdm_first = rdms[layer_indices[0]]
            el, _ = spearmanr(rdm_first, rdms[layer_indices[-1]])
            fm, _ = spearmanr(rdm_first, rdms[layer_indices[n_layers // 2]])
            adj = np.array(adjacent_corrs, dtype=np.float64)
            _fin = adj[np.isfinite(adj)]
            # Summaries over the DEFINED pairs only, with the count of
            # undefined ones disclosed; NaN when none is defined.
            return {
                "adjacent_mean": float(np.mean(_fin)) if _fin.size else float("nan"),
                "adjacent_min": float(np.min(_fin)) if _fin.size else float("nan"),
                "adjacent_std": float(np.std(_fin)) if _fin.size else float("nan"),
                "n_undefined_pairs": int(adj.size - _fin.size),
                "early_late": float(el),
                "first_middle": float(fm),
                # `np.nanargmin`, not `np.argmin`. Over an UNFILTERED vector
                # `np.argmin([0.9, nan, 0.2, 0.5])` returns 1 -- it NAMES THE
                # NaN PAIR as the least-continuous transition, which is the one
                # pair for which no continuity was measured at all
                # (independent Opus review, 2026-09-04).
                "min_continuity_layer": (int(np.nanargmin(adj))
                                         if _fin.size else -1),
            }

        euc = _summarize(_rdms("euclidean"))
        cor = _summarize(_rdms("correlation"))

        return {
            "rsa_adjacent_mean": euc["adjacent_mean"],
            "_meta_n_degenerate_rows_jittered": int(_n_degenerate[0]),
            "_meta_last_entry_is_block_output": bool(_last_is_block),
            "rsa_adjacent_min": euc["adjacent_min"],
            "rsa_adjacent_std": euc["adjacent_std"],
            "rsa_early_late": euc["early_late"],
            "rsa_first_middle": euc["first_middle"],
            "_meta_rsa_min_continuity_layer": euc["min_continuity_layer"],
            # COMPUTED AND THEN DROPPED until 2026-09-04: `_summarize` returned
            # `n_undefined_pairs` but no caller forwarded it, so the round-3
            # NaN-propagation fix was invisible end to end -- the same
            # "fix present, effect absent" failure this campaign has now hit
            # three times (independent Opus review, 2026-09-04).
            "_meta_rsa_n_undefined_pairs": euc["n_undefined_pairs"],
            # Kriegeskorte-default correlation-distance RDMs.
            "rsa_adjacent_mean_corrdist": cor["adjacent_mean"],
            "rsa_adjacent_min_corrdist": cor["adjacent_min"],
            "rsa_adjacent_std_corrdist": cor["adjacent_std"],
            "rsa_early_late_corrdist": cor["early_late"],
            "rsa_first_middle_corrdist": cor["first_middle"],
            "_meta_rsa_min_continuity_layer_corrdist": cor["min_continuity_layer"],
            "_meta_rsa_n_undefined_pairs_corrdist": cor["n_undefined_pairs"],
        }
