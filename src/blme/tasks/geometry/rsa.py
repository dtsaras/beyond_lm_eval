"""
Representational Similarity Analysis (RSA) across layers.

At each layer a Representational Dissimilarity Matrix (RDM) of pairwise
distances between the same sampled tokens is built, and RDMs are compared by
Spearman rank correlation (Kriegeskorte, Mur & Bandettini 2008, Step 4). Two
dissimilarities are reported: Euclidean, and Kriegeskorte's correlation
distance 1 - r (``*_corrdist``).

References:
- "Representational Similarity Analysis — Connecting the Branches of Systems
  Neuroscience" (Kriegeskorte et al., Frontiers 2008)
- "Blackbox meets blackbox: Representational Similarity and Stability Analysis
  of Neural Language Models and Brains" (Abnar et al., BlackboxNLP 2019)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .._tokens import (hidden_states_for_task, massive_token_share,
                       position0_share, select_rows)
from .utils import as_numpy, collected_layer_count
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
import logging
logger = logging.getLogger("blme")


@register_task("geometry_rsa")
class RepresentationalSimilarityTask(DiagnosticTask):
    """
    Computes Representational Similarity Analysis (RSA) across layers.

    For each layer, constructs an RDM from pairwise distances between sampled
    token representations, then computes the Spearman rank correlation between
    RDMs of adjacent layers and between early/late layers.

    Rows follow ``TOKEN_POLICY``: position 0 is dropped (RSA did not originate
    on LLMs, and sink rows take the largest distances, so they occupy the top
    ranks of every RDM; tests/tasks/invariance/test_rsa.py).
    """
    TOKEN_POLICY = "exclude_pos0"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representational Similarity Analysis (RSA)...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 20)
        max_tokens_per_layer = self.config.get("max_tokens", 200)
        use_cache = self.config.get("use_cache", True)

        per_doc, _info = hidden_states_for_task(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            replace_final_norm=True, layer_idx="all", use_cache=use_cache,
            per_sample=True)
        _last_is_block = bool(_info["last_is_block"])

        if not per_doc:
            return {"error": "No hidden states collected"}

        layer_indices = sorted(per_doc.keys())
        # Only a collection without the deepest-block hook still ends in the
        # post-norm entry; that entry is trimmed.
        if not _last_is_block:
            _keep = collected_layer_count(model)
            if _keep and len(layer_indices) > _keep:
                layer_indices = layer_indices[:_keep]
        n_layers = len(layer_indices)

        if n_layers < 2:
            return {"error": "Need at least 2 layers for RSA"}

        all_layers, _rows_meta = {}, None
        for li in layer_indices:
            all_layers[li], _rows_meta = select_rows(per_doc[li], self.TOKEN_POLICY)
        min_tokens = min(len(all_layers[li]) for li in layer_indices)
        n_tokens = min(min_tokens, max_tokens_per_layer)

        # Evenly spaced rows over the whole collection, so every document
        # contributes (a leading slice covered one or two documents).
        _avail = min_tokens
        if _avail > n_tokens:
            sel = np.linspace(0, _avail - 1, n_tokens).astype(int)
        else:
            sel = np.arange(_avail)

        if n_tokens < 3:
            return {"error": f"Too few tokens ({n_tokens}) for RSA"}

        # Rows that make correlation distance undefined (exactly constant).
        _n_degenerate = [0]

        def _rdms(metric):
            out = {}
            for li in layer_indices:
                X = np.asarray(as_numpy(all_layers[li])[sel], dtype=np.float64)
                if metric == "correlation":
                    # An exactly constant row has no correlation distance:
                    # NaN as scipy returns, counted, never jittered (a jitter
                    # breaks scale invariance; rsatoolbox masks such rows).
                    _bad = np.ptp(X, axis=1) == 0.0
                    if np.any(_bad):
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
                # Non-finite stays NaN: 0.0 is a legal RSA value.
                adjacent_corrs.append(float(rho))
            rdm_first = rdms[layer_indices[0]]
            el, _ = spearmanr(rdm_first, rdms[layer_indices[-1]])
            fm, _ = spearmanr(rdm_first, rdms[layer_indices[n_layers // 2]])
            adj = np.array(adjacent_corrs, dtype=np.float64)
            _fin = adj[np.isfinite(adj)]
            # Summaries over the defined pairs only; the undefined count is
            # reported, and nanargmin never names an undefined pair.
            return {
                "adjacent_mean": float(np.mean(_fin)) if _fin.size else float("nan"),
                "adjacent_min": float(np.min(_fin)) if _fin.size else float("nan"),
                "adjacent_std": float(np.std(_fin)) if _fin.size else float("nan"),
                "n_undefined_pairs": int(adj.size - _fin.size),
                "early_late": float(el),
                "first_middle": float(fm),
                "min_continuity_layer": (int(np.nanargmin(adj))
                                         if _fin.size else -1),
            }

        euc = _summarize(_rdms("euclidean"))
        cor = _summarize(_rdms("correlation"))

        _mid = per_doc[layer_indices[n_layers // 2]]
        out = {
            "rsa_adjacent_mean": euc["adjacent_mean"],
            "_meta_n_degenerate_rows_jittered": int(_n_degenerate[0]),
            "_meta_last_entry_is_block_output": bool(_last_is_block),
            "rsa_adjacent_min": euc["adjacent_min"],
            "rsa_adjacent_std": euc["adjacent_std"],
            "rsa_early_late": euc["early_late"],
            "rsa_first_middle": euc["first_middle"],
            "_meta_rsa_min_continuity_layer": euc["min_continuity_layer"],
            "_meta_rsa_n_undefined_pairs": euc["n_undefined_pairs"],
            "rsa_adjacent_mean_corrdist": cor["adjacent_mean"],
            "rsa_adjacent_min_corrdist": cor["adjacent_min"],
            "rsa_adjacent_std_corrdist": cor["adjacent_std"],
            "rsa_early_late_corrdist": cor["early_late"],
            "rsa_first_middle_corrdist": cor["first_middle"],
            "_meta_rsa_min_continuity_layer_corrdist": cor["min_continuity_layer"],
            "_meta_rsa_n_undefined_pairs_corrdist": cor["n_undefined_pairs"],
            "_meta_n_layers": int(n_layers),
            "_meta_n_tokens_used": int(len(sel)),
            # Sink diagnostics on the full (inclusive) middle-layer cloud.
            "_meta_position0_share": position0_share(_mid),
            "_meta_massive_token_share": massive_token_share(_mid),
        }
        out.update(_rows_meta.as_meta())
        return out
