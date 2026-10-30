"""
Positional Attention Decay (RoPE Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates the geometric degradation of relative positional embeddings
(like RoPE) by computing the Spearman rank correlation between the
relative discrete distance of two tokens and the magnitude of their
attention connection.

A structurally intact context window will show a strong negative correlation
(closer tokens have higher attention in positional/local heads). If this
correlation collapses, the model's positional geometry is degraded.

References:
- Su et al. 2021, RoFormer (arXiv:2104.09864): motivates the premise
  (RoPE's logit bound decays with relative distance); it defines no
  statistic of trained attention.
"""

from __future__ import annotations  # PEP 563: keep `float | None` lazy on py3.9

import torch
import numpy as np
from scipy.stats import spearmanr

from .._precision import effective_dtype_meta, fp32_compute
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


def _row_distance_correlation(attn_row: np.ndarray) -> float | None:
    """Spearman correlation for one causal query row.

    Constant rows have no distance preference; treat them as the null
    correlation instead of letting pooled row lengths manufacture decay.
    """
    if attn_row.size < 2:
        return None
    # Only an EXACTLY flat row is the null: attention rows are tiny, and any
    # tolerance (np.allclose, ptp <= 1e-12) also caught rows with small but
    # strictly ordered spread, by a model-dependent amount.
    if float(np.ptp(attn_row)) == 0.0:
        return 0.0
    distances = np.arange(attn_row.size, 0, -1, dtype=np.float64)
    corr, _ = spearmanr(distances, attn_row)
    if np.isnan(corr):
        return None
    return float(corr)


@register_task("geometry_positional_decay")
class PositionalAttentionDecayTask(DiagnosticTask):
    """
    Computes the Spearman correlation between target-source token distance
    and attention probability to measure the structural integrity of
    positional encodings.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Positional Attention Decay Analysis...")
        num_samples = self.config.get("num_samples", 5)

        # We need reasonably long sequences to measure positional decay gracefully
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device

        # Per-layer correlations: {layer_idx: [corr_sample1, corr_sample2, ...]}
        from collections import defaultdict
        layer_correlations = defaultdict(list)
        layer_correlations_nosink = defaultdict(list)
        layer_correlations_pooled = defaultdict(list)
        sink_mass = []
        _n_nonfinite_rows = 0
        _n_flat_rows = 0

        # The attention probabilities must be PRODUCED in fp32 (P5): bf16's
        # 8-bit mantissa makes about a fifth of pythia-70m's rows exactly flat.
        # fp32_compute promotes at every model size (the old <= 2e9-parameter
        # gate made the dtype a function of size) and restores parameters AND
        # buffers (tests/tasks/test_positional_decay_protocol.py).
        _dtype_ctx = fp32_compute(model)
        _dtype_info = _dtype_ctx.__enter__()
        try:
          with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

                seq_len = inputs.input_ids.shape[1]
                if seq_len < 4:
                   continue

                out = model(**inputs, output_attentions=True)
                # SDPA returns a tuple of the right length whose entries are all
                # None: name the attention backend, not the corpus.
                if (out.attentions is None or len(out.attentions) == 0
                        or all(a is None for a in out.attentions)):
                    return {"error": (
                        "Model returned no attention matrices. This is almost "
                        "always the attention backend, not the corpus: SDPA / "
                        "flash-attention never materialise the probabilities. "
                        "Load with attn_implementation='eager' to measure "
                        "positional decay."),
                        "_meta_attn_implementation": str(
                            getattr(getattr(model, "config", None),
                                    "_attn_implementation", "unknown")),
                    }

                for li, attn_entry in enumerate(out.attentions):
                    if attn_entry is None:
                        continue
                    attn_matrix = attn_entry[0]  # (num_heads, seq_len, seq_len)
                    # Each query row and head separately: pooling triangular
                    # causal entries confounds distance with row normalisation.
                    # .float() before any comparison.
                    a = attn_matrix.float().cpu().numpy()
                    _n_nonfinite_rows += int(
                        (~np.isfinite(a)).any(axis=-1).sum())
                    for head_idx in range(a.shape[0]):
                        # POOLED variant (one Spearman over every causal
                        # {query, key} pair, diagonal included, from q = 1), as
                        # the manuscript writes it; reported beside the rowwise
                        # statistic because it reads decay under a uniform null.
                        _pool_v, _pool_d = [], []
                        for _q in range(1, seq_len):
                            _fr = a[head_idx, _q, :_q + 1]
                            _pool_v.append(_fr)
                            _pool_d.append(_q - np.arange(_fr.size))
                        for query_idx in range(2, seq_len):
                            _row = a[head_idx, query_idx, :query_idx]
                            # The same row set for both twins: the sink-excluded
                            # one needs >= 3 keys (it drops one).
                            if _row.size <= 2:
                                continue
                            if float(np.ptp(_row)) == 0.0:
                                _n_flat_rows += 1
                            corr = _row_distance_correlation(_row)
                            if corr is not None:
                                layer_correlations[li].append(corr)
                            # Position 0 is the attention sink and also the
                            # maximum-distance key of every causal row, so the
                            # inclusive statistic mixes decay with sink strength.
                            if True:
                                _rs = float(_row.sum())
                                if _rs > 0:
                                    sink_mass.append(float(_row[0]) / _rs)
                                corr_ns = _row_distance_correlation(_row[1:])
                                if corr_ns is not None:
                                    layer_correlations_nosink[li].append(corr_ns)
                        if _pool_v:
                            _pv = np.concatenate(_pool_v)
                            _pd = np.concatenate(_pool_d).astype(np.float64)
                            if np.ptp(_pv) > 0 and np.ptp(_pd) > 0:
                                _rho = float(spearmanr(_pd, _pv).statistic)
                                if np.isfinite(_rho):
                                    layer_correlations_pooled[li].append(_rho)

        finally:
            # Restores on every exit, including the early error return above.
            _dtype_ctx.__exit__(None, None, None)

        if not layer_correlations:
            return {"error": "Could not compute positional correlations (sequences too short or nan)."}

        # Compute per-layer means
        layer_means = {}
        for li in sorted(layer_correlations.keys()):
            layer_means[f"layer_{li}"] = float(np.mean(layer_correlations[li]))

        all_corrs = [c for corrs in layer_correlations.values() for c in corrs]
        _pooled_all = [c for v in layer_correlations_pooled.values() for c in v]

        out = {
            # The ROWWISE statistic: exactly 0 under uniform causal attention
            # at every length, where the pooled one reads -0.43 to -0.47
            # (test_positional_decay_uniform_causal_attention_is_null).
            "mean_positional_decay_correlation": float(np.mean(all_corrs)),
            "mean_positional_decay_correlation_pooled": (
                float(np.mean(_pooled_all)) if _pooled_all else float("nan")),
            "layer_positional_decay": layer_means,
        }

        # Sink-excluded twin: the same Spearman over keys 1..q.
        if layer_correlations_nosink:
            _ns_layer = {
                f"layer_{li}": float(np.mean(layer_correlations_nosink[li]))
                for li in sorted(layer_correlations_nosink)
            }
            _ns_all = [c for v in layer_correlations_nosink.values() for c in v]
            out["_meta_mean_positional_decay_correlation_nosink"] = float(np.mean(_ns_all))
            out["_meta_layer_positional_decay_nosink"] = _ns_layer
        if sink_mass:
            out["_meta_attention_sink_mass"] = float(np.mean(sink_mass))
        # Many flat or non-finite rows mean the dtype, not the geometry, drives the number.
        out["_meta_n_nonfinite_attn_rows"] = int(_n_nonfinite_rows)
        out["_meta_n_flat_attn_rows"] = int(_n_flat_rows)
        out["_meta_model_dtype"] = str(getattr(model, "dtype", "unknown"))
        out["_meta_promoted_to_fp32"] = bool(_dtype_info["promoted"])
        out.update(effective_dtype_meta(_dtype_info))
        return out
