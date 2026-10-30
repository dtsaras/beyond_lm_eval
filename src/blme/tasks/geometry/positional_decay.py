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


_ATTN_ATTRS = ("self_attn", "attn", "attention", "self_attention")
_OUT_ATTRS = ("o_proj", "c_proj", "dense", "out_proj")


def _write_sites(model):
    """Each block's attention output projection, or None if any is missing."""
    from ..common import get_layers
    blocks = get_layers(model)
    if not blocks:
        return None
    sites = []
    for b in blocks:
        attn = next((getattr(b, a) for a in _ATTN_ATTRS
                     if isinstance(getattr(b, a, None), torch.nn.Module)), None)
        proj = None
        if attn is not None:
            proj = next((getattr(attn, a) for a in _OUT_ATTRS
                         if isinstance(getattr(attn, a, None), torch.nn.Module)
                         and getattr(getattr(attn, a), "weight", None) is not None), None)
        if proj is None:
            return None
        sites.append(proj)
    return sites


def _capture_input(store, i):
    def hook(module, args):
        if args and isinstance(args[0], torch.Tensor):
            store[i] = args[0].detach()
    return hook


def _head_write_norms(z, proj, n_heads: int):
    """||o_{h,q}||: what head h writes into the residual stream at query q.

    ``z`` is the output projection's input (1, T, H * d_head), the heads'
    outputs side by side; o_h = z_h W_O[:, h] (bias excluded: it is not any
    head's). Returns a (T, H) float64 array, or None when shapes disagree.
    """
    if z is None:
        return None
    z = z[0]
    W = proj.weight.detach().to(z.dtype)
    D = int(z.shape[-1])
    if n_heads <= 0 or D % n_heads:
        return None
    dh = D // n_heads
    zh = z.reshape(z.shape[0], n_heads, dh)
    if hasattr(proj, "nf"):                       # GPT-2 Conv1D: weight (in, out)
        if W.shape[0] != D:
            return None
        o = torch.einsum("thk,hkd->thd", zh, W.reshape(n_heads, dh, W.shape[1]))
    else:                                         # nn.Linear: weight (out, in)
        if W.shape[1] != D:
            return None
        o = torch.einsum("thk,dhk->thd", zh, W.reshape(W.shape[0], n_heads, dh))
    return torch.linalg.vector_norm(o.double(), dim=-1).cpu().numpy()


def _write_weighted_mean(pairs) -> float:
    """sum w rho / sum w over (write norm, correlation) rows.

    A head whose output projection is zero (an inserted identity block) has
    w = 0 and drops out of both sums; a global residual rescale multiplies
    every w by the same factor, which cancels. math.fsum is exactly rounded,
    so neither the extra zeros nor the order of the rows changes the bits.
    """
    import math
    tot = math.fsum(w for w, _ in pairs)
    if not tot > 0:
        return float("nan")
    return float(math.fsum(w * r for w, r in pairs) / tot)


@register_task("geometry_positional_decay")
class PositionalAttentionDecayTask(DiagnosticTask):
    """
    Computes the Spearman correlation between target-source token distance
    and attention probability to measure the structural integrity of
    positional encodings.
    """
    # The key at position 0 (the attention sink) is dropped from every row of
    # the headline correlation; the inclusive value is _meta_.
    TOKEN_POLICY = "exclude_pos0"
    SOURCES = {
        "paper": "arXiv:2104.09864 (RoFormer: the decay premise; no statistic of trained attention)",
        "code": "",
        "llm_precedent": ("arXiv:2004.10102 (Kobayashi et al. 2020: weigh attention by the norm "
                          "of what a head writes); arXiv:2309.17453 (attention sinks)"),
    }
    ADAPTATION_NOTE = (
        "Positional decay is a BLME construct: for every causal query row, the Spearman "
        "correlation between key distance and attention probability, negative when nearer keys "
        "receive more attention. RoFormer motivates the premise (RoPE's logit bound decays with "
        "relative distance) but defines no statistic of trained attention. The rowwise form is "
        "used because one pooled Spearman per head reads -0.43 to -0.47 on exactly uniform causal "
        "attention. Position 0 is dropped from the keys: the attention sink is distance-"
        "independent mass on the farthest key of every row; the inclusive value and the sink "
        "mass are _meta_. Rows are aggregated over heads and layers with weights equal to the "
        "norm of each head's write into the residual stream (after Kobayashi et al. 2020), "
        "normalised by the model total, which is exactly unchanged by identity blocks and by a "
        "residual rescale, where a uniform layer mean is not; the attention is computed in fp32."
    )
    HEADLINE_KEYS = ("positional_decay_nosink_write_weighted",)
    SIZE_INVARIANT = {
        "positional_decay_nosink_write_weighted": (
            ("insert_identity_blocks", "residual_scale_gauge"),
            "tests/tasks/invariance/test_geometry_positional_decay.py::"
            "test_weighted_mean_is_invariant_under_identity_blocks"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Positional Attention Decay Analysis...")
        _drop = 1 if self.TOKEN_POLICY == "exclude_pos0" else 0
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
        # (write norm, correlation) per (layer, head, query) row, for the
        # write-weighted depth aggregation (see _write_weighted_mean).
        _w_incl, _w_nosink = [], []
        _sites = _write_sites(model)
        _zcap = {}
        _n_layers_weighted = _n_layer_slots = 0
        _n_attn_layers = 0
        _n_nonfinite_rows = 0
        _n_flat_rows = 0

        # The attention probabilities must be PRODUCED in fp32 (P5): bf16's
        # 8-bit mantissa makes about a fifth of pythia-70m's rows exactly flat.
        # fp32_compute promotes at every model size (the old <= 2e9-parameter
        # gate made the dtype a function of size) and restores parameters AND
        # buffers (tests/tasks/test_positional_decay_protocol.py).
        _dtype_ctx = fp32_compute(model)
        _dtype_info = _dtype_ctx.__enter__()
        _hooks = [proj.register_forward_pre_hook(_capture_input(_zcap, i))
                  for i, proj in enumerate(_sites or [])]
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

                _n_attn_layers = max(_n_attn_layers, len(out.attentions))
                for li, attn_entry in enumerate(out.attentions):
                    if attn_entry is None:
                        continue
                    attn_matrix = attn_entry[0]  # (num_heads, seq_len, seq_len)
                    # Each query row and head separately: pooling triangular
                    # causal entries confounds distance with row normalisation.
                    # .float() before any comparison.
                    a = attn_matrix.float().cpu().numpy()
                    _wn = (_head_write_norms(_zcap.get(li), _sites[li], a.shape[0])
                           if _sites is not None and li < len(_sites) else None)
                    _n_layers_weighted += int(_wn is not None)
                    _n_layer_slots += 1
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
                                if _wn is not None:
                                    _w_incl.append((float(_wn[query_idx, head_idx]), corr))
                            # Position 0 is the attention sink and also the
                            # maximum-distance key of every causal row, so the
                            # inclusive statistic mixes decay with sink strength.
                            if True:
                                _rs = float(_row.sum())
                                if _rs > 0:
                                    sink_mass.append(float(_row[0]) / _rs)
                                corr_ns = _row_distance_correlation(_row[_drop:])
                                if corr_ns is not None:
                                    layer_correlations_nosink[li].append(corr_ns)
                                    if _wn is not None:
                                        _w_nosink.append(
                                            (float(_wn[query_idx, head_idx]), corr_ns))
                        if _pool_v:
                            _pv = np.concatenate(_pool_v)
                            _pd = np.concatenate(_pool_d).astype(np.float64)
                            if np.ptp(_pv) > 0 and np.ptp(_pd) > 0:
                                _rho = float(spearmanr(_pd, _pv).statistic)
                                if np.isfinite(_rho):
                                    layer_correlations_pooled[li].append(_rho)

        finally:
            # Restores on every exit, including the early error return above.
            for _h in _hooks:
                _h.remove()
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
            # Inclusive of the position-0 sink: _meta_ (decision 3b).
            "_meta_mean_positional_decay_correlation": float(np.mean(all_corrs)),
            # Uniform means over layers move under identity-block insertion
            # (P4), and the pooled form is not null-calibrated: _meta_.
            "_meta_mean_positional_decay_correlation_pooled": (
                float(np.mean(_pooled_all)) if _pooled_all else float("nan")),
            "_meta_layer_positional_decay": layer_means,
        }

        # HEADLINE (decision 3b): the same Spearman over keys 1..q-1. RoPE's
        # decay premise concerns relative distance; the sink is distance-
        # independent mass parked on the farthest key of every row.
        if layer_correlations_nosink:
            _ns_layer = {
                f"layer_{li}": float(np.mean(layer_correlations_nosink[li]))
                for li in sorted(layer_correlations_nosink)
            }
            _ns_all = [c for v in layer_correlations_nosink.values() for c in v]
            out["_meta_mean_positional_decay_correlation_nosink"] = float(np.mean(_ns_all))
            out["_meta_layer_positional_decay_nosink"] = _ns_layer
        # HEADLINE: the sink-excluded correlation, each row weighted by what its
        # head writes into the residual stream.
        complete = bool(_sites is not None and _n_layer_slots > 0
                        and _n_layers_weighted == _n_layer_slots)
        out["positional_decay_nosink_write_weighted"] = (
            _write_weighted_mean(_w_nosink) if complete else float("nan"))
        out["_meta_positional_decay_inclusive_write_weighted"] = (
            _write_weighted_mean(_w_incl) if complete else float("nan"))
        out["_meta_write_weight_site"] = (
            type(_sites[0]).__name__ if _sites else None)
        out["_meta_write_weights_complete"] = complete
        if sink_mass:
            out["_meta_attention_sink_mass"] = float(np.mean(sink_mass))
        # Many flat or non-finite rows mean the dtype, not the geometry, drives the number.
        out["_meta_n_nonfinite_attn_rows"] = int(_n_nonfinite_rows)
        out["_meta_n_flat_attn_rows"] = int(_n_flat_rows)
        out["_meta_model_dtype"] = str(getattr(model, "dtype", "unknown"))
        out["_meta_promoted_to_fp32"] = bool(_dtype_info["promoted"])
        out.update(effective_dtype_meta(_dtype_info))
        return out
