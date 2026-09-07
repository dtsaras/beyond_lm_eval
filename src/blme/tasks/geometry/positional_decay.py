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
- 2024-2025 Long-Context Extrapolation and RoPE literature.
"""

from __future__ import annotations  # PEP 563: keep `float | None` lazy on py3.9

import torch
import numpy as np
from scipy.stats import spearmanr

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
    # Only a TRULY constant row is the null. `np.allclose` uses a RELATIVE
    # tolerance (1e-5) and attention rows are tiny, so it also caught rows with
    # small-but-real spread and forced them to 0.0 -- 132 of 25,008 pythia-70m
    # rows, whose own mean rho is -0.3584, which is exactly what moved the
    # shipped value from -0.27426 to -0.272371. Those rows carry signal and are
    # now scored normally; only an EXACTLY flat row returns the null.
    #
    # The threshold must be == 0.0, not <= 1e-12: attention rows are tiny, and
    # on pythia-70m 61 of 24,768 rows (0.246%) have 0 < ptp <= 1e-12 while
    # being strictly ordered. Their mean rho is -0.3410, and
    # 61 * 0.3410 / 24768 = 0.000840 accounts exactly for the -0.280844 the
    # 1e-12 threshold produced against a true -0.28168. gpt2 has no such rows,
    # so the bias is model-dependent (adversarial review, 2026-09-04).
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

        # COMPUTE ATTENTION IN fp32.
        #
        # Upcasting the probabilities AFTER the model has already produced them
        # cannot recover the mantissa -- that earlier "fix" was a no-op, since
        # `attn_matrix.float()` was already present. Under core.py's bf16 cast,
        # bf16's 8-bit mantissa makes 5,172 of 24,768 pythia-70m attention rows
        # (20.88%) come out EXACTLY flat, which the estimator then scores as
        # the null: headline -0.251010 against -0.281684 in fp32 (-10.89%) and
        # sink mass -52.73%. In fp16 38.63% of rows are all-NaN and silently
        # dropped. The rows must be produced in fp32, not repaired afterwards.
        #
        # Promote at the CALL SITE with a per-parameter dtype restore, the same
        # pattern dynamics/sharpness.py uses, so the shared model is returned
        # exactly as it was found (adversarial review, 2026-09-04).
        _lowp = {torch.float16, torch.bfloat16}
        _saved_dtypes = None
        _n_params = sum(p_.numel() for p_ in model.parameters())
        if (getattr(model, "dtype", None) in _lowp) and _n_params <= 2_000_000_000:
            try:
                _saved_dtypes = [(p_, p_.dtype) for p_ in model.parameters()]
                model.float()
            except Exception as e:  # pragma: no cover - defensive
                logger.info(f"positional_decay: fp32 promotion failed ({e})")
                _saved_dtypes = None

        try:
          with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

                seq_len = inputs.input_ids.shape[1]
                if seq_len < 4:
                   continue

                out = model(**inputs, output_attentions=True)
                # SDPA returns a tuple of the right LENGTH whose entries are
                # all None, so a length check alone does not catch it -- which
                # is why this surfaced downstream as "sequences too short".
                if (out.attentions is None or len(out.attentions) == 0
                        or all(a is None for a in out.attentions)):
                    # Name the real cause. A default-loaded gpt2 uses the SDPA
                    # attention backend, which does not materialise the
                    # probability matrix, so `output_attentions=True` yields
                    # nothing -- and the downstream failure surfaced as
                    # "sequences too short or nan", blaming the corpus for an
                    # attention-backend problem (adversarial review,
                    # 2026-09-04).
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
                    # Compute each query-row/head separately. Pooling
                    # triangular causal entries first confounds distance
                    # with row normalisation (uniform rows shrink as
                    # context grows) and invents decay under the null.
                    # .float() BEFORE any comparison. Under core.py's bf16
                    # cast, bf16's 8-bit mantissa makes 20.88% of pythia-70m's
                    # attention rows come out EXACTLY flat -- they are then
                    # scored as the 0.0 null, moving the headline -10.9% and
                    # sink mass -52.7% -- against 0.23% for fp32. In raw
                    # fp16+eager, 38.63% of rows are all-NaN and silently
                    # dropped. The upcast happens here, before ptp/argsort, so
                    # the null branch only fires on genuinely flat rows
                    # (adversarial review, 2026-09-04).
                    a = attn_matrix.float().cpu().numpy()
                    _n_nonfinite_rows += int(
                        (~np.isfinite(a)).any(axis=-1).sum())
                    for head_idx in range(a.shape[0]):
                        # POOLED variant, emitted alongside the per-row one.
                        #
                        # The paper text (sections/metrics_geometry.tex:246-251)
                        # specifies ONE Spearman over all {query, key} pairs per
                        # head, against distance q - k. BLME's headline averages
                        # PER-QUERY Spearmans instead, deliberately: pooling
                        # triangular causal entries confounds distance with row
                        # normalisation, because a uniform row's entries shrink
                        # as the context grows, so the pooled statistic reports
                        # decay even under an exactly uniform null. The two
                        # genuinely differ -- a valid causal head reads
                        # +1.000000 per-row and -0.204614 pooled (adversarial
                        # review, 2026-09-04) -- so BOTH are reported rather
                        # than one silently standing in for the other. THE PAPER
                        # TEXT MUST BE RECONCILED WITH THIS (tracked in
                        # verification_campaign/RERUN_LEDGER.md).
                        # POOLED branch: EVERY causal {i, j} pair, exactly as
                        # the manuscript writes it -- diagonal included, and
                        # from q = 1 rather than q >= 3. Restricting it to the
                        # rowwise branch's own q >= 3, j < q domain silently
                        # made it a different statistic from the one the paper
                        # defines (adversarial review round 2, 2026-09-04).
                        _pool_v, _pool_d = [], []
                        for _q in range(1, seq_len):
                            _fr = a[head_idx, _q, :_q + 1]
                            _pool_v.append(_fr)
                            _pool_d.append(_q - np.arange(_fr.size))
                        for query_idx in range(2, seq_len):
                            _row = a[head_idx, query_idx, :query_idx]
                            # SAME ROW SET for both twins. The sink-excluded
                            # variant needs >= 3 keys (it drops one), so
                            # computing the raw variant on q >= 2 rows and the
                            # twin on q >= 3 made 8-12% of the apparent "sink
                            # effect" a row-set difference rather than the sink:
                            # the raw key read -0.29230 on the wider set against
                            # -0.30340 on the matched one (adversarial review,
                            # 2026-09-04). Skip rows the twin cannot use.
                            if _row.size <= 2:
                                continue
                            if float(np.ptp(_row)) == 0.0:
                                _n_flat_rows += 1
                            corr = _row_distance_correlation(_row)
                            if corr is not None:
                                layer_correlations[li].append(corr)
                            # ATTENTION-SINK CONTROL. Position 0 is the sink,
                            # and in a causal row it is also the MAXIMUM-
                            # distance key -- so a large, distance-independent
                            # mass sits exactly at the far end of the range the
                            # Spearman is measuring, and the reported number is
                            # a mixture of positional decay and sink strength.
                            # Sink mass ranges 14.5-47.3% across models
                            # (gpt2 47.3%, distilgpt2 41.3%, pythia-70m 14.5%),
                            # so the bias is model-dependent and does not cancel
                            # in a cross-model comparison.
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
            if _saved_dtypes is not None:
                for p_, dt in _saved_dtypes:
                    p_.data = p_.data.to(dt)

        if not layer_correlations:
             return {"error": "Could not compute positional correlations (sequences too short or nan)."}

        # Compute per-layer means
        layer_means = {}
        for li in sorted(layer_correlations.keys()):
            layer_means[f"layer_{li}"] = float(np.mean(layer_correlations[li]))

        all_corrs = [c for corrs in layer_correlations.values() for c in corrs]
        _pooled_all = [c for v in layer_correlations_pooled.values() for c in v]

        out = {
            # HEADLINE = the ROWWISE statistic, and this is now settled by
            # measurement rather than by preference.
            #
            # The manuscript (sections/metrics_geometry.tex:246-251) specifies
            # ONE pooled Spearman per head over all {i, j} pairs. That estimand
            # IS implemented, and matches an independent tie-ranked reference
            # over 50 randomised causal configurations to 5.55e-17 -- but it is
            # NOT NULL-CALIBRATED. On an exactly UNIFORM causal attention
            # pattern, where by construction there is no positional decay at
            # all, the pooled statistic reads
            #
            #     L =   8 -> -0.4344766895
            #     L =  64 -> -0.4659740104
            #     L = 256 -> -0.4666215371
            #
            # against a correct 0, while the rowwise statistic reads EXACTLY
            # 0.0 at every length
            # (`test_positional_decay_uniform_causal_attention_is_null`).
            #
            # CORRECTION (2026-09-04): an earlier version of this note quoted
            # +0.2545271629778672 here. That number was measured on a MALFORMED
            # fixture whose rows did not sum to 1 -- it left the diagonal at 0
            # and row 0 empty -- and an independent review caught it. The
            # fixture is fixed; the true null bias is LARGER and of the
            # OPPOSITE SIGN, converging to about -0.467 as the context grows,
            # so the conclusion is strengthened, not weakened.
            #
            # The cause is structural: pooling triangular causal entries
            # confounds distance with row normalisation, because a uniform
            # row's entries shrink as the context grows, so long distances --
            # which only occur in deep rows -- are systematically paired with
            # small values. A statistic that reports decay under its own null
            # cannot be the headline.
            #
            # THE PAPER TEXT MUST BE RECONCILED TO THE ROWWISE DEFINITION
            # (tracked in RERUN_LEDGER.md). Both are emitted; neither stands in
            # for the other (adversarial review rounds 1-3, 2026-09-04).
            "mean_positional_decay_correlation": float(np.mean(all_corrs)),
            "mean_positional_decay_correlation_pooled": (
                float(np.mean(_pooled_all)) if _pooled_all else float("nan")),
            "layer_positional_decay": layer_means,
        }

        # Sink-excluded twin: the same Spearman over keys 1..q, dropping the
        # position-0 sink. This is the comparable quantity -- the raw one mixes
        # decay with sink strength, and sink strength varies by model.
        if layer_correlations_nosink:
            _ns_layer = {
                f"layer_{li}": float(np.mean(layer_correlations_nosink[li]))
                for li in sorted(layer_correlations_nosink)
            }
            _ns_all = [c for v in layer_correlations_nosink.values() for c in v]
            # DEMOTED to _meta_ (2026-09-04). The sink-excluded twin claims to
            # be the cross-model-comparable quantity, but it fails the
            # mechanism test this campaign requires: inserting zero-output
            # IDENTITY attention blocks -- function-preserving, max logit delta
            # 0.0 -- moved it -1.000000 -> +0.750000 from 1 to 8 layers and
            # gave rho(depth) = +1.000000. Its better cross-model fit is
            # therefore not evidence that it removed the confound; it moved the
            # association onto depth.
            out["_meta_mean_positional_decay_correlation_nosink"] = float(np.mean(_ns_all))
            out["_meta_layer_positional_decay_nosink"] = _ns_layer
        if sink_mass:
            out["_meta_attention_sink_mass"] = float(np.mean(sink_mass))
        # Precision diagnostics: a large flat-row or non-finite-row count means
        # the model's dtype, not its geometry, is driving the number.
        out["_meta_n_nonfinite_attn_rows"] = int(_n_nonfinite_rows)
        out["_meta_n_flat_attn_rows"] = int(_n_flat_rows)
        out["_meta_model_dtype"] = str(getattr(model, "dtype", "unknown"))
        out["_meta_promoted_to_fp32"] = bool(_saved_dtypes is not None)
        return out
