from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_prediction_stats, strided_perplexity
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")

@register_task("geometry_perplexity")
class RarePPLTask(DiagnosticTask):
    """
    Token-frequency-stratified perplexity, corpus-level perplexity, and
    bits-per-character (BPC).

    BPC is the cross-entropy loss in bits divided by the number of UTF-8
    characters in the evaluation text. Unlike token-level perplexity, BPC
    is tokenizer-independent (modulo BPE split semantics) and therefore
    the cleanest single-number capability proxy for cross-architecture
    comparison.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Rare Token PPL + BPC Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # MATERIALISE ONCE. `evaluate` traverses the dataset twice -- once for
        # the per-document statistics and again for the strided pass -- so a
        # one-shot iterator was consumed by the first traversal and the second
        # silently fell back to the per-document protocol under the same key.
        # The same 4 texts gave strided PPL 25.964948 / BPB 1.146522 as a list
        # but per-document PPL 69.553105 (2.679x) as an iterator, with nothing
        # in the output to distinguish them (adversarial review round 2,
        # 2026-09-04).
        if not isinstance(dataset, (list, tuple)):
            dataset = list(dataset)

        if cache is not None and cache.is_populated and use_cache:
            stats, _ = cache.get_prediction_stats(num_samples=num_samples)
        else:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)

        # Categorize tokens by corpus frequency, but **only among
        # observed tokens**. With a ~12k-token eval corpus in a
        # 50–200k-vocab model, ~90% of vocab ids have count 0 — if we
        # argsort the full vocab, the bottom 20% is dominated by never-
        # seen ids and ``cnt_rare`` ends up 0 (so ``ppl_rare`` becomes
        # +inf for every model). Filter to observed ids before
        # percentile-thresholding so the buckets are actually populated.
        token_counts = np.asarray(stats["token_counts"])
        observed_ids = np.flatnonzero(token_counts > 0)
        observed_counts = token_counts[observed_ids]
        # Stable sort by count; ties broken by id for determinism.
        sort_order = np.lexsort((observed_ids, observed_counts))
        sorted_observed = observed_ids[sort_order]

        n_observed = sorted_observed.size
        if n_observed == 0:
            rare_ids, freq_ids = set(), set()
        else:
            rare_thresh = max(1, int(round(n_observed * 0.2)))
            # Keep the freq bucket non-empty on tiny vocab counts (audit
            # 2026-07-16: with 2 observed ids the old asymmetry emptied the
            # freq bucket -> ppl_freq = inf).
            freq_thresh = min(n_observed - 1,
                              max(1, n_observed - int(round(n_observed * 0.2))))
            rare_ids = set(sorted_observed[:rare_thresh].tolist())
            freq_ids = set(sorted_observed[freq_thresh:].tolist())

        nll_rare, cnt_rare = 0.0, 0
        nll_freq, cnt_freq = 0.0, 0
        nll_total, cnt_total = 0.0, 0

        for logits, labels in zip(stats["logits"], stats["labels"]):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), reduction="none")
            label_ids = labels.view(-1).tolist()
            loss_vals = loss.tolist()
            for l_val, tid in zip(loss_vals, label_ids):
                nll_total += l_val
                cnt_total += 1
                if tid in rare_ids:
                    nll_rare += l_val
                    cnt_rare += 1
                elif tid in freq_ids:
                    nll_freq += l_val
                    cnt_freq += 1

        # ---- Corpus perplexity by the STANDARD strided-window protocol ----
        #
        # ppl_overall and the bits-per-X family are the numbers a reader will
        # compare against published perplexities, so they must be computed the
        # way published perplexities are: one concatenated stream, a sliding
        # window with masked overlap. See strided_perplexity() for the full
        # rationale and the measured gpt2 discrepancy (37.91 -> 21.80).
        #
        # The frequency-stratified buckets above stay on the per-document
        # stats path: they need per-token identity, they are a RATIO between
        # two buckets scored under the same conditions, and the protocol
        # difference largely cancels in that ratio.
        bpc = float("nan")
        bpb = float("nan")
        ppl_strided = float("nan")
        _strided_error = None
        _byte_attr = "none"
        _extra_bounds = {}
        nll_strided = float("nan")
        n_strided = 0
        try:
            (ppl_strided, nll_strided, n_strided,
             n_bytes, n_chars, _ppl_window, _ppl_stride,
             _byte_attr) = strided_perplexity(
                model, tokenizer, dataset, num_samples=num_samples,
                max_length=self.config.get("ppl_max_length"),
                stride=self.config.get("ppl_stride"),
            )
            if n_strided > 0 and np.isfinite(nll_strided):
                bits_total = nll_strided * n_strided / float(np.log(2))
                # bits_per_byte is the tokenizer-independent standard (the
                # Pile / lm-evaluation-harness convention). bits_per_char is
                # kept for continuity but is Unicode-code-point based, so it
                # is NOT comparable across corpora with different scripts.
                # The PUBLIC keys carry EXACT values only.
                #
                # The offset-less branch can only bound the denominator from
                # above, and the bound is not always tight: when token 0 owns
                # 16 of a 22-byte corpus the bound is off by 3.7x, so exporting
                # it under `bits_per_byte` would contradict what that key
                # means (adversarial review round 3, 2026-09-04). It is emitted
                # under an explicitly-named `_meta_` key instead.
                if n_bytes > 0:
                    _bpb_val = float(bits_total / n_bytes)
                    if _byte_attr == "offsets":
                        bpb = _bpb_val
                    else:
                        _extra_bounds["_meta_bits_per_byte_lower_bound"] = _bpb_val
                if n_chars > 0:
                    _bpc_val = float(bits_total / n_chars)
                    if _byte_attr == "offsets":
                        bpc = _bpc_val
                    else:
                        _extra_bounds["_meta_bits_per_char_lower_bound"] = _bpc_val
        except Exception as e:
            # RECORDED, not swallowed. `ppl_overall` is documented as the
            # STRIDED number; when the strided pass fails it silently became
            # the per-document one under the same key, with no flag. The same
            # three-document input gave ppl_overall 59.735543 as a list and a
            # different value as a one-shot iterator, because the iterator was
            # consumed by the first pass -- indistinguishable in the output
            # (adversarial review, 2026-09-04).
            _strided_error = f"{type(e).__name__}: {e}"
            logger.info(f"  Strided PPL/BPC computation failed: {_strided_error}")

        # Per-document figures, retained under explicit names so the
        # protocol behind every number is visible at the call site.
        ppl_perdoc = (float(np.exp(nll_total / cnt_total))
                      if cnt_total > 0 else float("inf"))
        nll_perdoc = (nll_total / cnt_total) if cnt_total > 0 else float("nan")

        # ppl_overall / mean_nll_nats now carry the STRIDED value, falling
        # back to the per-document one only if the strided pass failed.
        # `ppl_overall` MEANS the strided value. When the strided pass fails
        # it is NaN, not silently re-pointed at the per-document number: a
        # synthetic joined-stream failure returned ppl_overall = 2.000000004,
        # identical to `ppl_overall_perdoc`, with only a `_meta_` field to
        # betray that the definition had changed (adversarial review round 3,
        # 2026-09-04). The per-document value keeps its own name.
        _strided_ok = bool(np.isfinite(ppl_strided))
        ppl_overall = ppl_strided if _strided_ok else float("nan")
        mean_nll_nats = nll_strided if np.isfinite(nll_strided) else float("nan")

        _extra = {
            # Which protocol `ppl_overall` is actually on, always emitted.
            "_meta_ppl_protocol": "strided" if _strided_ok else "per_document",
            "_meta_ppl_byte_attribution": _byte_attr,
            **_extra_bounds,
        }
        if _strided_error is not None:
            _extra["_meta_ppl_strided_error"] = _strided_error
        try:
            _extra["_meta_ppl_window"] = int(_ppl_window)
            _extra["_meta_ppl_stride"] = int(_ppl_stride)
        except NameError:
            pass

        return {
            # Frequency-stratified buckets. These come from the PER-DOCUMENT
            # pass -- they need per-token identity -- while ppl_overall is
            # strided, so forming ppl_rare/ppl_overall mixes protocols: on gpt2
            # that cross-protocol ratio is 5.31 against a within-protocol 3.05,
            # a 74% inflation of the apparent rare-token penalty. The
            # within-protocol ratio is emitted directly so nobody has to build
            # it, and the protocol is recorded in _meta_ppl_bucket_protocol.
            "ppl_rare": float(np.exp(nll_rare / cnt_rare)) if cnt_rare > 0 else float("inf"),
            "ppl_freq": float(np.exp(nll_freq / cnt_freq)) if cnt_freq > 0 else float("inf"),
            "ppl_rare_over_freq": (float(np.exp(nll_rare / cnt_rare - nll_freq / cnt_freq))
                                   if cnt_rare > 0 and cnt_freq > 0 else float("nan")),
            "_meta_ppl_bucket_protocol": "per_document",
            "ppl_overall": ppl_overall,
            "mean_nll_nats": mean_nll_nats,
            "bits_per_char": bpc,
            "bits_per_byte": bpb,
            "_meta_n_tokens_scored": int(n_strided) if n_strided > 0 else cnt_total,
            # Protocol-explicit duplicates for auditability.
            "_meta_ppl_overall_strided": ppl_strided,
            "ppl_overall_perdoc": ppl_perdoc,
            "_meta_n_tokens_strided": int(n_strided),
            "_meta_n_tokens_perdoc": int(cnt_total),
            **_extra,
        }
