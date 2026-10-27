from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_prediction_stats, strided_perplexity
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")

# Byte-attribution modes of `strided_perplexity` that give an exact byte count;
# any other mode only bounds it, and is emitted under a `_lower_bound` name.
_EXACT_BYTE_MODES = ("offsets", "token_pieces")


def _frequency_buckets(token_counts):
    """Bottom/top 20% of the token ids OBSERVED in the corpus (never-seen ids
    would fill the rare bucket and empty it of scored tokens)."""
    token_counts = np.asarray(token_counts)
    observed_ids = np.flatnonzero(token_counts > 0)
    observed_counts = token_counts[observed_ids]
    # Stable sort by count; ties broken by id.
    sorted_observed = observed_ids[np.lexsort((observed_ids, observed_counts))]
    n_observed = sorted_observed.size
    if n_observed == 0:
        return set(), set()
    rare_thresh = max(1, int(round(n_observed * 0.2)))
    # Keep the frequent bucket non-empty on tiny vocabularies.
    freq_thresh = min(n_observed - 1,
                      max(1, n_observed - int(round(n_observed * 0.2))))
    return (set(sorted_observed[:rare_thresh].tolist()),
            set(sorted_observed[freq_thresh:].tolist()))


@register_task("geometry_perplexity")
class RarePPLTask(DiagnosticTask):
    """
    Corpus perplexity (HF strided-window protocol), bits per byte and per
    character, and token-frequency-stratified perplexity.

    Bits per UTF-8 byte is invariant to the tokenisation (The Pile, Gao et al.
    2020, Sec 3.1) and is the headline; per-token perplexity is comparable only
    within a tokenizer.
    """

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2101.00027",
        "code": "huggingface/transformers docs 'Perplexity of fixed-length models' (strided window; web page, no commit)",
        "llm_precedent": "Gao et al. 2020 The Pile Sec 3.1 (bits per UTF-8 byte across tokenizers)",
    }
    # Size-linked by meaning (a capability): larger models reach lower loss, so
    # bpb is disclosed, not normalised. Pinned: never-emitted vocabulary rows.
    HEADLINE_KEYS = ("bits_per_byte",)
    SIZE_INVARIANT = {
        "bits_per_byte": (("pad_vocabulary_masked",),
                          "tests/tasks/invariance/test_geometry_perplexity.py::"
                          "test_bits_per_byte_invariant_under_masked_vocabulary_padding"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Rare Token PPL + BPC Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Materialise once: the dataset is traversed twice (per-document stats
        # and the strided pass), which would exhaust an iterator.
        if not isinstance(dataset, (list, tuple)):
            dataset = list(dataset)

        if cache is not None and cache.is_populated and use_cache:
            stats, _ = cache.get_prediction_stats(num_samples=num_samples)
        else:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)

        rare_ids, freq_ids = _frequency_buckets(stats["token_counts"])

        nll_rare, cnt_rare = 0.0, 0
        nll_freq, cnt_freq = 0.0, 0
        nll_total, cnt_total = 0.0, 0
        for logits, labels in zip(stats["logits"], stats["labels"]):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), reduction="none")
            for l_val, tid in zip(loss.tolist(), labels.view(-1).tolist()):
                nll_total += l_val
                cnt_total += 1
                if tid in rare_ids:
                    nll_rare += l_val
                    cnt_rare += 1
                elif tid in freq_ids:
                    nll_freq += l_val
                    cnt_freq += 1

        # Corpus perplexity by the standard strided-window protocol (one
        # concatenated stream, masked overlap); see strided_perplexity().
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
                # Public keys carry exact byte/char counts only; an offset-less
                # attribution only bounds the denominator.
                if n_bytes > 0:
                    _bpb_val = float(bits_total / n_bytes)
                    if _byte_attr in _EXACT_BYTE_MODES:
                        bpb = _bpb_val
                    else:
                        _extra_bounds["_meta_bits_per_byte_lower_bound"] = _bpb_val
                if n_chars > 0:
                    _bpc_val = float(bits_total / n_chars)
                    if _byte_attr in _EXACT_BYTE_MODES:
                        bpc = _bpc_val
                    else:
                        _extra_bounds["_meta_bits_per_char_lower_bound"] = _bpc_val
        except Exception as e:
            # Recorded: ppl_overall becomes NaN, never the per-document value.
            _strided_error = f"{type(e).__name__}: {e}"
            logger.info(f"  Strided PPL/BPC computation failed: {_strided_error}")

        ppl_perdoc = (float(np.exp(nll_total / cnt_total))
                      if cnt_total > 0 else float("inf"))

        _strided_ok = bool(np.isfinite(ppl_strided))
        ppl_overall = ppl_strided if _strided_ok else float("nan")
        mean_nll_nats = nll_strided if np.isfinite(nll_strided) else float("nan")

        _extra = {
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
            # The buckets come from the per-document pass (they need per-token
            # identity). _meta_: bucket membership is each model's own tokens'
            # bottom/top 20% on this corpus, so it changes with the tokenizer
            # and the corpus size (ppl_rare_over_freq 4.0 -> 7.0 from 25 to 200
            # docs on gpt2): neither tokenizer- nor N-comparable.
            "_meta_ppl_rare": float(np.exp(nll_rare / cnt_rare)) if cnt_rare > 0 else float("inf"),
            "_meta_ppl_freq": float(np.exp(nll_freq / cnt_freq)) if cnt_freq > 0 else float("inf"),
            "_meta_ppl_rare_over_freq": (float(np.exp(nll_rare / cnt_rare - nll_freq / cnt_freq))
                                         if cnt_rare > 0 and cnt_freq > 0 else float("nan")),
            "_meta_ppl_bucket_protocol": "per_document",
            "ppl_overall": ppl_overall,
            "mean_nll_nats": mean_nll_nats,
            "bits_per_char": bpc,
            "bits_per_byte": bpb,
            "_meta_n_tokens_scored": int(n_strided) if n_strided > 0 else cnt_total,
            "_meta_ppl_overall_strided": ppl_strided,
            "ppl_overall_perdoc": ppl_perdoc,
            "_meta_n_tokens_strided": int(n_strided),
            "_meta_n_tokens_perdoc": int(cnt_total),
            **_extra,
        }
