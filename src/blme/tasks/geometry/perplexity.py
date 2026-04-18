from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_prediction_stats
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
            freq_thresh = max(1, n_observed - int(round(n_observed * 0.2)))
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

        # Bits-per-character: re-tokenise the first num_samples documents
        # so we can count UTF-8 characters in each one. NLL is per token
        # in nats; convert to bits and divide by char count.
        bpc = float("nan")
        if cnt_total > 0:
            try:
                total_chars = 0
                total_tokens_for_chars = 0
                count = 0
                for sample in dataset:
                    if count >= num_samples:
                        break
                    text = sample["text"] if isinstance(sample, dict) else str(sample)
                    if not text:
                        continue
                    enc = tokenizer(text, truncation=False, add_special_tokens=False)
                    n_tok = len(enc["input_ids"])
                    if n_tok > 1:  # match the (T-1) labels-shift from collect_prediction_stats
                        total_tokens_for_chars += (n_tok - 1)
                        total_chars += len(text)
                    count += 1

                if total_chars > 0 and total_tokens_for_chars > 0:
                    # Average NLL per token (over the same shifted labels)
                    mean_nll_nats = nll_total / cnt_total
                    # bits per token
                    bits_per_token = mean_nll_nats / float(np.log(2))
                    # tokens per char (using the freshly tokenised counts)
                    tokens_per_char = total_tokens_for_chars / total_chars
                    bpc = float(bits_per_token * tokens_per_char)
            except Exception as e:
                logger.info(f"  BPC computation failed: {type(e).__name__}: {e}")

        ppl_overall = float(np.exp(nll_total / cnt_total)) if cnt_total > 0 else float("inf")
        mean_nll_nats = (nll_total / cnt_total) if cnt_total > 0 else float("nan")

        return {
            "ppl_rare": float(np.exp(nll_rare / cnt_rare)) if cnt_rare > 0 else float("inf"),
            "ppl_freq": float(np.exp(nll_freq / cnt_freq)) if cnt_freq > 0 else float("inf"),
            "ppl_overall": ppl_overall,
            "mean_nll_nats": mean_nll_nats,
            "bits_per_char": bpc,
            "n_tokens_scored": cnt_total,
        }
