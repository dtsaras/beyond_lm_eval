from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


@register_task("interpretability_prediction_entropy")
class PredictionEntropyTask(DiagnosticTask):
    """
    Computes the entropy of the output probability distribution at each
    token position, profiling the model's inherent uncertainty.

    The per-token entropy is the generic Shannon entropy of the next-token
    distribution (Shannon 1948), H = -Σ p log p — not a quantity defined by
    Holtzman et al. (whose contribution is nucleus/top-p sampling). Holtzman
    et al. 2020 (ICLR, arXiv:1904.09751) is the *degeneration* motivation
    for tracking output uncertainty, not the source of this metric.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Prediction Entropy Analysis...")

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(self.config.get("num_samples", 100))

        num_samples = self.config.get("num_samples", 100)
        # Decisiveness metrics: top-k subset entropy and top1-top2 logprob
        # gap. These complement the full-distribution entropy by isolating
        # how confidently the model commits to its top predictions, even
        # when the full distribution has long tails. Recent work (Chen et
        # al. 2024 "self-certainty"; Manvi et al. 2024) shows decisiveness
        # correlates with reasoning ability without needing labels.
        decisiveness_k = self.config.get("decisiveness_top_k", 5)

        all_entropies = []
        all_top1_probs = []
        all_top5_probs = []
        all_top1_top2_gap_logprob = []  # log p(top1) - log p(top2)
        all_topk_entropy = []            # H over the top-k renormalized

        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Computing Entropy")):
                if i >= num_samples:
                    break

                text = sample.get("text", "") if isinstance(sample, dict) else sample
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                outputs = model(**inputs)
                logits = outputs.logits  # (B, T, V)

                # Compute log-probabilities via ``log_softmax`` directly
                # on logits so low-probability tokens don't first
                # underflow to 0 inside ``softmax`` (in bf16/fp16 over a
                # 150k-vocab that bias is non-trivial) and then get
                # bumped back up by the ``clamp(min=1e-12)`` step. The
                # ``log-sum-exp`` trick inside ``log_softmax`` keeps the
                # full floating-point precision of the original logit.
                log_probs = torch.log_softmax(logits, dim=-1)  # (B, T, V)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
                all_entropies.extend(entropy[0].cpu().tolist())

                # Top-1 probability (confidence)
                top1 = probs.max(dim=-1).values  # (B, T)
                all_top1_probs.extend(top1[0].cpu().tolist())

                # Top-k probabilities and decisiveness metrics
                k = min(decisiveness_k, probs.shape[-1])
                topk_p = torch.topk(probs, k=k, dim=-1).values  # (B, T, k)
                top5_sum = topk_p[..., :min(5, k)].sum(dim=-1)  # (B, T)
                all_top5_probs.extend(top5_sum[0].cpu().tolist())

                # Renormalised top-k distribution and its entropy
                topk_norm = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                topk_log = torch.log(topk_norm.clamp(min=1e-12))
                topk_entropy = -(topk_norm * topk_log).sum(dim=-1)  # (B, T)
                all_topk_entropy.extend(topk_entropy[0].cpu().tolist())

                # top1 - top2 logprob gap (from raw probabilities, not the
                # renormalised top-k — this is what "decisiveness" means in
                # the literature).
                if k >= 2:
                    log_top = torch.log(topk_p.clamp(min=1e-12))
                    gap = (log_top[..., 0] - log_top[..., 1])  # (B, T)
                    all_top1_top2_gap_logprob.extend(gap[0].cpu().tolist())

        ent_arr = np.array(all_entropies)
        top1_arr = np.array(all_top1_probs)
        top5_arr = np.array(all_top5_probs)
        topk_ent_arr = np.array(all_topk_entropy) if all_topk_entropy else np.array([np.nan])
        gap_arr = np.array(all_top1_top2_gap_logprob) if all_top1_top2_gap_logprob else np.array([np.nan])

        return {
            "mean_entropy": float(np.mean(ent_arr)),
            "std_entropy": float(np.std(ent_arr)),
            "median_entropy": float(np.median(ent_arr)),
            "p90_entropy": float(np.percentile(ent_arr, 90)),
            "mean_top1_prob": float(np.mean(top1_arr)),
            "mean_top5_prob": float(np.mean(top5_arr)),
            f"mean_top{decisiveness_k}_entropy": float(np.nanmean(topk_ent_arr)),
            f"top{decisiveness_k}_entropy_p90": float(np.nanpercentile(topk_ent_arr, 90)),
            "mean_top1_top2_gap_logprob": float(np.nanmean(gap_arr)),
            "median_top1_top2_gap_logprob": float(np.nanmedian(gap_arr)),
        }
