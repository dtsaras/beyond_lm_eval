"""
Data Contamination Scoring — Min-k% Probability Method
──────────────────────────────────────────────────────────────────────
Scores whether text looks unusually high-likelihood under a model by
analyzing the distribution of per-token log probabilities. Detection
thresholds are only reported when labeled calibration examples are
provided.

The Min-k% method (Shi et al., 2023) identifies contamination by checking
if the lowest-probability tokens in a passage are still unusually high —
a signature of memorized (rather than generalized) text.

References:
- "Detecting Pretraining Data from Large Language Models"
  (Shi et al., 2023). arXiv:2310.16789
"""

import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


def _min_k_mean_logprob(token_logprobs, k_pct: float) -> float:
    """Shi et al. Min-K% probability score on token log-probabilities.

    The paper selects the lowest-probability tokens; in log space these
    are the most negative log-probabilities. BLME returns their mean log
    probability, so higher (less negative) means more memorization-like.
    """
    token_lps = np.asarray(token_logprobs, dtype=np.float64)
    token_lps = token_lps[np.isfinite(token_lps)]
    if token_lps.size == 0:
        return float("nan")
    k_count = max(1, int(token_lps.size * float(k_pct) / 100.0))
    return float(np.mean(np.sort(token_lps)[:k_count]))


@register_task("consistency_contamination")
class ContaminationDetectionTask(DiagnosticTask):
    """
    Scores potential data contamination using the Min-k% probability method.

    Computes per-token log probabilities and checks whether the bottom-k%
    tokens have unusually high probabilities, a memorization proxy.
    Primary score is the raw min-k% mean log probability (higher = more
    likely memorized). Also reports a calibrated threshold only when labels
    are supplied.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Data Contamination Detection (Min-k%%)...")
        num_samples = self.config.get("num_samples", 10)
        k_pct = self.config.get("k_pct", 20)  # bottom k% of tokens

        device = next(model.parameters()).device

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        all_min_k_probs = []
        all_mean_logprobs = []
        all_std_logprobs = []
        labels = []

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=512).to(device)
                input_ids = inputs["input_ids"]

                if input_ids.shape[1] < 3:
                    continue

                outputs = model(**inputs)
                logits = outputs.logits  # (1, T, V)

                # Shift: predict token t from context [0..t-1]
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                # Per-token log probabilities — cast to fp32 before
                # log_softmax so bf16 models (which otherwise return
                # bf16 log-probs that numpy cannot convert) work.
                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs.gather(
                    2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)  # (1, T-1)

                token_lps = token_log_probs[0].cpu().numpy()

                if len(token_lps) < 2:
                    continue

                # Mean and std of all log probs for this passage
                mean_lp = float(np.mean(token_lps))
                std_lp = float(np.std(token_lps))
                all_mean_logprobs.append(mean_lp)
                all_std_logprobs.append(std_lp)

                # Min-k%: take the bottom k% of token log probs.
                min_k_mean = _min_k_mean_logprob(token_lps, k_pct)
                all_min_k_probs.append(min_k_mean)
                if isinstance(s, dict) and "label" in s:
                    labels.append(_normalise_contamination_label(s["label"]))
                else:
                    labels.append(None)

        if not all_min_k_probs:
            return {"error": "No valid samples processed."}

        # Primary contamination score: raw min-k% mean log prob (Shi et al.)
        # Higher (less negative) = more likely memorized
        min_k_score = float(np.mean(all_min_k_probs))
        mean_logprob = float(np.mean(all_mean_logprobs))

        # Per-passage z-score: how unusual is the min-k% average relative to
        # the full passage distribution
        z_scores = []
        for mk, mu, sd in zip(all_min_k_probs, all_mean_logprobs, all_std_logprobs):
            if sd > 1e-8:
                z_scores.append((mk - mu) / sd)
            else:
                z_scores.append(0.0)

        # Legacy ratio metric (secondary)
        if mean_logprob != 0:
            contamination_ratio = float(min_k_score / mean_logprob)
        else:
            contamination_ratio = 0.0

        label_values = [label for label in labels if label is not None]
        has_calibration = (
            len(label_values) == len(all_min_k_probs)
            and len(set(label_values)) == 2
        )
        calibration_mode = self.config.get("calibration_mode", "in_sample")
        holdout_frac = float(self.config.get("calibration_holdout_frac", 0.3))

        result = {
            "score_semantics": "calibrated_min_k_detection" if has_calibration else "uncalibrated_min_k_score_only",
            "is_calibrated_detection": bool(has_calibration),
            "min_k_score": min_k_score,
            "contamination_z_score": float(np.mean(z_scores)),
            "contamination_ratio": contamination_ratio,
            "mean_token_logprob": mean_logprob,
            "k_pct": k_pct,
            "num_samples_analyzed": len(all_min_k_probs),
        }
        if has_calibration:
            if (
                calibration_mode == "held_out"
                and len(all_min_k_probs) >= 6
            ):
                split = max(2, int(len(all_min_k_probs) * (1.0 - holdout_frac)))
                if split < len(all_min_k_probs) - 1:
                    cal_scores = all_min_k_probs[:split]
                    cal_labels = label_values[:split]
                    eval_scores = all_min_k_probs[split:]
                    eval_labels = label_values[split:]
                    threshold, _ = _best_threshold(cal_scores, cal_labels)
                    eval_preds = [1 if score >= threshold else 0 for score in eval_scores]
                    eval_accuracy = sum(
                        int(pred == label)
                        for pred, label in zip(eval_preds, eval_labels)
                    ) / len(eval_labels)
                    result.update({
                        "calibration_mode": "held_out",
                        "held_out_threshold": threshold,
                        "held_out_accuracy": float(eval_accuracy),
                        "held_out_auroc": _binary_auroc(eval_labels, eval_scores),
                        "calibration_train_n": split,
                        "calibration_eval_n": len(eval_scores),
                    })
                else:
                    calibration_mode = "in_sample"

            if calibration_mode != "held_out" or "held_out_threshold" not in result:
                threshold, accuracy = _best_threshold(all_min_k_probs, label_values)
                result.update({
                    "calibration_mode": "in_sample",
                    "in_sample_threshold": threshold,
                    "in_sample_accuracy": accuracy,
                    "in_sample_auroc": _binary_auroc(label_values, all_min_k_probs),
                    "calibration_warning": (
                        "Threshold was fit on the same samples it was evaluated "
                        "against; use calibration_mode='held_out' for a held-out "
                        "calibration split."
                    ),
                    # Legacy aliases retained for downstream compatibility.
                    "calibrated_threshold": threshold,
                    "calibrated_accuracy": accuracy,
                    "calibrated_auroc": _binary_auroc(label_values, all_min_k_probs),
                })
        else:
            result["diagnostic_warning"] = (
                "Min-k scores are score-only without labeled calibration data; "
                "no contamination detection threshold was fitted."
            )
        return result


def _normalise_contamination_label(label):
    if label in (1, "1", True, "contaminated", "member", "train", "training"):
        return 1
    if label in (0, "0", False, "clean", "non_member", "non-member", "heldout", "test"):
        return 0
    return None


def _binary_auroc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    concordant = sum(1 for pos in positives for neg in negatives if pos > neg)
    ties = sum(0.5 for pos in positives for neg in negatives if pos == neg)
    return float((concordant + ties) / (len(positives) * len(negatives)))


def _best_threshold(scores, labels):
    """Choose the score threshold that maximizes in-sample calibration accuracy."""
    candidates = sorted(set(float(score) for score in scores))
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for threshold in candidates:
        preds = [1 if score >= threshold else 0 for score in scores]
        accuracy = sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return float(best_threshold), float(best_accuracy)
