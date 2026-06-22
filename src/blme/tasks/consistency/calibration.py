
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from ..geometry.utils import collect_prediction_stats
import logging
logger = logging.getLogger("blme")


def _calibration_from_confidences(confidences, correct, n_bins: int = 10) -> dict:
    """Guo-style ECE and Brier score from confidences and correctness."""
    conf_np = np.asarray(confidences, dtype=np.float64)
    acc_np = np.asarray(correct, dtype=np.float64)
    finite = np.isfinite(conf_np) & np.isfinite(acc_np)
    conf_np = conf_np[finite]
    acc_np = acc_np[finite]
    if conf_np.size == 0:
        return {
            "ece": float("nan"),
            "brier_score": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
            "bin_stats": [],
            "num_predictions": 0,
        }

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    total = conf_np.size
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (conf_np >= lo) & (conf_np <= hi)
        else:
            mask = (conf_np > lo) & (conf_np <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf_np[mask]))
        bin_acc = float(np.mean(acc_np[mask]))
        bin_prop = float(np.sum(mask) / total)
        ece += abs(bin_conf - bin_acc) * bin_prop
        bin_stats.append({
            "range": f"{lo:.2f}-{hi:.2f}",
            "confidence": bin_conf,
            "accuracy": bin_acc,
            "count": int(np.sum(mask)),
        })

    brier_score = float(np.mean((conf_np - acc_np) ** 2))
    if len(bin_stats) >= 3:
        try:
            slope, intercept = np.polyfit(
                np.array([b["confidence"] for b in bin_stats]),
                np.array([b["accuracy"] for b in bin_stats]),
                1,
            )
            calibration_slope = float(slope)
            calibration_intercept = float(intercept)
        except Exception:
            calibration_slope = float("nan")
            calibration_intercept = float("nan")
    else:
        calibration_slope = float("nan")
        calibration_intercept = float("nan")

    return {
        "ece": float(ece),
        "brier_score": brier_score,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "bin_stats": bin_stats,
        "num_predictions": int(total),
    }


@register_task("consistency_calibration")
class CalibrationTask(DiagnosticTask):
    """
    Computes language-model calibration diagnostics.

    References:
      * Guo, Pleiss, Sun & Weinberger 2017 — Expected Calibration
        Error for neural classifier confidence.
      * Brier 1950 — mean squared probabilistic error.

    BLME adapts these to next-token teacher-forcing: "accuracy" is
    whether the model's top-1 next token matches corpus text, not a
    downstream QA correctness label.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Calibration Analysis (ECE)...")
        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        
        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)
        if cache is not None and cache.is_populated and use_cache:
            stats, _ = cache.get_prediction_stats(num_samples=num_samples)
        else:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)
        
        logits = torch.cat(stats["logits"], dim=0)  # (TotalTokens, V)
        labels = torch.cat(stats["labels"], dim=0)  # (TotalTokens,)
        
        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = predictions.eq(labels)

        # Drop non-finite confidences before binning. Historic bug:
        # fp16 logits on pythia-6.9b / 12b overflow to NaN → every
        # ``mask = (conf > lo) & (conf <= hi)`` is False, ECE stays at
        # the initial 0.0, and the model looks perfectly calibrated.
        finite = torch.isfinite(confidences) & torch.isfinite(probs).all(dim=-1)
        if not bool(finite.any()):
            return {
                "error": "All confidences are NaN/Inf (fp16 overflow?)",
                "ece": float("nan"),
                "brier_score": float("nan"),
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "num_predictions": 0,
            }
        confidences = confidences[finite]
        predictions = predictions[finite]
        accuracies = accuracies[finite]
        labels = labels[finite]

        n_bins = self.config.get("n_bins", 10)
        metrics = _calibration_from_confidences(
            confidences.float().cpu().numpy(),
            accuracies.float().cpu().numpy(),
            n_bins=n_bins,
        )
        metrics.pop("bin_stats", None)
        return metrics
