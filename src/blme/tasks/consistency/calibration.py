
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from ..geometry.utils import collect_prediction_stats
import logging
logger = logging.getLogger("blme")

@register_task("consistency_calibration")
class CalibrationTask(DiagnosticTask):
    """
    Computes Expected Calibration Error (ECE).
    Ref: Guo et al. (2017)
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
        
        # Binning
        n_bins = self.config.get("n_bins", 10)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        bin_stats = []
        
        total_samples = confidences.numel()
        
        for i in range(n_bins):
            # Bin range: [bin_boundaries[i], bin_boundaries[i+1]] — first bin uses >= to include 0
            if i == 0:
                mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            else:
                mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            
            if mask.any():
                bin_conf = confidences[mask].mean().item()
                bin_acc = accuracies[mask].float().mean().item()
                bin_prop = mask.sum().item() / total_samples
                
                ece += np.abs(bin_conf - bin_acc) * bin_prop
                
                bin_stats.append({
                    "range": f"{bin_boundaries[i]:.2f}-{bin_boundaries[i+1]:.2f}",
                    "confidence": bin_conf,
                    "accuracy": bin_acc,
                    "count": mask.sum().item()
                })
                
        # Brier score: mean squared error between confidence and correctness
        # Brier = E[(confidence - correct)^2]. Lower = better calibrated.
        conf_np = confidences.float().cpu().numpy()
        acc_np = accuracies.float().cpu().numpy()
        brier_score = float(np.mean((conf_np - acc_np) ** 2))

        # Calibration slope: linear fit of bin_acc vs bin_conf.
        # A perfectly calibrated model has slope = 1.0. Slope < 1 means
        # overconfident; slope > 1 means underconfident.
        if len(bin_stats) >= 3:
            bin_confs = np.array([b["confidence"] for b in bin_stats])
            bin_accs = np.array([b["accuracy"] for b in bin_stats])
            try:
                slope, intercept = np.polyfit(bin_confs, bin_accs, 1)
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
            "num_predictions": total_samples,
        }
