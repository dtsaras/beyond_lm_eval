
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
        
        if dataset is None:
             try:
                 from datasets import load_dataset
                 dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                 dataset = []
                 for i in range(min(num_samples, len(dset))):
                     dataset.append({"text": dset[i]["text"]})
             except ImportError:
                 logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                 dataset = [{"text": "The quick brown fox jumps over the lazy dog."}]
        if cache is not None and cache.is_populated:
            stats, _ = cache.get_prediction_stats()
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
                
        return {
            "ece": float(ece),
            "num_predictions": total_samples,
            # "bin_detailed": bin_stats
        }
