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
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Rare Token PPL Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        if cache is not None and cache.is_populated:
            stats, _ = cache.get_prediction_stats()
        else:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=self.config.get("num_samples", 100))
        
        # Categorize tokens
        token_counts = stats["token_counts"]
        sorted_ids = np.argsort(token_counts)
        vocab_size = len(token_counts)
        
        rare_thresh = int(vocab_size * 0.2)
        freq_thresh = int(vocab_size * 0.8)
        
        rare_ids = set(sorted_ids[:rare_thresh])
        freq_ids = set(sorted_ids[freq_thresh:])
        
        nll_rare, cnt_rare = 0, 0
        nll_freq, cnt_freq = 0, 0
        
        for logits, labels in zip(stats["logits"], stats["labels"]):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            
            for l, t in zip(loss, labels.view(-1)):
                tid = t.item()
                if tid in rare_ids:
                    nll_rare += l.item()
                    cnt_rare += 1
                elif tid in freq_ids:
                    nll_freq += l.item()
                    cnt_freq += 1
                    
        return {
            "ppl_rare": float(np.exp(nll_rare / cnt_rare)) if cnt_rare > 0 else float("inf"),
            "ppl_freq": float(np.exp(nll_freq / cnt_freq)) if cnt_freq > 0 else float("inf")
        }
