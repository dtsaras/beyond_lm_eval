from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
from .utils import collect_prediction_stats
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")

@register_task("geometry_consistency")
class ConsistencyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Geometric Consistency Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        if cache is not None and cache.is_populated and use_cache:
            stats, embeddings = cache.get_prediction_stats(num_samples=num_samples)
        else:
            stats, embeddings = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)
        
        if embeddings is None:
            # Fallback to shared utility
            embeddings = get_embeddings(model)
        if embeddings is None:
            return {"error": "Could not access embeddings"}
            
        embeddings = embeddings.cpu()
        cosine_sims = []
        
        for h, labels in zip(stats["hidden"], stats["labels"]):
            # With flattened stats: h is (N, D), labels is (N,)
            if h.dim() == 3:
                B, T, D = h.shape
                h = h.reshape(-1, D)
                labels = labels.reshape(-1)

            target_embs = F.embedding(labels, embeddings)
            
            h_norm = F.normalize(h.float(), p=2, dim=-1)
            e_norm = F.normalize(target_embs.float(), p=2, dim=-1)
            
            cos = (h_norm * e_norm).sum(dim=-1)
            cosine_sims.extend(cos.tolist())
            
        return {
            "cosine_consistency_mean": float(np.mean(cosine_sims)),
            "cosine_consistency_std": float(np.std(cosine_sims))
        }
