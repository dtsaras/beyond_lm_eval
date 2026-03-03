from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM
import logging
logger = logging.getLogger("blme")

@register_task("dynamics_stability")
class NeighborhoodStabilityTask(DiagnosticTask):
    """
    Measures Jaccard stability of k-NN neighborhoods.
    If 'reference_model_path' is provided in config, compares current model to reference.
    Otherwise, compares model against itself (sanity check = 1.0) or different seed/noise if implemented.
    For now, standard usage requires a reference.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Neighborhood Stability Analysis...")
        k = self.config.get("k", 50)
        n_sample = self.config.get("n_sample", 5000)
        ref_path = self.config.get("reference_model_path", None)
        
        device = next(model.parameters()).device
        E1 = get_embeddings(model)
        if E1 is None: return {"error": "E1 not found"}
        E1 = E1.to(device)
        
        if ref_path:
            logger.info(f"Loading reference model from {ref_path}...")
            # This is heavy. Optimally we just load embeddings if possible.
            # But assume we load model for compatibility.
            # Warning: Memory usage.
            try:
                ref_model = AutoModelForCausalLM.from_pretrained(ref_path, trust_remote_code=True).to('cpu') # Keep on CPU first
                E2 = get_embeddings(ref_model)
                if E2 is None: return {"error": "E2 not found"}
                # Move E2 to device only if needed, or keep both on CPU if VRAM tight
                # For safety, let's do calculation on CPU if ref provided, or E2 to device
                E2 = E2.to(device)
                del ref_model
                torch.cuda.empty_cache()
            except Exception as e:
                return {"error": f"Failed to load ref model: {e}"}
        else:
            logger.info("No reference model provided. Calculating self-stability (should be 1.0).")
            E2 = E1
            
        # Compute Stability
        E1_np = E1.float().cpu().numpy()
        E2_np = E2.float().cpu().numpy()
        
        n_vocab = len(E1_np)
        if len(E2_np) != n_vocab:
            return {"error": f"Vocab mismatch: {n_vocab} vs {len(E2_np)}"}
            
        # Normalize
        E1_norm = E1_np / (np.linalg.norm(E1_np, axis=1, keepdims=True) + 1e-10)
        E2_norm = E2_np / (np.linalg.norm(E2_np, axis=1, keepdims=True) + 1e-10)
        
        np.random.seed(42)
        sample_idx = np.random.choice(n_vocab, min(n_sample, n_vocab), replace=False)
        
        jaccards = []
        
        for idx in tqdm(sample_idx, desc="Stability Jaccard"):
            # kNN 1
            sims1 = E1_norm @ E1_norm[idx]
            sims1[idx] = -np.inf
            nn1 = set(np.argsort(sims1)[-k:])
            
            # kNN 2
            sims2 = E2_norm @ E2_norm[idx]
            sims2[idx] = -np.inf
            nn2 = set(np.argsort(sims2)[-k:])
            
            jaccard = len(nn1 & nn2) / len(nn1 | nn2)
            jaccards.append(jaccard)
            
        return {
            "stability_mean": float(np.mean(jaccards)),
            "stability_std": float(np.std(jaccards))
        }
