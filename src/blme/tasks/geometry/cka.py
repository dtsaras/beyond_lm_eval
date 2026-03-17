
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from .utils import collect_hidden_states
import logging
logger = logging.getLogger("blme")

@register_task("geometry_cka")
class CKATask(DiagnosticTask):
    """
    Computes Centered Kernel Alignment (CKA) between all layers of the model.
    Focuses on Linear CKA which is efficient for N > D.
    Ref: Kornblith et al. (2019)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running CKA Layer Similarity Analysis...")
        
        if dataset is None:
            # Mock dataset if missing
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        # 1. Collect Activations
        # We need synchronization: same tokens for all layers.
        # collect_hidden_states(..., layer_idx="all") does this by collecting from the same forward pass.
        
        # Limit N to avoid OOM if we keep all X in memory?
        # For CKA, we need X (N, D). If N=10k, D=4k, X is 40MB. 32 layers -> 1.2GB. Feasible.
        
        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        logger.info(f"  Collecting hidden states for {num_samples} samples...")
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
        else:
            layer_activations = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples, layer_idx="all")
        
        layers = sorted(layer_activations.keys())
        n_layers = len(layers)
        
        # Pre-process: Center the columns (features) or rows?
        # Linear CKA: centered columns of X?
        # Actually, standard Linear CKA definition:
        # CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        # where X, Y are centered.
        # Centering: equivalent to subtracting mean from each column.
        
        logger.info("  Centering activations...")
        centered_acts = {}
        norms = {} # ||X^T X||_F
        
        for idx in tqdm(layers, desc="Centering"):
            X = layer_activations[idx].float() # (N, D)
            # Center X
            X = X - X.mean(dim=0, keepdim=True)
            centered_acts[idx] = X
            
            # Compute denominator term
            # ||X^T X||_F
            # This can be large.
            # Optimization: ||X^T X||_F = sqrt(sum((X^T X)^2))
            
            # If D is large, X^T X is (D, D). 4096^2 floats = 64MB. Fine.
            xtx = X.t() @ X
            norms[idx] = torch.norm(xtx, p='fro').item()
            
            # Free original X if memory tight? No need yet.
            
        logger.info("  Computing CKA Matrix...")
        cka_matrix = np.zeros((n_layers, n_layers))
        
        for i in tqdm(range(n_layers), desc="CKA Rows"):
            idx_i = layers[i]
            X = centered_acts[idx_i]
            norm_x = norms[idx_i]
            
            for j in range(i, n_layers): # Symmetric
                idx_j = layers[j]
                Y = centered_acts[idx_j]
                norm_y = norms[idx_j]
                
                # Numerator: ||Y^T X||_F^2
                # Y^T X is (D, D)
                ytx = Y.t() @ X
                numerator = torch.norm(ytx, p='fro').item() ** 2
                
                denom = norm_x * norm_y
                cka = numerator / denom if denom > 1e-12 else 0.0
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka
                
        # Structure results
        results = {
            "cka_matrix": cka_matrix.tolist(), # List of lists
            "layers": layers
        }
        
        # Add some summary stats
        # e.g. avg CKA between adjacent layers
        diagonal_off1 = [cka_matrix[i, i+1] for i in range(n_layers-1)]
        results["avg_adjacent_cka"] = float(np.mean(diagonal_off1)) if diagonal_off1 else 0.0
        
        return results
