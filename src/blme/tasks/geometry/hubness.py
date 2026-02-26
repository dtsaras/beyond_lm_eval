from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from scipy.stats import skew
from tqdm import tqdm

@register_task("geometry_hubness")
class GlobalHubnessTask(DiagnosticTask):
    """
    Analyzes the "hubness" of the embedding space: the skewness of the distribution
    of k-nearest neighbor occurrences.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Global Hubness Analysis...")
        k_values = self.config.get("k_values", [10, 50, 100])
        batch_size = self.config.get("batch_size", 2000)
        
        E = get_embeddings(model)
        if E is None:
            return {"error": "Could not extract embeddings"}
                
        device = E.device
        E_np = E.float().cpu().numpy()
        n_vocab = len(E_np)
        
        # Normalize for cosine similarity
        E_norm = E_np / (np.linalg.norm(E_np, axis=1, keepdims=True) + 1e-10)
        
        results = {}
        
        for k in k_values:
            n_occ = np.zeros(n_vocab, dtype=np.int32)
            
            # Compute k-NN for all tokens in batches
            for i in tqdm(range(0, n_vocab, batch_size), desc=f"Hubness k={k}"):
                end = min(i + batch_size, n_vocab)
                
                # Similarity matrix slice: (batch, n_vocab)
                sims = E_norm[i:end] @ E_norm.T
                
                # Mask self-similarity (set to -inf)
                # dims: (batch_size, n_vocab)
                # For row j (which corresponds to token i+j), we mask column i+j
                for j in range(end - i):
                    sims[j, i + j] = -np.inf
                
                # Get top-k indices
                # argsort is expensive on full vocab, substitute with argpartition used carefully
                # or just topk if using torch?
                # Using numpy argsort on (2000, 50000) is reasonable.
                # optimization: argpartition for top k
                top_k_indices = np.argpartition(sims, -k, axis=1)[:, -k:]
                
                # Update occurrence counts
                # flattening is faster
                for idx in top_k_indices.flatten():
                    n_occ[idx] += 1
            
            # Metrics
            hub_skew = skew(n_occ)
            hub_max = int(n_occ.max())
            
            # Top 1% mass (concentration)
            top_1pct_threshold = np.percentile(n_occ, 99)
            top_1pct_mass = n_occ[n_occ >= top_1pct_threshold].sum() / n_occ.sum()
            
            # Gini Coefficient
            n_occ_sorted = np.sort(n_occ)
            n = len(n_occ_sorted)
            # Gini formula: (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
            # where i is 1-based rank
            if n_occ_sorted.sum() > 0:
                gini = (2 * np.sum((np.arange(1, n+1) * n_occ_sorted))) / (n * n_occ_sorted.sum()) - (n + 1) / n
            else:
                gini = 0.0
                
            results[f'hubness_k{k}_skew'] = float(hub_skew)
            results[f'hubness_k{k}_max'] = hub_max
            results[f'hubness_k{k}_top1pct'] = float(top_1pct_mass)
            results[f'hubness_k{k}_gini'] = float(gini)
            
        return results
