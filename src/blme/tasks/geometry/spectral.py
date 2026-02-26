
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm

@register_task("geometry_spectral")
class WeightSpectralTask(DiagnosticTask):
    """
    Analyzes the spectral properties of weight matrices.
    Metrics:
    - Stable Rank: ||W||_F^2 / ||W||_2^2 (Bartlett et al., 2020)
    - Power Law Alpha: Fit to singular value distribution (Martin & Mahoney, 2021)
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Weight Spectral Analysis...")
        
        # No dataset needed for weight analysis
        # But base class expects it. Usually ignored.
        
        # Iterate over all linear layers
        # Focus on attention and MLP layers
        
        results = {}
        layer_stats = {}
        
        TARGET_MODULES = (torch.nn.Linear, torch.nn.Conv1d) # Add others if needed
        
        alphas = []
        stable_ranks = []
        
        # Traverse modules
        modules_to_scan = []
        for name, module in model.named_modules():
            if isinstance(module, TARGET_MODULES):
                # Filter out small projections/heads/embeddings if desired?
                # Usually we want the main matrices: Q, K, V, O, Up, Down, Gate
                if "weight" in module._parameters and module.weight is not None:
                    modules_to_scan.append((name, module))
                    
        print(f"  Found {len(modules_to_scan)} linear modules.")
        
        for name, module in tqdm(modules_to_scan, desc="Analyzing Weights"):
            W = module.weight.detach().float()
            
            # Conv1D weights in GPT2 are (F_in, F_out)? Usually (Out, In) for Linear
            if W.dim() != 2:
                continue
                
            # SVD
            # Use randomized SVD for speed if matrix is huge, or full SVD
            # torch.linalg.svd returns U, S, Vh
            # S are singular values in descending order
            
            try:
                S = torch.linalg.svdvals(W)
                     
                S_np = S.cpu().numpy()
                
                # 1. Stable Rank
                # ||W||_F^2 = sum(S^2)
                # ||W||_2^2 = max(S)^2 = S[0]^2
                if len(S_np) > 0 and S_np[0] > 0:
                    fro_sq = np.sum(S_np**2)
                    spec_sq = S_np[0]**2
                    stable_rank = fro_sq / spec_sq
                else:
                    stable_rank = 0.0
                    
                # 2. Power Law Alpha (Martin & Mahoney)
                # Fit ESD to P(x) ~ x^-alpha
                # Use Hill estimator on the tail (e.g. top 20% or values > threshold)
                # Or simplistic log-log regression.
                # Martin & Mahoney typically look at the "bulk" vs "tail".
                # For simplicity, we fit the top 50% of singular values?
                # Or use the Hill estimator on the largest K values.
                # ideally K is determined dynamically. Let's pick top 20% by count.
                
                k = max(2, int(0.2 * len(S_np)))
                top_k = S_np[:k]
                
                # Hill Estimator: alpha = 1 + k / sum(ln(x_i / x_min))
                # where x_min = S_np[k-1] (the k-th largest value)
                if k > 0 and top_k[-1] > 1e-6:
                    x_min = top_k[-1]
                    # Filter out any exactly equal to x_min to avoid log(1)=0 division issues?
                    # The formula sums ln(x_i / x_min).
                    # x_i >= x_min.
                    
                    log_sum = np.sum(np.log(top_k / x_min))
                    if log_sum > 0:
                        alpha = 1 + k / log_sum
                    else:
                        alpha = 0.0
                else:
                    alpha = 0.0
                    
                layer_stats[name] = {
                    "stable_rank": float(stable_rank),
                    "alpha": float(alpha),
                    "spectral_norm": float(S_np[0]) if len(S_np)>0 else 0
                }
                
                alphas.append(alpha)
                stable_ranks.append(stable_rank)
                
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                continue
                
        # Aggregate
        results["avg_alpha"] = float(np.mean(alphas)) if alphas else 0.0
        results["avg_stable_rank"] = float(np.mean(stable_ranks)) if stable_ranks else 0.0
        results["min_alpha"] = float(np.min(alphas)) if alphas else 0.0
        results["max_alpha"] = float(np.max(alphas)) if alphas else 0.0
        
        # Detailed stats?
        # results["details"] = layer_stats 
        
        return results
