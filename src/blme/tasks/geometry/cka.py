
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
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)
            
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
                
        # Compute reviewer-meaningful scalar summaries and suppress
        # ``cka_matrix`` + ``layers`` from the top-level dict.
        #
        # Historic bug: the aggregator's ``_flatten_dict`` dropped
        # ``cka_matrix`` (list-of-lists, not 1-D) and then produced
        # ``cka.layers.{mean,std,min,max,slope,...}`` as descriptive
        # statistics of the layer-index vector [0, 1, …, N-1]. That
        # contaminated ~8 feature columns with pure layer-count
        # proxies — reviewer-visible if anyone inspects what those
        # columns actually encode.
        n = n_layers
        iu = np.triu_indices(n, k=1)  # off-diagonal upper triangle
        off_diag = cka_matrix[iu] if iu[0].size else np.array([], dtype=float)

        diagonal_off1 = np.array(
            [cka_matrix[i, i + 1] for i in range(n - 1)],
            dtype=float,
        ) if n > 1 else np.array([], dtype=float)

        def _stat(arr, fn, fallback=float("nan")):
            return float(fn(arr)) if arr.size else fallback

        results = {
            # Core adjacent-layer measure (paper's headline quantity).
            "avg_adjacent_cka": _stat(diagonal_off1, np.mean, 0.0),
            "min_adjacent_cka": _stat(diagonal_off1, np.min),
            "max_adjacent_cka": _stat(diagonal_off1, np.max),
            "std_adjacent_cka": _stat(diagonal_off1, np.std),
            # Global off-diagonal summary: picks up "all pairs of layers".
            "mean_offdiag_cka": _stat(off_diag, np.mean),
            "std_offdiag_cka": _stat(off_diag, np.std),
            "min_offdiag_cka": _stat(off_diag, np.min),
            "max_offdiag_cka": _stat(off_diag, np.max),
            # Early-vs-late comparison: layer 0 vs layer N-1.
            "early_late_cka": (
                float(cka_matrix[0, -1]) if n >= 2 else float("nan")
            ),
            # First-to-middle (captures early representation drift).
            "first_middle_cka": (
                float(cka_matrix[0, n // 2]) if n >= 3 else float("nan")
            ),
            # First-to-last normalised depth distance; lower => layers
            # diverge more across depth.
            "n_layers": int(n),
        }
        return results
