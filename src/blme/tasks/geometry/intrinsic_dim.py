from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from .utils import collect_hidden_states


@register_task("geometry_intrinsic_dim")
class IntrinsicDimensionTask(DiagnosticTask):
    """
    Estimates the Intrinsic Dimension (ID) of the embedding manifold using the Two-NN estimator.
    Ref: 'Estimating the intrinsic dimension of datasets by a minimal neighborhood information' (Facco et al., 2017)
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Intrinsic Dimension Estimation (Two-NN)...")
        
        # Check mode: Embeddings (static) or Layer-wise Activations (dynamic)
        layerwise = self.config.get("layerwise", False)
        
        if layerwise:
            print("  Mode: Layer-wise Activations")
            if dataset is None:
                # Mock dataset if missing
                dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
                
            # Collect states from all layers
            print("  Collecting hidden states...")
            # Use 'all' to get dict of {layer_idx: tensor}
            layer_activations = collect_hidden_states(model, tokenizer, dataset, num_samples=self.config.get("num_samples", 100), layer_idx="all")
            
            results = {}
            # Compute ID for each layer
            sorted_layers = sorted(layer_activations.keys())
            id_trend = []
            
            for layer_idx in tqdm(sorted_layers, desc="Computing Layer IDs"):
                X = layer_activations[layer_idx].float().numpy()
                # Subsample if too large
                if len(X) > 20000:
                    indices = np.random.choice(len(X), 20000, replace=False)
                    X = X[indices]
                    
                lid_result = self._compute_id(X)
                lid = lid_result["intrinsic_dimension"]
                results[f"lid_layer_{layer_idx}"] = lid
                id_trend.append(lid)
                
            results["lid_trend"] = id_trend
            return results
            
        else:
             print("  Mode: Static Embeddings")
             # 1. Get Embeddings
             from ..common import get_embeddings as _get_emb
             E = _get_emb(model)
             if E is None:
                 return {"error": "Could not extract embeddings"}
             
             E_np = E.float().cpu().numpy()
             return self._compute_id(E_np, sample_size=self.config.get("sample_size", None))

    def _compute_id(self, X, sample_size=None):
        n_vocab = len(X)
        if sample_size and sample_size < n_vocab:
            indices = np.random.choice(n_vocab, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        N = len(X_sample)
        
        # Use sklearn for Two-NN
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='euclidean', n_jobs=-1).fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            
            mus = []
            for j in range(len(distances)):
                d = distances[j]
                d = d[d > 1e-6]
                if len(d) >= 2:
                    r1 = d[0]
                    r2 = d[1]
                    if r1 > 0:
                        mus.append(r2 / r1)
                        
            mus = np.array(mus)
            
            if len(mus) > 0:
                # Facco et al. Eq 11
                full_sum = np.sum(np.log(mus))
                intrinsic_dim = len(mus) / full_sum
            else:
                intrinsic_dim = 0.0
                
        except ImportError:
            print("sklearn not installed, skipping ID estimation")
            intrinsic_dim = 0.0

        return {
            "intrinsic_dimension": float(intrinsic_dim),
            "sample_size": N
        }

