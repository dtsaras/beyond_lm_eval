"""
Lipschitz Continuity Analysis — measures how smoothly representations
change between adjacent layers.

The Lipschitz constant of a layer transformation f: R^d -> R^d bounds
how much the output can change relative to the input:
    ||f(x) - f(y)|| <= L * ||x - y||

A high Lipschitz constant means the layer is "explosive" (small input 
changes cause large output changes), while a low constant means the 
layer is contractive (smoothing). This is closely related to training 
stability, gradient flow, and the model's sensitivity to perturbations.

We estimate this empirically by computing:
    L_hat(layer) = max_i ||h_{l+1}(x_i) - h_l(x_i)|| / ||h_l(x_i)||

References:
- "Spectral Normalization for Generative Adversarial Networks"
  (Miyato et al., ICLR 2018) — Lipschitz constraints for stability
- "Lipschitz Regularity of Deep Neural Networks" (Virmaux & Scaman, 2018)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch


@register_task("geometry_lipschitz")
class LipschitzContinuityTask(DiagnosticTask):
    """
    Estimates the empirical Lipschitz constant between consecutive layers.
    
    For each pair of adjacent layers (l, l+1), computes:
      - The ratio ||h_{l+1}(x) - h_l(x)|| / ||h_l(x)|| for each token x
      - Takes the mean and max over all tokens as estimates
    
    A model with uniform, moderate Lipschitz constants across layers has
    smoother, more stable transformations. Spike patterns indicate layers
    where representations undergo dramatic restructuring.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Lipschitz Continuity Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        num_samples = self.config.get("num_samples", 20)
        
        # Collect hidden states from ALL layers
        all_layers = collect_hidden_states(model, tokenizer, dataset,
                                           num_samples=num_samples,
                                           layer_idx="all")
        
        if not all_layers:
            return {"error": "No hidden states collected"}
            
        layer_indices = sorted(all_layers.keys())
        n_layers = len(layer_indices)
        
        if n_layers < 2:
            return {"error": "Need at least 2 layers for Lipschitz analysis"}
        
        lipschitz_means = []
        lipschitz_maxes = []
        contraction_rates = []  # ratio < 1 = contractive, > 1 = expansive
        
        for i in range(n_layers - 1):
            h_l = all_layers[layer_indices[i]].numpy()
            h_next = all_layers[layer_indices[i + 1]].numpy()
            
            # Align shapes (use minimum token count)
            n_tokens = min(len(h_l), len(h_next))
            h_l = h_l[:n_tokens]
            h_next = h_next[:n_tokens]
            
            # Compute norms
            diff_norms = np.linalg.norm(h_next - h_l, axis=1)
            input_norms = np.linalg.norm(h_l, axis=1)
            
            # Avoid division by zero
            valid = input_norms > 1e-8
            if not np.any(valid):
                lipschitz_means.append(0.0)
                lipschitz_maxes.append(0.0)
                contraction_rates.append(1.0)
                continue
                
            ratios = diff_norms[valid] / input_norms[valid]
            
            lipschitz_means.append(float(np.mean(ratios)))
            lipschitz_maxes.append(float(np.max(ratios)))
            
            # Contraction rate: ||h_{l+1}(x)|| / ||h_l(x)||
            output_norms = np.linalg.norm(h_next, axis=1)
            cr = output_norms[valid] / input_norms[valid]
            contraction_rates.append(float(np.mean(cr)))
        
        lip_arr = np.array(lipschitz_means)
        
        return {
            "lipschitz_mean": float(np.mean(lip_arr)),
            "lipschitz_max": float(np.max(lip_arr)),
            "lipschitz_std": float(np.std(lip_arr)),
            "lipschitz_max_layer": int(np.argmax(lip_arr)),
            "mean_contraction_rate": float(np.mean(contraction_rates)),
            "contraction_std": float(np.std(contraction_rates)),
        }
