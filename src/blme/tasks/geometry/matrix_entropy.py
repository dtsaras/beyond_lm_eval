"""
Matrix Entropy (Information Bottleneck) Task
──────────────────────────────────────────────────────────────────────
Evaluates the data compression (information bottleneck) properties of
representations by computing the von Neumann Spectral Entropy of the 
normalized covariance matrix of the hidden states at each layer.

As LLMs process tokens, they compress redundant information. A decreasing 
or low matrix entropy indicates the model is actively filtering noise 
and forming a tight semantic bottleneck.

References:
- "Matrix Entropy as an Intrinsic Metric for LLMs" (Wei et al., 2024).
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states


@register_task("geometry_matrix_entropy")
class MatrixEntropyTask(DiagnosticTask):
    """
    Computes the Information Bottleneck geometry of hidden representations
    via von Neumann spectral entropy over the internal covariance matrix.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Matrix Entropy (Information Bottleneck) Analysis...")
        num_samples = self.config.get("num_samples", 10)
        
        if dataset is None:
             dataset = [{"text": "Deep neural networks enforce an information bottleneck over layers."}] * num_samples
             
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        all_hidden_states = []
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                
                out = model(**inputs, output_hidden_states=True)
                # Filter out the embedding layer (index 0)
                hiddens = out.hidden_states[1:] 
                # Take average over sequence length for macro-level state
                h_mean = [h.mean(dim=1).detach().cpu() for h in hiddens]
                all_hidden_states.append(h_mean)
                
        if not all_hidden_states:
            return {"error": "Could not collect hidden states."}
            
        num_layers = len(all_hidden_states[0])
        layer_entropies = {}
        
        for l in range(num_layers):
            # Gather all batch samples for this layer
            # Shape: (num_samples, hidden_dim)
            H_l = torch.cat([batch[l] for batch in all_hidden_states], dim=0)
            
            # Center the data
            H_l = H_l - H_l.mean(dim=0, keepdim=True)
            
            # Compute Covariance Matrix: C = (H^T H) / N
            # For numerical stability on high-dim spaces, we can work directly with the SVD
            # of H, since the eigenvalues of C are the squared singular values of H.
            U, S, V = torch.svd(H_l, compute_uv=False)
            
            # True singular values, square to get covariance eigenvalues
            eigenvalues = (S ** 2) / (H_l.shape[0] - 1 + 1e-12)
            
            # Normalize to form a valid probability distribution (Trace(rho) = 1)
            rho = eigenvalues / torch.sum(eigenvalues)
            rho = rho[rho > 0] # Filter exactly 0
            
            # Compute von Neumann Entropy (Information Bottleneck capacity)
            entropy = -torch.sum(rho * torch.log(rho)).item()
            
            layer_entropies[f"layer_{l}"] = entropy
            
        return {
            "mean_matrix_entropy": sum(layer_entropies.values()) / len(layer_entropies),
            "layer_matrix_entropies": layer_entropies,
            "interpretation": "Lower downstream matrix entropy indicates a stronger information bottleneck (semantic compression)."
        }
