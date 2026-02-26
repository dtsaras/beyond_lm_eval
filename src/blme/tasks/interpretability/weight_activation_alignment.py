"""
Weight-Activation Alignment (WAA) Task
──────────────────────────────────────────────────────────────────────
Evaluates mechanistic capacity utilization by measuring the cosine similarity
between the principal components of the actual forward-pass activations and 
the principal singular vectors of the static layer weights.

A high alignment score suggests the model is efficiently utilizing the 
feature directions inherently encoded in its weights during inference. A low 
score implies representation collapse or underutilization of parameter capacity.

References:
- General mechanistic interpretability and capacity utilization (2024-2025).
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task

def get_layers(model):
    """
    Heuristic to extract the transformer layers from various HF architectures.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers # Llama / Mistral
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h # GPT-2
    elif hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        return model.base_model.encoder.layer # BERT
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    else:
        # Fallback to searching for ModuleLists
        for name, module in model.named_modules():
             if isinstance(module, torch.nn.ModuleList) and ("layer" in name.lower() or "h" in name.lower() or "block" in name.lower()):
                 return module
        raise ValueError(f"Could not automatically locate transformer layers in architecture: {model.__class__.__name__}")


@register_task("interpretability_waa")
class WeightActivationAlignmentTask(DiagnosticTask):
    """
    Computes structural alignment between static layer weights (via SVD) 
    and empirical activation vectors (via PCA).
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Weight-Activation Alignment...")
        num_samples = self.config.get("num_samples", 5)
        
        if dataset is None:
             try:
                 from datasets import load_dataset
                 dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                 dataset = []
                 for i in range(min(num_samples, len(dset))):
                     dataset.append({"text": dset[i]["text"]})
             except ImportError:
                 print("Warning: `datasets` library not found. Falling back to default examples.")
                 dataset = [{"text": "The geometry of activations defines the expressivity of the model."}] * num_samples
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        layers = get_layers(model)
        
        # We need to hook into the MLP/FFN output projection layer
        # Heuristic to find the main output projection matrix per layer
        target_modules = []
        for l_idx, layer in enumerate(layers):
            # Try to find the down projection or c_proj (GPT style) or dense (BERT style)
            # Typically, the second linear layer in the MLP
            mlp = getattr(layer, "mlp", None) or getattr(layer, "output", None) or getattr(layer, "feed_forward", None)
            
            proj = None
            if mlp is not None:
                if hasattr(mlp, "c_proj"): # GPT2
                    proj = mlp.c_proj
                elif hasattr(mlp, "down_proj"): # Llama
                    proj = mlp.down_proj
                elif hasattr(mlp, "dense"): # BERT
                    proj = mlp.dense
            
            if proj is not None and hasattr(proj, "weight"):
                 target_modules.append((l_idx, proj))
                 
        if not target_modules:
             return {"error": "Could not identify standard MLP projection layers for WAA computation."}
             
        # Dictionary to store mean alignment per layer
        alignments = {}
        
        for l_idx, proj in target_modules:
            W = proj.weight.detach().float()
            # If Conv1D (like GPT2), weights are transposed
            if len(W.shape) == 2 and W.shape[0] < W.shape[1] and hasattr(proj, "nf"): 
                pass # Already right shape for out = in @ W
            else:
                W = W.T # Standard linear out = in @ W^T, so we transpose
                
            # Get the top singular vector of the weight matrix
            # W has shape (in_features, out_features)
            U, S, V = torch.svd(W, compute_uv=True)
            top_weight_vector = U[:, 0].unsqueeze(0) # (1, in_features)
            
            # Hook to collect activations entering this projection
            activations = []
            def hook_fn(module, input_args, output):
                # input_args[0] is typically the activation entering the layer
                act = input_args[0].detach().cpu().float()
                activations.append(act.view(-1, act.shape[-1])) # Flatten batch and seq
                
            handle = proj.register_forward_hook(hook_fn)
            
            # Run forward pass
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    model(**inputs)
                    
            handle.remove()
            
            if not activations:
                continue
                
            # Compute principal component of activations
            all_acts = torch.cat(activations, dim=0) # (N, in_features)
            
            # Center activations
            all_acts = all_acts - all_acts.mean(dim=0, keepdim=True)
            
            # Since N can be large, we can compute PCA via SVD or Covariance
            # If N > 10000, covariance is better
            if all_acts.shape[0] > 5000:
                cov = (all_acts.T @ all_acts) / (all_acts.shape[0] - 1)
                L, Q = torch.linalg.eigh(cov)
                top_act_vector = Q[:, -1].unsqueeze(0) # eigh sorts ascending
            else:
                U_a, S_a, V_a = torch.svd(all_acts, compute_uv=True)
                top_act_vector = V_a[:, 0].unsqueeze(0)
                
            # Align shapes
            top_weight_vector = top_weight_vector.to(top_act_vector.device)
            
            if top_weight_vector.shape[-1] != top_act_vector.shape[-1]:
                # Transpose Mismatch fallback
                top_weight_vector = V[:, 0].unsqueeze(0).to(top_act_vector.device)
                
            # Compute absolute cosine similarity (we only care about axis alignment, not sign)
            cos_sim = torch.nn.functional.cosine_similarity(top_weight_vector, top_act_vector)
            alignment = float(torch.abs(cos_sim).mean().item())
            
            alignments[str(l_idx)] = alignment
            
        if not alignments:
             return {"error": "Failed to collect layer activations for WAA."}
             
        return {
            "mean_waa_alignment": sum(alignments.values()) / len(alignments),
            "layer_waa_alignments": alignments
        }
