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
from ..common import get_layers
import logging
logger = logging.getLogger("blme")


@register_task("interpretability_waa")
class WeightActivationAlignmentTask(DiagnosticTask):
    """
    Computes structural alignment between static layer weights (via SVD) 
    and empirical activation vectors (via PCA).
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Weight-Activation Alignment...")
        num_samples = self.config.get("num_samples", 5)
        
        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)
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
        
        # Detect Conv1D class once. GPT-2 uses transformers.pytorch_utils.Conv1D,
        # whose `weight` is shape (in, out) and the forward op is `x @ W`.
        # nn.Linear's `weight` is shape (out, in) and the forward op is `x @ W^T`.
        try:
            from transformers.pytorch_utils import Conv1D as _HFConv1D
        except Exception:
            _HFConv1D = None

        for l_idx, proj in target_modules:
            W = proj.weight.detach().float()
            # Normalize to (in, out) so U[:, 0] lives in the *input* feature
            # space and matches the activation principal component below.
            if _HFConv1D is not None and isinstance(proj, _HFConv1D):
                pass  # Already (in, out)
            else:
                # nn.Linear
                W = W.T  # (out, in) -> (in, out)

            # SVD: W = U @ diag(S) @ V^T, with U in input space.
            try:
                U, S, V = torch.svd(W, compute_uv=True)
            except Exception as e:
                logger.info(f"  SVD failed on layer {l_idx}: {e}")
                continue
            top_weight_vector = U[:, 0].unsqueeze(0)  # (1, in_features)

            # Hook to collect activations entering this projection.
            activations = []
            def hook_fn(module, input_args, output):
                act = input_args[0].detach().cpu().float()
                activations.append(act.reshape(-1, act.shape[-1]))

            handle = proj.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    for s in samples:
                        text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                        inputs = tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=128).to(device)
                        model(**inputs)
            finally:
                handle.remove()

            if not activations:
                continue

            all_acts = torch.cat(activations, dim=0)  # (N, in_features)
            all_acts = all_acts - all_acts.mean(dim=0, keepdim=True)

            # Top principal component of activations (in input space)
            if all_acts.shape[0] > 5000:
                cov = (all_acts.T @ all_acts) / (all_acts.shape[0] - 1)
                L_eig, Q = torch.linalg.eigh(cov)
                top_act_vector = Q[:, -1].unsqueeze(0)  # eigh sorts ascending
            else:
                U_a, S_a, V_a = torch.svd(all_acts, compute_uv=True)
                top_act_vector = V_a[:, 0].unsqueeze(0)

            top_weight_vector = top_weight_vector.to(top_act_vector.device)

            # Both vectors should now be in the same (input feature) space.
            # If they aren't, there's a real bug — fail loudly instead of
            # silently falling back to a different vector.
            if top_weight_vector.shape[-1] != top_act_vector.shape[-1]:
                logger.info(
                    f"  WAA layer {l_idx}: dimension mismatch "
                    f"(weight={top_weight_vector.shape[-1]}, act={top_act_vector.shape[-1]}) — skipping"
                )
                continue

            # Absolute cosine similarity (sign doesn't matter for axis alignment).
            cos_sim = torch.nn.functional.cosine_similarity(top_weight_vector, top_act_vector)
            alignment = float(torch.abs(cos_sim).mean().item())

            alignments[str(l_idx)] = alignment
            
        if not alignments:
             return {"error": "Failed to collect layer activations for WAA."}
             
        return {
            "mean_waa_alignment": sum(alignments.values()) / len(alignments),
            "layer_waa_alignments": alignments
        }
