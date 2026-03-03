import torch
import numpy as np
import warnings

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")

try:
    from sae_lens import SAE
    HAS_SAE_LENS = True
except ImportError:
    HAS_SAE_LENS = False

@register_task("interpretability_sae_features")
class SAEFeatureDimensionalityTask(DiagnosticTask):
    """
    Measures true feature dimensionality using Sparse Autoencoders (SAEs).
    Optionally relies on `sae_lens` to load a pretrained SAE for a specific model layer
    and measures the average number of active features (L0 norm of feature activations)
    per token.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SAE Feature Dimensionality (Optional)...")
        num_samples = self.config.get("num_samples", 5)
        
        sae_release = self.config.get("sae_release", "gpt2-small-res-jb")
        sae_id = self.config.get("sae_id", "blocks.8.hook_resid_pre")
        
        if not HAS_SAE_LENS:
            msg = "sae_lens library not installed. Skipping SAE Feature Dimensionality module. Install with: pip install sae-lens"
            logger.info("  " + msg)
            return {"error": msg}
            
        device = next(model.parameters()).device
        
        try:
            logger.info(f"  Attempting to load SAE: release={sae_release}, id={sae_id}")
            # Loading the SAE requires an internet connection on first run to download from HF
            sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=str(device))
            sae.eval()
        except Exception as e:
            msg = f"Failed to load SAE {sae_release}/{sae_id}. This might be due to a mismatch with the model or internet access. Error: {e}"
            logger.info("  " + msg)
            return {"error": msg}

        if dataset is None:
            dataset = [
                {"text": "Sparse autoencoders help us understand true feature dimensions."}
            ] * num_samples
            
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample with 'text' key"}
            
        # Typically, SAEs in HookedTransformer format depend on specific layer blocks.
        # Since BLME is architecture-agnostic, we will loosely try to attach to the
        # middle layer or the specific layer the SAE was trained for.
        # To avoid complex hook injection for every possible model architecture, we will
        # run the model and extract the hidden states of the middle layer to pass to the SAE.
        
        layers = get_layers(model)
        num_layers = len(layers)
        target_layer = num_layers // 2 # Approximation for arbitrary models
        
        active_features_counts = []
        max_active_features = []
        
        with torch.no_grad():
            for s in samples:
                if isinstance(s, dict) and "text" in s:
                    text = s["text"]
                else:
                    text = str(s)
                    
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                
                # Get the target hidden state
                h = out.hidden_states[target_layer + 1][0] # shape (seq_len, hidden_dim)
                
                # Try to map hidden state to SAE
                # sae() returns sae_out, feature_acts, loss, ...
                try:
                    feature_acts = sae.encode(h)
                    
                    # Compute L0 (number of non-zero features per token)
                    l0_per_token = (feature_acts > 0).float().sum(dim=-1) # shape (seq_len,)
                    
                    active_features_counts.append(l0_per_token.mean().item()) # Average active per token in this sequence
                    max_active_features.append(l0_per_token.max().item())
                except RuntimeError as e:
                    return {"error": f"SAE dimension mismatch applied to layer {target_layer}. Error: {e}"}

        if not active_features_counts:
             return {"error": "Failed to collect any activations"}
             
        results = {
            "mean_active_features_l0": float(np.mean(active_features_counts)),
            "max_active_features_l0": float(np.max(max_active_features)),
            "sae_total_dict_size": sae.cfg.d_sae
        }
            
        return results
