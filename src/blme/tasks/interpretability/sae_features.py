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


def _select_sae_hidden_state_index(sae_id, num_layers):
    """Map a TransformerLens SAE hook name to HF ``hidden_states`` index.

    HuggingFace hidden states are ``(embed, after block 0, ..., after block N)``.
    TransformerLens ``blocks.N.hook_resid_pre`` is the input to block N, so it
    lives at ``hidden_states[N]`` rather than ``hidden_states[N + 1]``.

    Unsupported TransformerLens hooks are rejected rather than silently
    mapped to an arbitrary residual tensor.
    """
    import re as _re

    fallback_layer = num_layers // 2
    fallback_index = min(num_layers, fallback_layer + 1)

    m = _re.search(r"blocks\.(\d+)\.([^.\s]+)", sae_id or "")
    if m is None:
        return max(0, min(num_layers - 1, fallback_layer)), fallback_index

    parsed_layer = int(m.group(1))
    hook_name = m.group(2)
    target_layer = max(0, min(num_layers - 1, parsed_layer))

    hook_to_index = {
        "hook_resid_pre": parsed_layer,
        "hook_resid_post": parsed_layer + 1,
        "hook_resid_mid": parsed_layer + 1,
    }
    if hook_name not in hook_to_index:
        raise ValueError(
            f"Unsupported SAE hook '{hook_name}' in sae_id={sae_id!r}. "
            "Supported hooks: hook_resid_pre, hook_resid_post, hook_resid_mid."
        )

    hidden_state_index = hook_to_index[hook_name]
    return target_layer, max(0, min(num_layers, hidden_state_index))


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

        # Check model compatibility — default SAE config is GPT2-specific.
        # This task is opt-in: it should be silently skipped (with a clear
        # warning) on any model the configured SAE wasn't trained for, so
        # that batch runs on heterogeneous model zoos don't get error noise.
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "")
        if model_name and "gpt2" not in model_name.lower():
            if sae_release == "gpt2-small-res-jb":
                msg = (
                    f"Default SAE config (release={sae_release}, id={sae_id}) "
                    f"is specific to GPT2. Current model: {model_name}. "
                    f"Skipping. Provide a model-appropriate sae_release and sae_id "
                    f"in the task config to enable this task on other architectures."
                )
                logger.warning("  " + msg)
                return {"skipped": True, "reason": msg}

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
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
            
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
        # Extract the target layer from the SAE id if possible, and map
        # TransformerLens hook points onto HuggingFace hidden-state indices.
        try:
            target_layer, hidden_state_index = _select_sae_hidden_state_index(
                sae_id, num_layers
            )
        except ValueError as e:
            return {"error": str(e)}
        
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
                
                # Get the target hidden state. For hook_resid_pre this is the
                # input to the parsed block, i.e. hidden_states[target_layer].
                h = out.hidden_states[hidden_state_index][0] # shape (seq_len, hidden_dim)
                
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
            "sae_total_dict_size": sae.cfg.d_sae,
            "sae_target_layer": int(target_layer),
            "sae_hidden_state_index": int(hidden_state_index),
        }
            
        return results
