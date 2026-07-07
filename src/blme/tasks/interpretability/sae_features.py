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


def _sae_metadata(sae):
    return getattr(getattr(sae, "cfg", None), "metadata", None)


def _sae_prepend_bos(sae) -> bool:
    meta = _sae_metadata(sae)
    return bool(getattr(meta, "prepend_bos", False)) if meta is not None else False


def _prepare_sae_input(h, sae):
    """Match the SAE's training-time activation convention before encoding.

    The gpt2-small-res-jb SAE was trained on HookedTransformer activations with
    ``center_writing_weights=True`` (the residual stream is mean-centered over
    d_model at each position). Feeding raw HF hidden states is out-of-
    distribution and inflates L0 ~37x, so subtract the per-token mean when the
    SAE metadata requests centering; otherwise pass the input through unchanged.
    """
    meta = _sae_metadata(sae)
    center = False
    if meta is not None:
        kw = getattr(meta, "model_from_pretrained_kwargs", None) or {}
        try:
            center = bool(kw.get("center_writing_weights", False))
        except AttributeError:
            center = False
    if center:
        return h - h.mean(dim=-1, keepdim=True)
    return h


def _is_gpt2_small(model_name: str) -> bool:
    """True only for gpt2-small (the model the default gpt2-small-res-jb SAE was
    trained on). Uses an exact basename match, so the old substring test that
    let ``distilgpt2`` (and ``gpt2-medium``, etc.) through no longer passes."""
    base = (model_name or "").lower().rsplit("/", 1)[-1]
    return base in ("gpt2", "gpt2-small")


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

    # Reject an SAE trained for a block this model does not have (e.g. the
    # gpt2-small blocks.8 SAE on distilgpt2's 6 layers) instead of silently
    # clamping to the wrong layer.
    if parsed_layer >= num_layers:
        raise ValueError(
            f"SAE parsed layer {parsed_layer} >= model num_layers {num_layers} "
            f"(sae_id={sae_id!r}); the SAE targets a block this model lacks."
        )

    if hook_name == "hook_resid_pre":
        # Input to block N == HF hidden_states[N].
        hidden_state_index = parsed_layer
    elif hook_name == "hook_resid_post":
        # resid_post of block N == HF hidden_states[N+1], BUT the last entry
        # hidden_states[num_layers] is ln_f(block output) — post-final-norm,
        # not the raw residual TL exposes. Reject the final block.
        if parsed_layer == num_layers - 1:
            raise ValueError(
                f"blocks.{parsed_layer}.hook_resid_post maps to hidden_states"
                f"[{num_layers}], which HF returns AFTER ln_f (post-final-norm), "
                "not the raw block output; refusing to feed it to the SAE."
            )
        hidden_state_index = parsed_layer + 1
    elif hook_name == "hook_resid_mid":
        # Post-attention / pre-MLP residual: HF exposes no such tensor.
        raise ValueError(
            f"hook_resid_mid (sae_id={sae_id!r}) has no HF hidden_states tensor "
            "(post-attention, pre-MLP residual); refusing to map it to resid_post."
        )
    else:
        raise ValueError(
            f"Unsupported SAE hook '{hook_name}' in sae_id={sae_id!r}. "
            "Supported hooks: hook_resid_pre, hook_resid_post."
        )

    return parsed_layer, hidden_state_index


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
        if model_name and not _is_gpt2_small(model_name):
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
            loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=str(device))
            # sae-lens API drift: older versions return a (sae, cfg_dict, sparsity)
            # tuple; sae-lens >=6.x returns the SAE object directly. Handle both so
            # the task runs regardless of the installed sae-lens version.
            sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
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

        prepend_bos = _sae_prepend_bos(sae)
        active_features_counts = []
        max_active_features = []
        
        with torch.no_grad():
            for s in samples:
                if isinstance(s, dict) and "text" in s:
                    text = s["text"]
                else:
                    text = str(s)
                    
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                # The SAE was trained with a BOS token prepended (prepend_bos);
                # match that so the encoded activations are in-distribution.
                if prepend_bos and getattr(tokenizer, "bos_token_id", None) is not None:
                    bos = torch.full((1, 1), int(tokenizer.bos_token_id),
                                     dtype=inputs["input_ids"].dtype, device=device)
                    inputs["input_ids"] = torch.cat([bos, inputs["input_ids"]], dim=1)
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.cat(
                            [torch.ones_like(bos), inputs["attention_mask"]], dim=1)
                out = model(**inputs, output_hidden_states=True)

                # Get the target hidden state. For hook_resid_pre this is the
                # input to the parsed block, i.e. hidden_states[target_layer].
                h = out.hidden_states[hidden_state_index][0] # shape (seq_len, hidden_dim)
                # Match the SAE's training-time activation convention
                # (center_writing_weights) before encoding.
                h = _prepare_sae_input(h, sae)

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
