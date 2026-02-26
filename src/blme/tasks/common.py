"""Shared utilities for diagnostic tasks — universal HuggingFace compatibility.

All model introspection goes through these functions so that tasks work
with any AutoModelForCausalLM architecture (LLaMA, GPT2, Pythia, Phi,
Falcon, BLOOM, MPT, OLMo, Qwen, Gemma, etc.).
"""

import torch


# ── Embedding Access ───────────────────────────────────────────────────

def get_embeddings(model):
    """Extract the input embedding weight matrix from any HF causal LM.

    Returns:
        Embedding weight tensor (V, D) detached, or None if not found.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb.weight.detach()
    except Exception:
        pass
    return None


# ── Layer Access ───────────────────────────────────────────────────────

_LAYER_ATTRS = [
    # (parent_chain, layer_list_attr)
    ("model", "layers"),          # LLaMA, Mistral, Qwen2, Gemma, Phi-3
    ("transformer", "h"),         # GPT2, GPT-Neo, BLOOM, CodeGen
    ("gpt_neox", "layers"),       # Pythia, GPT-NeoX
    ("model.decoder", "layers"),  # OPT
    ("transformer", "blocks"),    # MPT, Falcon (new)
    ("model", "embed_tokens"),    # Skip — not layers, but check next
]


def _resolve_attr(obj, dotted_path):
    """Resolve a dotted attribute path like 'model.decoder'."""
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def get_layers(model):
    """Extract the list of transformer layers from any HF causal LM.

    Returns:
        nn.ModuleList of layers, or None if not found.
    """
    for parent_chain, attr in _LAYER_ATTRS:
        parent = _resolve_attr(model, parent_chain)
        if parent is not None and hasattr(parent, attr):
            candidate = getattr(parent, attr)
            if isinstance(candidate, torch.nn.ModuleList):
                return candidate
    return None


def get_num_layers(model):
    """Get number of transformer layers from model config or introspection.

    Returns:
        int number of layers, or 0 if not found.
    """
    # Prefer config (most universal)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
    # Fallback to counting layers
    layers = get_layers(model)
    if layers is not None:
        return len(layers)
    return 0


# ── LM Head Access ────────────────────────────────────────────────────

def get_lm_head(model):
    """Get the output projection (lm_head) module.

    Returns the nn.Linear module, or None if not found / tied without head.
    """
    # Standard attribute
    if hasattr(model, "lm_head") and isinstance(model.lm_head, torch.nn.Linear):
        return model.lm_head
    # Some models use embed_out (GPT-NeoX / Pythia)
    if hasattr(model, "embed_out") and isinstance(model.embed_out, torch.nn.Linear):
        return model.embed_out
    # HF generic API
    try:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and isinstance(out_emb, torch.nn.Linear):
            return out_emb
    except Exception:
        pass
    return None


def apply_lm_head(model, hidden_states):
    """Project hidden states to vocabulary logits using the LM head.

    Works universally: uses model.lm_head if available, otherwise falls
    back to computing h @ E^T using the input embedding matrix.

    Args:
        model: HuggingFace causal LM
        hidden_states: tensor of shape (..., D) — can be (T, D) or (1, D)

    Returns:
        Logits tensor of shape (..., V)
    """
    head = get_lm_head(model)
    if head is not None:
        dtype = next(head.parameters()).dtype
        return head(hidden_states.to(dtype)).float()
    # Fallback: h @ E^T (works for tied embeddings)
    E = get_embeddings(model)
    if E is not None:
        return hidden_states.float() @ E.float().T
    raise RuntimeError("Cannot project hidden states to vocab: no lm_head or embeddings found")


# ── Final Layer Norm ──────────────────────────────────────────────────

_NORM_ATTRS = [
    ("model", "norm"),            # LLaMA, Mistral, Qwen2, Gemma
    ("transformer", "ln_f"),      # GPT2, GPT-Neo, CodeGen
    ("gpt_neox", "final_layer_norm"),  # Pythia, GPT-NeoX
    ("model.decoder", "final_layer_norm"),  # OPT
    ("transformer", "norm_f"),    # MPT, Falcon
]


def get_final_norm(model):
    """Get the final layer normalization module.

    Returns:
        The norm module, or None.
    """
    for parent_chain, attr in _NORM_ATTRS:
        parent = _resolve_attr(model, parent_chain)
        if parent is not None and hasattr(parent, attr):
            return getattr(parent, attr)
    return None
