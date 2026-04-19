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


def get_vocab_size(model):
    """Get the text vocabulary size from any HF model config.

    Handles multimodal "composed config" models (Gemma 4, Llava, Idefics,
    Pixtral, etc.) where ``vocab_size`` lives under ``config.text_config``
    rather than at the top level.

    Resolution order:
      1. ``model.config.vocab_size`` (plain causal LMs)
      2. ``model.config.text_config.vocab_size`` (multimodal wrappers)
      3. ``model.get_input_embeddings().num_embeddings`` (last resort)

    Returns:
        int vocab size, or None if nothing works.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        v = getattr(cfg, "vocab_size", None)
        if v is not None:
            return int(v)
        # Multimodal: look inside nested sub-configs.
        for sub_attr in ("text_config", "language_config", "llm_config"):
            sub = getattr(cfg, sub_attr, None)
            if sub is not None:
                v = getattr(sub, "vocab_size", None)
                if v is not None:
                    return int(v)
    # Last resort: ask the embedding module directly.
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
    except Exception:
        pass
    return None


# ── Layer Access ───────────────────────────────────────────────────────

_LAYER_ATTRS = [
    # (parent_chain, layer_list_attr)
    ("model", "layers"),          # LLaMA, Mistral, Qwen2, Gemma 2/3, Phi-3
    ("model.language_model", "layers"),  # Gemma 4 multimodal (ForConditionalGeneration)
    ("transformer", "h"),         # GPT2, GPT-Neo, BLOOM, CodeGen
    ("gpt_neox", "layers"),       # Pythia, GPT-NeoX
    ("model.decoder", "layers"),  # OPT
    ("transformer", "blocks"),    # MPT, Falcon (new)
    ("bert.encoder", "layer"),    # BertLMHeadModel
    ("encoder", "layer"),         # BERT-encoder style
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
        params = next(head.parameters())
        # On ``device_map=auto`` models (e.g. pythia-12b, qwen3.5-27b
        # with device_map sharding) the lm_head can live on a different
        # GPU than the caller's hidden state. Move the input to the
        # head's device before projecting.
        return head(hidden_states.to(params.device).to(params.dtype)).float()
    # Fallback: h @ E^T (works for tied embeddings)
    E = get_embeddings(model)
    if E is not None:
        return hidden_states.to(E.device).float() @ E.float().T
    raise RuntimeError("Cannot project hidden states to vocab: no lm_head or embeddings found")


# ── Final Layer Norm ──────────────────────────────────────────────────

_NORM_ATTRS = [
    ("model", "norm"),            # LLaMA, Mistral, Qwen2, Gemma 2/3
    ("model.language_model", "norm"),  # Gemma 4 multimodal (ForConditionalGeneration)
    ("transformer", "ln_f"),      # GPT2, GPT-Neo, CodeGen
    ("gpt_neox", "final_layer_norm"),  # Pythia, GPT-NeoX
    ("model.decoder", "final_layer_norm"),  # OPT
    ("transformer", "norm_f"),    # MPT, Falcon
]


def score_continuation(model, tokenizer, prompt: str, answer: str):
    """Score ``answer`` as a continuation of ``prompt`` without the
    BPE-boundary bug that independent tokenisation of the two pieces
    introduces.

    Returns ``(mean_nll_nats, n_answer_tokens, answer_token_ids)`` or
    ``None`` if scoring is not possible (e.g. the answer tokenises to
    zero tokens, model has no logits, etc.).

    How the boundary is found:

      1. Tokenise ``prompt + answer`` once with
         ``return_offsets_mapping=True`` (when the tokenizer supports
         it) so each token carries its character range in the joined
         string.
      2. The first token whose *start* offset is ≥ ``len(prompt)`` is
         the first answer token; everything before is prompt.
      3. Fall back to ``len(tokenize(prompt))`` only if offset
         mapping is unavailable — older slow tokenizers (SentencePiece
         without offsets) use the prefix-length approximation, which
         is the same approximation the historic BLME tasks used.

    This function is used by every task that scores a string
    continuation (``format_robustness``, ``icl_slope``,
    ``consistency_logical``, ``consistency_paraphrase``,
    ``consistency_contrastive``) to eliminate a systematic
    tokenisation-boundary bias across models with different BPE
    vocabularies.
    """
    import torch
    import torch.nn.functional as F

    if not answer:
        return None
    device = next(model.parameters()).device
    full = prompt + answer

    # Prefer the fast tokenizer's offset mapping. Not all tokenizer
    # classes accept ``return_offsets_mapping=True`` — in that case
    # fall back to the prefix-length approximation.
    prompt_len = None
    try:
        enc_full = tokenizer(
            full, return_tensors="pt", return_offsets_mapping=True,
        )
        offsets = enc_full.get("offset_mapping", None)
        if offsets is not None:
            offs = offsets[0].tolist()
            for i, (s, _e) in enumerate(offs):
                if s >= len(prompt):
                    prompt_len = i
                    break
            if prompt_len is None:
                # Every token overlaps with the prompt — no answer
                # tokens survived tokenisation.
                return None
        # offset_mapping is not a valid model kwarg; drop it.
        enc_full = {k: v for k, v in enc_full.items() if k != "offset_mapping"}
    except (TypeError, ValueError, Exception):  # pragma: no cover
        enc_full = tokenizer(full, return_tensors="pt")

    # Prefix-length fallback when offset mapping isn't available.
    if prompt_len is None:
        enc_prompt = tokenizer(prompt, return_tensors="pt")
        prompt_len = int(enc_prompt["input_ids"].shape[1])

    full_ids = enc_full["input_ids"][0]
    if full_ids.shape[0] <= prompt_len:
        return None

    enc_full_dev = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc_full.items()
    }
    with torch.no_grad():
        out = model(**enc_full_dev)
    logits = out.logits[0]

    pred_logits = logits[prompt_len - 1: -1]
    targets = full_ids[prompt_len:].to(pred_logits.device)
    if pred_logits.shape[0] != targets.shape[0] or pred_logits.shape[0] == 0:
        return None
    losses = F.cross_entropy(pred_logits, targets, reduction="none")
    return (
        float(losses.mean().item()),
        int(targets.shape[0]),
        targets.cpu().tolist(),
    )


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
