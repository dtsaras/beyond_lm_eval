"""Attention head knockout — head specialisation diagnostic.

The classical head-ablation experiment (Michel et al. 2019, Voita et al.
2019) zeroes a single head's output *before* the output projection
(``o_proj`` in Llama/Qwen/Gemma, ``c_proj`` in GPT-2, ``dense`` in
Pythia/GPT-NeoX). That tensor has shape ``(B, T, num_heads * head_dim)``
with each head occupying a contiguous ``head_dim``-sized slice, so the
knockout is semantically a per-head mask.

Zeroing a slice of the **post-projection** attention output (as the
previous implementation did) instead blanks a chunk of the residual
stream that mixes *all* heads. On Gemma / Qwen where
``num_heads * head_dim != hidden_size`` the slice also overflowed the
tensor silently and half the heads registered impact = 0.

We now attach a ``register_forward_pre_hook`` to the output projection
module, derive ``head_dim`` from that module's ``in_features`` (so GQA
and Gemma's per-layer head dim are handled), and zero the correct
contiguous slice of the pre-projection tensor.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:  # pragma: no cover - optional dep
    _HFConv1D = None


# Output-projection attribute names, ordered so the most specific match
# wins. All of these accept a tensor of shape
# ``(B, T, num_heads * head_dim)`` as input.
_OUT_PROJ_ATTRS = (
    "o_proj",        # Llama, Qwen, Gemma, Mistral
    "out_proj",      # some encoder-decoder blocks, OPT
    "c_proj",        # GPT-2 (transformers.Conv1D)
    "dense",         # Pythia / GPT-NeoX
    "proj",          # generic fallback
)


def _find_attn_module(layer):
    """Return the attention sub-module of a transformer block, or None."""
    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    # Last-resort: scan children for one whose name mentions attention.
    for name, module in layer.named_children():
        if "attn" in name.lower() or "attention" in name.lower():
            return module
    return None


def _find_out_proj(attn_module):
    """Return the output projection module (the one that consumes the
    concatenated per-head representations) and its ``in_features`` —
    which equals ``num_heads * head_dim`` for every standard
    architecture, even when that differs from the residual stream width.
    """
    for attr in _OUT_PROJ_ATTRS:
        proj = getattr(attn_module, attr, None)
        if proj is None:
            continue
        if isinstance(proj, torch.nn.Linear):
            return proj, int(proj.in_features)
        if _HFConv1D is not None and isinstance(proj, _HFConv1D):
            # transformers Conv1D stores weights as (in, out); in_features
            # is the first dimension.
            w = getattr(proj, "weight", None)
            if w is not None and w.dim() == 2:
                return proj, int(w.shape[0])
    return None, 0


@register_task("causality_attention_knockout")
class AttentionKnockoutTask(DiagnosticTask):
    """
    Zero-ablates each attention head and records the resulting NLL
    increase. Reports the Gini coefficient of per-head impacts as a
    specialisation measure.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Knockout Specialisation...")
        num_samples = self.config.get("num_samples", 3)

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect transformer layers"}
        layers = list(layers)
        num_layers = len(layers)
        if num_layers == 0:
            return {"error": "Model has zero layers"}

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        encodings = []
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            ids = tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=128
            )
            if ids is None:
                continue
            ids = ids.to(device)
            if ids.shape[1] > 2:
                encodings.append(ids)

        if not encodings:
            return {"error": "No valid sequences"}

        def _resolve_num_heads(cfg):
            if cfg is None:
                return None
            for attr in ("num_attention_heads", "n_head"):
                v = getattr(cfg, attr, None)
                if v is not None:
                    return int(v)
            return None

        num_heads = _resolve_num_heads(getattr(model, "config", None))
        if num_heads is None:
            # Gemma 4 and other multimodal wrappers nest the transformer
            # config under ``config.text_config`` / ``language_config`` /
            # ``llm_config``; fall through those before giving up.
            cfg = getattr(model, "config", None)
            for sub_attr in ("text_config", "language_config", "llm_config"):
                sub = getattr(cfg, sub_attr, None)
                num_heads = _resolve_num_heads(sub)
                if num_heads is not None:
                    break
        if num_heads is None:
            return {
                "error": (
                    "Could not determine num_attention_heads from model.config"
                )
            }

        # Pre-resolve (attention module, out_proj module, in_features) per
        # layer. Layers where we can't find an output projection are
        # skipped explicitly rather than silently falling back to a
        # destructive residual-stream ablation.
        layer_targets = []  # list of (layer_idx, attn_module, out_proj, head_dim)
        for l_idx, layer in enumerate(layers):
            attn = _find_attn_module(layer)
            if attn is None:
                continue
            out_proj, in_features = _find_out_proj(attn)
            if out_proj is None or in_features <= 0:
                continue
            if in_features % num_heads != 0:
                # Fall back to the config's head_dim if the projection
                # uses a non-standard stride.
                head_dim = getattr(
                    model.config, "head_dim", in_features // num_heads
                )
                if head_dim <= 0 or num_heads * head_dim > in_features:
                    continue
            else:
                head_dim = in_features // num_heads
            layer_targets.append((l_idx, attn, out_proj, head_dim))

        if not layer_targets:
            return {"error": "Could not locate any attention output projection"}

        def get_loss(batch_ids: torch.Tensor) -> float:
            outputs = model(batch_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return float(loss.item())

        with torch.no_grad():
            baseline_losses = [get_loss(ids) for ids in encodings]
        baseline_mean_loss = float(np.mean(baseline_losses))

        # For very deep models, sample 5 evenly-spaced layers to keep the
        # task tractable; otherwise analyse every layer.
        if len(layer_targets) > 10:
            idx = np.linspace(0, len(layer_targets) - 1, 5).round().astype(int)
            targets = [layer_targets[i] for i in idx]
        else:
            targets = layer_targets

        per_head_impacts = []

        def make_pre_hook(start: int, end: int):
            def pre_hook(module, inputs):
                if not inputs:
                    return inputs
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return inputs
                if x.dim() < 2 or end > x.shape[-1]:
                    return inputs
                patched = x.clone()
                patched[..., start:end] = 0.0
                return (patched,) + tuple(inputs[1:])
            return pre_hook

        with torch.no_grad():
            for l_idx, _attn, out_proj, head_dim in targets:
                for h_idx in range(num_heads):
                    start = h_idx * head_dim
                    end = start + head_dim
                    handle = out_proj.register_forward_pre_hook(
                        make_pre_hook(start, end)
                    )
                    try:
                        losses = [get_loss(ids) for ids in encodings]
                        impact = float(np.mean(losses) - baseline_mean_loss)
                    finally:
                        handle.remove()
                    per_head_impacts.append(impact)

        impacts = np.array(per_head_impacts, dtype=np.float64)
        # Zero-out improvements — they're noise, not specialisation.
        impacts_pos = np.maximum(0.0, impacts)

        def gini(array: np.ndarray) -> float:
            if array.size == 0 or float(np.sum(array)) == 0.0:
                return 0.0
            arr = np.sort(array)
            n = arr.size
            idx = np.arange(1, n + 1)
            return float(np.sum((2 * idx - n - 1) * arr) / (n * float(np.sum(arr))))

        return {
            "baseline_loss": baseline_mean_loss,
            "max_knockout_impact": float(np.max(impacts_pos)) if impacts_pos.size else 0.0,
            "mean_knockout_impact": float(np.mean(impacts_pos)) if impacts_pos.size else 0.0,
            "head_impact_gini_coefficient": gini(impacts_pos),
            "per_head_impacts": [float(v) for v in impacts],
            "num_layers_analyzed": len(targets),
            "num_heads": int(num_heads),
        }
