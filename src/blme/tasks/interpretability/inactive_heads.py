"""Inactive-head fraction: the share of attention heads whose output is small
next to the other heads of their layer (Sandoval-Segura et al. 2026).

Reference: "Identifying and Evaluating Inactive Heads in Pretrained LLMs",
arXiv:2504.03889 (ICLR 2026); code psandovalsegura/inactive-heads
(``monkey_patch_llama.py::my_eager_attention_forward``, the ``HonorHeads``
branch; ``evaluate_attention_heads_drop.py``'s ``model_proportion``;
``create-CDF.py``; ``monkey_patch_head_types.py::x_intercept_for_cdf``).

The score is the paper's Avg Head Output Norm (LN), the release's HONOR. For
one sequence, layer and query head h, a_h is the L2 norm of the head's
attention output (attn_weights @ V, head h's slice of the o_proj input)
averaged over the sequence's tokens (padding excluded, the first token kept),
and the score is a_h over the mean of a across the layer's heads. A head is
inactive on a sequence when its score is below tau. The fraction is the share
of (sequence, head) pairs below tau in each layer, averaged over the layers.
The release sets tau per model at the quantiles of all (sequence, layer, head)
scores that capture 0, 5, ..., 30% of heads on MMLU (5-shot).
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..causality.attention_knockout import _find_attn_module, _find_out_proj, _resolve_head_dim
from ..common import get_layers, get_max_positions

logger = logging.getLogger("blme")

# generate_all_thresholds: thresholds capturing these shares of the heads.
CDF_SHARES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TAU_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# transformers' ALLOWED_ATTENTION_LAYER_TYPES less "linear_attention".
_SOFTMAX_LAYER_TYPES = ("full_attention", "sliding_attention", "chunked_attention")


def head_output_norms(x, n_heads, head_dim):
    """Per-token L2 norm of each head's slice of the o_proj input, in float32:
    (..., T, n_heads * head_dim) -> (..., T, n_heads)."""
    x = x[..., : n_heads * head_dim].float()
    return x.unflatten(-1, (n_heads, head_dim)).norm(dim=-1)


def honor_scores(avg_norms):
    """HonorHeads: each head's token-averaged output norm over the layer's
    mean, (N, H) -> (N, H). A row whose heads are all exactly silent is 0/0
    (NaN), as in the release."""
    return avg_norms / avg_norms.mean(dim=-1, keepdim=True)


def layer_fractions(scores, tau):
    """Per layer, the share of (sequence, head) scores below tau
    (``dormant_mask = relative_avg_norm_per_head < threshold``). Rows without
    a layer mean are left out; a layer with none left is NaN."""
    out = []
    for s in scores:
        s = s[~torch.isnan(s).any(dim=-1)]
        out.append(float((s < tau).double().mean()) if s.numel() else float("nan"))
    return out


def inactive_fraction(scores, tau):
    """``model_proportion``: the mean over layers of the per-layer shares."""
    f = [v for v in layer_fractions(scores, tau) if not math.isnan(v)]
    return math.fsum(f) / len(f) if f else float("nan")


def x_intercept_for_cdf(cdf, y):
    """The release's ``x_intercept_for_cdf``: the x at which the (N, 2) CDF
    reaches y, linearly interpolated and clamped to the CDF's range."""
    y = torch.as_tensor(y, dtype=cdf.dtype, device=cdf.device)
    x_vals, y_vals = cdf[:, 0].contiguous(), cdf[:, 1].contiguous()
    y = torch.clamp(y, min=y_vals[0], max=y_vals[-1])
    i = torch.clamp(torch.searchsorted(y_vals, y), min=1, max=len(y_vals) - 1)
    x1, y1, x2, y2 = x_vals[i - 1], y_vals[i - 1], x_vals[i], y_vals[i]
    den = y2 - y1
    den = torch.where(den == 0, torch.ones_like(den), den)
    return x1 + (y - y1) / den * (x2 - x1)


def cdf_thresholds(scores, shares=CDF_SHARES):
    """create-CDF.py's CDF of every (sequence, layer, head) score, read at
    ``shares`` (share 0 gives the minimum). None with fewer than 2 scores."""
    flat = torch.cat([s.flatten() for s in scores]).float()
    flat = flat[~torch.isnan(flat)]
    if flat.numel() < 2:
        return None
    x = torch.sort(flat).values
    cdf = torch.stack((x, torch.arange(1, len(x) + 1) / len(x)), dim=1)
    return [float(x_intercept_for_cdf(cdf, p)) for p in shares]


def _text_config(model):
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "get_text_config"):
        return cfg.get_text_config()
    return cfg


def attention_targets(model, layers):
    """``(layer index, o_proj, n_heads, head_dim)`` for every softmax-attention
    layer, plus the number of layers skipped as non-softmax (a declared
    non-softmax ``layer_types`` entry, or no attention module) and as having
    no slicable output projection."""
    cfg = _text_config(model)
    n_heads = None
    for attr in ("num_attention_heads", "n_head"):
        if getattr(cfg, attr, None) is not None:
            n_heads = int(getattr(cfg, attr))
            break
    layer_types = getattr(cfg, "layer_types", None)
    targets, n_non_softmax, n_no_proj = [], 0, 0
    for i, layer in enumerate(layers):
        lt = layer_types[i] if layer_types is not None and i < len(layer_types) else None
        attn = _find_attn_module(layer) if lt is None or lt in _SOFTMAX_LAYER_TYPES else None
        if attn is None:
            n_non_softmax += 1
            continue
        proj, in_features = _find_out_proj(attn)
        head_dim = (_resolve_head_dim(cfg, in_features, n_heads)
                    if proj is not None and n_heads else None)
        if head_dim is None:
            n_no_proj += 1
            continue
        targets.append((i, proj, n_heads, head_dim))
    return targets, n_non_softmax, n_no_proj


def collect_head_norms(model, tokenizer, texts, targets, max_length):
    """One forward per text (batch 1, so no padding); for each target layer an
    (N, H) tensor of token-averaged head output norms, the tokens seen."""
    device = next(model.parameters()).device
    store = [[] for _ in targets]

    def hook(j, n_heads, head_dim):
        def fn(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            store[j].append(head_output_norms(x, n_heads, head_dim).mean(dim=-2).cpu())
        return fn

    n_tokens = []
    handles = []
    try:
        for j, (_, proj, n_heads, head_dim) in enumerate(targets):
            handles.append(proj.register_forward_pre_hook(hook(j, n_heads, head_dim),
                                                          with_kwargs=True))
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_length)["input_ids"]
                if ids.shape[-1] == 0:
                    continue
                model(input_ids=ids.to(device), use_cache=False)
                n_tokens.append(int(ids.shape[-1]))
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(s) if s else torch.empty(0, t[2]) for s, t in zip(store, targets)], n_tokens


@register_task("interpretability_inactive_heads")
class InactiveHeadsTask(DiagnosticTask):
    """Fraction of inactive attention heads by Avg Head Output Norm (LN)."""

    TOKEN_POLICY = ("paper_protocol:SandovalSegura2026 every token of each sequence, the "
                    "first included, averaged per head (padding excluded)")
    SOURCES = {
        "paper": ("arXiv:2504.03889 (Sandoval-Segura et al., ICLR 2026, Identifying and "
                  "Evaluating Inactive Heads in Pretrained LLMs)"),
        "code": "psandovalsegura/inactive-heads@9dd884ce6ad18905392024eb34314d49b12f401a",
        "llm_precedent": ("arXiv:2504.03889 (Llama-3.1/3.2, OLMo-2 base/SFT/DPO/Instruct, "
                          "Qwen2.5 0.5B-14B)"),
    }
    ADAPTATION_NOTE = (
        "The score (Avg Head Output Norm (LN), the release's HONOR), its mean over every token "
        "of a sequence, the per-layer share below tau averaged over layers, and the CDF "
        "thresholds follow the release; norms are taken in float32 where the release keeps "
        "the model dtype. The release sets tau per model at the score quantiles capturing "
        "0-30% of heads on MMLU 5-shot; BLME scores the first 1000 corpus passages (up to 512 "
        "tokens) at a fixed tau = 0.5, near the tau (median 0.54, 0.45-0.58 in 13 of 14 "
        "models) at which zeroing AHON (LN) heads costs the paper's 1 MMLU point in the "
        "release's own results (class 2). inactive_head_threshold_5pct is the release's 5% "
        "threshold itself and needs no tau. The head output is read at the o_proj input, "
        "attn_weights @ V per query head (heads sharing a GQA value head are scored "
        "separately, as in the release); on gated attention (Qwen3.5, Qwen3-Next) it is the "
        "gated output o_proj receives, a case the release never met. Non-softmax layers are "
        "skipped and counted, and a sequence on which every head of a layer is exactly silent "
        "has no layer mean and is left out (the release's 0/0 counts it active). Identity "
        "blocks move the fraction and no rewrite can stop it: their heads write nothing, so "
        "they are inactive by definition, yet a score read before W_O rates them as ordinary "
        "heads; zero-padding the width adds no heads and leaves every score unchanged."
    )
    HEADLINE_KEYS = ("inactive_head_fraction", "inactive_head_threshold_5pct")
    SIZE_INVARIANT = {}

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Inactive-Head Fraction...")
        num_samples = int(self.config.get("num_samples", 1000))
        max_length = int(self.config.get("max_length", 512))
        max_length = min(max_length, get_max_positions(model, max_length))
        tau = float(self.config.get("tau", 0.5))

        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect transformer layers"}
        targets, n_non_softmax, n_no_proj = attention_targets(model, layers)
        if not targets:
            return {"error": "No softmax-attention layer with a slicable output projection"}
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        # The first num_samples passages; none is replaced by a later one.
        texts = [s["text"] if isinstance(s, dict) else str(s) for s in list(dataset)[:num_samples]]
        norms, n_tokens = collect_head_norms(model, tokenizer, texts, targets, max_length)
        if len({len(a) for a in norms}) != 1 or len(norms[0]) != len(n_tokens):
            return {"error": "The attention layers saw different numbers of sequences"}
        if len(n_tokens) < 1:
            return {"error": "No non-empty sequence"}

        scores = [honor_scores(a) for a in norms]
        per_layer = layer_fractions(scores, tau)
        thresholds = cdf_thresholds(scores)
        n_undefined = sum(int(torch.isnan(s).any(dim=-1).sum()) for s in scores)
        heads = sorted({t[2] for t in targets})
        return {
            "inactive_head_fraction": inactive_fraction(scores, tau),
            "inactive_head_threshold_5pct": thresholds[1] if thresholds else float("nan"),
            "inactive_head_fraction_per_layer": per_layer,
            "_meta_tau": tau,
            "_meta_tau_grid": list(TAU_GRID),
            "_meta_inactive_head_fraction_curve": [inactive_fraction(scores, t) for t in TAU_GRID],
            "_meta_cdf_shares": list(CDF_SHARES),
            "_meta_cdf_thresholds": thresholds,
            "_meta_score_median": float(np.nanmedian(torch.cat([s.flatten() for s in scores]))),
            "_meta_scored_layers": [t[0] for t in targets],
            "_meta_n_layers_scored": len(targets),
            "_meta_n_layers_skipped_non_softmax": n_non_softmax,
            "_meta_n_layers_skipped_no_out_proj": n_no_proj,
            "_meta_n_heads": heads[0] if len(heads) == 1 else heads,
            "_meta_n_sequences": len(n_tokens),
            "_meta_n_tokens": int(sum(n_tokens)),
            "_meta_n_undefined_rows": n_undefined,
        }
