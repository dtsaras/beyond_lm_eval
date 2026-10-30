"""Effective depth / Block Influence — layer-utilization diagnostics.

References:
  * Men et al. 2024/2025 — "ShortGPT: Layers in Large Language Models are
    More Redundant Than You Expect", arXiv:2403.03853 (Findings of ACL
    2025). Defines BLOCK INFLUENCE: BI_l = 1 - E_t[cos(h_{l,t}, h_{l+1,t})],
    the angular change a block applies to the residual stream; low-BI
    layers can be removed with little loss ("layer redundancy").
  * Csordas, Manning & Potts 2025 — "Do Language Models Use Their Depth
    Efficiently?", arXiv:2505.13898 (NeurIPS 2025). Measures per-layer
    residual CONTRIBUTION norms ||h_{l+1} - h_l|| relative to the stream
    norm and shows the second half of layers contributes much less —
    models have an "effective depth" smaller than their nominal depth.

Reported (one private forward pass — the shared cache is NOT used because
it drops the embedding output and its final entry is post-final-norm; a
hook recovers the last block's pre-norm output):
  * ``block_influence_per_layer`` and mean/min/max summaries.
  * ``frac_low_influence``: fraction of layers with BI below a threshold
    (default 0.05). The threshold is a BLME choice — ShortGPT itself ranks
    layers by BI and prunes to a target ratio; it defines no threshold.
  * ``effective_depth_90``: smallest normalized depth k/L at which the
    cumulative contribution mass reaches 90% of the total. This is a BLME
    summary INSPIRED BY Csordas et al.'s depth-utilization analysis; their
    own statistic is an importance-weighted depth score, not this.

Analytic anchors: identity layers -> BI = 0, contribution 0; an
orthogonal-rotation layer -> BI = 1; a sign-flip layer -> BI = 2.
"""

import logging
import math
from typing import List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")

# The released Csordas et al. analysis/analyze_norms.py clamps at 1e-6.
_CSORDAS_EPS = 1e-6


def _block_influence(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """ShortGPT BI = 1 - E_t[cos(h_in_t, h_out_t)] over token positions."""
    a = h_in.float()
    b = h_out.float()
    # A scale-free cosine: each row is divided by its largest coordinate first
    # (exact), so neither an absolute eps nor an underflowing norm enters
    # (tests/test_scale_invariance.py).
    from .utils import safe_unit_rows_torch as _unit
    cos = (_unit(a) * _unit(b)).sum(dim=-1)
    return float(1.0 - cos.mean().item())


def _depth_profile(layer_states: List[torch.Tensor]) -> dict:
    """BI + contribution profiles from a list of per-layer (T, D) states
    (index 0 = block-1 input, i.e. the embedding output when available;
    otherwise the first cached layer)."""
    L = len(layer_states) - 1
    if L < 1:
        return {}
    bis, contribs = [], []
    for l in range(L):
        h_in, h_out = layer_states[l], layer_states[l + 1]
        bis.append(_block_influence(h_in, h_out))
        # Csordas et al. 2025 Sec. 3.1 / analyze_norms.py: the MEAN OF PER-TOKEN
        # ratios ||h_{l+1,t} - h_{l,t}|| / ||h_{l,t}||, computed scale-exactly by
        # factoring each row's largest magnitude out of both norms.
        _a = h_in.float()
        _b = h_out.float()
        _m = _a.abs().amax(dim=-1, keepdim=True)
        _tiny = torch.finfo(_a.dtype).tiny
        _msafe = _m.clamp_min(_tiny)
        _num = ((_b - _a) / _msafe).norm(dim=-1)
        _den = (_a / _msafe).norm(dim=-1)
        # Only an exactly zero input row has no defined relative change.
        per_token = torch.where(_den > 0, _num / _den.clamp_min(_tiny),
                                torch.zeros_like(_num))
        _valid = (_m.squeeze(-1) > 0) & torch.isfinite(per_token)
        contribs.append(float(per_token[_valid].mean().item())
                        if bool(_valid.any()) else float("nan"))

    contribs_arr = np.asarray(contribs, dtype=np.float64)
    total = float(np.nansum(contribs_arr))
    if total > 0:
        cum = np.nancumsum(contribs_arr) / total
        k90 = int(np.searchsorted(cum, 0.9) + 1)
        effective_depth_90 = k90 / L
    else:
        effective_depth_90 = float("nan")

    return {
        "block_influence_per_layer": bis,
        "mean_block_influence": float(np.mean(bis)),
        "min_block_influence": float(np.min(bis)),
        "max_block_influence": float(np.max(bis)),
        "frac_low_influence": float(np.mean(np.asarray(bis) < 0.05)),
        "contribution_ratio_per_layer": contribs,
        "effective_depth_90": float(effective_depth_90),
    }


@register_task("geometry_effective_depth")
class EffectiveDepthTask(DiagnosticTask):
    """Block-Influence profile + effective-depth ratio (forward pass only).

    ``effective_depth_90`` is a CAPACITY RATIO: the fraction of nominal depth
    (k90 / L) that carries 90% of the summed relative contributions. Appending
    identity blocks lowers it by design, and it is already divided by L, so it
    must not be divided by L again.
    """

    TOKEN_POLICY = ("paper_protocol:ShortGPT2024+Csordas2025 every token of each sample, "
                    "position 0 included")
    SOURCES = {
        "paper": "arXiv:2403.03853; arXiv:2505.13898",
        "code": ("robertcsordas/llm_effective_depth@393e5c740e2a46d8b06d60129b1a3fb4b581a01a; "
                 "RitAreaSciencePark/ZigZagLLMs@2d10c07 (short-transformers BI)"),
        "llm_precedent": "arXiv:2403.03853 and arXiv:2505.13898 (proposed for LLMs)",
    }
    HEADLINE_KEYS = ("effective_depth_90", "total_block_influence", "max_block_influence")
    _INV = "tests/tasks/invariance/test_geometry_effective_depth.py::"
    SIZE_INVARIANT = {
        "total_block_influence": (("insert_identity_blocks",),
                                  _INV + "test_depth_additive_keys_under_identity_blocks"),
        "max_block_influence": (("insert_identity_blocks",),
                                _INV + "test_depth_additive_keys_under_identity_blocks"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Effective Depth / Block Influence...")
        num_samples = self.config.get("num_samples", 20)

        if dataset is None:
            dataset = [{"text": f"Depth-utilization probe sentence {i}."}
                       for i in range(num_samples)]

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers"}

        # Per-layer token states of every sample, concatenated (the per-token
        # expectations are unchanged). The last state is the hooked raw output
        # of the last block, so every transition is block input -> block output
        # as ShortGPT's X_i requires (HF's last entry is ln_f's output).
        _max_len = int(self.config.get("max_length", 512))
        last_block_out = {}

        def _capture_last(module, inputs, output):
            out_t = output[0] if isinstance(output, tuple) else output
            last_block_out["h"] = out_t.detach()

        hook = layers[-1].register_forward_hook(_capture_last)
        per_layer: List[List[torch.Tensor]] = None
        try:
            with torch.no_grad():
                for i, s in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = s["text"] if isinstance(s, dict) else str(s)
                    # The window (512, the cached tasks' window) is part of the
                    # protocol: BI is a per-position mean. ShortGPT scores PG19
                    # chunks and Csordas et al. 20 GSM8K prompts, so absolute
                    # values compare within this zoo only.
                    enc = tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=_max_len).to(device)
                    last_block_out.pop("h", None)
                    out = model(**enc, output_hidden_states=True)
                    hs = list(out.hidden_states)
                    # post-final-norm: SUBSTITUTED -- embedding output + L block
                    # outputs, the last replaced by the hooked pre-norm output.
                    if "h" in last_block_out:
                        hs[-1] = last_block_out["h"]
                    if per_layer is None:
                        per_layer = [[] for _ in range(len(hs))]
                    for li, h in enumerate(hs):
                        per_layer[li].append(h[0].float().cpu())
        finally:
            hook.remove()

        if not per_layer or len(per_layer) < 2:
            return {"error": "Not enough hidden states collected"}

        layer_states = [torch.cat(chunks, dim=0) for chunks in per_layer]
        results = _depth_profile(layer_states)
        if not results:
            return {"error": "Depth profile could not be computed"}
        results["_meta_n_layers_profiled"] = len(layer_states) - 1
        # Depth-additive twin: an identity block has BI = 0, so the sum over
        # blocks is unchanged by inserting one, while the mean is diluted as
        # L0 / L and the minimum and the BI < 0.05 fraction count the inserted
        # blocks (P4): those three are _meta_.
        results["total_block_influence"] = float(math.fsum(results["block_influence_per_layer"]))
        for k in ("mean_block_influence", "min_block_influence", "frac_low_influence"):
            results[f"_meta_{k}"] = results.pop(k)
        return results
