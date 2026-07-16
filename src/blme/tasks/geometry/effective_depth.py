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
  * ``contribution_ratio_per_layer``: E_t ||h_{l+1}-h_l|| / E_t ||h_l||
    (ratio-of-means; Csordas et al. use per-token relative norms).
  * ``effective_depth_90``: smallest normalized depth k/L at which the
    cumulative contribution mass reaches 90% of the total. This is a BLME
    summary INSPIRED BY Csordas et al.'s depth-utilization analysis; their
    own statistic is an importance-weighted depth score, not this.

Analytic anchors: identity layers -> BI = 0, contribution 0; an
orthogonal-rotation layer -> BI = 1; a sign-flip layer -> BI = 2.
"""

import logging
from typing import List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


def _block_influence(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """ShortGPT BI = 1 - E_t[cos(h_in_t, h_out_t)] over token positions."""
    a = h_in.float()
    b = h_out.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
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
        num = (h_out.float() - h_in.float()).norm(dim=-1).mean()
        den = h_in.float().norm(dim=-1).mean().clamp_min(1e-12)
        contribs.append(float((num / den).item()))

    contribs_arr = np.asarray(contribs, dtype=np.float64)
    total = contribs_arr.sum()
    if total > 0:
        cum = np.cumsum(contribs_arr) / total
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
    """Block-Influence profile + effective-depth ratio (forward pass only)."""

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

        # Accumulate per-layer token states over samples, then compute the
        # profile on the concatenated token clouds (position-aligned per
        # sample; concatenating preserves the per-token expectations).
        #
        # PROTOCOL FIX (audit 2026-07-16): HF ``hidden_states[-1]`` is
        # POST-final-norm; on gpt2 ~90% of the last transition's BI was the
        # ln_f artifact, not the last block's transformation. We capture the
        # last block's raw (pre-norm) output with a forward hook and use it
        # as the final state, so every transition is block-input -> block
        # -output as ShortGPT's X_i requires.
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
                    enc = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    out = model(**enc, output_hidden_states=True)
                    hs = list(out.hidden_states)
                    # embedding output + L block outputs; replace the
                    # post-norm final entry with the hooked pre-norm output.
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
        return results
