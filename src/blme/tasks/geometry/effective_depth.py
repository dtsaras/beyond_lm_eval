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
from typing import List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")

# The released Csordas et al. `analysis/analyze_norms.py` uses
# `.clamp(min=1e-6)`, not 1e-12. The difference is not academic on small-norm
# states: for a 1e-7-per-coordinate input the two clamps give a contribution of
# 1.0 vs 0.360555 (adversarial review, 2026-09-03). Matching the reference.
_CSORDAS_EPS = 1e-6


def _block_influence(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """ShortGPT BI = 1 - E_t[cos(h_in_t, h_out_t)] over token positions."""
    a = h_in.float()
    b = h_out.float()
    # torch's default cosine eps is an ABSOLUTE 1e-8 floor on the denominator,
    # so it is not the scale-invariant cosine the ShortGPT paper and its
    # released code define. Rescaling one identity state from norm 5.477 to
    # 5.477e-9 moved BI from ~0.0 to 0.699999988 (adversarial review,
    # 2026-09-04). Only an exactly zero vector has no direction.
    # `eps` guards the DIVISION but not the norm's own arithmetic, which
    # squares and therefore underflows first: on a nonzero scale-1e-23 identity
    # input (norm 5.294e-23) the true BI is 0 and BLME returned -0.070435762
    # (adversarial review round 2, 2026-09-04). Normalise each vector by its
    # own largest coordinate first -- exact in binary floating point -- then
    # take the plain dot product of the unit vectors.
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
        # Csordas et al. 2025 §3.1 measures E_t[ ||h_{l+1,t}-h_{l,t}|| / ||h_{l,t}|| ]
        # -- the MEAN OF PER-TOKEN RATIOS. Taking a ratio of means instead
        # (adversarial review, 2026-09-02) is a different quantity whenever the
        # residual norm and the relative update size covary across tokens, which
        # they do: on 22,747 GPT-2 token states the two aggregations differed by
        # 27.12% per layer on average and 108.37% at the worst layer
        # analyze_norms.py forms the ratio per token before averaging.
        # SCALE-EXACT per-token ratio.
        #
        # `.norm()` squares first and `_CSORDAS_EPS` is an ABSOLUTE 1e-6 floor,
        # so the ratio -- which is scale-free by construction -- was neither.
        # Measured: the branch returned NaN at cloud scale 1e20 and 1e30, and
        # 0.0 at 1e-23 where the exact value is 0.5, which made
        # `effective_depth_90` NaN instead of 1.0 (independent Opus review,
        # 2026-09-04). Factoring each row's largest magnitude out of BOTH the
        # numerator and the denominator is exact and cancels in the ratio.
        _a = h_in.float()
        _b = h_out.float()
        _m = _a.abs().amax(dim=-1, keepdim=True)
        _tiny = torch.finfo(_a.dtype).tiny
        _msafe = _m.clamp_min(_tiny)
        _num = ((_b - _a) / _msafe).norm(dim=-1)
        _den = (_a / _msafe).norm(dim=-1)
        # Only an EXACTLY zero input row has no defined relative change; the
        # reference's 1e-6 clamp is kept solely for that case.
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
                    # CONFIGURABLE, and no longer a hard-coded 128. BI is a
                    # per-position mean, so the prefix length is part of the
                    # protocol: the first cached PG19 validation document holds
                    # 17,035 GPT-2 tokens, and scoring its first 128 rather
                    # than four 256-token chunks over the first 1,024 moved
                    # mean BI 0.12100935 -> 0.12332957 (+1.92%), with one layer
                    # moving 0.0127302 (adversarial review, 2026-09-04). 512
                    # matches the window every other cached task uses, so the
                    # tasks are at least on the same protocol.
                    #
                    # PROTOCOL DIVERGENCE, recorded rather than hidden:
                    # released ShortGPT scores 256-token PG19 chunks and
                    # Csordas et al. use 10 random GSM8K examples. BLME uses
                    # its own shared 512-token corpus window so every cached
                    # task in a run sees the same tokens, so absolute BI values
                    # are NOT directly comparable to either paper's published
                    # figures -- only within this zoo (adversarial review round
                    # 2, 2026-09-04).
                    enc = tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=_max_len).to(device)
                    out = model(**enc, output_hidden_states=True)
                    hs = list(out.hidden_states)
                    # post-final-norm: SUBSTITUTED. Block influence compares
                    # each block's input against its output, so every entry
                    # must be a block boundary.
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
