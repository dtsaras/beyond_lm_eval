"""
Per-layer weight norm profiles — Tier 1 (weight-only, no data dependency).

For each transformer block, computes the Frobenius norm, spectral norm
(largest singular value), and stable rank (||W||_F^2 / ||W||_2^2) of
all linear weight matrices and reports per-layer aggregates.

These complement `geometry_spectral` (which fits a power-law exponent to
the aggregate singular value distribution) by giving a *layer-resolved*
view of the weight magnitude structure. Models where norms grow with
depth may have exploding-gradient tendencies; models with uniform norms
are typically better-conditioned.

Reports:
  - **frobenius_norm_per_layer**: mean Frobenius norm of all weight
    matrices in each transformer block.
  - **spectral_norm_per_layer**: mean largest singular value per block.
  - **stable_rank_per_layer**: mean stable rank per block.
  - **norm_uniformity**: 1 - CV(frobenius_per_layer). Higher = more
    uniform weight magnitudes across layers.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


@register_task("geometry_weight_norms")
class WeightNormProfileTask(DiagnosticTask):
    """Per-layer weight matrix norm profiles (Tier 1, weight-only)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Per-Layer Weight Norm Analysis...")

        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)

        try:
            from transformers.pytorch_utils import Conv1D as _HFConv1D
        except Exception:
            _HFConv1D = None

        frob_per_layer = []
        spec_per_layer = []
        srank_per_layer = []

        for li, layer in enumerate(layers):
            frob_vals = []
            spec_vals = []
            srank_vals = []

            for name, param in layer.named_parameters():
                if param.ndim < 2:
                    continue  # skip biases and norms
                W = param.detach().float()
                # For Conv1D (GPT-2), weight is (in, out); for Linear, (out, in).
                # SVD works on either shape — just compute norms directly.
                frob = float(W.norm().item())
                frob_vals.append(frob)

                # Spectral norm = largest singular value. Stable rank
                # ``||W||_F² / ||W||_2²`` is bounded above by the matrix
                # rank ``min(W.shape)`` by construction — any value
                # above that is a numerical artefact (has been observed
                # at 10⁴ on Qwen3.5 shared-embedding matrices where
                # SVD-on-bf16 returned a tiny ``S[0]`` relative to the
                # Frobenius norm). Clip explicitly so one outlier
                # module doesn't dominate the per-layer average.
                try:
                    S = torch.linalg.svdvals(W)
                    spectral = float(S[0].item())
                    spec_vals.append(spectral)
                    if spectral > 0:
                        raw_srank = frob ** 2 / spectral ** 2
                        max_rank = int(min(W.shape))
                        srank_vals.append(min(raw_srank, float(max_rank)))
                    else:
                        srank_vals.append(0.0)
                except Exception:
                    spec_vals.append(frob)  # fallback
                    srank_vals.append(1.0)

            frob_per_layer.append(float(np.mean(frob_vals)) if frob_vals else 0.0)
            spec_per_layer.append(float(np.mean(spec_vals)) if spec_vals else 0.0)
            srank_per_layer.append(float(np.mean(srank_vals)) if srank_vals else 0.0)

        frob_arr = np.array(frob_per_layer)
        spec_arr = np.array(spec_per_layer)
        srank_arr = np.array(srank_per_layer)

        # Norm uniformity: 1 - coefficient of variation
        if frob_arr.mean() > 0:
            cv = float(frob_arr.std() / frob_arr.mean())
            norm_uniformity = max(0.0, 1.0 - cv)
        else:
            norm_uniformity = float("nan")

        return {
            "frobenius_norm_per_layer": frob_arr.tolist(),
            "spectral_norm_per_layer": spec_arr.tolist(),
            "stable_rank_per_layer": srank_arr.tolist(),
            "mean_frobenius_norm": float(frob_arr.mean()),
            "mean_spectral_norm": float(spec_arr.mean()),
            "mean_stable_rank": float(srank_arr.mean()),
            "norm_uniformity": norm_uniformity,
            "n_layers": n_layers,
        }
