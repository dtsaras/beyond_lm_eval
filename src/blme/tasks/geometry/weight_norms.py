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


def _weight_matrix_stats(W):
    """Frobenius norm, spectral norm (sigma_max) and stable rank for a weight
    tensor. A >2-D parameter (e.g. fused MoE experts, shape (E, in, out)) is a
    stack of independent 2-D matrices; each is measured separately. The old code
    ran svdvals on the whole >2-D tensor and then crashed float(S[0].item()) on
    the batched singular-value vector, falling back to spectral=Frobenius (too
    high) and a fabricated stable_rank=1.0. Returns (frobs, specs, sranks) lists.
    """
    mats = W.reshape(-1, W.shape[-2], W.shape[-1]) if W.ndim > 2 else W.unsqueeze(0)
    frobs, specs, sranks = [], [], []
    for M in mats:
        frob = float(M.norm().item())
        frobs.append(frob)
        try:
            S = torch.linalg.svdvals(M)
            spectral = float(S[0].item())
            specs.append(spectral)
            if spectral > 0:
                raw_srank = frob ** 2 / spectral ** 2
                sranks.append(min(raw_srank, float(min(M.shape))))
            else:
                sranks.append(0.0)
        except Exception:
            specs.append(frob)  # fallback
            sranks.append(1.0)
    return frobs, specs, sranks


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
                # A >2-D param (fused MoE experts) is a stack of matrices, each
                # measured separately. Stable rank ``||W||_F²/||W||_2²`` is
                # clipped to the matrix rank ``min(W.shape)`` inside the helper
                # (values above are numerical artefacts, e.g. SVD-on-bf16 on
                # shared-embedding matrices returning a tiny sigma_max).
                fr, sp, sr = _weight_matrix_stats(W)
                frob_vals.extend(fr)
                spec_vals.extend(sp)
                srank_vals.extend(sr)

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
