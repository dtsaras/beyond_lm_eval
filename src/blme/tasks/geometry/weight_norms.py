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
  - **norm_uniformity**: max(0, 1 - CV(frobenius_per_layer)) — clamped
    at zero when the layer norms vary wildly (CV > 1). Higher = more
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
    stack of independent 2-D matrices, each measured separately. Returns
    (frobs, specs, sranks, sranks / min(m, n), ||W||_F / sqrt(mn)) lists.
    """
    mats = W.reshape(-1, W.shape[-2], W.shape[-1]) if W.ndim > 2 else W.unsqueeze(0)
    frobs, specs, sranks = [], [], []
    sranks_norm, rms_scales = [], []
    for M in mats:
        # Promote, then factor out the largest magnitude: exact, and both norms
        # and their ratio survive denormal and near-overflow scales
        # (tests/test_scale_invariance.py).
        _Md = M.double()
        _g = _Md.abs().amax()
        _gf = float(_g)
        _Mn = _Md / _g.clamp_min(torch.finfo(_Md.dtype).tiny) if _gf > 0 else _Md
        _frob_n = float(_Mn.norm().item())          # in units of the max entry
        frob = _gf * _frob_n
        frobs.append(frob)
        _m, _n = int(M.shape[-2]), int(M.shape[-1])
        rms_scales.append(frob / float(np.sqrt(max(1, _m * _n))))
        try:
            S = torch.linalg.svdvals(_Mn)
            _spec_n = float(S[0].item())            # also in units of the max
            spectral = _gf * _spec_n
            specs.append(spectral)
            if _spec_n > 0:
                # The `_g` factor cancels exactly: both norms carry it once.
                raw_srank = _frob_n ** 2 / _spec_n ** 2
                _sr = min(raw_srank, float(min(M.shape)))
                sranks.append(_sr)
                # stable rank in [0, 1]: it is bounded above by min(m, n).
                sranks_norm.append(_sr / float(max(1, min(_m, _n))))
            else:
                sranks.append(0.0)
                sranks_norm.append(0.0)
        except Exception:
            # NaN, not fabricated values; excluded by the nan-aware means.
            specs.append(float("nan"))
            sranks.append(float("nan"))
            sranks_norm.append(float("nan"))
    return frobs, specs, sranks, sranks_norm, rms_scales


@register_task("geometry_weight_norms")
class WeightNormProfileTask(DiagnosticTask):
    """Per-layer weight matrix norm profiles (Tier 1, weight-only)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Per-Layer Weight Norm Analysis...")

        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)

        frob_per_layer = []
        spec_per_layer = []
        srank_per_layer = []
        srank_norm_per_layer, rms_per_layer = [], []

        for li, layer in enumerate(layers):
            frob_vals = []
            spec_vals = []
            srank_vals = []
            srank_norm_vals, rms_vals = [], []

            for name, param in layer.named_parameters():
                if param.ndim < 2:
                    continue  # skip biases and norms
                # Conv1D (GPT-2) stores (in, out), Linear (out, in); every
                # statistic here is transpose-invariant. Stable rank is clipped
                # to min(W.shape) inside the helper.
                W = param.detach().float()
                fr, sp, sr, srn, rms = _weight_matrix_stats(W)
                frob_vals.extend(fr)
                spec_vals.extend(sp)
                srank_vals.extend(sr)
                srank_norm_vals.extend(srn)
                rms_vals.extend(rms)

            srank_norm_per_layer.append(
                float(np.nanmean(srank_norm_vals)) if srank_norm_vals else 0.0)
            rms_per_layer.append(float(np.nanmean(rms_vals)) if rms_vals else 0.0)
            frob_per_layer.append(float(np.nanmean(frob_vals)) if frob_vals else 0.0)
            spec_per_layer.append(float(np.nanmean(spec_vals)) if spec_vals else 0.0)
            srank_per_layer.append(float(np.nanmean(srank_vals)) if srank_vals else 0.0)

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
            # Raw stable rank is the headline: zero-padding a matrix leaves it
            # unchanged. sr / min(m, n) halves under the same padding (the
            # operator does not change), so it is a capacity ratio, _meta_.
            "mean_stable_rank": float(srank_arr.mean()),
            "_meta_mean_stable_rank_normalized": float(
                np.nanmean(np.asarray(srank_norm_per_layer, dtype=float))
                if srank_norm_per_layer else float("nan")),
            # ||W||_F / sqrt(mn) carries the 1/sqrt(fan-in) initialisation
            # scale, so it falls with width: a per-model diagnostic only.
            "_meta_mean_rms_weight_scale": float(
                np.nanmean(np.asarray(rms_per_layer, dtype=float))
                if rms_per_layer else float("nan")),
            "norm_uniformity": norm_uniformity,
            "_meta_n_layers": n_layers,
        }
