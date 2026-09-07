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
    stack of independent 2-D matrices; each is measured separately. The old code
    ran svdvals on the whole >2-D tensor and then crashed float(S[0].item()) on
    the batched singular-value vector, falling back to spectral=Frobenius (too
    high) and a fabricated stable_rank=1.0. Returns (frobs, specs, sranks) lists.
    """
    mats = W.reshape(-1, W.shape[-2], W.shape[-1]) if W.ndim > 2 else W.unsqueeze(0)
    frobs, specs, sranks = [], [], []
    # DIMENSIONLESS COMPANIONS. The raw norms are architecture-conditioned:
    # ||W||_F grows with sqrt(mn) and stable rank is bounded by min(m, n), so
    # mean_stable_rank has Spearman +0.683 with d_model over 61 models. These
    # two carry the same information with the shape term divided out, and
    # neither needs a divisor from outside the matrix itself.
    sranks_norm, rms_scales = [], []
    for M in mats:
        # FACTOR OUT THE LARGEST MAGNITUDE FIRST. `M.norm()` squares before
        # summing, so it underflows to 0.0 and overflows to inf long before the
        # matrix does -- and stable rank is ||W||_F^2 / ||W||_2^2, a ratio of
        # squared norms and therefore EXACTLY scale-invariant. Measured on
        # `diag(10..1)/10`, whose exact stable rank is 3.85 (independent
        # gpt-5.6-sol review, 2026-09-06; reproduced here):
        #
        #     scale 1      3.850000256   (float32 rounding, fine)
        #     scale 1e-25  0.0           frob underflowed, `spectral > 0` false
        #     scale 1e30   10.0          frob = inf, so min(inf, min(m,n)) = m
        #
        # 100% and 159.7% errors on a quantity that cannot depend on scale at
        # all. Dividing by the max only shifts exponents, so the ratio below is
        # exact and the factor cancels identically.
        # CAST BEFORE SCALING. The max-factoring below is correct in form and
        # fixes the stable-rank RATIO (3.849999320273-3.850000255895 across all
        # seven rungs from 1e-30 to 2.406e38, worst 1.77e-7 against the exact
        # 3.85), because the `_g` factor cancels identically there.
        #
        # It did NOT fix the two norms themselves, because the division ran in
        # the input dtype. At scale 1e-38 a float32 matrix is DENORMAL -- it has
        # already lost mantissa bits, and dividing denormal by denormal loses
        # more -- so BLME reported Frobenius/spectral
        # 1.669205318748e-38 / 8.507058067882e-39 against independent
        # 1.962141623860e-38 / 9.999999350456e-39 (**-14.93% each**), reaching
        # **-99.15%** at 1e-40 (gpt-5.6-sol round 9, 2026-09-07). The ratio
        # survived because both errors are the same factor; the absolute
        # quantities did not.
        #
        # Promoting first is exact -- every float32 denormal is a normal double
        # -- so the division is then error-free and the multiply-back is a
        # single rounding. Same repair as `safe_svdvals` received today.
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
            # NaN, not fabricated values (audit 2026-07-16: spectral=frob +
            # stable_rank=1.0 was exactly the silent-fallback pathology the
            # module docstring criticizes). NaN rows are excluded by the
            # nan-aware aggregation.
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

        try:
            from transformers.pytorch_utils import Conv1D as _HFConv1D
        except Exception:
            _HFConv1D = None

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
                W = param.detach().float()
                # For Conv1D (GPT-2), weight is (in, out); for Linear, (out, in).
                # A >2-D param (fused MoE experts) is a stack of matrices, each
                # measured separately. Stable rank ``||W||_F²/||W||_2²`` is
                # clipped to the matrix rank ``min(W.shape)`` inside the helper
                # (values above are numerical artefacts, e.g. SVD-on-bf16 on
                # shared-embedding matrices returning a tiny sigma_max).
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
            # `_meta_`: THE DEMOTION PICKED THE WEAKER PROXY FIRST.
            # `mean_rms_weight_scale` was demoted at rho -0.543 with
            # d_model while THIS key, sitting two lines above a comment
            # that calls its normalized twin "comparable across widths",
            # is rho **+0.830** with d_model and fits d^0.748 -- a
            # stronger width dependence than the column that was demoted
            # for having one (independent Opus review, 2026-09-05).
            # `mean_stable_rank_normalized` divides by the shape bound
            # min(m, n) and is the feature.
            "_meta_mean_stable_rank": float(srank_arr.mean()),
            # Dimensionless companions (see _weight_matrix_stats): stable rank
            # divided by its own bound min(m, n), and ||W||_F / sqrt(mn). Both
            # carry the same information with the matrix-shape term divided
            # out, so they are comparable across widths without an external
            # divisor.
            "mean_stable_rank_normalized": float(
                np.nanmean(np.asarray(srank_norm_per_layer, dtype=float))
                if srank_norm_per_layer else float("nan")),
            # `_meta_`: NOT dimensionless after all, despite sitting under the
            # comment above that says it is. ||W||_F / sqrt(mn) divides out the
            # matrix SHAPE but not the initialisation scale, which is itself
            # width-dependent: standard fan-in init puts the per-coordinate sd
            # at ~1/sqrt(d), so this quantity falls as a power of width.
            # Measured across the archive it has rho(d_model) = -0.582 and
            # fits d^-0.471 -- i.e. essentially the 1/sqrt(d) the init
            # prescribes (red-team review, 2026-09-05). It stays emitted,
            # because per-model it is the right diagnostic; it must not enter a
            # cross-model regression as a structural predictor of capability.
            # `mean_stable_rank_normalized` above IS dimensionless and stays a
            # feature.
            "_meta_mean_rms_weight_scale": float(
                np.nanmean(np.asarray(rms_per_layer, dtype=float))
                if rms_per_layer else float("nan")),
            "norm_uniformity": norm_uniformity,
            "_meta_n_layers": n_layers,
        }
