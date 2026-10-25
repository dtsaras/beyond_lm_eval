"""
Relative-change ratio (sometimes called empirical Lipschitz) — measures
how smoothly representations change between adjacent layers.

The Lipschitz constant of a layer transformation f: R^d -> R^d bounds
how much the output can change relative to the input:
    ||f(x) - f(y)|| <= L * ||x - y||

A high Lipschitz constant means the layer is "explosive" (small input 
changes cause large output changes), while a low constant means the 
layer is contractive (smoothing). This is closely related to training 
stability, gradient flow, and the model's sensitivity to perturbations.

We estimate this empirically per layer as the MEAN over tokens of the
relative change, and report the maximum of these per-layer means:
    ratio(layer, x_i) = ||h_{l+1}(x_i) - h_l(x_i)|| / ||h_l(x_i)||
    L_hat(layer)      = mean_i ratio(layer, x_i)
    lipschitz_max     = max_layer L_hat(layer)

References:
- "Spectral Normalization for Generative Adversarial Networks"
  (Miyato et al., ICLR 2018) — Lipschitz constraints for stability
- "Lipschitz Regularity of Deep Neural Networks" (Virmaux & Scaman, 2018)
"""

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states, collected_layer_count
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")


def _rownorm(M):
    """Row-wise L2 norms that are exact at any representable magnitude.

    `np.linalg.norm` squares before summing, so with every coordinate finite
    and nonzero, scaling [1,-2,3] and [-4,5,-6] by 2^-80 made both underflow
    and the ratio NaN, where the exact answer is 3.327375628 (adversarial
    review round 3, 2026-09-04). Factoring out each row's largest magnitude
    only shifts exponents.
    """
    M = np.asarray(M, dtype=np.float64)
    m = np.max(np.abs(M), axis=1)
    out = np.zeros(M.shape[0], dtype=np.float64)
    nz = m > 0
    out[nz] = m[nz] * np.linalg.norm(M[nz] / m[nz, None], axis=1)
    return out


@register_task("geometry_lipschitz")
class LipschitzContinuityTask(DiagnosticTask):
    """
    Estimates the relative layer-wise change ratio between consecutive layers.

    Note: This metric is sometimes called "empirical Lipschitz constant", but
    it measures ||h_{l+1}(x) - h_l(x)|| / ||h_l(x)|| — the relative change
    in representations — rather than the true operator Lipschitz constant of
    the layer transformation (which would require sup over all input pairs).

    For each pair of adjacent layers (l, l+1), computes:
      - The ratio ||h_{l+1}(x) - h_l(x)|| / ||h_l(x)|| for each token x
      - Takes the MEAN over tokens as the per-layer estimate; ``lipschitz_max``
        is the maximum of these per-layer means across depth

    A model with uniform, moderate ratios across layers has smoother, more
    stable transformations. Spike patterns indicate layers where
    representations undergo dramatic restructuring.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Lipschitz Continuity Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            
        num_samples = self.config.get("num_samples", 20)
        use_cache = self.config.get("use_cache", True)
        
        # Collect hidden states from ALL layers
        all_layers = None
        if cache is not None and cache.is_populated and use_cache:
            # INCLUDE THE EMBEDDING. The paper
            # (sections/metrics_geometry.tex:101) defines the statistic over
            # all L residual updates, i.e. starting from the embedding INPUT to
            # block 0. Starting at block 0's OUTPUT scores only L-1 of them,
            # and the first update is systematically the largest, so the
            # omission is not a wash (adversarial review round 2,
            # 2026-09-04).
            all_layers = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples,
                replace_final_norm=True, include_embedding=True)
            # Derived from whether the swap ACTUALLY fired, not from the
            # mere presence of a hooked tensor (a shape mismatch is
            # rejected inside get_hidden_states).
            _last_is_block = bool(getattr(cache, "_last_block_substituted", False))
            if all_layers is None:
                # A POPULATED CACHE IS NOT NECESSARILY A USABLE ONE. This task ran c
                # orrectly with NO cache and returned an error with an attention-onl
                # y one, i.e. it refused rather than self-collecting -- the defect a
                # stra RED-flagged on geometry_vendi_score. Found by EXECUTING the c
                # heck across all 51 tasks: the class is 15 sites, not the 5 a read-
                # only sweep reported (2026-09-09).
                logger.info("  shared cache is populated but carries no hidden states; self-collecting.")
        if all_layers is None:
            _st = {}
            all_layers = collect_hidden_states(model, tokenizer, dataset,
                                           num_samples=num_samples,
                                           layer_idx="all",
                                           replace_final_norm=True, status=_st,
                                           include_embedding=True)
            # NOT unconditionally True. The hook only fires when `get_layers`
            # locates the blocks and the captured shape matches the slot; a
            # model where it does not silently keeps `ln_f(b_{L-1})`, and
            # asserting otherwise reports a LayerNorm output as a block output
            # (adversarial review, 2026-09-04).
            _last_is_block = bool(_st.get("last_is_block", False))
        
        if not all_layers:
            return {"error": "No hidden states collected"}
            
        layer_indices = sorted(all_layers.keys())
        # RESOLVED by a hook, not a truncation. Adversarial review measured
        # that the truncation drops what is the LARGEST transition in the
        # profile on all 7 models tested -- hook-recovered deepest-block value
        # vs the shipped max: 4.6698 vs 1.0408 (pythia-70m, 4.49x), 3.6345 vs
        # 0.7618 (pythia-160m, 4.77x), 1.3579 vs 0.6539 (gpt2), 1.8303 vs
        # 0.9040 (distilgpt2) -- so `relative_change_max` was not the max
        # across depth at all. Adopting the hook moves relative_change_mean by
        # +85.1%/+42.0%/+41.8%/+26.0%/+6.3%/+4.7%/+3.6%, and because that shift
        # is DEPTH-DEPENDENT the truncation was itself adding to the depth
        # confound. The truncation now survives only on the cache path.
        #
        # Historical rationale for the truncation it replaces:
        # Drop the post-final-norm entry. HuggingFace's last hidden_states
        # element has already passed through ln_f for most architectures, so the
        # final "adjacent pair" spans block output -> LayerNorm, not a layer
        # transformation. That pair WAS the reported maximum: pythia-70m's
        # per-pair ratios are [0.78, 1.00, 0.81, 0.96, 28.64], and
        # lipschitz_max_layer equalled n_layers-2 in 8 of 12 models
        # (adversarial review, 2026-09-03).
        if not _last_is_block:
            # INDEX-INDEPENDENT: drop the LAST entry, which is the one that has
            # passed through ln_f. `collected_layer_count(model)` counted
            # BLOCKS, so with the embedding now at key 0 it would have cut the
            # deepest block instead (adversarial review round 2, 2026-09-04).
            # This branch only runs when the hook failed to fire on every
            # forward, i.e. when there is no raw deepest-block output to use.
            if len(layer_indices) > 2:
                layer_indices = layer_indices[:-1]
        n_layers = len(layer_indices)
        
        if n_layers < 2:
            return {"error": "Need at least 2 layers for Lipschitz analysis"}
        
        lipschitz_means = []
        contraction_rates = []  # ratio < 1 = contractive, > 1 = expansive
        
        for i in range(n_layers - 1):
            h_l = all_layers[layer_indices[i]].numpy()
            h_next = all_layers[layer_indices[i + 1]].numpy()
            
            # Align shapes (use minimum token count)
            n_tokens = min(len(h_l), len(h_next))
            h_l = h_l[:n_tokens]
            h_next = h_next[:n_tokens]
            
            # Compute norms
            # SCALE-EXACT row norms. `np.linalg.norm` squares before summing,
            # so with every stored coordinate finite and nonzero, scaling
            # [1,-2,3] and [-4,5,-6] by 2^-80 made both inputs underflow to 0
            # and the ratio undefined (adversarial review round 3,
            # 2026-09-04). Factoring out each row's largest magnitude first
            # only shifts exponents.
            diff_norms = _rownorm(h_next - h_l)
            input_norms = _rownorm(h_l)
            
            # EXACT zero only, and no absolute floor.
            #
            # The paper defines the mean over EVERY token's ||dh||/||h||;
            # deleting rows below an absolute 1e-8 both changes the population
            # and breaks the scale invariance the ratio has by construction
            # (adversarial review round 2, 2026-09-04). Only a token whose
            # residual state is exactly the zero vector has no defined relative
            # change, and there is nothing to divide by there.
            valid = input_norms > 0
            if not np.any(valid):
                # NaN, not 0.0: a sentinel that looks legal silently drags the
                # mean and can win the argmax. 0.0 is a legal relative-change
                # value, so it is indistinguishable from a real measurement.
                lipschitz_means.append(float("nan"))
                contraction_rates.append(float("nan"))
                continue
                
            ratios = diff_norms[valid] / input_norms[valid]
            
            lipschitz_means.append(float(np.mean(ratios)))
            
            # Contraction rate: ||h_{l+1}(x)|| / ||h_l(x)||
            output_norms = _rownorm(h_next)
            cr = output_norms[valid] / input_norms[valid]
            contraction_rates.append(float(np.mean(cr)))
        
        # The paper's literal denominator, computed alongside.
        _paper_eps = 1e-8
        _pe = []
        for i in range(n_layers - 1):
            _a = all_layers[layer_indices[i]].numpy()
            _b = all_layers[layer_indices[i + 1]].numpy()
            _n = min(len(_a), len(_b))
            if _n == 0:
                continue
            _dn = _rownorm(_b[:_n] - _a[:_n])
            _in = _rownorm(_a[:_n])
            _pe.append(float(np.mean(_dn / (_in + _paper_eps))))
        _paper_eps_mean = float(np.mean(_pe)) if _pe else float("nan")

        lip_arr = np.array(lipschitz_means)
        # PAPER-REQUESTED OUTPUT, previously absent: the fraction of layers
        # whose mean relative change is below 1, i.e. contractive
        # (metrics_geometry.tex:101, metrics_catalogue.tex:31). For layer means
        # [0.5, 2.0] the required value is 0.500000 and the result carried no
        # such key at all (adversarial review round 3, 2026-09-04).
        _lf = lip_arr[np.isfinite(lip_arr)]
        _contractive_fraction = (float(np.mean(_lf < 1.0)) if _lf.size
                                 else float("nan"))
        _cr_arr = np.asarray(contraction_rates, dtype=np.float64)
        if _cr_arr.size and np.any(np.isfinite(_cr_arr)):
            _cr_mean = float(np.nanmean(_cr_arr))
            _cr_std = float(np.nanstd(_cr_arr))
        else:
            _cr_mean = _cr_std = float("nan")

        # The keys named `lipschitz_*` are kept for backward compatibility,
        # but this metric is the *relative-change ratio* ||Δh||/||h||, not
        # the operator Lipschitz constant. The `relative_change_*` keys are
        # the preferred names going forward.
        # nan-aware: degenerate layers now record NaN rather than a
        # legal-looking 0.0, so the aggregates must skip them explicitly.
        if not np.any(np.isfinite(lip_arr)):
            mean_v = max_v = std_v = float("nan")
            max_layer = -1
        else:
            mean_v = float(np.nanmean(lip_arr))
            max_v = float(np.nanmax(lip_arr))
            std_v = float(np.nanstd(lip_arr))
            max_layer = int(np.nanargmax(lip_arr))

        return {
            # `relative_change_*` are the real names: this is the
            # relative-change ratio ||dh||/||h||, NOT an operator Lipschitz
            # constant. The `lipschitz_*` aliases are kept for backward
            # compatibility but `_meta_`-prefixed, because they were BIT-
            # IDENTICAL duplicates -- eight keys carrying four values -- and a
            # sparse selection then splits one coefficient across two perfectly
            # collinear columns (adversarial review, 2026-09-03).
            #
            # Note also that `*_max` is the maximum over PER-LAYER MEANS, not a
            # token-level maximum. The docstring said so; the key name did not.
            "relative_change_mean": mean_v,
            # NOTE: this is the maximum over PER-LAYER MEANS, not a
            # token-level maximum. Keeping the established key name and
            # documenting the semantics here, rather than renaming a key the
            # study tables already carry.
            "relative_change_max": max_v,
            "relative_change_std": std_v,
            # DEMOTED to _meta_ (2026-09-04). With the embedding transition now
            # included, the first residual update is essentially always the
            # largest, so this is 0 on 4 of 4 models tested -- a zero-variance
            # ABSOLUTE-DEPTH column in the feature matrix (independent Opus
            # review, 2026-09-04). It is still informative as provenance, and
            # the normalized position is emitted alongside for anyone who wants
            # a comparable "where in the stack" quantity.
            "_meta_relative_change_max_layer": max_layer,
            "_meta_relative_change_max_layer_frac": (
                float(max_layer) / max(1, len(lip_arr) - 1)
                if max_layer >= 0 else float("nan")),
            # The paper's own requested statistic (see the note above).
            "contractive_fraction": _contractive_fraction,
            # THE PAPER'S LITERAL FORM, emitted so both exist.
            #
            # `sections/metrics_geometry.tex:101` writes
            # L_l = n^-1 sum_i ||dh_i|| / (||h_i|| + epsilon) over ALL n tokens.
            # The code instead drops exactly-zero-norm rows and divides the rest
            # by ||h_i|| alone, because the paper's own literal epsilon destroys
            # the scale invariance the SAME paper claims for the quantity: with
            # it, a 1e-10 rescale of an unchanged representation moves the ratio
            # 0.8679 -> 0.0554, while the epsilon-free form is exactly constant
            # (independent Opus review, 2026-09-05). So the code is right and
            # the paper text needs correcting -- but the paper's number is
            # reproducible from here, under its own name, rather than being
            # unavailable.
            "_meta_relative_change_mean_paper_eps": _paper_eps_mean,
            "_meta_lipschitz_mean": mean_v,
            "_meta_lipschitz_max": max_v,
            "_meta_lipschitz_std": std_v,
            "_meta_lipschitz_max_layer": max_layer,
            "_meta_last_entry_is_block_output": bool(_last_is_block),
            # nan-aware, like relative_change_*. The degenerate branch appends
            # NaN, and plain np.mean/np.std propagate it: ONE degenerate layer
            # would NaN both of these while the relative_change_* keys survived.
            # Latent (never fires on real activations) but inconsistent.
            "mean_contraction_rate": _cr_mean,
            "contraction_std": _cr_std,
        }
