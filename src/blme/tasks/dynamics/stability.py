from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM
import logging
logger = logging.getLogger("blme")

@register_task("dynamics_stability")
class NeighborhoodStabilityTask(DiagnosticTask):
    """
    Measures Jaccard stability of k-NN neighborhoods.
    If 'reference_model_path' is provided in config, compares current model to reference.
    Otherwise, compares the embedding neighborhood graph against a seeded
    small Gaussian perturbation of the embeddings.

    This is a BLME embedding-graph diagnostic, not a Lyapunov exponent or
    Jacobian spectral-norm calculation.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Neighborhood Stability Analysis...")
        k = self.config.get("k", 50)
        n_sample = self.config.get("num_samples", 5000)
        ref_path = self.config.get("reference_model_path", None)
        
        device = next(model.parameters()).device
        E1 = get_embeddings(model)
        if E1 is None: return {"error": "E1 not found"}
        E1 = E1.to(device)
        
        if ref_path:
            stability_mode = "reference_model"
            logger.info(f"Loading reference model from {ref_path}...")
            # This is heavy. Optimally we just load embeddings if possible.
            # But assume we load model for compatibility.
            # Warning: Memory usage.
            try:
                ref_model = AutoModelForCausalLM.from_pretrained(ref_path, trust_remote_code=True).to('cpu') # Keep on CPU first
                E2 = get_embeddings(ref_model)
                if E2 is None: return {"error": "E2 not found"}
                # Move E2 to device only if needed, or keep both on CPU if VRAM tight
                # For safety, let's do calculation on CPU if ref provided, or E2 to device
                E2 = E2.to(device)
                del ref_model
                torch.cuda.empty_cache()
            except Exception as e:
                return {"error": f"Failed to load ref model: {e}"}
        else:
            stability_mode = "embedding_noise"
            noise_std = float(self.config.get("noise_std", 0.01))
            seed = int(self.config.get("seed", 42))
            logger.info(
                "No reference model provided. Calculating embedding-noise "
                f"stability with noise_std={noise_std}."
            )
            generator = torch.Generator(device=E1.device)
            generator.manual_seed(seed)
            E1_float = E1.float()
            # SCALE-EXACT row norms. `E1_float.norm(dim=1).clamp_min(1e-10)`
            # had both of this campaign's most repeated numeric defects at
            # once, in a file whose own comment 30 lines below condemns the
            # first of them:
            #
            #  1. an ABSOLUTE epsilon (1e-10) inside a scale-free operation;
            #  2. `.norm()` SQUARES before summing, so it overflows to inf (and
            #     underflows to 0) long before the data does.
            #
            # Together they broke the defining axiom. With `noise_std = 0` the
            # two embeddings are identical, so self-stability must be exactly
            # 1.0 -- but at scale 1e25 the row norm overflowed to inf, `noise`
            # became `inf * 0 = NaN`, and the task returned **0.048611111111**
            # (gpt-5.6-sol round 9, 2026-09-07). The same overflow moved the
            # ordinary mean to 0.031756902891 / 0.042763078846 / 0.044048734770
            # at 1e-30 / 1e-13 / 1e25 against 0.966494845361.
            #
            # Factoring out the largest per-row magnitude first only shifts
            # exponents, so ||x|| = rmax * ||x / rmax|| is exact, and every
            # coordinate is inside [-1, 1] before anything is squared. A row
            # that is exactly zero gets scale 0 and therefore no noise, which
            # is what "1% of this row" means for a zero row.
            _rmax = E1_float.abs().amax(dim=1, keepdim=True)
            _tiny = torch.finfo(E1_float.dtype).tiny
            row_scale = _rmax * (
                E1_float / _rmax.clamp_min(_tiny)).norm(dim=1, keepdim=True)
            # Normalize the noise DIRECTION before scaling, so the realized
            # perturbation is exactly `noise_std` relative to each row.
            #
            # Multiplying an isotropic `randn` elementwise by the row norm makes
            # ||noise_row|| ~ sqrt(d) * noise_std * ||E_row||, i.e. the realized
            # perturbation is 0.01*sqrt(d), NOT the documented 1%: measured
            # 0.2261 / 0.2770 / 0.3199 / 0.3577 / 0.4525 at d = 512 / 768 /
            # 1024 / 1280 / 2048, matching 0.01*sqrt(d) to four decimals. So
            # the perturbation size was a deterministic function of d_model,
            # and spearman(as-implemented ranking, equal-perturbation ranking)
            # was **+0.500** -- half the cross-model ordering this metric
            # produced was that artifact (adversarial review, 2026-09-03).
            #
            # The sibling `perturbation_growth.py` already normalizes the
            # direction; this brings the two into agreement.
            raw = torch.randn(
                E1_float.shape,
                device=E1.device,
                dtype=E1_float.dtype,
                generator=generator,
            )
            raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            noise = raw * row_scale * noise_std
            E2 = E1_float + noise
            
        # Compute Stability
        E1_np = E1.float().cpu().numpy()
        E2_np = E2.float().cpu().numpy()
        
        n_vocab = len(E1_np)
        if len(E2_np) != n_vocab:
            return {"error": f"Vocab mismatch: {n_vocab} vs {len(E2_np)}"}
            
        # `safe_unit_rows`, not `norm + 1e-10`. An ABSOLUTE epsilon inside a
        # scale-free row normalisation is this campaign's most repeated numeric
        # defect: it hard-codes a magnitude into an operation that has none, and
        # the norm underflows before the data does. `safe_unit_rows` factors out
        # the largest per-row magnitude first, which only shifts exponents and
        # is therefore exact.
        from ..geometry.utils import safe_unit_rows
        E1_norm = safe_unit_rows(E1_np)
        E2_norm = safe_unit_rows(E2_np)
        
        # Local generator — np.random.seed(42) mutated GLOBAL numpy RNG
        # state, an order-dependence hazard for co-resident tasks (audit
        # 2026-07-16). Row sampling now also honors the config seed.
        # SAMPLE SIZE IS THE WHOLE STORY HERE, so it is measured and reported
        # rather than assumed. The study runs `num_samples: 100` against a
        # package default of 5000, and at 100 rows the estimator does not
        # resolve the models it is asked to rank. Ten seeds, gpt2 vs distilgpt2
        # (red-team review, 2026-09-05):
        #
        #   gpt2        mean 0.950968  sd 0.003644  range 0.012055
        #   distilgpt2  mean 0.957275  sd 0.002970  range 0.009665
        #   cross-model mean difference                    0.006307
        #
        # The within-model seed RANGE exceeds the between-model difference, the
        # ordering FLIPS at 1 of the 10 seeds (seed 13: -0.000641), and the
        # study's seed 42 happens to give the second-largest gap of the ten.
        # fp32 -> bf16 moves distilgpt2 a further -0.0019, a quarter of the gap.
        #
        # The ESTIMATOR is sound -- the direction-normalisation fix is correct
        # and the realized perturbation is exactly `noise_std` at every width --
        # so this is a `confounds` matter, not a fidelity one. What the harness
        # can do is stop hiding it: the standard error of the mean over the
        # sampled rows is emitted, so a caller can see whether a difference it
        # is about to report is inside its own noise.
        rng = np.random.default_rng(int(self.config.get("seed", 42)))
        sample_idx = rng.choice(n_vocab, min(n_sample, n_vocab), replace=False)

        jaccards = []

        for idx in tqdm(sample_idx, desc="Stability Jaccard"):
            # DIRECT chord distances with a STABLE tie rule, for both models.
            #
            # Two defects in the previous `argsort(E_norm @ E_norm[idx])`:
            #  1. cancellation -- unit rows make every dot product sit near 1,
            #     so the separating quantity is the difference of two ~1.0 sums
            #     (confirmed in `hubness.py` 2026-09-07; this site found by
            #     sweeping the class).
            #  2. `np.argsort` defaults to quicksort, which is UNSTABLE, so
            #     tied neighbours were ordered arbitrarily.
            # Both defects bias this metric in the SAME direction: they perturb
            # the two neighbour sets independently, which can only shrink their
            # intersection. A spurious ranking difference is indistinguishable
            # from genuine representational drift, so the error manufactures
            # apparent INSTABILITY -- it cannot manufacture stability.
            def _knn(E):
                d2 = ((E - E[idx]) ** 2).sum(axis=1)
                d2[idx] = np.inf
                return set(np.lexsort((np.arange(len(d2)), d2))[:k].tolist())

            nn1 = _knn(E1_norm)
            nn2 = _knn(E2_norm)
            
            jaccard = len(nn1 & nn2) / len(nn1 | nn2)
            jaccards.append(jaccard)
            
        _n = len(jaccards)
        _sem = float(np.std(jaccards) / np.sqrt(_n)) if _n > 1 else float("nan")
        if np.isfinite(_sem) and _sem > 0.002:
            logger.warning(
                f"  dynamics_stability: standard error of the mean is {_sem:.4f} "
                f"at num_samples={_n}. The measured gpt2-vs-distilgpt2 "
                f"difference is 0.0063, so a difference of that size is inside "
                f"this run's own noise. Raise num_samples (the package default "
                f"is 5000) before ranking models on it.")
        return {
            "diagnostic_semantics": "embedding_neighborhood_jaccard_stability",
            "stability_mean": float(np.mean(jaccards)),
            "stability_std": float(np.std(jaccards)),
            # The quantity a caller needs to decide whether a between-model
            # difference is real. sd/sqrt(n), not sd: the headline is a MEAN.
            "_meta_stability_sem": _sem,
            "_meta_stability_n_rows": int(_n),
            "stability_mode": stability_mode,
        }
