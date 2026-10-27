from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
from ..geometry.utils import safe_unit_rows
import torch
import numpy as np
from tqdm import tqdm
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
            # The whole reference model is loaded on CPU for its embeddings.
            try:
                ref_model = AutoModelForCausalLM.from_pretrained(ref_path, trust_remote_code=True).to('cpu') # Keep on CPU first
                E2 = get_embeddings(ref_model)
                if E2 is None: return {"error": "E2 not found"}
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
            # Scale-exact row norms: the row max is factored out and the product
            # taken in double, so noise_std = 0 gives self-stability exactly 1.0
            # at every scale up to float32's max.
            _E64 = E1_float.double()
            _rmax = _E64.abs().amax(dim=1, keepdim=True)
            _tiny = torch.finfo(_E64.dtype).tiny
            row_scale = _rmax * (
                _E64 / _rmax.clamp_min(_tiny)).norm(dim=1, keepdim=True)
            # The noise DIRECTION is normalised before scaling, so the realised
            # perturbation is exactly noise_std of each row at every width (an
            # elementwise randn would realise noise_std * sqrt(d)).
            raw = torch.randn(
                E1_float.shape,
                device=E1.device,
                dtype=E1_float.dtype,
                generator=generator,
            )
            raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            noise = (raw.double() * row_scale * noise_std).to(E1_float.dtype)
            E2 = E1_float + noise

        # Compute Stability
        E1_np = E1.float().cpu().numpy()
        E2_np = E2.float().cpu().numpy()

        # Only rows the tokenizer can emit (ids < len(tokenizer)) are queries or
        # neighbours: embedding matrices are padded past the tokenizer (pythia
        # 27 rows, Qwen3 267), and such rows are not tokens.
        n_rows_total = len(E1_np)
        n_tok = len(tokenizer) if hasattr(tokenizer, "__len__") else None
        n_vocab = min(n_rows_total, n_tok) if n_tok else n_rows_total
        E1_np = E1_np[:n_vocab]
        if len(E2_np) < n_vocab:
            return {"error": f"Vocab mismatch: {n_vocab} vs {len(E2_np)}"}
        E2_np = E2_np[:n_vocab]

        E1_norm = safe_unit_rows(E1_np)
        E2_norm = safe_unit_rows(E2_np)

        # Local, seeded row sampler (no global RNG). The SEM of the mean over
        # sampled rows is emitted: at 100 rows the seed range exceeds the
        # gpt2-vs-distilgpt2 difference.
        rng = np.random.default_rng(int(self.config.get("seed", 42)))
        sample_idx = rng.choice(n_vocab, min(n_sample, n_vocab), replace=False)

        jaccards = []

        for idx in tqdm(sample_idx, desc="Stability Jaccard"):
            # Direct chord distances (no near-1 dot-product cancellation) with a
            # stable index tie-break, for both embeddings.
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
            "_meta_stability_sem": _sem,
            "_meta_stability_n_rows": int(_n),
            "_meta_n_rows_emittable": int(n_vocab),
            "_meta_n_rows_excluded": int(n_rows_total - n_vocab),
            "_meta_k": int(k),
            "stability_mode": stability_mode,
        }
