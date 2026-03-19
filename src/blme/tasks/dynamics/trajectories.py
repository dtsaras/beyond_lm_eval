from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger("blme")


def _slerp(h1, h2, alpha):
    """Spherical linear interpolation between two vectors."""
    h1_norm = F.normalize(h1, dim=-1)
    h2_norm = F.normalize(h2, dim=-1)
    dot = torch.clamp((h1_norm * h2_norm).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if omega.abs() < 1e-6:
        # Vectors are nearly parallel, fall back to lerp
        return (1 - alpha) * h1 + alpha * h2
    sin_omega = torch.sin(omega)
    return (torch.sin((1 - alpha) * omega) / sin_omega) * h1 + \
           (torch.sin(alpha * omega) / sin_omega) * h2


@register_task("dynamics_interpolation")
class LatentInterpolationTask(DiagnosticTask):
    """
    Interpolates between two hidden states in latent space.
    Measures entropy of decoded predictions along the path (convexity check).
    Uses norm-corrected linear interpolation and slerp for comparison.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Latent Interpolation...")
        num_pairs = self.config.get("num_pairs", 50)
        steps = self.config.get("steps", 10)
        num_samples = self.config.get("num_samples", 10)

        device = next(model.parameters()).device

        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    dataset.append({"text": dset[i]["text"]})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [{"text": f"Sample {i}"} for i in range(num_samples)]

        samples = list(dataset)
        if len(samples) < 2: return {"error": "Need at least 2 samples"}

        entropies = defaultdict(list)
        slerp_entropies = defaultdict(list)
        alphas = np.linspace(0, 1, steps)

        count = 0
        with torch.no_grad():
            while count < num_pairs:
                import random
                s1, s2 = random.sample(samples, 2)

                h_states = []
                for s in [s1, s2]:
                    if isinstance(s, str):
                        inp = tokenizer(s, return_tensors="pt").to(device)
                    elif 'text' in s:
                        inp = tokenizer(s['text'][:128], return_tensors="pt").to(device)
                    elif 'input_ids' in s:
                        inp = {'input_ids': torch.tensor(s['input_ids']).long().unsqueeze(0).to(device)}
                    else: continue

                    out = model(**inp, output_hidden_states=True)
                    h_states.append(out.hidden_states[-1][0, -1].float())

                if len(h_states) < 2: continue

                h1, h2 = h_states

                for alpha in alphas:
                    # Norm-corrected linear interpolation
                    h_interp = (1 - alpha) * h1 + alpha * h2
                    target_norm = (1 - alpha) * h1.norm() + alpha * h2.norm()
                    h_interp = h_interp * (target_norm / (h_interp.norm() + 1e-10))

                    try:
                        logits = apply_lm_head(model, h_interp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits = h_interp.unsqueeze(0) @ E.float().T

                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
                    entropies[f"{alpha:.1f}"].append(entropy)

                    # Slerp interpolation for comparison
                    h_slerp = _slerp(h1, h2, alpha)
                    # Restore magnitude via same norm target
                    h_slerp = h_slerp * (target_norm / (h_slerp.norm() + 1e-10))

                    try:
                        logits_s = apply_lm_head(model, h_slerp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits_s = h_slerp.unsqueeze(0) @ E.float().T

                    probs_s = F.softmax(logits_s, dim=-1)
                    entropy_s = -(probs_s * (probs_s + 1e-10).log()).sum(dim=-1).item()
                    slerp_entropies[f"{alpha:.1f}"].append(entropy_s)

                count += 1

        results = {}
        for alpha_key, vals in entropies.items():
            results[f"interp_entropy_{alpha_key}"] = float(np.mean(vals))
        for alpha_key, vals in slerp_entropies.items():
            results[f"slerp_entropy_{alpha_key}"] = float(np.mean(vals))

        mid = results.get("interp_entropy_0.5", 0)
        end = (results.get("interp_entropy_0.0", 0) + results.get("interp_entropy_1.0", 0)) / 2
        results["convexity_gap"] = mid - end

        mid_s = results.get("slerp_entropy_0.5", 0)
        end_s = (results.get("slerp_entropy_0.0", 0) + results.get("slerp_entropy_1.0", 0)) / 2
        results["slerp_convexity_gap"] = mid_s - end_s

        return results
