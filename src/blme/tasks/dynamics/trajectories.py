from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import logging
logger = logging.getLogger("blme")


def _canonical_alpha(alpha: float) -> float:
    """Round alpha to avoid string-key collisions in convexity lookups."""
    return round(float(alpha), 6)


def _alpha_label(alpha: float) -> str:
    """Human-readable alpha label for exported metric keys."""
    a = _canonical_alpha(alpha)
    if a == 0.0:
        return "0.0"
    if a == 1.0:
        return "1.0"
    if a == 0.5:
        return "0.5"
    return f"{a:.6g}"


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
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)
        if len(samples) < 2: return {"error": "Need at least 2 samples"}

        entropies = defaultdict(list)
        slerp_entropies = defaultdict(list)
        base_alphas = np.linspace(0, 1, steps)
        alphas = np.array(sorted(set(np.concatenate([
            base_alphas,
            np.array([0.0, 0.5, 1.0]),
        ]))))

        # Seeded RNG for reproducible pair sampling across runs.
        import random as _random
        _rng = _random.Random(0)
        count = 0
        with torch.no_grad():
            while count < num_pairs:
                s1, s2 = _rng.sample(samples, 2)

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
                    alpha_key = _canonical_alpha(alpha)
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
                    entropies[alpha_key].append(entropy)

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
                    slerp_entropies[alpha_key].append(entropy_s)

                count += 1

        results = {}
        for alpha_key, vals in entropies.items():
            results[f"interp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))
        for alpha_key, vals in slerp_entropies.items():
            results[f"slerp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))

        mid = float(np.mean(entropies[_canonical_alpha(0.5)]))
        end = (
            float(np.mean(entropies[_canonical_alpha(0.0)]))
            + float(np.mean(entropies[_canonical_alpha(1.0)]))
        ) / 2
        results["convexity_gap"] = mid - end

        mid_s = float(np.mean(slerp_entropies[_canonical_alpha(0.5)]))
        end_s = (
            float(np.mean(slerp_entropies[_canonical_alpha(0.0)]))
            + float(np.mean(slerp_entropies[_canonical_alpha(1.0)]))
        ) / 2
        results["slerp_convexity_gap"] = mid_s - end_s

        return results
