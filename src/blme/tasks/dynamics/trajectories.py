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
    # Scale-exact unit vectors, float32-first: the plain norm is used wherever it
    # is finite and nonzero; the max-factored path only where it under/overflows
    # (slerp is positively homogeneous, so a rescale must not move the angle).
    def _unit(v):
        n = v.norm(dim=-1, keepdim=True)
        if bool(torch.isfinite(n).all()) and float(n.min()) > 0.0:
            return v / n
        m = v.abs().amax(dim=-1, keepdim=True)
        if not bool(torch.isfinite(m).all()) or float(m.max()) == 0.0:
            return F.normalize(v, dim=-1)
        vs = v / m.clamp_min(torch.finfo(v.dtype).tiny)
        return vs / vs.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(vs.dtype).tiny)

    h1_norm = _unit(h1)
    h2_norm = _unit(h2)
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

        # The pair pool is the first `num_samples` passages, not the whole shared
        # corpus, which core.py sizes to the largest request of any co-run task.
        samples = list(dataset)[:num_samples] if num_samples else list(dataset)
        if len(samples) < 2: return {"error": "Need at least 2 samples"}

        entropies = defaultdict(list)
        slerp_entropies = defaultdict(list)
        base_alphas = np.linspace(0, 1, steps)
        alphas = np.array(sorted(set(np.concatenate([
            base_alphas,
            np.array([0.0, 0.5, 1.0]),
        ]))))

        # Seeded pair sampler; `seed` is honoured, default 0
        # (tests/tasks/test_round5_determinism.py).
        import random as _random
        _rng = _random.Random(int(self.config.get("seed", 0)))
        count = 0
        attempts = 0
        # Attempts are capped so unusable rows cannot spin the loop forever
        # (tests/tasks/test_dynamics.py::test_interpolation_does_not_hang_on_unusable_samples).
        max_attempts = max(20, num_pairs * 20)
        with torch.no_grad():
            while count < num_pairs and attempts < max_attempts:
                attempts += 1
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
                    # post-final-norm: required, because the interpolant is decoded by
                    # the LM head, which only ever sees post-norm vectors.
                    h_states.append(out.hidden_states[-1][0, -1].float())

                if len(h_states) < 2: continue

                h1, h2 = h_states

                for alpha in alphas:
                    alpha_key = _canonical_alpha(alpha)
                    # Norm-corrected linear interpolation
                    h_interp = (1 - alpha) * h1 + alpha * h2
                    target_norm = (1 - alpha) * h1.norm() + alpha * h2.norm()
                    # Rescale to the interpolated norm (no midpoint norm collapse);
                    # clamp_min(tiny) guards only the zero vector, unlike an absolute eps.
                    h_interp = h_interp * (target_norm / h_interp.norm().clamp_min(
                        torch.finfo(h_interp.dtype).tiny))

                    try:
                        logits = apply_lm_head(model, h_interp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits = h_interp.unsqueeze(0) @ E.float().T

                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
                    entropies[alpha_key].append(entropy)

                    # Slerp comparison arm, renormalised to the same target norm. At
                    # alpha in {0, 0.5, 1} it then equals the lerp arm exactly, so its
                    # summaries are _meta_ (tests/tasks/test_dynamics.py lerp==slerp).
                    h_slerp = _slerp(h1, h2, alpha)
                    h_slerp = h_slerp * (target_norm / h_slerp.norm().clamp_min(
                        torch.finfo(h_slerp.dtype).tiny))

                    try:
                        logits_s = apply_lm_head(model, h_slerp.unsqueeze(0))
                    except RuntimeError:
                        E = get_embeddings(model).to(device)
                        logits_s = h_slerp.unsqueeze(0) @ E.float().T

                    probs_s = F.softmax(logits_s, dim=-1)
                    entropy_s = -(probs_s * (probs_s + 1e-10).log()).sum(dim=-1).item()
                    slerp_entropies[alpha_key].append(entropy_s)

                count += 1

        if count == 0:
            return {"error": "No usable interpolation pairs (samples lack text/input_ids)"}

        results = {}
        # The whole per-alpha entropy profile is _meta_: its endpoints are the
        # model's own next-token entropy and every interior point tracks them.
        for alpha_key, vals in entropies.items():
            results[f"_meta_interp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))
        # The slerp arm is a comparison diagnostic, so all of it is _meta_.
        for alpha_key, vals in slerp_entropies.items():
            results[f"_meta_slerp_entropy_{_alpha_label(alpha_key)}"] = float(np.mean(vals))

        mid = float(np.mean(entropies[_canonical_alpha(0.5)]))
        end = (
            float(np.mean(entropies[_canonical_alpha(0.0)]))
            + float(np.mean(entropies[_canonical_alpha(1.0)]))
        ) / 2
        results["convexity_gap"] = mid - end
        # Disclosure: the gap contains the model's own endpoint entropy at
        # coefficient -1; the endpoint mean is emitted so an analysis can separate it.
        results["_meta_convexity_gap_endpoint_entropy_mean"] = end

        mid_s = float(np.mean(slerp_entropies[_canonical_alpha(0.5)]))
        end_s = (
            float(np.mean(slerp_entropies[_canonical_alpha(0.0)]))
            + float(np.mean(slerp_entropies[_canonical_alpha(1.0)]))
        ) / 2
        # _meta_: equal to convexity_gap by construction (see the slerp arm).
        results["_meta_slerp_convexity_gap"] = mid_s - end_s

        return results
