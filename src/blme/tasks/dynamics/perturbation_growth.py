"""Layer perturbation growth — depth-wise sensitivity (chaoticity) profile.

Grounding:
  * Li et al. 2025 — "Cognitive Activation and Chaotic Dynamics in Large
    Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms",
    arXiv:2503.13530. Defines a quasi-Lyapunov exponent: the growth rate
    of hidden-state divergence across layers under small perturbations.
  * Fernando & Guitchounts 2025/2026 — residual-stream dynamical-systems
    line (arXiv:2502.12131, 2605.14258): layer-Jacobian spectra shape how
    perturbations amplify or contract through depth.

BLME's clean-room implementation (documented BLME formulation, cited
grounding above): perturb the input embeddings with a SEEDED Gaussian of
relative size ``rel_eps`` (scaled per token to rel_eps * ||h_token||), run
the clean and perturbed forward passes, and measure the per-layer
divergence d_l = E_t ||h_l - h'_l||. Reported:

  * ``perturbation_growth_per_layer``: log(d_l / d_0) profile.
  * ``growth_slope``: least-squares slope of log d_l against NORMALIZED
    depth in [0, 1] — a quasi-Lyapunov exponent per unit depth. Positive =
    perturbations amplify (chaotic regime); negative = contract.
  * ``growth_log_ratio``: log(d_L / d_0).

Analytic anchor: for a purely linear layer stack h_{l+1} = a * h_l the
divergence grows exactly by |a| per layer, so growth_slope = (L-1) * ln|a| /
(normalized-depth span) and growth_log_ratio = (L-1) * ln|a|.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _growth_profile(clean_layers, perturbed_layers) -> dict:
    """Divergence profile from two lists of per-layer (T, D) hidden states."""
    L = min(len(clean_layers), len(perturbed_layers))
    if L < 2:
        return {}
    d = []
    for l in range(L):
        diff = (clean_layers[l].float() - perturbed_layers[l].float())
        d.append(float(diff.norm(dim=-1).mean().item()))
    d = np.asarray(d, dtype=np.float64)
    if d[0] <= 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        return {}
    log_d = np.log(d)
    x = np.arange(L, dtype=np.float64) / max(1, L - 1)
    slope = float(np.polyfit(x, log_d, 1)[0])
    return {
        "perturbation_growth_per_layer": (log_d - log_d[0]).tolist(),
        "growth_slope": slope,
        "growth_log_ratio": float(log_d[-1] - log_d[0]),
        "initial_divergence": float(d[0]),
    }


@register_task("dynamics_perturbation_growth")
class PerturbationGrowthTask(DiagnosticTask):
    """Seeded input-embedding perturbation; per-layer divergence growth."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Layer Perturbation Growth (quasi-Lyapunov)...")
        num_samples = self.config.get("num_samples", 10)
        rel_eps = float(self.config.get("rel_eps", 1e-3))
        seed = int(self.config.get("seed", 42))

        if dataset is None:
            dataset = [{"text": f"Perturbation growth probe sentence {i}."}
                       for i in range(num_samples)]

        device = next(model.parameters()).device
        try:
            emb_layer = model.get_input_embeddings()
        except Exception:
            return {"error": "Model does not expose input embeddings"}
        if emb_layer is None:
            return {"error": "Model does not expose input embeddings"}

        gen = torch.Generator(device="cpu").manual_seed(seed)
        clean_acc, pert_acc = None, None
        n_used = 0
        with torch.no_grad():
            for i, s in enumerate(dataset):
                if i >= num_samples:
                    break
                text = s["text"] if isinstance(s, dict) else str(s)
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128).to(device)
                if enc["input_ids"].shape[1] < 2:
                    continue
                E = emb_layer(enc["input_ids"])  # (1, T, D)
                noise = torch.randn(E.shape, generator=gen).to(
                    device=E.device, dtype=E.dtype)
                # Per-token relative scaling so every token is perturbed by
                # rel_eps of its own norm.
                scale = E.norm(dim=-1, keepdim=True) * rel_eps
                noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                E_pert = E + noise * scale

                kw = {}
                if "attention_mask" in enc:
                    kw["attention_mask"] = enc["attention_mask"]
                out_c = model(inputs_embeds=E, output_hidden_states=True, **kw)
                out_p = model(inputs_embeds=E_pert, output_hidden_states=True, **kw)

                hs_c = [h[0].float().cpu() for h in out_c.hidden_states]
                hs_p = [h[0].float().cpu() for h in out_p.hidden_states]
                if clean_acc is None:
                    clean_acc = [[] for _ in hs_c]
                    pert_acc = [[] for _ in hs_p]
                for li in range(min(len(hs_c), len(clean_acc))):
                    clean_acc[li].append(hs_c[li])
                    pert_acc[li].append(hs_p[li])
                n_used += 1

        if not n_used:
            return {"error": "No usable samples for perturbation growth"}

        clean_layers = [torch.cat(c, dim=0) for c in clean_acc]
        pert_layers = [torch.cat(c, dim=0) for c in pert_acc]
        results = _growth_profile(clean_layers, pert_layers)
        if not results:
            return {"error": "Degenerate divergence profile"}
        results["rel_eps"] = rel_eps
        results["n_samples_used"] = n_used
        return results
