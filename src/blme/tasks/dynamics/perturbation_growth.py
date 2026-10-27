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

  * ``growth_log_ratio`` (headline): log(d_L / d_0), the end-to-end
    amplification. Positive = perturbations amplify; negative = contract.
  * ``perturbation_growth_per_layer``: log(d_l / d_0) profile.
  * ``_meta_growth_slope``: least-squares slope of log d_l against
    normalised depth in [0, 1].
  * ``_meta_growth_rate_per_block``: growth_log_ratio / L.
  * ``_meta_initial_divergence``: d_0, set by rel_eps and the embedding scale.

Analytic anchor: for a purely linear layer stack h_{l+1} = a * h_l the
divergence grows exactly by |a| per layer, so growth_log_ratio = (L-1) * ln|a|
and the slope over normalised depth equals it.
"""

import logging

import numpy as np
import torch

from ...registry import register_task
from .._precision import effective_dtype_meta, fp32_compute
from ..geometry.utils import block_output_count, safe_unit_rows_torch
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _growth_profile(clean_layers, perturbed_layers) -> dict:
    """Divergence profile from two lists of per-layer (T, D) hidden states."""
    L = min(len(clean_layers), len(perturbed_layers))
    if L < 2:
        return {}
    # Both operands in double before the subtraction: a float32 norm under- and
    # overflows at extreme scales, and casting before subtracting quantises a
    # common offset away (tests/tasks/test_perturbation_growth.py scale/offset).
    d = []
    for l in range(L):
        diff = (clean_layers[l].double() - perturbed_layers[l].double())
        d.append(float(diff.norm(dim=-1).mean().item()))
    d = np.asarray(d, dtype=np.float64)
    if d[0] <= 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
        return {}
    log_d = np.log(d)
    x = np.arange(L, dtype=np.float64) / max(1, L - 1)
    slope = float(np.polyfit(x, log_d, 1)[0])
    n_transitions = max(1, L - 1)
    # Only the end-to-end log ratio survives function-preserving identity-block
    # insertion; the per-block rate is it times 1/L, the slope moves with where
    # blocks are inserted, and the initial divergence echoes rel_eps
    # (tests/tasks/invariance/test_dynamics_perturbation_growth.py).
    return {
        "perturbation_growth_per_layer": (log_d - log_d[0]).tolist(),
        "growth_log_ratio": float(log_d[-1] - log_d[0]),
        "_meta_growth_slope": slope,
        "_meta_growth_rate_per_block": float((log_d[-1] - log_d[0]) / n_transitions),
        "_meta_initial_divergence": float(d[0]),
        "_meta_n_transitions": int(n_transitions),
    }


@register_task("dynamics_perturbation_growth")
class PerturbationGrowthTask(DiagnosticTask):
    """Seeded input-embedding perturbation; per-layer divergence growth."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Layer Perturbation Growth (quasi-Lyapunov)...")
        num_samples = self.config.get("num_samples", 10)
        # 1e-2: the smallest relative displacement fp32, fp16 and bf16 storage
        # all realise exactly (bf16 realises only 82% of a 1e-3 request); the
        # realised size is measured and emitted as _meta_realized_rel_eps.
        rel_eps = float(self.config.get("rel_eps", 1e-2))
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
        _realized = []
        from ..common import get_layers as _gl
        _blocks = _gl(model)
        _sink, _used_hook, _hook = {}, [False], None
        if _blocks is not None and len(_blocks) > 0:
            def _cap_last(module, inputs, output):
                _sink["h"] = (output[0] if isinstance(output, tuple)
                              else output).detach()
            _hook = _blocks[-1].register_forward_hook(_cap_last)
        n_used = 0
        # The hook is removed on every path, exceptions included
        # (tests/tasks/test_forward_hook_leaks.py). fp32 arithmetic on a
        # bf16/fp16 model: bf16 rounding inside the forward pass is a noise floor
        # comparable to the perturbation (tests/tasks/test_perturbation_growth.py).
        try:
            with torch.no_grad(), fp32_compute(model) as _prec:
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
                    # Per-token scale rel_eps * ||e_t||, with the row max factored
                    # out and the product taken in double, so it neither under-
                    # nor overflows (tests/tasks/test_perturbation_growth.py).
                    _E64 = E.double()
                    _rmax = _E64.abs().amax(dim=-1, keepdim=True)
                    _tiny = torch.finfo(_E64.dtype).tiny
                    scale = ((_rmax * (_E64 / _rmax.clamp_min(_tiny)).norm(
                        dim=-1, keepdim=True)) * rel_eps).to(E.dtype)
                    noise = safe_unit_rows_torch(noise)
                    E_pert = E + noise * scale
                    _realized.append(float(
                        ((E_pert - E).float().norm(dim=-1)
                         / E.float().norm(dim=-1).clamp_min(_tiny)).mean().item()))

                    kw = {}
                    if "attention_mask" in enc:
                        kw["attention_mask"] = enc["attention_mask"]
                    out_c = model(inputs_embeds=E, output_hidden_states=True, **kw)
                    _sink_c = dict(_sink)
                    out_p = model(inputs_embeds=E_pert, output_hidden_states=True, **kw)
                    _sink_p = dict(_sink)

                    # HF's last hidden_states entry is post-final-norm; the
                    # deepest block's raw output (hooked) replaces it so the
                    # profile ends at a block output. States keep their dtype
                    # here: narrowing before the subtraction loses the signal.
                    _hc = list(out_c.hidden_states)
                    _hp = list(out_p.hidden_states)
                    if (_sink_c.get("h") is not None
                            and tuple(_sink_c["h"].shape) == tuple(_hc[-1].shape)):
                        _hc[-1] = _sink_c["h"]
                        _hp[-1] = _sink_p["h"]
                        _used_hook[0] = True
                        hs_c = [h[0].cpu() for h in _hc]
                        hs_p = [h[0].cpu() for h in _hp]
                    else:
                        _used_hook[0] = False
                        _keep = block_output_count(model)
                        hs_c = [h[0].cpu() for h in _hc][:_keep]
                        hs_p = [h[0].cpu() for h in _hp][:_keep]
                    if clean_acc is None:
                        clean_acc = [[] for _ in hs_c]
                        pert_acc = [[] for _ in hs_p]
                    for li in range(min(len(hs_c), len(clean_acc))):
                        clean_acc[li].append(hs_c[li])
                        pert_acc[li].append(hs_p[li])
                    n_used += 1

        finally:
            if _hook is not None:
                _hook.remove()
                _hook = None

        if not n_used:
            return {"error": "No usable samples for perturbation growth"}

        clean_layers = [torch.cat(c, dim=0) for c in clean_acc]
        pert_layers = [torch.cat(c, dim=0) for c in pert_acc]
        results = _growth_profile(clean_layers, pert_layers)
        if not results:
            return {"error": "Degenerate divergence profile"}
        results["rel_eps"] = rel_eps
        _rz = float(np.mean(_realized)) if _realized else float("nan")
        results["_meta_realized_rel_eps"] = _rz
        if np.isfinite(_rz) and rel_eps > 0 and abs(_rz - rel_eps) / rel_eps > 0.05:
            logger.warning(
                f"  perturbation_growth: requested rel_eps={rel_eps:g} but the "
                f"model realized {_rz:.4g} ({_rz / rel_eps * 100:.1f}%) -- the "
                f"perturbation is below this dtype's resolution "
                f"({torch.finfo(next(model.parameters()).dtype).eps:.3g}). "
                f"Growth is not comparable against a model in a wider dtype.")
        results["n_samples_used"] = n_used
        results["_meta_last_entry_is_block_output"] = bool(_used_hook[0])
        results.update(effective_dtype_meta(_prec))
        return results
