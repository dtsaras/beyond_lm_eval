"""
Loss-landscape sharpness metrics:

  - **Hutchinson trace estimate** of the Hessian (Hutchinson 1990;
    Bai, Fahey & Golub 1996; Yao et al. 2020 PyHessian) — average
    curvature of the loss surface.
  - **Top-1 Hessian eigenvalue** via power iteration (Yao et al. 2020) —
    the dominant direction of curvature.
  - **SAM-style sharpness** — Foret et al. 2021, arXiv:2010.01412 —
    L(theta + rho * g/||g||) - L(theta), measured at the gradient
    direction. Closely related to the operator-norm sharpness used in
    sharpness-aware minimisation.

For large LLMs the full-parameter Hessian is too big to handle, so by
default we restrict the curvature computation to the **final transformer
layer's parameters**. This is the most prediction-relevant subset and
keeps memory bounded for models up to ~30B. The user can override the
parameter scope via the `param_scope` config field:

  - "last_layer" (default): final transformer block
  - "lm_head": final unembedding projection
  - "all": all parameters (use only on small models)

References:
  - Foret, Kleiner, Mobahi, Neyshabur, "Sharpness-Aware Minimization for
    Efficiently Improving Generalization", ICLR 2021. arXiv:2010.01412.
  - Yao, Gholami, Keutzer, Mahoney, "PyHessian: Neural Networks Through
    the Lens of the Hessian", IEEE Big Data 2020. arXiv:1912.07145.
  - Hutchinson, "A stochastic estimator of the trace of the influence
    matrix for Laplacian smoothing splines", Commun. Stat. Simul.
    Comput. 19(2), 1990. Bai, Fahey & Golub, "Some large-scale matrix
    computation problems", J. Comput. Appl. Math. 74, 1996.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers, get_lm_head

logger = logging.getLogger("blme")


def _select_parameters(model, scope: str) -> List[torch.nn.Parameter]:
    """Select the parameter subset over which to compute curvature.

    Returns all parameters of the target module unconditionally (ignoring
    the current ``requires_grad`` state, which may have been cleared by a
    previous task). The caller is responsible for enabling/disabling grad.
    """
    if scope == "all":
        return [p for p in model.parameters()]
    if scope == "lm_head":
        head = get_lm_head(model)
        if head is None:
            return []
        return [p for p in head.parameters()]
    # Default: last transformer block.
    layers = get_layers(model)
    if layers is None or len(layers) == 0:
        return []
    return [p for p in layers[-1].parameters()]


def _flatten_grads(grads: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _hvp(loss, params, v_list):
    """Hessian-vector product via double backward.

    Args:
        loss: scalar loss tensor with grad_fn
        params: list of nn.Parameter with requires_grad=True
        v_list: list of tensors with the same shapes as `params`
    Returns:
        list of tensors with the same shapes as `params`, equal to H @ v
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_g = _flatten_grads(list(grads))
    flat_v = _flatten_grads(v_list)
    gv = (flat_g * flat_v).sum()
    hv = torch.autograd.grad(gv, params, retain_graph=True)
    return [h.detach() for h in hv]


def _power_iteration_top_eigenvalue(hvp, v, n_iter: int) -> float:
    """Largest-magnitude Hessian eigenvalue via power iteration, returned WITH
    its sign — the Rayleigh quotient ``v^T H v`` at the converged eigenvector,
    matching PyHessian's ``eigenvalues(top_n=1)``. Reporting ``||Hv||`` alone
    (as the old code did) drops the sign and misreports a dominant negative
    curvature (a saddle) as positive.

    ``hvp(v)`` returns ``H @ v`` as a list of tensors shaped like ``v``.
    """
    flat = _flatten_grads(v)
    vn = float(flat.norm().item())
    if vn > 0:
        v = [vi / vn for vi in v]
    for _ in range(max(1, n_iter)):
        hv = hvp(v)
        mag = float(_flatten_grads(hv).norm().item())
        if mag <= 0:
            return 0.0
        v = [h / mag for h in hv]
    hv = hvp(v)
    return float(sum((h * vi).sum() for h, vi in zip(hv, v)).item())


def _sam_sharpness(compute_loss, params, sam_rho: float,
                   baseline_loss: float) -> Tuple[float, float, float]:
    """SAM-style sharpness (Foret et al. 2021, Eq. 2 with p=2):
    ``L(theta + rho * g/||g||_2) - L(theta)`` along the normalized
    full-batch gradient direction. Perturbs the selected parameters in
    place, measures the perturbed loss, and restores them exactly.

    Returns ``(sam_sharpness, perturbed_loss, grad_norm)``; NaNs when the
    gradient is exactly zero (no ascent direction).
    """
    loss_for_grad = compute_loss()
    grads = torch.autograd.grad(loss_for_grad, params, create_graph=False)
    grad_flat = _flatten_grads(list(grads))
    grad_norm = float(grad_flat.norm().item())

    if grad_norm > 0:
        with torch.no_grad():
            saved = [p.detach().clone() for p in params]
            for p, g in zip(params, grads):
                p.add_(g.detach() * (sam_rho / grad_norm))
            perturbed_loss = float(compute_loss().item())
            # Restore
            for p, s in zip(params, saved):
                p.copy_(s)
        return perturbed_loss - baseline_loss, perturbed_loss, grad_norm
    return float("nan"), float("nan"), grad_norm


def _critical_step_size(compute_loss, params, grads, grad_norm: float,
                        baseline_loss: float, max_doublings: int = 30,
                        bisect_iters: int = 6,
                        eta0: float = 1e-6) -> Tuple[float, float]:
    """Critical sharpness (campaign-4 B8, after Kalra et al.,
    arXiv:2601.16979): the largest stable step size eta* of the raw
    gradient-descent update — loss(theta - eta*g) decreases for
    eta < eta* and increases beyond it. Reported with
    lambda_c = 2/eta*, the edge-of-stability curvature scale (exact for a
    quadratic: eta* = 2/lambda). Kalra et al. probe the optimizer's update
    direction; this is the gradient-direction variant (documented).

    Cost: <= max_doublings + bisect_iters forward passes. Parameters are
    perturbed in place and restored exactly after every probe. The probe
    step is the RAW gradient-descent update theta - eta*g (eta in
    learning-rate units): for a quadratic with curvature lambda this gives
    theta' = (1 - eta*lambda)*theta, so eta* = 2/lambda EXACTLY and the
    identity lambda_c = 2/eta* recovers the curvature. (A normalized-
    direction probe would instead measure 2*distance-to-reflection, not
    curvature.)

    Returns (eta*, lambda_c); (nan, nan) when no stable step exists at
    eta0 or the gradient is zero.
    """
    if grad_norm <= 0:
        return float("nan"), float("nan")
    gvecs = [g.detach() for g in grads]

    saved = [p.detach().clone() for p in params]

    def probe(eta: float) -> float:
        with torch.no_grad():
            for p_, g_ in zip(params, gvecs):
                p_.add_(g_, alpha=-eta)
            val = float(compute_loss().item())
            for p_, s_ in zip(params, saved):
                p_.copy_(s_)
        return val

    # Doubling phase: find the first unstable eta.
    eta = eta0
    if probe(eta) >= baseline_loss:
        return float("nan"), float("nan")   # unstable already at eta0
    lo = eta
    hi = None
    for _ in range(max_doublings):
        eta *= 2.0
        if probe(eta) >= baseline_loss:
            hi = eta
            break
        lo = eta
    if hi is None:
        # Never became unstable within the search range; report the last
        # stable step as a lower bound.
        return float(lo), float(2.0 / lo)

    # Bisection between last stable (lo) and first unstable (hi).
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if probe(mid) >= baseline_loss:
            hi = mid
        else:
            lo = mid
    eta_star = 0.5 * (lo + hi)
    return float(eta_star), float(2.0 / eta_star)


def _make_random_vec(params, rademacher: bool = True,
                     generator: "torch.Generator | None" = None,
                     ) -> List[torch.Tensor]:
    """Random probe vectors shaped like ``params``. Pass a seeded
    ``generator`` (on the params' device) for reproducible Hutchinson /
    power-iteration estimates — unseeded draws made the trace and
    top-eigenvalue estimates vary run to run."""
    out = []
    for p in params:
        if rademacher:
            v = torch.randint(0, 2, p.shape, device=p.device,
                              generator=generator) * 2 - 1
            v = v.to(p.dtype)
        else:
            v = torch.randn(p.shape, device=p.device, dtype=p.dtype,
                            generator=generator)
        out.append(v)
    return out


@register_task("dynamics_sharpness")
class LossSharpnessTask(DiagnosticTask):
    """Loss-landscape sharpness via Hessian + SAM metrics."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Loss-Landscape Sharpness Analysis...")

        param_scope = self.config.get("param_scope", "last_layer")
        n_hutchinson = self.config.get("n_hutchinson", 5)
        n_power_iter = self.config.get("n_power_iter", 8)
        sam_rho = self.config.get("sam_rho", 0.05)
        num_samples = self.config.get("num_samples", 4)
        seed = int(self.config.get("seed", 42))

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog. " * 4},
                {"text": "Machine learning models are trained on large corpora. " * 4},
            ]

        device = next(model.parameters()).device
        params = _select_parameters(model, param_scope)
        if not params:
            return {"error": f"No parameters found for scope='{param_scope}'"}
        n_params = sum(p.numel() for p in params)
        logger.info(f"  Sharpness scope='{param_scope}', n_params={n_params}")

        # SDPA / Flash Attention don't have second-order derivatives, which
        # breaks Hessian-vector products. Detect early and bail out cleanly.
        attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if attn_impl in ("sdpa", "flash_attention_2"):
            return {
                "error": (
                    f"Model loaded with attn_implementation='{attn_impl}', which does "
                    "not support double backward. Reload with attn_implementation='eager' "
                    "to enable Hessian-based sharpness analysis."
                )
            }

        # Build a single batched loss using `num_samples` short sequences.
        # We accumulate per-sample losses, then take the mean for HVP.
        encs = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample["text"] if isinstance(sample, dict) else str(sample)
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=64).to(device)
            if enc["input_ids"].shape[1] >= 4:
                encs.append(enc)
        if not encs:
            return {"error": "No valid encodings"}

        # The forward pass needs requires_grad=True; ensure model is in
        # eval mode but parameters can compute gradients.
        was_training = model.training
        model.eval()
        # Save original requires_grad state so we can restore it — other
        # tasks or user code may depend on the model's grad configuration.
        orig_grad_state = {p: p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        try:
            # Build a fresh computational graph and compute the per-batch loss.
            def compute_loss():
                losses = []
                for enc in encs:
                    out = model(**enc)
                    logits = out.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = enc["input_ids"][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    losses.append(loss)
                return torch.stack(losses).mean()

            loss = compute_loss()
            baseline_loss = float(loss.item())

            # 1. Hutchinson trace estimator
            # Seeded generator on the params' device: Hutchinson and
            # power-iteration probes are otherwise irreproducible run to run.
            gen = torch.Generator(device=params[0].device)
            gen.manual_seed(seed)
            trace_estimates = []
            for _ in range(n_hutchinson):
                v = _make_random_vec(params, rademacher=True, generator=gen)
                # Build a fresh graph for each HVP because retain_graph
                # over many calls leaks memory.
                loss_fresh = compute_loss()
                hv = _hvp(loss_fresh, params, v)
                vhv = sum((h * vi).sum() for h, vi in zip(hv, v)).item()
                trace_estimates.append(vhv)
            mean_trace = float(np.mean(trace_estimates))
            trace_per_param = mean_trace / max(1, n_params)

            # 2. Top-1 eigenvalue via power iteration. Report the SIGNED
            # largest-magnitude eigenvalue (Rayleigh quotient at the converged
            # eigenvector), like PyHessian — ||Hv|| alone drops the sign.
            v = _make_random_vec(params, rademacher=False, generator=gen)
            # Fresh graph per HVP (retain_graph over many calls leaks memory).
            top_eig = _power_iteration_top_eigenvalue(
                lambda vv: _hvp(compute_loss(), params, vv), v, n_power_iter
            )

            # 3. SAM-style sharpness (module-level helper so parity tests
            # drive the exact code evaluate() runs, not a transcription).
            sam_sharpness, perturbed_loss, grad_norm = _sam_sharpness(
                compute_loss, params, sam_rho, baseline_loss
            )

            # 4. Critical sharpness (B8): largest stable descent step along
            # the normalized gradient; lambda_c = 2/eta* (edge of stability).
            loss_for_cs = compute_loss()
            grads_cs = torch.autograd.grad(loss_for_cs, params,
                                           create_graph=False)
            gnorm_cs = float(_flatten_grads(list(grads_cs)).norm().item())
            critical_step, critical_sharpness = _critical_step_size(
                compute_loss, params, list(grads_cs), gnorm_cs, baseline_loss
            )

        finally:
            # Restore original requires_grad state and training mode.
            for p, grad_flag in orig_grad_state.items():
                p.requires_grad_(grad_flag)
            if was_training:
                model.train()

        return {
            "param_scope": param_scope,
            "n_params": int(n_params),
            "baseline_loss": baseline_loss,
            "hutchinson_trace_estimate": mean_trace,
            "hutchinson_trace_per_param": trace_per_param,
            "top_eigenvalue_estimate": top_eig,
            "sam_sharpness": sam_sharpness,
            "sam_perturbed_loss": perturbed_loss,
            "critical_step_size": critical_step,
            "critical_sharpness": critical_sharpness,
            "sam_rho": sam_rho,
            "n_hutchinson_samples": n_hutchinson,
            "n_power_iterations": n_power_iter,
        }
