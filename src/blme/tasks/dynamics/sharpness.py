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

import contextlib
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._precision import effective_dtype_meta, fp32_compute
from ..common import get_layers, get_lm_head
from ..geometry.utils import safe_norm as _safe_norm

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


class _HessianOperator:
    """PyHessian's single-batch operator (hessian.py ``__init__`` and
    ``utils.hessian_vector_product``): the gradient graph is built once with
    ``create_graph=True`` and every HVP is one more backward through it."""

    def __init__(self, compute_loss, params):
        self.params = list(params)
        self.loss = compute_loss()
        grads = torch.autograd.grad(self.loss, self.params, create_graph=True)
        # A gradient that does not depend on the parameters has a zero HVP block.
        self._live = [i for i, g in enumerate(grads) if g.requires_grad]
        self.grads = grads

    def __call__(self, v: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self._live:
            return [torch.zeros_like(p) for p in self.params]
        outs = [self.grads[i] for i in self._live]
        gos = [v[i] for i in self._live]
        hv = torch.autograd.grad(outs, self.params, grad_outputs=gos,
                                 retain_graph=True, allow_unused=True)
        return [torch.zeros_like(p) if h is None else h.detach()
                for h, p in zip(hv, self.params)]


def _dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    return float(sum((x * y).sum() for x, y in zip(a, b)).item())


def _relative_change(prev: float, cur: float) -> float:
    """|cur - prev| / |prev|; PyHessian's denominator adds an absolute 1e-6,
    which makes the stopping rule depend on the Hessian's units (not reproduced)."""
    if cur == prev:
        return 0.0
    return abs(prev - cur) / abs(prev) if prev != 0.0 else float("inf")


def _power_iteration_top_eigenvalue(hvp, v, n_iter: int = 100, status=None,
                                    tol: float = 1e-3) -> float:
    """Largest-magnitude Hessian eigenvalue, signed, by PyHessian's
    ``eigenvalues(maxIter=n_iter, tol=tol, top_n=1)`` loop: ONE HVP per
    iteration, Rayleigh quotient <Hv, v> of the current unit vector, v <- Hv /
    ||Hv||, stop once the quotient changes by less than ``tol`` (relative) and
    return the last accepted quotient. ``tol=0`` runs the full budget.

    Two PyHessian details are not reproduced: its normalisation divides by
    ||v|| + 1e-6 and its stopping test by |lambda| + 1e-6; both are absolute
    epsilons inside scale-free operations (exact scale-safe norms are used).

    ``status`` receives the iterations used, the last relative change, whether
    the rule fired, and the residual ||Hv - lambda v|| / |lambda| of the
    returned pair. A residual below 1 certifies (Weinstein's bound) that the
    Hessian has an eigenvalue of the reported sign in [lambda - r, lambda + r].
    """
    vn = _safe_norm(_flatten_grads(v))
    if vn > 0:
        v = [vi / vn for vi in v]
    eigenvalue = None
    resid = float("nan")
    rel = float("nan")
    converged = False
    n_used = 0
    for _ in range(max(1, int(n_iter))):
        hv = hvp(v)
        n_used += 1
        tmp = _dot(hv, v)
        mag = _safe_norm(_flatten_grads(hv))
        r = _safe_norm(_flatten_grads([h - tmp * vi for h, vi in zip(hv, v)]))
        tmp_resid = r / abs(tmp) if tmp != 0.0 else float("inf")
        if mag <= 0:
            eigenvalue, resid, converged = 0.0, 0.0, True
            break
        v = [h / mag for h in hv]
        if eigenvalue is None:
            eigenvalue, resid = tmp, tmp_resid
            continue
        rel = _relative_change(eigenvalue, tmp)
        if rel < tol:
            converged = True
            break
        eigenvalue, resid = tmp, tmp_resid
    if status is not None:
        status.update(n_iter=n_used, rel_change=rel, converged=converged,
                      residual=resid)
    return float(eigenvalue)


def _hutchinson_trace(hvp, params, generator=None, n_iter: int = 100,
                      tol: float = 1e-3, status=None) -> List[float]:
    """Hutchinson trace samples v^T H v by PyHessian's ``trace(maxIter, tol)``
    loop: Rademacher probes, stop once the running mean changes by less than
    ``tol`` (relative; PyHessian's +1e-6 in the denominator is not reproduced).
    Returns the samples; their mean is the estimate."""
    samples: List[float] = []
    trace = 0.0
    converged = False
    for _ in range(max(1, int(n_iter))):
        v = _make_random_vec(params, rademacher=True, generator=generator)
        samples.append(_dot(hvp(v), v))
        mean = float(np.mean(samples))
        if _relative_change(trace, mean) < tol:
            converged = True
            break
        trace = mean
    if status is not None:
        status.update(n_iter=len(samples), converged=converged)
    return samples


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
    grad_norm = _safe_norm(grad_flat)

    if grad_norm > 0:
        # try/finally: a failing loss must not leave the model perturbed.
        with torch.no_grad():
            saved = [p.detach().clone() for p in params]
            try:
                for p, g in zip(params, grads):
                    p.add_(g.detach() * (sam_rho / grad_norm))
                perturbed_loss = float(compute_loss().item())
            finally:
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

    # eta0 is raised until theta - eta*g is representable in the parameters'
    # dtype (in bf16, 1e-6 * g rounds away and every probe returns the baseline).
    try:
        _eps = float(torch.finfo(params[0].dtype).eps)
        _pnorm = _safe_norm(torch.cat([p_.detach().float().reshape(-1)
                                       for p_ in params]))
        # Exact guards on physical magnitudes, no absolute floor.
        if not (grad_norm > 0.0):
            return float("nan"), float("nan")
        _min_eta = 8.0 * _eps * _pnorm / grad_norm
        if np.isfinite(_min_eta) and _min_eta > eta0:
            eta0 = float(_min_eta)
    except Exception:
        pass

    saved = [p.detach().clone() for p in params]

    def probe(eta: float) -> float:
        # try/finally: a failing loss must not leave the model stepped.
        with torch.no_grad():
            try:
                for p_, g_ in zip(params, gvecs):
                    p_.add_(g_, alpha=-eta)
                val = float(compute_loss().item())
            finally:
                for p_, s_ in zip(params, saved):
                    p_.copy_(s_)
        return val

    # A step is unstable when the loss strictly INCREASES (Kalra Def. 2.1).
    # Doubling phase: find the first unstable eta; if eta0 is already unstable,
    # halve downward first so very sharp scopes still resolve eta*.
    eta = eta0
    if probe(eta) > baseline_loss:
        hi0 = eta
        for _ in range(max_doublings):
            eta /= 2.0
            if eta < 1e-30:
                return float("nan"), float("nan")
            # Equality is stable, mirroring the upward branch's `> baseline`.
            if probe(eta) <= baseline_loss:
                lo, hi = eta, eta * 2.0
                for _ in range(bisect_iters):
                    mid = 0.5 * (lo + hi)
                    if probe(mid) > baseline_loss:
                        hi = mid
                    else:
                        lo = mid
                # Bracket midpoint, as the upward branch returns.
                eta_star = 0.5 * (lo + hi)
                return float(eta_star), float(2.0 / eta_star)
        return float("nan"), float("nan")
    lo = eta
    hi = None
    for _ in range(max_doublings):
        eta *= 2.0
        if probe(eta) > baseline_loss:
            hi = eta
            break
        lo = eta
    if hi is None:
        # eta* is beyond the search range: NaN, not the cap as a sentinel.
        return float("nan"), float("nan")

    # Bisection between last stable (lo) and first unstable (hi).
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if probe(mid) > baseline_loss:
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
        # PyHessian's defaults: at most 100 HVPs per estimator, stopping once the
        # estimate changes by less than tol = 1e-3 (relative).
        n_hutchinson = int(self.config.get("n_hutchinson", 100))
        n_power_iter = int(self.config.get("n_power_iter", 100))
        hutchinson_tol = float(self.config.get("hutchinson_tol", 1e-3))
        power_iter_tol = float(self.config.get("power_iter_tol", 1e-3))
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

        # The try opens before the first mutation of the caller's model (eval
        # mode, requires_grad, dtype), so every exit path restores all three.
        was_training = model.training
        orig_grad_state = {}
        _stack = contextlib.ExitStack()
        try:
            model.eval()
            orig_grad_state = {p: p.requires_grad for p in model.parameters()}
            for p in model.parameters():
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)

            # Every probe runs in fp32 arithmetic on a bf16/fp16 model, parameters
            # AND buffers: bf16 arithmetic biases curvature (~21% on
            # tiny-random-gpt2's top eigenvalue). Restored on every exit path.
            _prec = _stack.enter_context(fp32_compute(model))
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

            # PyHessian's single-batch operator: one gradient graph, reused by
            # every HVP of both estimators, then released.
            H = _HessianOperator(compute_loss, params)
            baseline_loss = float(H.loss.item())

            # 1. Hutchinson trace, PyHessian trace(maxIter=n_hutchinson,
            # tol=hutchinson_tol), seeded on the params' device.
            gen = torch.Generator(device=params[0].device)
            gen.manual_seed(seed)
            _tr_status = {}
            trace_estimates = _hutchinson_trace(H, params, generator=gen,
                                                n_iter=n_hutchinson, tol=hutchinson_tol,
                                                status=_tr_status)
            mean_trace = float(np.mean(trace_estimates))
            trace_per_param = mean_trace / max(1, n_params)
            # Monte Carlo uncertainty of the trace, reported as a relative SE.
            _tr_sd = float(np.std(trace_estimates, ddof=1)) if len(trace_estimates) > 1 else float("nan")
            _tr_se = _tr_sd / np.sqrt(len(trace_estimates)) if len(trace_estimates) > 1 else float("nan")
            _tr_rel_se = (abs(_tr_se / mean_trace) if mean_trace not in (0.0,)
                          and np.isfinite(_tr_se) else float("nan"))

            # 2. Signed top eigenvalue, PyHessian eigenvalues(maxIter=n_power_iter,
            # tol=power_iter_tol, top_n=1), from its own seeded start vector so
            # it does not depend on how many trace probes were drawn.
            gen_pi = torch.Generator(device=params[0].device)
            gen_pi.manual_seed(seed + 1)
            v = _make_random_vec(params, rademacher=False, generator=gen_pi)
            _pi_status = {}
            top_eig = _power_iteration_top_eigenvalue(
                H, v, n_power_iter, status=_pi_status, tol=power_iter_tol)
            del H

            # 3. SAM-style sharpness (module-level helper so parity tests
            # drive the exact code evaluate() runs, not a transcription).
            sam_sharpness, perturbed_loss, grad_norm = _sam_sharpness(
                compute_loss, params, sam_rho, baseline_loss
            )

            # 4. Critical sharpness: largest stable descent step along the
            # gradient; lambda_c = 2/eta* (edge of stability).
            loss_for_cs = compute_loss()
            grads_cs = torch.autograd.grad(loss_for_cs, params,
                                           create_graph=False)
            gnorm_cs = _safe_norm(_flatten_grads(list(grads_cs)))
            critical_step, critical_sharpness = _critical_step_size(
                compute_loss, params, list(grads_cs), gnorm_cs, baseline_loss
            )

        finally:
            _stack.close()
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
            "_meta_hutchinson_trace_rel_se": _tr_rel_se,
            "_meta_hutchinson_n_used": int(_tr_status.get("n_iter", 0)),
            "_meta_hutchinson_converged": bool(_tr_status.get("converged", False)),
            "_meta_power_iter_rel_change": float(_pi_status.get("rel_change", float("nan"))),
            "_meta_power_iter_converged": bool(_pi_status.get("converged", False)),
            "_meta_power_iter_n_used": int(_pi_status.get("n_iter", 0)),
            # Sign check at no extra HVP: the residual of the returned pair. Below
            # 1, an eigenvalue of the reported sign provably exists within
            # |lambda| * residual of it; a near-degenerate +/- pair caught
            # mid-rotation (pythia-70m: -104737 at 10 iterations, +111530 at 64)
            # shows a residual of order 1.
            "_meta_top_eigenvalue_residual": float(_pi_status.get("residual", float("nan"))),
            "_meta_top_eigenvalue_sign_certified": bool(
                np.isfinite(_pi_status.get("residual", float("nan")))
                and _pi_status.get("residual") < 1.0),
            "hutchinson_trace_per_param": trace_per_param,
            "top_eigenvalue_estimate": top_eig,
            "sam_sharpness": sam_sharpness,
            "sam_perturbed_loss": perturbed_loss,
            "critical_step_size": critical_step,
            "critical_sharpness": critical_sharpness,
            "sam_rho": sam_rho,
            "n_hutchinson_samples": int(_tr_status.get("n_iter", 0)),
            "n_power_iterations": int(_pi_status.get("n_iter", 0)),
            "_meta_hutchinson_max_iter": n_hutchinson,
            "_meta_hutchinson_tol": hutchinson_tol,
            "_meta_power_iter_max_iter": n_power_iter,
            "_meta_power_iter_tol": power_iter_tol,
            **effective_dtype_meta(_prec),
        }
