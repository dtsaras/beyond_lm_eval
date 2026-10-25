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

import math
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
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


def _power_iteration_top_eigenvalue(hvp, v, n_iter: int, status=None) -> float:
    """Largest-magnitude Hessian eigenvalue via power iteration, returned WITH
    its sign — the Rayleigh quotient ``v^T H v`` at the converged eigenvector,
    matching PyHessian's ``eigenvalues(top_n=1)``. Reporting ``||Hv||`` alone
    (as the old code did) drops the sign and misreports a dominant negative
    curvature (a saddle) as positive.

    ``hvp(v)`` returns ``H @ v`` as a list of tensors shaped like ``v``.
    """
    flat = _flatten_grads(v)
    # Scale-exact throughout: `.norm()` squares first. On the end-to-end
    # cross-entropy anchor every Hessian entry was the representable
    # 1.0000000787060493e-23 with a true top eigenvalue of
    # 2.0000001574120985e-23, yet the HVP norm underflowed to 0.0 and the
    # headline collapsed (adversarial review round 3, 2026-09-04).
    vn = _safe_norm(flat)
    if vn > 0:
        v = [vi / vn for vi in v]
    # CONVERGENCE IS MEASURED AND REPORTED, never used to stop early.
    #
    # The shipped 8 iterations returned 2.019681747104697 against an exact
    # 2.056211316963325 on the analytic MLP (-1.78%), so the count was raised
    # to 64 (adversarial review round 2, 2026-09-04). It is NOT an early-exit
    # criterion: a start vector that lies near a NON-dominant eigenvector makes
    # the Rayleigh quotient look stationary while the dominant component is
    # still growing -- diag(2, 1) started at [1e-6, 1] sits at lambda = 1 for
    # several iterations before turning toward the true 2. Any local test is
    # fooled there, so the loop always runs its full budget and the last
    # relative change is emitted as a diagnostic instead.
    _tol = 1e-7
    _prev = None
    _iters = 0
    _rel = float("nan")
    for _ in range(max(1, n_iter)):
        hv = hvp(v)
        mag = _safe_norm(_flatten_grads(hv))
        if mag <= 0:
            if status is not None:
                status.update(n_iter=_iters, rel_change=0.0, converged=True)
            return 0.0
        v = [h / mag for h in hv]
        _iters += 1
        _cur = float(sum((h * vi).sum() for h, vi in zip(hvp(v), v)).item())
        if _prev is not None and _cur != 0.0:
            _rel = abs(_cur - _prev) / abs(_cur)
        _prev = _cur
    if status is not None:
        status.update(n_iter=_iters, rel_change=_rel,
                      converged=bool(np.isfinite(_rel) and _rel < _tol))
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
    # Two nonzero gradients of -5.000000097707407e-25 produced norm 0.0 and
    # SAM NaN under the naive form; max-coordinate-first gives the exact
    # 7.071067629665599e-25 (adversarial review round 3, 2026-09-04).
    grad_norm = _safe_norm(grad_flat)

    if grad_norm > 0:
        # try/finally: an exception raised by `compute_loss` used to leave the
        # model PERTURBED. Injecting a loss failure left a scalar parameter at
        # 1.1 instead of 1.0 -- absolute corruption 0.1 that then silently
        # poisoned every later task in the run (adversarial review round 2,
        # 2026-09-04).
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

    # eta0 must be large enough that theta - eta*g is REPRESENTABLE.
    #
    # In bf16 the 1e-6 default is BELOW parameter resolution (bf16 eps
    # 7.81e-03, so `p - 1e-6*g` returns exactly `p`): every probe returned the
    # unperturbed loss, and critical_step_size / critical_sharpness were NaN on
    # EVERY model in the study -- which core.py forces to bf16 precisely
    # because this task needs eager attention.
    #
    # The CALLER additionally promotes the whole model to float32 for this
    # measurement when it is stored in bf16/fp16. That matters beyond
    # representability: lambda_c is dtype-BIASED by +13.8% (gpt2 bf16) to
    # +34.7% (fp16), and an ablation isolates the bias to bf16 ARITHMETIC
    # rather than weight rounding -- pythia-160m in bf16 gives loss 2.328 /
    # ||g|| 171 against fp32 1.652 / 18.90, while bf16-ROUNDED weights
    # evaluated in fp32 math give 1.654 / 20.34. Without the promotion,
    # `dtype="auto"` would split the zoo by checkpoint storage format rather
    # than by any property of the model (adversarial review, 2026-09-03).
    try:
        _eps = float(torch.finfo(params[0].dtype).eps)
        _pnorm = _safe_norm(torch.cat([p_.detach().float().reshape(-1)
                                       for p_ in params]))
        # Exact guards: these are physical magnitudes, and clamping them at an
        # absolute 1e-12 makes the floor depend on the units the weights happen
        # to be in. A zero gradient norm means there is no step to take at all.
        if not (grad_norm > 0.0):
            return float("nan"), float("nan")
        _min_eta = 8.0 * _eps * _pnorm / grad_norm
        if np.isfinite(_min_eta) and _min_eta > eta0:
            eta0 = float(_min_eta)
    except Exception:
        pass

    saved = [p.detach().clone() for p in params]

    def probe(eta: float) -> float:
        # Same try/finally reason as _sam_sharpness: a loss failure mid-probe
        # left the model stepped (0.9 instead of 1.0 on the injected-exception
        # anchor) rather than restored.
        with torch.no_grad():
            try:
                for p_, g_ in zip(params, gvecs):
                    p_.add_(g_, alpha=-eta)
                val = float(compute_loss().item())
            finally:
                for p_, s_ in zip(params, saved):
                    p_.copy_(s_)
        return val

    # STRICT inequality, per Kalra & Barkeshli Definition 2.1: a step is
    # unstable when the loss INCREASES, so equality is stable. With `>=`, the
    # exactly-flat case was classified unstable and the search converged to
    # eta = 0.9999999999995453 / lambda_c = 2.0000000000009095 on
    # D(eta) = -eta(eta-1)^2(4-eta), whose first strict increase is elsewhere
    # (adversarial review, 2026-09-04).
    #
    # Doubling phase: find the first unstable eta. If already unstable at
    # eta0, HALVE downward first (Kalra et al. search in both directions;
    # audit 2026-07-16) so extremely sharp scopes still resolve eta*.
    eta = eta0
    if probe(eta) > baseline_loss:
        hi0 = eta
        for _ in range(max_doublings):
            eta /= 2.0
            if eta < 1e-30:
                return float("nan"), float("nan")
            # `<=`, not `<`. Kalra Def. 2.1 calls a step unstable when the
            # loss INCREASES, so equality is STABLE -- and the upward branch
            # already uses `> baseline`. The asymmetry made the downward search
            # return NaN on a loss that is exactly flat for eta <= 1e-9, where
            # eta* = 1e-9 and the mirrored upward case returns 0.000996
            # (independent Opus review, 2026-09-04).
            if probe(eta) <= baseline_loss:
                lo, hi = eta, eta * 2.0
                for _ in range(bisect_iters):
                    mid = 0.5 * (lo + hi)
                    if probe(mid) > baseline_loss:
                        hi = mid
                    else:
                        lo = mid
                # Bracket MIDPOINT, exactly as the upward branch does. The
                # downward branch used to return `lo`, which is the last
                # KNOWN-STABLE point rather than Kalra's approximation of the
                # critical step: on the lambda = 100 quadratic with 6
                # bisections it reported 101.135802 where the midpoint gives
                # 100.515337 (adversarial review, 2026-09-04).
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
        # Never became unstable within the search range. Returning the cap
        # (eta0 * 2^max_doublings, a CONSTANT) polluted cross-model
        # regressions with an indistinguishable sentinel (audit 2026-07-16:
        # random-init gpt2 returned exactly 1073.74). NaN is honest: eta*
        # exceeds the search range.
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
        # THE try MUST OPEN BEFORE THE FIRST MUTATION.
        #
        # `model.eval()`, the `requires_grad_` rewrite and the float32 promotion
        # below all mutate the CALLER'S model, and the `finally` that restores
        # them used to begin only after all three had run. An exception in that
        # window — an OOM during the promotion is the realistic one — left the
        # model in eval mode, with `requires_grad=False` on every parameter
        # except the last block, and in float32; `core.py:439` then continued
        # the run with that same object, so every later task in the sweep
        # measured a mutated model (Opus round 11, 2026-09-08).
        #
        # This file already fixed exactly this class once, at
        # `_sam_sharpness` (see its comment: an exception left a parameter at
        # 1.1 instead of 1.0 and "silently poisoned every later task in the
        # run"). The inner perturbation was wrapped; the outer setup was not.
        #
        # The restore variables are initialised to inert values first so the
        # `finally` can never reference an unbound name when the failure
        # happens early.
        was_training = model.training
        orig_grad_state = {}
        _saved_param_dtypes = []
        _promoted = False
        try:
            model.eval()
            # Save original requires_grad state so we can restore it — other
            # tasks or user code may depend on the model's grad configuration.
            orig_grad_state = {p: p.requires_grad for p in model.parameters()}
            for p in model.parameters():
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)

            # Promote to float32 for EVERY measurement, not only the critical step.
            #
            # Curvature is a second-order quantity, so storage precision biases all
            # four probes, not just the one that takes an explicit step. Measured on
            # cached tiny-random-gpt2: bf16 power iteration gave 2.4375 against
            # 3.109532833 in fp32, and evaluating the SAME bf16-rounded weights in
            # fp32 arithmetic gave 3.111037254 -- isolating a ~21% error to the
            # ARITHMETIC rather than to the weights (adversarial review,
            # 2026-09-04). Without this, trace/eigenvalue/SAM split the zoo by
            # checkpoint storage format rather than by model.
            #
            # Parameters ONLY. `model.float()` then `model.to(dtype)` is not a round
            # trip -- it downcasts buffers that were deliberately float32 (on
            # pythia-160m it turned `rotary_emb.inv_freq` float32 -> bfloat16, rel
            # 3.65e-03, and PERSISTED into every later task in the run).
            _model_dtype = next(model.parameters()).dtype
            _promoted = _model_dtype in (torch.bfloat16, torch.float16)
            _saved_param_dtypes = []
            if _promoted:
                logger.info(f"  sharpness: promoting model {_model_dtype} -> "
                            f"float32 for all curvature probes")
                _saved_param_dtypes = [(p_, p_.dtype) for p_ in model.parameters()]
                with torch.no_grad():
                    for p_ in model.parameters():
                        p_.data = p_.data.float()
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
            # UNCERTAINTY, reported. Hutchinson is a Monte Carlo estimator and
            # the shipped 5 probes are few: on the exact-trace MLP the 5-probe
            # mean was 4.464863030275032 against an exact 5.0693436557608935
            # (-11.92%) at a relative standard error of 86.16% (adversarial
            # review round 2, 2026-09-04). A reader cannot judge the number
            # without this.
            _tr_sd = float(np.std(trace_estimates, ddof=1)) if len(trace_estimates) > 1 else float("nan")
            _tr_se = _tr_sd / np.sqrt(len(trace_estimates)) if len(trace_estimates) > 1 else float("nan")
            _tr_rel_se = (abs(_tr_se / mean_trace) if mean_trace not in (0.0,)
                          and np.isfinite(_tr_se) else float("nan"))

            # 2. Top-1 eigenvalue via power iteration. Report the SIGNED
            # largest-magnitude eigenvalue (Rayleigh quotient at the converged
            # eigenvector), like PyHessian — ||Hv|| alone drops the sign.
            v = _make_random_vec(params, rademacher=False, generator=gen)
            # Fresh graph per HVP (retain_graph over many calls leaks memory).
            _pi_status = {}
            top_eig = _power_iteration_top_eigenvalue(
                lambda vv: _hvp(compute_loss(), params, vv), v, n_power_iter,
                status=_pi_status,
            )

            # SIGN STABILITY, MEASURED — because the reported sign can be an
            # artefact of the iteration count.
            #
            # Power iteration cannot separate a near-degenerate +/-lambda pair:
            # on pythia-70m the SAME model reports **-104737 at 10 iterations
            # and +111530 at 64**, on 3/3 seeds, with |lambda| ~ 1.24e5 and a
            # zero-crossing near iteration 37 (Opus round 11, 2026-09-08).
            #
            # `_meta_power_iter_converged` cannot warn about it: it was False on
            # 5/5 real models, and True in 0/120 runs where the answer was exact
            # to 4.3e-07 — a flag that is false when the answer is right and
            # silent when it is wrong is worse than absent, because it invites
            # the reader to trust it.
            #
            # So the flag is DERIVED from a second run at a different iteration
            # count rather than asserted from an internal convergence test. If
            # the two disagree in sign, the sign is not a property of the model
            # and the caller is told so. Costs one extra power iteration.
            _sign_stable = None
            try:
                _alt_status = {}
                _alt = _power_iteration_top_eigenvalue(
                    lambda vv: _hvp(compute_loss(), params, vv),
                    _make_random_vec(params, rademacher=False, generator=gen),
                    max(2, n_power_iter * 4), status=_alt_status,
                )
                if math.isfinite(_alt) and math.isfinite(top_eig):
                    _sign_stable = bool(
                        (_alt >= 0) == (top_eig >= 0)) if (
                        _alt != 0.0 and top_eig != 0.0) else True
                    _pi_status["alt_estimate"] = float(_alt)
                    _pi_status["alt_n_iter"] = int(max(2, n_power_iter * 4))
            except Exception:
                _sign_stable = None

            # 3. SAM-style sharpness (module-level helper so parity tests
            # drive the exact code evaluate() runs, not a transcription).
            sam_sharpness, perturbed_loss, grad_norm = _sam_sharpness(
                compute_loss, params, sam_rho, baseline_loss
            )

            # 4. Critical sharpness (B8): largest stable descent step along
            # the normalized gradient; lambda_c = 2/eta* (edge of stability).
            # Measure in float32. bf16 cannot represent the probe step, and
            # its ARITHMETIC biases lambda_c by 14-35% in a way that tracks the
            # checkpoint's storage format rather than the model. Promote the
            # WHOLE model, not just the differentiated parameters: promoting
            # params alone leaves buffers in the original dtype and the forward
            # pass raises on mixed dtypes.
            # The model is already in float32 here (hoisted above), so
            # `baseline_loss` was itself measured in float32 and needs no
            # separate re-measurement.
            loss_for_cs = compute_loss()
            grads_cs = torch.autograd.grad(loss_for_cs, params,
                                           create_graph=False)
            gnorm_cs = _safe_norm(_flatten_grads(list(grads_cs)))
            critical_step, critical_sharpness = _critical_step_size(
                compute_loss, params, list(grads_cs), gnorm_cs, baseline_loss
            )

        finally:
            # Restore parameters to their ORIGINAL dtypes individually.
            # Buffers were never touched, so they keep the checkpoint's dtype.
            if _promoted:
                with torch.no_grad():
                    for p_, dt in _saved_param_dtypes:
                        p_.data = p_.data.to(dt)
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
            "_meta_power_iter_rel_change": float(_pi_status.get("rel_change", float("nan"))),
            "_meta_power_iter_converged": bool(_pi_status.get("converged", False)),
            # DERIVED from a second run at 4x the iteration count -- see the
            # note at the call site. None when the second run could not be made.
            "_meta_top_eigenvalue_sign_stable": _sign_stable,
            "_meta_top_eigenvalue_alt_estimate": _pi_status.get("alt_estimate"),
            "_meta_top_eigenvalue_alt_n_iter": _pi_status.get("alt_n_iter"),
            "_meta_power_iter_n_used": int(_pi_status.get("n_iter", 0)),
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
