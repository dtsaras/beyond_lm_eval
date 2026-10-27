"""
Gradient flow per layer — per-layer gradient norms measuring how strongly
each layer's hidden state influences the prediction loss.

For each layer l, computes ||d L_CE / d h_l||_F where L_CE is the shifted
next-token cross-entropy loss averaged over positions (the same objective
the model was trained on). This is the "signal strength" reaching the
output from each layer. Smooth flow indicates healthy propagation; sharp
drops indicate vanishing gradients; spikes indicate exploding gradients.

SIGN CONVENTION. The layer index runs FORWARD, but backpropagation runs
backward, so for a chain with per-layer Jacobian scale ``a`` the activation
gradient at layer i is proportional to ``a^(L-i)`` and
``d log||g_i|| / d i = -log a``. A CONTRACTIVE chain (the Pascanu et al.
vanishing regime) therefore produces a POSITIVE slope, and an
expansive/exploding chain a NEGATIVE one: on matched six-layer chains a = 0.5
gives slope +3.465736 and a = 2 gives -3.465736.

Reported metrics:
  - **gradient_norm_per_layer**: per-layer Frobenius norm of dL/dh_l.
  - **gradient_flow_entropy**: Shannon entropy of the normalised per-layer
    norms. High = signal distributed across layers; low = concentrated.
  - **gradient_flow_slope**: linear regression slope of log(norm) vs.
    NORMALIZED depth (layer index / (n_layers - 1)). Depth-corrected but not
    architecture-free. POSITIVE = gradients shrink toward the input
    (vanishing); NEGATIVE = they grow toward the input (exploding).
  - **gradient_vanishing_ratio**: fraction of layers where the norm is
    < 10% of the maximum norm. A DISPERSION statistic, direction-free by
    construction (the matched a = 0.5 and a = 2 chains both return 0.333333).
  - **gradient_flow_log_ratio_in_out**: log(||g_first|| / ||g_last||).
    Direction-aware and unit-free -- NEGATIVE means the input-side gradient
    is the smaller one, i.e. vanishing; POSITIVE means exploding.

Implementation: registers forward pre-hooks on each transformer block to
capture the residual-stream input, calls `.retain_grad()`, then
backpropagates the shifted next-token cross-entropy loss (result key
``loss: 'cross_entropy'``).
"""

import contextlib
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .._precision import effective_dtype_meta, fp32_compute
from ..common import get_layers
from ..geometry.utils import safe_norm as _safe_norm

logger = logging.getLogger("blme")


@register_task("dynamics_gradient_flow")
class GradientFlowTask(DiagnosticTask):
    """Per-layer gradient norm profile."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Gradient Flow Analysis...")

        num_samples = self.config.get("num_samples", 4)

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "Machine learning models are trained on large text corpora."},
                {"text": "A federal judge ruled the policy unconstitutional."},
                {"text": "The history of mathematics begins with counting."},
            ]

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)

        was_training = model.training
        model.eval()
        orig_grad_state = {p: p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)

        all_norms = []  # list of (n_layers,) arrays

        # fp32 arithmetic on a bf16/fp16 model, at every size: bf16 forward and
        # backward double the slope on pythia-160m even with fp32 logits
        # (tests/tasks/invariance/test_dynamics_gradient_flow.py).
        _stack = contextlib.ExitStack()
        try:
            _prec = _stack.enter_context(fp32_compute(model))
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                text = sample["text"] if isinstance(sample, dict) else str(sample)
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=64).to(device)
                if enc["input_ids"].shape[1] < 2:
                    continue

                captured: Dict[int, torch.Tensor] = {}

                def make_hook(li):
                    def hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0]
                            x.requires_grad_(True)
                            x.retain_grad()
                            captured[li] = x
                    return hook

                handles = [layers[li].register_forward_pre_hook(make_hook(li))
                           for li in range(n_layers)]
                try:
                    out = model(**enc)
                    # Shifted next-token cross-entropy (the training objective),
                    # not the argmax logit, whose target differs across models
                    # (tests/tasks/test_dynamics.py::test_gradient_flow_uses_cross_entropy_loss).
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = enc["input_ids"][..., 1:].contiguous()
                    # Half-precision logits are lifted to fp32 before the loss,
                    # as the model's own forward(labels=...) does: the seed
                    # derivative softmax(z) - onehot(y) is otherwise computed in bf16.
                    if shift_logits.dtype in (torch.float16, torch.bfloat16):
                        shift_logits = shift_logits.float()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss.backward()
                finally:
                    for h in handles:
                        h.remove()

                norms = np.zeros(n_layers, dtype=np.float64)
                for li in range(n_layers):
                    if li in captured and captured[li].grad is not None:
                        g = captured[li].grad.detach().float()
                        # Scale-exact norm: `.norm()` underflows long before the data.
                        norms[li] = _safe_norm(g)
                all_norms.append(norms)

                model.zero_grad(set_to_none=True)

        finally:
            _stack.close()
            for p, grad_flag in orig_grad_state.items():
                p.requires_grad_(grad_flag)
            if was_training:
                model.train()

        if not all_norms:
            return {"error": "No samples produced gradient norms"}

        mean_norms = np.mean(np.stack(all_norms), axis=0)  # (n_layers,)

        # Entropy of the normalised norm profile. It is bounded by log(n_layers),
        # so it is depth-linked; exp(H)/L is emitted as _meta_ only (it moves
        # under non-uniform identity-block insertion).
        total = mean_norms.sum()
        if total > 0:
            p = mean_norms / total
            p_pos = p[p > 0]
            flow_entropy = float(-np.sum(p_pos * np.log(p_pos)))
            flow_entropy_norm = (float(np.exp(flow_entropy) / n_layers)
                                 if n_layers > 0 else float("nan"))
        else:
            flow_entropy = float("nan")
            flow_entropy_norm = float("nan")

        # Slope of log(norm) vs normalised depth i/(n_layers - 1). Exactly-zero
        # layers are dropped before the log rather than clipped.
        positive = mean_norms > 0
        if positive.sum() >= 2:
            denom = max(1, n_layers - 1)
            xs = np.arange(n_layers)[positive] / float(denom)
            ys = np.log(mean_norms[positive])
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = float("nan")

        # Direction-aware companion to the vanishing ratio: NEGATIVE = the
        # input-side gradient is the smaller one = vanishing.
        if mean_norms[0] > 0 and mean_norms[-1] > 0:
            log_ratio_in_out = float(np.log(mean_norms[0]) - np.log(mean_norms[-1]))
        else:
            log_ratio_in_out = float("nan")

        max_norm = mean_norms.max()
        if max_norm > 0:
            vanishing_ratio = float(np.mean(mean_norms < 0.1 * max_norm))
        else:
            vanishing_ratio = float("nan")

        return {
            "gradient_norm_per_layer": mean_norms.tolist(),
            "gradient_flow_entropy": flow_entropy,
            "_meta_gradient_flow_entropy_norm": flow_entropy_norm,
            "gradient_flow_slope": slope,
            "gradient_vanishing_ratio": vanishing_ratio,
            "gradient_flow_log_ratio_in_out": log_ratio_in_out,
            "gradient_norm_mean": float(mean_norms.mean()),
            "gradient_norm_max": float(mean_norms.max()),
            "gradient_norm_min": float(mean_norms.min()),
            "_meta_n_layers": n_layers,
            "loss": "cross_entropy",
            **effective_dtype_meta(_prec),
        }
