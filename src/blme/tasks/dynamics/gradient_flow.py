"""
Gradient flow per layer — per-layer Jacobian norms measuring how strongly
each layer's hidden state influences the final prediction.

For each layer l, computes ||d logit(y*) / d h_l||_F where y* is the
predicted token at the final position. This is the "signal strength"
reaching the output from each layer. Smooth, monotonically decreasing
flow indicates healthy gradient propagation; sharp drops indicate
vanishing gradients; spikes indicate exploding gradients.

Reported metrics:
  - **gradient_norm_per_layer**: per-layer Frobenius norm of the Jacobian.
  - **gradient_flow_entropy**: Shannon entropy of the normalised per-layer
    norms. High = signal distributed across layers; low = concentrated.
  - **gradient_flow_slope**: linear regression slope of log(norm) vs. layer
    index. Negative = gradients decay with depth (vanishing); positive =
    gradients grow (exploding).
  - **gradient_vanishing_ratio**: fraction of layers where the norm is
    < 10% of the maximum norm.

Implementation: registers forward pre-hooks on each transformer block to
capture the residual-stream input, calls `.retain_grad()`, then
backpropagates a single logit.
"""

import logging
from typing import Dict, List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

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

        try:
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
                    logits = out.logits[0, -1]
                    target_id = int(logits.argmax().item())
                    logits[target_id].backward()
                finally:
                    for h in handles:
                        h.remove()

                norms = np.zeros(n_layers, dtype=np.float64)
                for li in range(n_layers):
                    if li in captured and captured[li].grad is not None:
                        g = captured[li].grad.detach().float()
                        norms[li] = float(g.norm().item())
                all_norms.append(norms)

                # Zero grads for next sample
                model.zero_grad(set_to_none=True)

        finally:
            for p, grad_flag in orig_grad_state.items():
                p.requires_grad_(grad_flag)
            if was_training:
                model.train()

        if not all_norms:
            return {"error": "No samples produced gradient norms"}

        mean_norms = np.mean(np.stack(all_norms), axis=0)  # (n_layers,)

        # Entropy of normalised norms
        total = mean_norms.sum()
        if total > 0:
            p = mean_norms / total
            p_pos = p[p > 0]
            flow_entropy = float(-np.sum(p_pos * np.log(p_pos)))
        else:
            flow_entropy = float("nan")

        # Slope of log(norm) vs layer index (gradient decay/growth rate)
        log_norms = np.log(mean_norms.clip(min=1e-30))
        valid = np.isfinite(log_norms)
        if valid.sum() >= 2:
            xs = np.arange(n_layers)[valid]
            ys = log_norms[valid]
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = float("nan")

        # Vanishing ratio: fraction of layers < 10% of max
        max_norm = mean_norms.max()
        if max_norm > 0:
            vanishing_ratio = float(np.mean(mean_norms < 0.1 * max_norm))
        else:
            vanishing_ratio = float("nan")

        return {
            "gradient_norm_per_layer": mean_norms.tolist(),
            "gradient_flow_entropy": flow_entropy,
            "gradient_flow_slope": slope,
            "gradient_vanishing_ratio": vanishing_ratio,
            "gradient_norm_mean": float(mean_norms.mean()),
            "gradient_norm_max": float(mean_norms.max()),
            "gradient_norm_min": float(mean_norms.min()),
            "n_layers": n_layers,
        }
