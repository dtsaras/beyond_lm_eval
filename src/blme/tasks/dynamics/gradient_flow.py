"""
Gradient flow per layer — per-layer gradient norms measuring how strongly
each layer's hidden state influences the prediction loss.

For each layer l, computes ||d L_CE / d h_l||_F where L_CE is the shifted
next-token cross-entropy loss averaged over positions (the same objective
the model was trained on). This is the "signal strength" reaching the
output from each layer. Smooth flow indicates healthy propagation; sharp
drops indicate vanishing gradients; spikes indicate exploding gradients.

SIGN CONVENTION (corrected 2026-09-04). The layer index runs FORWARD, but
backpropagation runs backward, so for a chain with per-layer Jacobian scale
``a`` the activation gradient at layer i is proportional to ``a^(L-i)`` and
``d log||g_i|| / d i = -log a``. A CONTRACTIVE chain -- the Pascanu et al.
Eq. 7 vanishing regime -- therefore produces a POSITIVE slope, and an
expansive/exploding chain a NEGATIVE one. Measured on matched six-layer
chains: a = 0.5 gives slope +3.465736 and a = 2 gives -3.465736
(adversarial review, 2026-09-04). The previous docstring asserted the
opposite mapping.

Reported metrics:
  - **gradient_norm_per_layer**: per-layer Frobenius norm of dL/dh_l.
  - **gradient_flow_entropy**: Shannon entropy of the normalised per-layer
    norms. High = signal distributed across layers; low = concentrated.
  - **gradient_flow_slope**: linear regression slope of log(norm) vs.
    NORMALIZED depth (layer index / (n_layers - 1)), so the slope is
    depth-corrected across models with different layer counts -- but NOT
    architecture-free: r(slope, log d_model) = +0.860 on real models.
    POSITIVE = gradients shrink toward the input (vanishing); NEGATIVE =
    they grow toward the input (exploding). See the sign convention above.
  - **gradient_vanishing_ratio**: fraction of layers where the norm is
    < 10% of the maximum norm. This is a DISPERSION statistic and is
    DIRECTION-FREE by construction: the matched a = 0.5 and a = 2 chains
    both return 0.333333 despite opposite slopes, so it cannot on its own
    distinguish vanishing from exploding flow. Read it together with
    ``gradient_flow_log_ratio_in_out``.
  - **gradient_flow_log_ratio_in_out**: log(||g_first|| / ||g_last||).
    Direction-aware and unit-free -- NEGATIVE means the input-side gradient
    is the smaller one, i.e. vanishing; POSITIVE means exploding. On the
    matched chains it is -3.465736 for a = 0.5 and +3.465736 for a = 2.

Implementation: registers forward pre-hooks on each transformer block to
capture the residual-stream input, calls `.retain_grad()`, then
backpropagates the shifted next-token cross-entropy loss. (An earlier
variant backpropagated the single argmax logit; the CE-loss form is what
ships and what the tests pin — result key ``loss: 'cross_entropy'``.)
"""

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask
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
                    # Shifted next-token cross-entropy. This is the loss
                    # Pascanu et al. 2013 uses to measure gradient flow.
                    # Backpropagating the argmax-logit, as the legacy
                    # code did, ties the measurement to the model's own
                    # greedy prediction — different targets across
                    # models make the resulting norms incommensurable.
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = enc["input_ids"][..., 1:].contiguous()
                    # PROMOTE half precision before the loss, as the released
                    # model's own `forward(labels=...)` does.
                    #
                    # `F.cross_entropy` on bf16/fp16 logits computes the softmax
                    # in half precision, and the SEED DERIVATIVE of this whole
                    # measurement is (softmax(z) - one_hot(y))/(T-1) -- so the
                    # error enters at the very first backward step and is then
                    # propagated through every layer norm this task reports. On
                    # a bf16 Pythia-70m, layer 1 read 162.675113677979 against
                    # the native/analytic 53.704253345424 (+202.9%), and a
                    # 256-class CONSTANT-loss anchor, whose true layer norms are
                    # exactly [0, 0, 0, 0], emitted four positive values
                    # (codex round 15, 2026-09-09).
                    #
                    # float64 is preserved rather than forced to float32: this
                    # only lifts half precision to the width the reference uses.
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
                        # Scale-exact norm: `.norm()` squares first, so it
                        # underflows to 0.0 long before the data does. A
                        # 4-layer model whose stored gradients each had 8
                        # nonzero coordinates at 1.250000039e-23 reported all
                        # four norms as 0.0 against a true 3.535534017e-23
                        # (adversarial review round 3, 2026-09-04).
                        norms[li] = _safe_norm(g)
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

        # Entropy of the normalised norm profile, reported BOTH raw and
        # divided by log(n_layers).
        #
        # The raw entropy is bounded above by log(n_layers), so it encodes
        # depth rather than gradient flow: on five synthetic models with a
        # BYTE-IDENTICAL normalized-depth profile (exact 1000x decay from first
        # to last layer) it ranged 0.750 -> 3.430 as depth went 6 -> 80, and on
        # real models it correlates r = 0.974 with log(n_layers) (adversarial
        # review, 2026-09-02). The slope below was already depth-normalized for
        # exactly this reason; the entropy was missed.
        total = mean_norms.sum()
        if total > 0:
            p = mean_norms / total
            p_pos = p[p > 0]
            flow_entropy = float(-np.sum(p_pos * np.log(p_pos)))
            # exp(H)/L -- the participation ratio: what FRACTION of the
            # layers effectively carries the gradient mass.
            #
            # This is an EMPIRICAL depth correction, not a theorem, and on
            # REAL models it is the WORSE of the two.
            #
            # An earlier version of this comment claimed real models sit in the
            # "fixed TOTAL decay" family, where exp(H)/L is the invariant one.
            # That assignment was wrong. The profile is dominated by a LAYER-0
            # ARTIFACT: layer 0 alone holds 0.545 / 0.654 / 0.735 of the total
            # gradient mass, because ||h_0|| ~ 23 against ~3265 deeper in, and
            # corr(log||dL/dh||, -log||h||) is +0.87 / +0.95. Measured spread
            # across depth on real models is 1.88x for the RAW entropy against
            # 4.15x for exp(H)/L (adversarial review, 2026-09-03).
            #
            # Both are emitted. Prefer the RAW entropy for cross-model work
            # until the layer-0 artifact is addressed; the synthetic families
            # below describe where each normalization would help, not where
            # real models actually sit.
            #     fixed TOTAL decay      raw H 2.7-10.9x  exp(H)/L 1.06-1.12x
            #     fixed PER-LAYER decay  raw H 1.06-1.83x exp(H)/L 3.1 -12.3x
            #     top-k layers carry it  raw H 1.00-1.16x exp(H)/L 10.0-13.3x
            flow_entropy_norm = (float(np.exp(flow_entropy) / n_layers)
                                 if n_layers > 0 else float("nan"))
        else:
            flow_entropy = float("nan")
            flow_entropy_norm = float("nan")

        # Slope of log(norm) vs normalised depth (gradient decay/growth
        # rate per unit of depth, cross-model comparable). Regressing on
        # raw layer index yields a slope that scales as 1/n_layers — a
        # 32-layer model and a 64-layer model with identical gradient
        # profiles would get different slopes. Using ``xs / (n_layers-1)``
        # gives a per-depth-unit slope. NOTE: "comparable across architectures"
        # is an OVER-CLAIM, the same one the entropy normalization carried until
        # it was corrected. Measured on the textbook vanishing-gradient profile
        # (fixed per-layer decay), the slope runs -1.78 -> -28.18 over
        # L = 6..80, and on real models r(slope, log d_model) = +0.860
        # (adversarial review, 2026-09-03). Normalizing by depth removes the
        # n_layers scaling but not the width dependence; treat the slope as
        # depth-corrected, not architecture-free.
        # Drop exactly-zero layers before taking the log rather than
        # clipping to 1e-30 — a clip injects spurious -inf-like values
        # that corrupt the linear fit.
        positive = mean_norms > 0
        if positive.sum() >= 2:
            denom = max(1, n_layers - 1)
            xs = np.arange(n_layers)[positive] / float(denom)
            ys = np.log(mean_norms[positive])
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = float("nan")

        # Direction-aware companion to gradient_vanishing_ratio, which is a
        # dispersion statistic and cannot tell vanishing from exploding (both
        # matched chains score 0.333333). NEGATIVE = input-side gradient is
        # smaller = vanishing.
        if mean_norms[0] > 0 and mean_norms[-1] > 0:
            log_ratio_in_out = float(np.log(mean_norms[0]) - np.log(mean_norms[-1]))
        else:
            log_ratio_in_out = float("nan")

        # Vanishing ratio: fraction of layers < 10% of max
        max_norm = mean_norms.max()
        if max_norm > 0:
            vanishing_ratio = float(np.mean(mean_norms < 0.1 * max_norm))
        else:
            vanishing_ratio = float("nan")

        return {
            "gradient_norm_per_layer": mean_norms.tolist(),
            # Raw entropy is depth-bounded; prefer the normalized twin as a
            # cross-model feature. Both are emitted (see note above).
            "gradient_flow_entropy": flow_entropy,
            # DEMOTED to _meta_ (2026-09-04). exp(H)/L survives UNIFORM
            # identity duplication (0.614327268485 to 1.1e-16) but not the
            # NONUNIFORM case: inserting identity blocks at uneven depths
            # leaves the logits exact and still moves it 0.614327 -> 0.430603,
            # -29.9%. A twin whose whole claim is "removes the depth artefact"
            # must survive any function-preserving re-decomposition, not just
            # the symmetric one. Same bar that removed all eleven registered
            # normalizers (adversarial review round 2, 2026-09-04).
            "_meta_gradient_flow_entropy_norm": flow_entropy_norm,
            "gradient_flow_slope": slope,
            # gradient_vanishing_ratio is a FRACTION of layers, so it is not
            # width-confounded, but it correlates r = 0.978 with log(n_layers)
            # on real models because deeper stacks put more layers below a
            # fixed 10%-of-max cut. It is retained but should be treated as
            # depth-linked, not as an independent structural signal.
            "gradient_vanishing_ratio": vanishing_ratio,
            "gradient_flow_log_ratio_in_out": log_ratio_in_out,
            "gradient_norm_mean": float(mean_norms.mean()),
            "gradient_norm_max": float(mean_norms.max()),
            "gradient_norm_min": float(mean_norms.min()),
            "_meta_n_layers": n_layers,
            "loss": "cross_entropy",
        }
