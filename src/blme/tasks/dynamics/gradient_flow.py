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
  - **gradient_flow_log_ratio_in_out** (headline): log(||g_emb|| / ||g_final||),
    the end-to-end backward gain from the final residual to the embeddings.
    Direction-aware and unit-free -- NEGATIVE means the input-side gradient
    is the smaller one, i.e. vanishing; POSITIVE means exploding.
  - **gradient_norm_per_layer**: Frobenius norm of dL/dh_l at the input of
    every block, then at the final residual (the input of the final norm).
  - ``_meta_`` profile summaries, which move under identity-block insertion:
    the entropy of the normalised norms, the slope of log(norm) vs normalised
    depth i / (n_points - 1) (POSITIVE = vanishing toward the input), and the
    vanishing ratio (fraction of points below 10% of the maximum; direction-
    free: the matched a = 0.5 and a = 2 chains both return 0.333333). The raw
    norm mean/max/min halve under a residual x2 rescale and are _meta_ too.

Implementation: registers forward pre-hooks on each transformer block to
capture the residual-stream input, calls `.retain_grad()`, then
backpropagates the shifted next-token cross-entropy loss (result key
``loss: 'cross_entropy'``). The token pool of every norm drops the sink rows
(position 0 and massive-activation tokens): one mask per token, from a no-grad
pass over the same passages before the backward passes, used at every point.
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
from .._sink_norms import SinkNormCollector
from .._tokens import massive_token_share, position0_share, select_rows, sink_meta
from ..common import get_final_norm, get_layers
from ..geometry.utils import safe_norm as _safe_norm

logger = logging.getLogger("blme")


@register_task("dynamics_gradient_flow")
class GradientFlowTask(DiagnosticTask):
    """Per-layer gradient norm profile."""

    # The per-layer Frobenius norm pools over tokens; the sink rows (position 0
    # and massive-activation tokens) are dropped from that pool (non-LLM
    # origin), the loss keeps every position.
    TOKEN_POLICY = "exclude_sinks"
    SOURCES = {
        "paper": "arXiv:1211.5063",
        "code": None,
        "llm_precedent": "Xiong et al. 2020 (arXiv:2002.04745) Sec. 3.4: per-layer "
                         "gradient norms vs depth in Pre-/Post-LN Transformers",
    }
    ADAPTATION_NOTE = (
        "Pascanu et al. (2013) study how the training-loss gradient shrinks or grows as it "
        "is propagated back through time in an RNN; BLME maps time to depth. It "
        "backpropagates the model's own next-token cross-entropy on held-out text and "
        "records ||dL/dh||_F, over every position but the sink rows, at the input of every "
        "block and at the final residual (the input of the final norm). Activation "
        "gradients are used instead of the parameter-gradient norms of Xiong et al. (2020), "
        "because they are defined for a frozen pretrained model and do not depend on block "
        "width or parameterisation; Jacobian spectra are too costly at 34B. The headline is "
        "the end-to-end gain log(||g_emb|| / ||g_final||), which identity blocks anywhere and "
        "a common residual rescale leave unchanged; entropy, slope, vanishing ratio and raw "
        "norms move under one of them and are _meta_. The backward pass runs in fp32 "
        "arithmetic, because bf16 doubles the slope on pythia-160m."
    )
    HEADLINE_KEYS = ("gradient_flow_log_ratio_in_out",)
    SIZE_INVARIANT = {
        "gradient_flow_log_ratio_in_out": (
            ("insert_identity_blocks", "residual_scale_gauge(rescale_eps=True)"),
            "tests/tasks/invariance/test_dynamics_gradient_flow.py::"
            "test_end_to_end_log_ratio_invariant_under_identity_blocks",
        ),
    }

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

        # The profile is the input of every block plus the FINAL residual (the
        # input of the final norm, or the last block's output when there is no
        # final norm), so its first-to-last ratio is end-to-end.
        final_norm = get_final_norm(model)
        n_points = n_layers + 1
        _endpoint = ["missing"]
        all_norms = []  # list of (n_points,) arrays
        all_norms_incl = []
        n_rows = 0

        encodings = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample["text"] if isinstance(sample, dict) else str(sample)
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=64).to(device)
            if enc["input_ids"].shape[1] < 2:
                continue
            encodings.append(enc)

        # fp32 arithmetic on a bf16/fp16 model, at every size: bf16 forward and
        # backward double the slope on pythia-160m even with fp32 logits
        # (tests/tasks/invariance/test_dynamics_gradient_flow.py).
        _stack = contextlib.ExitStack()
        try:
            _prec = _stack.enter_context(fp32_compute(model))
            # The sink threshold is a median over every passage, so the masks
            # come from a no-grad pass over all of them first, in the same
            # arithmetic as the backward passes; the same mask then applies at
            # every profile point (tests/tasks/invariance/
            # test_dynamics_gradient_flow.py).
            _sinks = SinkNormCollector()
            with torch.no_grad():
                for enc in encodings:
                    _sinks.add(model(**enc, output_hidden_states=True).hidden_states[1:])
            masks, sink_info = _sinks.masks()

            for d, enc in enumerate(encodings):
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
                if final_norm is not None:
                    handles.append(final_norm.register_forward_pre_hook(
                        make_hook(n_layers)))
                    _endpoint[0] = "final_norm_input"
                else:
                    def _out_hook(module, args, output):
                        y = output[0] if isinstance(output, tuple) else output
                        y.retain_grad()
                        captured[n_layers] = y
                    handles.append(layers[-1].register_forward_hook(_out_hook))
                    _endpoint[0] = "last_block_output"
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

                norms = np.zeros(n_points, dtype=np.float64)
                norms_incl = np.zeros(n_points, dtype=np.float64)
                row_meta = None
                for li in range(n_points):
                    if li in captured and captured[li].grad is not None:
                        g = captured[li].grad.detach().float()
                        # Scale-exact norm: `.norm()` underflows long before the data.
                        norms_incl[li] = _safe_norm(g)
                        rows, row_meta = select_rows([g[0]], self.TOKEN_POLICY,
                                                     keep=[masks[d]])
                        norms[li] = _safe_norm(rows)
                all_norms.append(norms)
                all_norms_incl.append(norms_incl)
                if row_meta is not None:
                    n_rows += row_meta.n_rows

                model.zero_grad(set_to_none=True)

        finally:
            _stack.close()
            for p, grad_flag in orig_grad_state.items():
                p.requires_grad_(grad_flag)
            if was_training:
                model.train()

        if not all_norms:
            return {"error": "No samples produced gradient norms"}

        mean_norms = np.mean(np.stack(all_norms), axis=0)  # (n_points,)
        mean_norms_incl = np.mean(np.stack(all_norms_incl), axis=0)

        # Entropy of the normalised norm profile. It is bounded by log(n_points),
        # so it is depth-linked; exp(H)/n is emitted as _meta_ only (it moves
        # under non-uniform identity-block insertion).
        total = mean_norms.sum()
        if total > 0:
            p = mean_norms / total
            p_pos = p[p > 0]
            flow_entropy = float(-np.sum(p_pos * np.log(p_pos)))
            flow_entropy_norm = float(np.exp(flow_entropy) / n_points)
        else:
            flow_entropy = float("nan")
            flow_entropy_norm = float("nan")

        # Slope of log(norm) vs normalised depth i/(n_points - 1). Exactly-zero
        # layers are dropped before the log rather than clipped.
        positive = mean_norms > 0
        if positive.sum() >= 2:
            denom = max(1, n_points - 1)
            xs = np.arange(n_points)[positive] / float(denom)
            ys = np.log(mean_norms[positive])
            slope = float(np.polyfit(xs, ys, 1)[0])
        else:
            slope = float("nan")

        # End-to-end backward gain log(g_emb / g_final): NEGATIVE = the
        # input-side gradient is the smaller one = vanishing. Identity blocks
        # anywhere and a common residual rescale leave it unchanged.
        def _log_ratio(a, b):
            return float(np.log(a) - np.log(b)) if a > 0 and b > 0 else float("nan")
        log_ratio_in_out = _log_ratio(mean_norms[0], mean_norms[-1])
        # The former endpoint, the input of the last block, moves when an
        # identity block is appended.
        log_ratio_block_inputs = _log_ratio(mean_norms[0], mean_norms[n_layers - 1])
        log_ratio_incl_pos0 = _log_ratio(mean_norms_incl[0], mean_norms_incl[-1])

        max_norm = mean_norms.max()
        if max_norm > 0:
            vanishing_ratio = float(np.mean(mean_norms < 0.1 * max_norm))
        else:
            vanishing_ratio = float("nan")

        return {
            "gradient_norm_per_layer": mean_norms.tolist(),
            # Headline: the end-to-end gain. The profile summaries below move
            # under function-preserving identity-block insertion and the raw
            # norms under a residual rescale, so they are _meta_
            # (tests/tasks/invariance/test_dynamics_gradient_flow.py).
            "gradient_flow_log_ratio_in_out": log_ratio_in_out,
            "_meta_gradient_flow_entropy": flow_entropy,
            "_meta_gradient_flow_entropy_norm": flow_entropy_norm,
            "_meta_gradient_flow_slope": slope,
            "_meta_gradient_vanishing_ratio": vanishing_ratio,
            "_meta_gradient_flow_log_ratio_block_inputs": log_ratio_block_inputs,
            "_meta_gradient_flow_log_ratio_in_out_incl_pos0": log_ratio_incl_pos0,
            "_meta_gradient_norm_per_layer_incl_pos0": mean_norms_incl.tolist(),
            "_meta_rows_policy": self.TOKEN_POLICY,
            "_meta_rows_n_rows": int(n_rows),
            **sink_meta(sink_info),
            "_meta_position0_share": position0_share(_sinks.mid_docs),
            "_meta_massive_token_share": massive_token_share(_sinks.mid_docs),
            "_meta_final_endpoint": _endpoint[0],
            "_meta_n_profile_points": n_points,
            "_meta_gradient_norm_mean": float(mean_norms.mean()),
            "_meta_gradient_norm_max": float(mean_norms.max()),
            "_meta_gradient_norm_min": float(mean_norms.min()),
            "_meta_n_layers": n_layers,
            "loss": "cross_entropy",
            **effective_dtype_meta(_prec),
        }
