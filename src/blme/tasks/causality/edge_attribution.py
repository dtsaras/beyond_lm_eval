"""
Layer attribution proxy inspired by attribution patching.

For compatibility the registered task name remains
``causality_edge_attribution``, but this implementation does not perform
true edge attribution patching over an activation graph. Instead, for
each transformer layer it approximates the contribution to the model's
prediction by:

    attr(layer l) = (h_l_clean - h_l_corrupted) · grad(logit | h_l_clean)

where h_l is the residual stream entering layer l. This is the first-order
linear approximation to the effect of "patching in the corrupted hidden
state at layer l on the clean input". The corrupted input is the same
text with shuffled tokens.

Summary metrics:
  - **attribution_gini**: Gini coefficient across layers — how concentrated
    is the causal effect in a few layers?
  - **top1_layer_share**: fraction of total attribution in the most
    important layer.
  - **peak_attribution_layer**: which layer carries the most attribution,
    normalised to [0, 1].
  - **attribution_entropy**: Shannon entropy of the per-layer attribution
    distribution.

Implementation note: this HF-generic proxy only requires one clean
forward + backward and one corrupted forward. Total cost: ~3x a single
forward pass.
"""

import logging
from typing import Dict, List

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


_EAP_PROMPTS: List[str] = [
    "The capital of France is Paris",
    "Water boils at 100 degrees Celsius",
    "The chemical symbol for gold is Au",
    "Beethoven was born in Germany",
    "The largest planet is Jupiter",
    "Albert Einstein developed the theory of relativity",
    "The Great Wall is in China",
    "The Mona Lisa was painted by Leonardo",
    "The speed of light is approximately 300000 kilometers per second",
    "DNA stands for deoxyribonucleic acid",
]


@register_task("causality_edge_attribution")
class EdgeAttributionTask(DiagnosticTask):
    """Per-layer residual-stream attribution proxy."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running layer attribution proxy...")

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and "text" in dataset[0]
        ):
            prompts = [d["text"] for d in dataset[:self.config.get("num_samples", 10)]]
        elif dataset is not None and isinstance(dataset, list) and dataset and isinstance(dataset[0], str):
            prompts = list(dataset)[:self.config.get("num_samples", 10)]
        else:
            prompts = _EAP_PROMPTS

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

        all_layer_attr_normed = []  # per-prompt: (n_layers,) attribution vector
        all_ginis = []
        all_entropies = []
        all_top1_shares = []
        all_peak_layers = []

        for pi, text in enumerate(prompts):
            try:
                enc = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128).to(device)
                input_ids = enc["input_ids"]
                if input_ids.shape[1] < 4:
                    continue

                # --- Corrupted pass: shuffle the tokens to destroy meaning.
                # Seed the permutation per-prompt so reruns are
                # reproducible; otherwise the corrupted baseline changes
                # every invocation and the resulting attribution scores
                # are noisy. A real counterfactual pair (clean/corrupted)
                # would be better but requires a curated dataset.
                _g = torch.Generator(device="cpu").manual_seed(pi * 997 + 11)
                perm = torch.randperm(
                    input_ids.shape[1], generator=_g
                ).to(device)
                corrupted_ids = input_ids[:, perm]

                # Collect corrupted residual-stream states via hidden_states.
                with torch.no_grad():
                    c_out = model(corrupted_ids, output_hidden_states=True)
                    # hidden_states[1:] are per-layer outputs
                    corrupted_hs = [h.detach() for h in c_out.hidden_states[1:]]

                # --- Clean pass: capture residual-stream inputs at each layer
                # IN the autograd graph, so we can read their .grad.
                captured: Dict[int, torch.Tensor] = {}

                def make_pre_hook(li):
                    def hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0]
                        else:
                            return
                        x.requires_grad_(True)
                        x.retain_grad()
                        captured[li] = x
                    return hook

                handles = []
                for li in range(n_layers):
                    handles.append(
                        layers[li].register_forward_pre_hook(make_pre_hook(li))
                    )

                try:
                    clean_out = model(input_ids=input_ids)
                    # Score: logit of the actual final-token prediction
                    logits = clean_out.logits[0, -1]
                    target_id = int(logits.argmax().item())
                    logits[target_id].backward()
                finally:
                    for h in handles:
                        h.remove()

                # --- Compute per-layer attribution
                layer_attr = np.zeros(n_layers, dtype=np.float64)
                for li in range(n_layers):
                    if li not in captured or captured[li].grad is None:
                        continue
                    clean_h = captured[li].detach().float().cpu().numpy()[0]   # (T, D)
                    grad_h = captured[li].grad.detach().float().cpu().numpy()[0]  # (T, D)
                    # Corrupted residual at the same layer: we approximate
                    # it using the corrupted hidden_states. corrupted_hs[li]
                    # is the output of layer li, while we want the *input*
                    # to layer li. For li=0 the input is the embedding
                    # output (hidden_states[0]), and for li>0 it's the
                    # output of layer li-1. But we stored
                    # hidden_states[1:] above, so corrupted_hs[li-1] =
                    # output of layer li-1 = input to layer li.
                    if li == 0:
                        c_h = c_out.hidden_states[0].detach().float().cpu().numpy()[0]
                    else:
                        c_h = corrupted_hs[li - 1].float().cpu().numpy()[0]
                    # Align shapes (the corrupted pass may have same token
                    # count but shuffled; shapes should match.)
                    T = min(clean_h.shape[0], c_h.shape[0], grad_h.shape[0])
                    diff = clean_h[:T] - c_h[:T]
                    attr = np.abs((diff * grad_h[:T]).sum())
                    layer_attr[li] = attr

                total = layer_attr.sum()
                if total == 0:
                    continue
                normed = layer_attr / total
                all_layer_attr_normed.append(normed)

                # Gini (sorted ascending per the standard formula)
                sorted_attr = np.sort(layer_attr)  # ascending
                n = len(sorted_attr)
                if n > 1 and total > 0:
                    cum = np.cumsum(sorted_attr)
                    g = float((n + 1 - 2 * cum.sum() / cum[-1]) / n)
                    all_ginis.append(g)

                # Top-1 share
                all_top1_shares.append(float(sorted_attr[-1] / total))

                # Peak layer
                all_peak_layers.append(float(np.argmax(layer_attr)) / max(1, n_layers - 1))

                # Entropy
                p = normed[normed > 0]
                H = float(-np.sum(p * np.log(p)))
                all_entropies.append(H)

            except Exception as e:
                logger.info(f"  EAP failed for '{text[:40]}': {type(e).__name__}: {e}")
                continue

        for p, grad_flag in orig_grad_state.items():
            p.requires_grad_(grad_flag)
        if was_training:
            model.train()

        if not all_ginis:
            return {"error": "No prompts produced attribution"}

        # Average per-layer profile
        mean_profile = np.mean(np.stack(all_layer_attr_normed), axis=0)

        return {
            "diagnostic_method": "residual_layer_gradient_patch_proxy",
            "attribution_unit": "transformer_layer",
            "n_prompts": len(all_ginis),
            "attribution_gini": float(np.mean(all_ginis)),
            "top1_layer_share": float(np.mean(all_top1_shares)),
            "peak_attribution_layer": float(np.mean(all_peak_layers)),
            "attribution_entropy": float(np.mean(all_entropies)),
            "mean_layer_attribution_profile": mean_profile.tolist(),
        }
