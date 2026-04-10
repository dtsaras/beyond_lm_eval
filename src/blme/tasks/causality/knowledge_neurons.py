"""
Knowledge neuron localization — Dai et al. 2022, arXiv:2104.08696.

For each (prompt, target_token) pair, compute saliency of every MLP
intermediate neuron with respect to the target token's logit, then
aggregate across facts to find which neurons consistently contribute
to factual recall.

Metrics reported:
  - **mean_attribution_gini**: average Gini coefficient of the
    per-neuron attribution magnitude (across MLP intermediates) over
    facts. High Gini = knowledge concentrated in a few neurons; low
    Gini = distributed.
  - **top1_share**: average fraction of total attribution magnitude
    carried by the single most-important neuron, across facts.
  - **top1pct_share**: same for the top 1% most-important neurons.
  - **localization_layer_mean**: average index of the layer with the
    largest summed attribution magnitude (normalised to [0, 1]).
  - **attribution_layer_entropy**: Shannon entropy of the per-layer
    attribution distribution, averaged across facts. Low = localized
    to a few layers; high = distributed.

Caveats:
  - This is the *gradient × activation* approximation, not full
    integrated gradients (no path integral). It is a fixed-point
    saliency rather than IG, but is several orders of magnitude
    cheaper and the qualitative signal is the same.
  - Requires autograd (no `torch.no_grad()`); may need
    `attn_implementation='eager'` for some models that route through
    SDPA's specialized kernels.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")


_FACT_BUNDLE: List[Tuple[str, str]] = [
    ("The capital of France is", " Paris"),
    ("The Eiffel Tower is located in", " Paris"),
    ("The currency of Japan is the", " yen"),
    ("Mount Everest is the tallest mountain in the world. Its height is over 8 thousand", " meters"),
    ("The chemical symbol for gold is", " Au"),
    ("Beethoven was born in the country of", " Germany"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The Pacific Ocean is the largest body of", " water"),
    ("Albert Einstein was born in the country of", " Germany"),
    ("The longest river in the world is the", " Nile"),
]


def _find_mlp_down_proj(layer: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Locate the down-projection of an MLP block (the second linear layer).

    Returns the module whose forward input is the MLP intermediate
    activation (after the up-projection / activation function).
    """
    mlp = (getattr(layer, "mlp", None)
           or getattr(layer, "feed_forward", None)
           or getattr(layer, "output", None))
    if mlp is None:
        return None
    for name in ("c_proj", "down_proj", "dense_4h_to_h", "dense", "wo", "fc2"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    return None


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values. 0 = uniform, 1 = maximally
    concentrated in a single element."""
    v = np.abs(values).flatten()
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


@register_task("causality_knowledge_neurons")
class KnowledgeNeuronsTask(DiagnosticTask):
    """Per-fact saliency on MLP intermediates (Dai et al. 2022)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Knowledge Neurons Localization...")

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and {"prompt", "target"} <= set(dataset[0])
        ):
            facts = [(d["prompt"], d["target"]) for d in dataset]
        else:
            facts = list(_FACT_BUNDLE)

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)

        # Find the MLP down projections per layer; the *input* to each is
        # the intermediate activation we want to saliency-attribute.
        down_projs = []
        for li, layer in enumerate(layers):
            dp = _find_mlp_down_proj(layer)
            if dp is None:
                return {"error": f"Could not find MLP down-projection in layer {li}"}
            down_projs.append(dp)

        # SDPA double-backward bail-out (the gradient of cross-entropy
        # logits through SDPA actually works fine; we only need single
        # backward, so this is OK without the eager check).

        was_training = model.training
        model.eval()
        orig_grad_state = {p: p.requires_grad for p in model.parameters()}
        for p in model.parameters():
            p.requires_grad_(False)

        # Per-fact aggregates
        all_layer_norms = []        # per-layer summed |attribution|
        all_top_neuron_layers = []  # which layer has the peak attribution
        all_attribution_ginis = []  # gini over the full neuron list
        all_top1_shares = []
        all_top1pct_shares = []
        all_layer_entropies = []

        for prompt, target in facts:
            try:
                enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
                target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
                if not target_ids:
                    continue
                target_token_id = target_ids[0]

                # We need a forward where the prompt's last position
                # predicts the target token. Use the prompt as-is (no
                # appended target) and read logits at position -1.
                input_ids = enc_prompt["input_ids"]

                # Capture intermediate activations (input to down_proj).
                captured: Dict[int, torch.Tensor] = {}
                def make_hook(li):
                    def pre_hook(module, args):
                        x = args[0]
                        x.requires_grad_(True)
                        x.retain_grad()
                        captured[li] = x
                        return None
                    return pre_hook

                handles = []
                for li, dp in enumerate(down_projs):
                    handles.append(dp.register_forward_pre_hook(make_hook(li)))

                try:
                    out = model(input_ids=input_ids)
                    logits = out.logits[0, -1]
                    target_logit = logits[target_token_id]
                    target_logit.backward()
                finally:
                    for h in handles:
                        h.remove()

                # Compute saliency = grad * activation per neuron, summed
                # across the sequence (we care about per-neuron importance,
                # not per-position).
                per_layer_attr = []
                for li in range(n_layers):
                    if li not in captured or captured[li].grad is None:
                        per_layer_attr.append(np.zeros(0, dtype=np.float64))
                        continue
                    act = captured[li][0].detach().float().cpu().numpy()  # (T, hidden)
                    grad = captured[li].grad[0].detach().float().cpu().numpy()  # (T, hidden)
                    saliency = (act * grad).sum(axis=0)  # (hidden,)
                    per_layer_attr.append(np.abs(saliency))
                # Concatenate all layers' neuron attributions
                flat = np.concatenate(per_layer_attr) if any(p.size for p in per_layer_attr) else np.zeros(0)
                if flat.size == 0 or flat.sum() == 0:
                    continue

                # Layer-wise summed attribution (for the localization metric)
                layer_norms = np.array([p.sum() for p in per_layer_attr], dtype=np.float64)
                all_layer_norms.append(layer_norms)

                # Most-important layer
                top_layer = int(np.argmax(layer_norms))
                all_top_neuron_layers.append(top_layer / max(1, n_layers - 1))

                # Gini over flat attribution
                all_attribution_ginis.append(_gini(flat))

                # Top1 / top1% shares
                sorted_flat = np.sort(flat)[::-1]
                total = flat.sum()
                if total > 0:
                    all_top1_shares.append(float(sorted_flat[0] / total))
                    one_pct = max(1, int(0.01 * len(flat)))
                    all_top1pct_shares.append(float(sorted_flat[:one_pct].sum() / total))

                # Layer-distribution entropy
                if layer_norms.sum() > 0:
                    p = layer_norms / layer_norms.sum()
                    p = p[p > 0]
                    H = float(-np.sum(p * np.log(p)))
                    all_layer_entropies.append(H)

            except Exception as e:
                logger.info(f"  Knowledge neuron extraction failed for '{prompt[:40]}': {e}")
                continue

        # Restore original requires_grad state and training mode.
        for p, grad_flag in orig_grad_state.items():
            p.requires_grad_(grad_flag)
        if was_training:
            model.train()

        if not all_attribution_ginis:
            return {"error": "No facts produced usable attribution"}

        return {
            "n_facts": len(all_attribution_ginis),
            "mean_attribution_gini": float(np.mean(all_attribution_ginis)),
            "mean_top1_share": float(np.mean(all_top1_shares)),
            "mean_top1pct_share": float(np.mean(all_top1pct_shares)),
            "localization_layer_mean": float(np.mean(all_top_neuron_layers)),
            "attribution_layer_entropy": float(np.mean(all_layer_entropies)),
            "n_layers": n_layers,
        }
