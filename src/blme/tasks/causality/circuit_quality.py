"""
Circuit Quality Metrics — Faithfulness and Minimality
──────────────────────────────────────────────────────────────────────
Identifies critical model components via causal effect, ablates everything
else, and measures circuit faithfulness (does the circuit reproduce the
model's behavior?) and minimality (is the circuit compact?).

References:
- "Causal Scrubbing" (Chan et al., 2022)
- "Towards Automated Circuit Discovery for Mechanistic Interpretability"
  (Conmy et al., 2023). arXiv:2304.14997
"""

import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")


@register_task("causality_circuit_quality")
class CircuitQualityTask(DiagnosticTask):
    """
    Identifies critical components via causal effect, ablates everything
    else, and measures circuit faithfulness x minimality.

    Returns circuit_faithfulness, circuit_minimality, and
    circuit_quality_score.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Circuit Quality Analysis...")
        num_samples = self.config.get("num_samples", 3)
        top_k_pct = self.config.get("top_k_pct", 25)  # top 25% components

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers."}
        num_layers = len(layers)

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        encodings = []
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            ids = tokenizer.encode(text, return_tensors="pt",
                                   truncation=True, max_length=128).to(device)
            if ids.shape[1] > 1:
                encodings.append(ids)

        if not encodings:
            return {"error": "No valid sequences."}

        def get_loss_and_probs(input_ids):
            outputs = model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            probs = F.softmax(logits[0, -1], dim=-1)
            return loss.item(), probs

        # Step 1: Get baseline performance
        with torch.no_grad():
            baseline_losses = []
            baseline_probs = []
            for ids in encodings:
                loss, probs = get_loss_and_probs(ids)
                baseline_losses.append(loss)
                baseline_probs.append(probs)

        baseline_mean_loss = float(np.mean(baseline_losses))

        # Step 2: Compute *dataset-mean* hidden state per layer (averaged
        # across all encodings, not per-sequence). This is closer to a true
        # mean-ablation baseline and avoids contaminating layer importance
        # with per-sequence content.
        dataset_mean_states = {}
        with torch.no_grad():
            sums = {l_idx: None for l_idx in range(num_layers)}
            counts = {l_idx: 0 for l_idx in range(num_layers)}
            for ids in encodings:
                clean_out = model(ids, output_hidden_states=True)
                for l_idx in range(num_layers):
                    h = clean_out.hidden_states[l_idx + 1]  # (1, T, D)
                    # Sum over tokens, accumulate count
                    sums[l_idx] = (h.sum(dim=1) if sums[l_idx] is None
                                   else sums[l_idx] + h.sum(dim=1))
                    counts[l_idx] += h.shape[1]
            for l_idx in range(num_layers):
                if counts[l_idx] > 0:
                    # shape (1, 1, D) — broadcastable to any (1, T, D)
                    dataset_mean_states[l_idx] = (sums[l_idx] / counts[l_idx]).unsqueeze(1)

        # Step 3: Measure causal importance of each layer via dataset-mean ablation
        layer_importances = []

        def make_mean_ablation_hook(mean_val):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    return (mean_val.expand_as(output[0]),) + output[1:]
                return mean_val.expand_as(output)
            return hook

        with torch.no_grad():
            for l_idx in range(num_layers):
                if l_idx not in dataset_mean_states:
                    layer_importances.append(0.0)
                    continue
                ablated_losses = []
                handle = layers[l_idx].register_forward_hook(
                    make_mean_ablation_hook(dataset_mean_states[l_idx])
                )
                try:
                    for ids in encodings:
                        loss, _ = get_loss_and_probs(ids)
                        ablated_losses.append(loss)
                finally:
                    handle.remove()
                mean_ablated = float(np.mean(ablated_losses))
                importance = mean_ablated - baseline_mean_loss
                layer_importances.append(max(0.0, importance))

        # Step 4: Identify "circuit" — top-k% most important layers
        importances = np.array(layer_importances)
        n_circuit = max(1, int(num_layers * top_k_pct / 100))
        circuit_layers = set(np.argsort(importances)[-n_circuit:].tolist())
        non_circuit_layers = set(range(num_layers)) - circuit_layers

        # Step 5: Measure faithfulness — ablate non-circuit layers, compare
        # to baseline via Jensen–Shannon divergence (symmetric, bounded in
        # [0, log 2]). We report `faith = 1 - JSD/log(2)` so the metric stays
        # in [0, 1] linearly and doesn't saturate exponentially the way
        # exp(-kl) does.
        log2 = float(np.log(2))
        faithfulness_scores = []

        with torch.no_grad():
            for idx, ids in enumerate(encodings):
                # Use dataset-mean activations (already computed) for the
                # non-circuit ablation hooks.
                hooks = []
                for l_idx in non_circuit_layers:
                    if l_idx not in dataset_mean_states:
                        continue
                    hooks.append(
                        layers[l_idx].register_forward_hook(
                            make_mean_ablation_hook(dataset_mean_states[l_idx])
                        )
                    )

                try:
                    circuit_loss, circuit_probs = get_loss_and_probs(ids)
                    base_probs = baseline_probs[idx]

                    # Jensen–Shannon divergence
                    m = 0.5 * (circuit_probs + base_probs)
                    log_m = torch.log(m.clamp(min=1e-12))
                    kl_cm = F.kl_div(log_m, circuit_probs,
                                     reduction='sum', log_target=False).item()
                    kl_bm = F.kl_div(log_m, base_probs,
                                     reduction='sum', log_target=False).item()
                    jsd = 0.5 * (kl_cm + kl_bm)
                    faith = float(max(0.0, min(1.0, 1.0 - jsd / log2)))
                    faithfulness_scores.append(faith)
                finally:
                    for h in hooks:
                        h.remove()

        # Step 5: Compute minimality
        minimality = 1.0 - (n_circuit / num_layers)

        # Step 6: Aggregate
        mean_faithfulness = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0

        # Quality = faithfulness * minimality (harmonic mean)
        if mean_faithfulness + minimality > 0:
            quality = 2.0 * mean_faithfulness * minimality / (mean_faithfulness + minimality)
        else:
            quality = 0.0

        return {
            "circuit_faithfulness": mean_faithfulness,
            "circuit_minimality": float(minimality),
            "circuit_quality_score": float(quality),
            "circuit_size_layers": n_circuit,
            "total_layers": num_layers,
            "layer_importances": layer_importances,
        }
