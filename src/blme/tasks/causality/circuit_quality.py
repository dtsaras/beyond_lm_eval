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
            dataset = [
                {"text": "The capital of France is Paris"},
                {"text": "Water boils at 100 degrees Celsius"},
                {"text": "The quick brown fox jumps over the lazy dog"},
            ] * max(1, num_samples // 3 + 1)

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

        # Step 2: Measure causal importance of each layer via mean ablation
        # For each layer, ablate it and measure loss increase
        layer_importances = []

        with torch.no_grad():
            for l_idx in range(num_layers):
                ablated_losses = []

                for ids in encodings:
                    # Compute mean activation for this layer
                    clean_out = model(ids, output_hidden_states=True)
                    h_state = clean_out.hidden_states[l_idx + 1]
                    seq_mean = h_state.mean(dim=1, keepdim=True)

                    def get_mean_ablation_hook(mean_val):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                return (mean_val.expand_as(output[0]),) + output[1:]
                            return mean_val.expand_as(output)
                        return hook

                    handle = layers[l_idx].register_forward_hook(
                        get_mean_ablation_hook(seq_mean)
                    )
                    try:
                        loss, _ = get_loss_and_probs(ids)
                        ablated_losses.append(loss)
                    finally:
                        handle.remove()

                mean_ablated = float(np.mean(ablated_losses))
                importance = mean_ablated - baseline_mean_loss
                layer_importances.append(max(0.0, importance))

        # Step 3: Identify "circuit" — top-k% most important layers
        importances = np.array(layer_importances)
        n_circuit = max(1, int(num_layers * top_k_pct / 100))
        circuit_layers = set(np.argsort(importances)[-n_circuit:].tolist())
        non_circuit_layers = set(range(num_layers)) - circuit_layers

        # Step 4: Measure faithfulness — ablate non-circuit layers, compare
        faithfulness_scores = []

        with torch.no_grad():
            for idx, ids in enumerate(encodings):
                # Compute mean activations for non-circuit layers
                clean_out = model(ids, output_hidden_states=True)
                mean_states = {}
                for l_idx in non_circuit_layers:
                    h = clean_out.hidden_states[l_idx + 1]
                    mean_states[l_idx] = h.mean(dim=1, keepdim=True)

                # Ablate all non-circuit layers
                hooks = []
                for l_idx in non_circuit_layers:
                    def get_ablation_hook(mean_val):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                return (mean_val.expand_as(output[0]),) + output[1:]
                            return mean_val.expand_as(output)
                        return hook
                    hooks.append(
                        layers[l_idx].register_forward_hook(
                            get_ablation_hook(mean_states[l_idx])
                        )
                    )

                try:
                    circuit_loss, circuit_probs = get_loss_and_probs(ids)
                    base_probs = baseline_probs[idx]

                    # Faithfulness: KL divergence between circuit output and full model
                    circuit_log_probs = torch.log(circuit_probs + 1e-10)
                    kl = F.kl_div(circuit_log_probs, base_probs,
                                  reduction='sum', log_target=False).item()
                    # Lower KL = more faithful; convert to 0-1 score
                    faith = float(np.exp(-kl))
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
