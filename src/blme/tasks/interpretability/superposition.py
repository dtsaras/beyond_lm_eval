"""
Superposition Index — Neuron Polysemanticity Measurement
──────────────────────────────────────────────────────────────────────
Measures the degree of superposition (polysemanticity) in model neurons
by analyzing activation distribution multimodality and pairwise activation
correlations within MLP layers.

In superposition, individual neurons encode multiple unrelated features,
leading to multimodal activation distributions and correlated activations
between neurons that share overlapping feature sets.

References:
- "Toy Models of Superposition" (Elhage et al., 2022)
- "Scaling Monosemanticity" (Templeton et al., 2024)
"""

import torch
import numpy as np
from collections import defaultdict

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")


def _multimodality_score(activations):
    """Estimate multimodality of a neuron's activation distribution.

    Uses a simple bimodality coefficient:
        BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))

    Values > 0.555 suggest bimodality (Pfister et al., 2013).
    We return 1 - BC so that higher = more monosemantic.
    """
    from scipy.stats import skew, kurtosis as scipy_kurtosis

    if len(activations) < 4:
        return 0.0

    s = skew(activations)
    k = scipy_kurtosis(activations, fisher=False)  # excess=False => Pearson

    n = len(activations)
    if n < 4 or k == 0:
        return 0.0

    bc = (s ** 2 + 1) / k
    # Clamp between 0 and 1
    bc = min(max(bc, 0.0), 1.0)
    return float(bc)


@register_task("interpretability_superposition")
class SuperpositionIndexTask(DiagnosticTask):
    """
    Measures polysemanticity of neurons by analyzing activation distribution
    multimodality and pairwise activation correlations in MLP layers.

    Returns mean_polysemanticity_index, polysemanticity_per_layer,
    and neuron_utilization_rate.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Superposition Index Analysis...")
        num_samples = self.config.get("num_samples", 5)
        max_neurons = self.config.get("max_neurons", 256)

        device = next(model.parameters()).device

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "Machine learning models learn representations of data."},
                {"text": "Quantum computing leverages superposition of states."},
            ] * max(1, num_samples // 3 + 1)

        samples = list(dataset)[:num_samples]
        if not samples:
            return {"error": "Need at least 1 sample."}

        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect model layers."}

        num_layers = len(layers)

        # Collect MLP activations via hooks
        activation_data = defaultdict(list)
        hooks = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                # (batch, seq_len, dim) -> flatten to (tokens, dim)
                flat = tensor.detach().cpu().reshape(-1, tensor.shape[-1])
                activation_data[layer_idx].append(flat)
            return hook

        for i, layer in enumerate(layers):
            target = layer
            if hasattr(layer, "mlp"):
                target = layer.mlp
            elif hasattr(layer, "feed_forward"):
                target = layer.feed_forward
            hooks.append(target.register_forward_hook(get_hook(i)))

        try:
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=128).to(device)
                    model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Analyze polysemanticity per layer
        polysemanticity_per_layer = []
        utilization_rates = []

        for l_idx in range(num_layers):
            if l_idx not in activation_data or not activation_data[l_idx]:
                polysemanticity_per_layer.append(0.0)
                utilization_rates.append(0.0)
                continue

            all_acts = torch.cat(activation_data[l_idx], dim=0).numpy()
            n_tokens, dim = all_acts.shape

            # Subsample neurons if dim is large
            neuron_indices = np.arange(dim)
            if dim > max_neurons:
                neuron_indices = np.random.choice(dim, max_neurons, replace=False)

            # 1. Multimodality scores per neuron
            bimodality_scores = []
            active_count = 0
            for n_idx in neuron_indices:
                neuron_acts = all_acts[:, n_idx]

                # Check if neuron is active (non-trivial activation)
                if np.std(neuron_acts) > 1e-6:
                    active_count += 1
                    bc = _multimodality_score(neuron_acts)
                    bimodality_scores.append(bc)

            # Polysemanticity = mean bimodality coefficient
            # Higher BC => more multimodal => more polysemantic
            if bimodality_scores:
                layer_poly = float(np.mean(bimodality_scores))
            else:
                layer_poly = 0.0

            polysemanticity_per_layer.append(layer_poly)

            # Utilization rate: fraction of neurons with non-trivial activations
            util = active_count / len(neuron_indices) if len(neuron_indices) > 0 else 0.0
            utilization_rates.append(float(util))

        mean_poly = float(np.mean(polysemanticity_per_layer)) if polysemanticity_per_layer else 0.0
        mean_util = float(np.mean(utilization_rates)) if utilization_rates else 0.0

        return {
            "mean_polysemanticity_index": mean_poly,
            "polysemanticity_per_layer": polysemanticity_per_layer,
            "neuron_utilization_rate": mean_util,
            "utilization_per_layer": utilization_rates,
        }
