import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy.stats import kurtosis

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers

@register_task("interpretability_sparsity")
class ActivationSparsityTask(DiagnosticTask):
    """
    Measures the Activation Sparsity (L0 norm) and Kurtosis of MLP blocks in the LLM.
    These metrics are crucial for understanding representation efficiency, finding
    feature polysemanticity, and preparing for quantization (e.g., KurTail).
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Activation Sparsity...")
        num_samples = self.config.get("num_samples", 5)
        
        device = next(model.parameters()).device
        
        if dataset is None:
            dataset = [{"text": "Sample text for activation sparsity."}] * num_samples
        
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        layers = get_layers(model)
        num_layers = len(layers)
        
        # We need to register hooks to capture the output of the MLP blocks
        activation_stats = defaultdict(lambda: {"l0_rates": [], "kurtosis_vals": []})
        hooks = []
        
        # Define a hook function generator
        def get_hook(layer_idx):
            def hook(module, input, output):
                # Output might be a tuple, grab the first tensor
                tensor = output[0] if isinstance(output, tuple) else output
                # We expect shape (batch, seq_len, hidden_dim)
                # Calculate L0 sparsity rate (percentage of active neurons > e-5)
                # L0 norm essentially counts non-zeros.
                active_elements = (tensor.abs() > 1e-5).float()
                l0_rate = active_elements.mean().item()
                
                # Calculate Kurtosis to measure the 'tailedness' of the activation distribution
                # Flatten the tensor to compute kurtosis across all tokens and batch
                flat_tensor = tensor.detach().cpu().numpy().flatten()
                
                # We use Fisher's definition (normal ==> 0.0) by default in scipy
                k = float(kurtosis(flat_tensor, fisher=True))
                
                activation_stats[layer_idx]["l0_rates"].append(l0_rate)
                activation_stats[layer_idx]["kurtosis_vals"].append(k)
            return hook
        
        # Try to attach hooks to the MLP/FFN blocks specifically.
        # This requires some architecture heuristics
        for i, layer in enumerate(layers):
            target_module = None
            if hasattr(layer, "mlp"):
                target_module = layer.mlp
            elif hasattr(layer, "feed_forward"):
                target_module = layer.feed_forward
            else:
                target_module = layer  # Fallback to the whole layer output
                
            h = target_module.register_forward_hook(get_hook(i))
            hooks.append(h)
            
        try:
            with torch.no_grad():
                for s in samples:
                    if isinstance(s, dict) and "text" in s:
                        text = s["text"]
                    else:
                        text = str(s)
                        
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                    model(**inputs)
        finally:
            # Always remove hooks!
            for h in hooks:
                h.remove()
                
        # Aggregate results
        results = {}
        mean_l0_rates = []
        mean_kurtosis = []
        
        for i in range(num_layers):
            if i in activation_stats:
                layer_l0 = np.mean(activation_stats[i]["l0_rates"])
                layer_kurt = np.mean(activation_stats[i]["kurtosis_vals"])
                mean_l0_rates.append(float(layer_l0))
                mean_kurtosis.append(float(layer_kurt))
            else:
                mean_l0_rates.append(0.0)
                mean_kurtosis.append(0.0)
                
        results["layer_l0_rates"] = mean_l0_rates
        results["layer_kurtosis"] = mean_kurtosis
        results["global_mean_l0"] = float(np.mean(mean_l0_rates))
        results["global_mean_kurtosis"] = float(np.mean(mean_kurtosis))
        
        return results
