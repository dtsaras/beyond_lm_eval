import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")

@register_task("causality_ablation")
class AblationRobustnessTask(DiagnosticTask):
    """
    Measures Ablation Robustness (Circuit Redundancy).
    Randomly mean-ablates k% of MLP neurons or attention heads and measures 
    the degradation in Cross-Entropy loss over a random dataset. Generates a 
    degradation curve to evaluate model brittleness vs. redundancy.
    
    References:
    - "Investigating Neuron Ablation in Attention Heads: The Case for Peak Activation Centering" (2024)
    - "Causal Scrubbing: a method for rigorously testing interpretability hypotheses" (Chan et al., 2022)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Ablation Robustness Analysis...")
        num_samples = self.config.get("num_samples", 5)
        ablation_percentages = self.config.get("ablation_percentages", [0.01, 0.05, 0.1, 0.25])
        
        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)
        
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("NeelNanda/counterfact-tracing", split="train")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    # We just need text that the model can process and predict on.
                    # The prompt + target_true provides a factual statement.
                    dataset.append({"text": item["prompt"] + item["target_true"]})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {"text": "A quick brown fox jumps over the lazy dog."}
                ] * num_samples
        
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}
            
        encodings = []
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            # We want sequences of at least length 2 to compute loss
            if ids.shape[1] > 1:
                encodings.append(ids)
                
        if not encodings:
            return {"error": "No valid sequences to evaluate"}
            
        def get_loss(batch_ids):
            # Compute cross entropy loss
            outputs = model(batch_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss.item()
            
        # 1. Baseline Loss
        with torch.no_grad():
            baseline_losses = [get_loss(ids) for ids in encodings]
        baseline_mean_loss = np.mean(baseline_losses)
        
        # 2. Ablation Studies
        # We will ablate random units in the hidden states of intermediate layers
        # A simple approximation instead of finding specific MLP weight matrices:
        # We apply a mask to the output of intermediate layers.
        
        results = {"baseline_loss": float(baseline_mean_loss)}
        degradation_curve = []
        
        # We randomly ablate across the middle 50% of layers
        target_layers = list(range(num_layers // 4, 3 * num_layers // 4))
        if not target_layers:
            target_layers = [num_layers // 2]
            
        for k_pct in ablation_percentages:
            ablation_losses = []
            
            with torch.no_grad():
                for ids in encodings:
                    # We compute the mean activation first over the sequence
                    clean_out = model(ids, output_hidden_states=True)
                    clean_states = [h.detach() for h in clean_out.hidden_states]
                    
                    hooks = []
                    
                    # Create hooks for target layers
                    for l_idx in target_layers:
                        # Find the mean activation over the sequence length for this specific input
                        # (A true mean ablation would average over a dataset, but sequence-mean is a common surrogate)
                        h_state = clean_states[l_idx + 1] # shape: (batch, seq, dim)
                        dim = h_state.shape[-1]
                        
                        # Select k% random indices to ablate
                        num_ablate = max(1, int(dim * k_pct))
                        ablate_indices = torch.randperm(dim)[:num_ablate].to(device)
                        
                        seq_mean = h_state.mean(dim=1, keepdim=True) # shape: (batch, 1, dim)
                        
                        def get_ablation_hook(indices, mean_vals):
                            def hook(module, input, output):
                                out_tensor = (output[0] if isinstance(output, tuple) else output).clone()
                                # Replace selected features with their sequence-mean
                                out_tensor[..., indices] = mean_vals[..., indices]
                                if isinstance(output, tuple):
                                    return (out_tensor,) + output[1:]
                                return out_tensor
                            return hook
                            
                        hooks.append(layers[l_idx].register_forward_hook(get_ablation_hook(ablate_indices, seq_mean)))
                        
                    try:
                        loss = get_loss(ids)
                        ablation_losses.append(loss)
                    finally:
                        for h in hooks:
                            h.remove()
                            
            mean_ablation_loss = np.mean(ablation_losses)
            loss_increase = mean_ablation_loss - baseline_mean_loss
            
            # Record degradation
            results[f"loss_ablate_{int(k_pct*100)}pct"] = float(mean_ablation_loss)
            results[f"degradation_{int(k_pct*100)}pct"] = float(loss_increase)
            degradation_curve.append(float(loss_increase))
            
        # Summary metrics
        # If the curve grows very fast, the model is brittle.
        # If the curve grows slowly, the model is redundant.
        results["area_under_degradation_curve"] = float(np.trapezoid(degradation_curve, ablation_percentages))
        
        return results
