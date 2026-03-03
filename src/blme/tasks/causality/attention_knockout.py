import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")

@register_task("causality_attention_knockout")
class AttentionKnockoutTask(DiagnosticTask):
    """
    Measures Attention Specialization using Attention Head Knockouts.
    Independently knocks out specific attention heads and records the 
    resulting drop in performance. Calculates the Gini Coefficient of the 
    performance drops. High Gini = High Specialization (few heads do all the work).
    Low Gini = High Polysemanticity/Distribution.
    
    References:
    - "Mechanisms of Prompt-Induced Hallucination in Vision-Language Models" (2026) 
      (for targeted attention head knockouts via mean ablation).
    - "Lost in the Prompt Order: Revealing the Limitations of Causal Attention..." (2026)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Knockout Specialization...")
        num_samples = self.config.get("num_samples", 3)
        
        device = next(model.parameters()).device
        layers = list(get_layers(model))
        num_layers = len(layers)
        
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("fahamu/ioi", split="train")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    dataset.append({"text": item["text"]})
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {"text": "John gave a book to Mary. Mary gave a pencil to"}
                ] * num_samples
            
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}
            
        encodings = []
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            if ids.shape[1] > 2:
                encodings.append(ids)
                
        if not encodings:
            return {"error": "No valid sequences"}
            
        def get_loss(batch_ids):
            outputs = model(batch_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss.item()
            
        # Baseline
        with torch.no_grad():
            baseline_losses = [get_loss(ids) for ids in encodings]
        baseline_mean_loss = np.mean(baseline_losses)
        
        # Determine number of heads (Architecture agnostic heuristic)
        # Try to inspect model config
        num_heads = getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", None))
        
        if num_heads is None:
            return {"error": "Could not determine the number of attention heads from model.config"}
            
        # We will ablate heads across a sampled set of layers to save time
        if num_layers > 10:
            target_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        else:
            target_layers = list(range(num_layers))
            
        knockout_impacts = []
        
        with torch.no_grad():
            for l_idx in target_layers:
                layer = layers[l_idx]
                
                # Attempt to find the specific attention module to avoid ablating MLP outputs
                attn_module = layer
                for name, module in layer.named_modules():
                    if "attn" in name.lower() or "attention" in name.lower():
                        attn_module = module
                        break
                        
                hidden_size = model.config.hidden_size
                head_size = hidden_size // num_heads
                
                for h_idx in range(num_heads):
                    start_dim = h_idx * head_size
                    end_dim = (h_idx + 1) * head_size
                    
                    def get_head_knockout_hook(start, end):
                        def hook(module, input, output):
                            out_tensor = output[0] if isinstance(output, tuple) else output
                            # Clone to avoid in-place modification errors if any
                            patched = out_tensor.clone()
                            patched[..., start:end] = 0.0
                            if isinstance(output, tuple):
                                return (patched,) + output[1:]
                            return patched
                        return hook
                        
                    hook = attn_module.register_forward_hook(get_head_knockout_hook(start_dim, end_dim))
                    
                    try:
                        losses = [get_loss(ids) for ids in encodings]
                        impact = np.mean(losses) - baseline_mean_loss
                        knockout_impacts.append(impact)
                    finally:
                        hook.remove()
                        
        # Ensure we only have meaningful impacts
        knockout_impacts = np.array(knockout_impacts)
        # ReLU out negative impacts (where knockout improved performance, usually noise)
        knockout_impacts = np.maximum(0, knockout_impacts)
        
        # Calculate Gini Coefficient of the impacts
        def gini(array):
            if len(array) == 0 or np.sum(array) == 0:
                return 0.0
            array = np.sort(array)
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return float((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

        results = {
            "baseline_loss": float(baseline_mean_loss),
            "max_knockout_impact": float(np.max(knockout_impacts)),
            "mean_knockout_impact": float(np.mean(knockout_impacts)),
            "head_impact_gini_coefficient": gini(knockout_impacts)
        }
        
        return results
