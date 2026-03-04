from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_layers, apply_lm_head
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")

@register_task("interpretability_logit_lens")
class LogitLensTask(DiagnosticTask):
    """
    Decodes hidden states at each layer using the final LM head (Logit Lens).
    Computes accuracy of intermediate layers relative to the final prediction.

    Caveat: Assumes the residual stream is directly interpretable through the
    unembedding matrix at every layer. This may not hold when features are
    stored in superposition or when intermediate layer norms differ
    significantly from the final layer norm.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Logit Lens Analysis...")
        num_samples = self.config.get("num_samples", 100)
        
        device = next(model.parameters()).device
        
        if dataset is None:
             dataset = [{"text": "Sample text for logit lens."} for _ in range(5)]
             
        # Detect layers (universal)
        layers = get_layers(model)
        if layers is None:
            # Fallback: use config
            from ..common import get_num_layers
            n_layers = get_num_layers(model)
            if n_layers == 0:
                return {"error": "Could not detect layers"}
        else:
            n_layers = len(layers)
        
        layer_accs = {i: [] for i in range(n_layers)}
        layer_entropies = {i: [] for i in range(n_layers)}
        
        count = 0
        with torch.no_grad():
            for sample in dataset:
                if count >= num_samples: break
                
                if isinstance(sample, str):
                    inputs = tokenizer(sample, return_tensors="pt").to(device)
                elif isinstance(sample, dict) and 'text' in sample:
                    inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, max_length=128).to(device)
                elif 'input_ids' in sample:
                     inputs = {'input_ids': torch.tensor(sample['input_ids']).long().unsqueeze(0).to(device)}
                else: continue
                count += 1
                
                outputs = model(**inputs, output_hidden_states=True)
                final_preds = outputs.logits[0].argmax(dim=-1)
                
                # Hidden states: [embed, layer1, layer2 ... layerN]
                hidden_states = outputs.hidden_states
                start_idx = 1 if len(hidden_states) > n_layers else 0
                
                for i in range(n_layers):
                    if start_idx + i >= len(hidden_states): break
                    
                    h = hidden_states[start_idx + i][0]  # (T, D)
                    
                    try:
                        logits = apply_lm_head(model, h)
                    except RuntimeError:
                        continue
                        
                    preds = logits.argmax(dim=-1)
                    acc = (preds == final_preds).float().mean().item()
                    layer_accs[i].append(acc)
                    
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                    layer_entropies[i].append(entropy)
                    
        results = {}
        for i in range(n_layers):
            if layer_accs[i]:
                results[f"layer{i}_acc"] = float(np.mean(layer_accs[i]))
                results[f"layer{i}_entropy"] = float(np.mean(layer_entropies[i]))
                
        return results
