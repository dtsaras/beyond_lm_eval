
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm

@register_task("interpretability_attention_entropy")
class AttentionEntropyTask(DiagnosticTask):
    """
    Computes the entropy of attention distributions.
    Ref: Clark et al., "What Does BERT Look At?" (2019)
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Attention Entropy Analysis...")
        
        if dataset is None:
             dataset = [{"text": "The quick brown fox jumps over the lazy dog."}]
             
        # We need attention weights: (B, H, T, T)
        # Ensure model outputs attentions
        
        batch_size = self.config.get("batch_size", 1)
        num_samples = self.config.get("num_samples", 100)
        
        entropies = [] # List of (num_layers, num_heads) arrays
        
        with torch.no_grad():
            for i, sample in enumerate(tqdm(dataset, desc="Analyzing Attention")):
                if i >= num_samples: break
                
                if isinstance(sample, str):
                    inputs = tokenizer(sample, return_tensors="pt").to(model.device)
                else:
                    text = sample.get('text', '')
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                
                # Forward pass with attentions
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions # Tuple of (B, H, T, T) tensors, one per layer
                
                # attention[layer] shape: (B, H, T, T)
                # Compute entropy per head
                # H(p) = - sum p log p
                
                layer_entropies = []
                for layer_att in attentions:
                    if layer_att is None:
                        # SDPA/Flash attention doesn't return weights
                        return {"error": "Model does not return attention weights. Reload with attn_implementation='eager'."}
                    # Layer shape: (B, H, T, T)
                    # We compute entropy over the last dim (attention to other tokens)
                    # Avg over B and T (query tokens)
                    
                    # Add epsilon for log
                    atts = layer_att + 1e-9
                    entropy = -torch.sum(atts * torch.log(atts), dim=-1) # (B, H, T)
                    
                    # Avg over Batch and Query Tokens
                    avg_head_entropy = entropy.mean(dim=[0, 2]).cpu().numpy() # (H,)
                    layer_entropies.append(avg_head_entropy)
                    
                entropies.append(np.array(layer_entropies)) # (L, H)
        
        # Average over all samples
        if not entropies:
            return {"error": "No attentions computed"}
            
        avg_entropies = np.mean(np.stack(entropies), axis=0) # (L, H)
        
        # Aggregate results
        results = {
            "avg_entropy_per_layer": np.mean(avg_entropies, axis=1).tolist(),
            "avg_entropy_total": float(np.mean(avg_entropies)),
            "min_entropy_head": float(np.min(avg_entropies)),
            "max_entropy_head": float(np.max(avg_entropies)),
            # Return detailed map? Maybe too large.
            # "head_entropies": avg_entropies.tolist() 
        }
        
        return results
