from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_num_layers
import torch
import numpy as np

@register_task("interpretability_attribution")
class ComponentAttributionTask(DiagnosticTask):
    """
    Analyzes the coherence of layer updates (deltas) in the token space.
    Projects delta = h_out - h_in onto vocabulary and checks if top-k tokens are semantically related.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Component Attribution Analysis...")
        num_samples = self.config.get("num_samples", 50)
        
        device = next(model.parameters()).device
        E = get_embeddings(model)
        if E is None: return {"error": "Embeddings not found"}
        E = E.to(device)
        
        if dataset is None: dataset = [{"text": "Sample"}]
        
        coherence_scores = []
        
        # Detect layers (universal)
        n_layers = get_num_layers(model)
        if n_layers == 0:
            return {"error": "Could not detect layers"}

        # Analyze last few layers where semantics are richer
        start_layer = max(0, n_layers - 4)
        
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
                hidden_states = outputs.hidden_states  # (embed, layer0, ..., layerN)
                
                if len(hidden_states) <= start_layer + 1: continue
                
                for i in range(start_layer, n_layers - 1):
                    if i + 1 >= len(hidden_states): break
                    
                    h_prev = hidden_states[i]
                    h_curr = hidden_states[i+1]
                    
                    if h_prev.shape != h_curr.shape: continue
                    
                    delta = h_curr - h_prev
                    delta = delta[0]  # (T, D)
                    
                    limit = min(delta.shape[0], 10)
                    for pos in range(limit):
                        d_vec = delta[pos].float()
                        
                        sims = d_vec @ E.float().T
                        top_k_idx = sims.argsort(descending=True)[:5]
                        
                        E_top = E[top_k_idx].float()
                        E_top_norm = E_top / (E_top.norm(dim=1, keepdim=True) + 1e-10)
                        pair_sims = E_top_norm @ E_top_norm.T
                        
                        mask = ~torch.eye(5, dtype=bool, device=device)
                        coherence = pair_sims[mask].mean().item()
                        coherence_scores.append(coherence)
                        
        return {
            "component_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "component_coherence_std": float(np.std(coherence_scores)) if coherence_scores else 0.0
        }
