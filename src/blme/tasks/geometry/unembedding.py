from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_lm_head
import torch
import numpy as np
import json
import os

@register_task("geometry_unembedding")
class UnembeddingDiagnosticsTask(DiagnosticTask):
    """
    Analyzes the output embedding matrix (unembedding).
    Checks if weights are tied and measures effective rank and category purity of the unembedding space.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Unembedding Diagnostics...")
        n_sample = self.config.get("n_sample", 2000)
        k = self.config.get("k", 20)
        categories_path = self.config.get("categories_path", None)
        
        # Get Output Embeddings (W_out)
        head = get_lm_head(model)
        if head is not None:
            W_out = head.weight.detach()
        else:
            # If no head, try input embeddings (tied weights)
            W_out = get_embeddings(model)
            if W_out is None:
                return {"error": "Could not find lm_head or output embeddings"}
            
        device = W_out.device
        W_out_np = W_out.float().cpu().numpy()
        
        # Get Input Embeddings (E_in)
        E_in = get_embeddings(model)
        is_tied = False
        if E_in is not None:
            E_in_np = E_in.float().cpu().numpy()
            if E_in_np.shape == W_out_np.shape:
                is_tied = np.allclose(W_out_np, E_in_np, atol=1e-5)
        
        # Effective Rank
        W_centered = W_out_np - W_out_np.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
            S_norm = S / S.sum()
            eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))
        except:
            eff_rank = 0.0
            
        # Category Purity
        cat_labels = {} 
        if not categories_path:
             candidate = os.path.join(os.path.dirname(__file__), "../../assets/categories.json")
             if os.path.exists(candidate):
                 categories_path = candidate
        
        purity_mean = 0.0
        if categories_path and os.path.exists(categories_path):
            try:
                with open(categories_path, 'r') as f:
                    categories = json.load(f)
                
                for cat, words in categories.items():
                    for w in words:
                        ids = tokenizer.encode(w, add_special_tokens=False)
                        if len(ids) == 1:
                            cat_labels[ids[0]] = cat
                        ids_sp = tokenizer.encode(" " + w, add_special_tokens=False)
                        if len(ids_sp) == 1:
                            cat_labels[ids_sp[0]] = cat
                            
                n_vocab = len(W_out_np)
                W_norm = W_out_np / (np.linalg.norm(W_out_np, axis=1, keepdims=True) + 1e-10)
                
                np.random.seed(42)
                sample_idx = np.random.choice(n_vocab, min(n_sample, n_vocab), replace=False)
                
                scores = []
                for idx in sample_idx:
                    if idx not in cat_labels: continue
                    
                    my_cat = cat_labels[idx]
                    sims = W_norm @ W_norm[idx]
                    sims[idx] = -np.inf
                    top_k_idx = np.argsort(sims)[-k:]
                    
                    match_count = sum(1 for t in top_k_idx if cat_labels.get(t) == my_cat)
                    scores.append(match_count / k)
                    
                if scores:
                    purity_mean = np.mean(scores)
                    
            except Exception as e:
                print(f"Error computing purity: {e}")

        return {
            "unembedding_is_tied": is_tied,
            "unembedding_eff_rank": float(eff_rank),
            "unembedding_purity_mean": float(purity_mean)
        }
