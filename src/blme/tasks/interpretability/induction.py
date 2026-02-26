
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
import random
from tqdm import tqdm

@register_task("interpretability_induction_heads")
class InductionHeadTask(DiagnosticTask):
    """
    Identifies induction heads by measuring their ability to copy the token 
    that followed a previous occurrence of the current token.
    Ref: Olsson et al., "In-context Learning and Induction Heads" (2022)
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Induction Head Analysis...")
        
        # We need a synthetic dataset of repeated random tokens to isolate induction behavior
        # "A B ... A B" pattern.
        # Construct random sequences of tokens.
        
        vocab_size = tokenizer.vocab_size
        seq_len = self.config.get("seq_len", 30) # Short sequence
        num_samples = self.config.get("num_samples", 20)
        
        scores = [] # (L, H)
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Analyzing Heads"):
                # Generate random sequence
                rand_tokens = torch.randint(0, vocab_size, (1, seq_len))
                
                # Repeat it: [A B C ... A B C ...]
                input_ids = torch.cat([rand_tokens, rand_tokens], dim=1).to(model.device)
                
                # Forward pass
                outputs = model(input_ids, output_attentions=True)
                attentions = outputs.attentions # (L, B, H, T, T)
                
                if attentions is None or attentions[0] is None:
                    return {"error": "Model does not return attention weights. Reload with attn_implementation='eager'."}
                
                # We analyze the second half of the sequence (the repetition)
                # For a token at pos `i` (in 2nd half), we check if it attends to `i - seq_len + 1`?
                # No. Induction head: content at `i` matches content at `j`. 
                # Head at `i+1` (next token prediction) should attend to `j+1`.
                # Here we are looking at attention *at* token `i`.
                # If current token is X (at pos i), and previous X was at pos j.
                # Induction head at `i` should attend to `j+1`.
                
                # In our repeated sequence:
                # Sequence 1: 0 to N-1
                # Sequence 2: N to 2N-1
                # Token at `k` (where k >= N) corresponds to token at `k-N`.
                # Previous occurrence of token `input_ids[k]` is at `k-N`.
                # We want to predict `input_ids[k+1]`.
                # So head at `k` should attend to `(k-N) + 1`.
                
                # Wait, standard definition:
                # Induction head attends to the token *after* the previous copy of the current token.
                # Current token is `input_ids[k]`. Previous copy is `input_ids[j]`.
                # Head at `k` should attend to `j+1`.
                # In our setup: `k` is in [N, 2N-2].
                # `input_ids[k] == input_ids[k-N]`.
                # We want head at `k` to attend to `(k-N) + 1`.
                
                T_total = input_ids.shape[1]
                N = seq_len
                
                sample_scores = []
                
                for layer_idx, layer_att in enumerate(attentions):
                    # layer_att: (B, H, T, T)
                    # Squeeze batch
                    att = layer_att[0] # (H, T, T)
                    
                    # We only care about queries in the second half
                    # From N to 2N-2 (last token 2N-1 has no next token in this tensor usually, or it does?)
                    # Attention matrix is TxT.
                    
                    head_scores = []
                    for h in range(att.shape[0]):
                        # Compute score for this head
                        # For each position k in [N, 2N-2]:
                        # Target attention index = (k - N) + 1
                        # Check attention weight `att[h, k, target_idx]`
                        
                        induction_score = 0.0
                        count = 0
                        
                        for k in range(N, 2*N - 1):
                            target_idx = (k - N) + 1
                            if target_idx < k: # Causal mask check
                                weight = att[h, k, target_idx].item()
                                induction_score += weight
                                count += 1
                                
                        avg_score = induction_score / max(1, count)
                        head_scores.append(avg_score)
                    
                    sample_scores.append(head_scores)
                
                scores.append(np.array(sample_scores)) # (L, H)
                
        # Average over samples
        if not scores:
            return {"error": "No scores computed"}
            
        avg_scores = np.mean(np.stack(scores), axis=0) # (L, H)
        
        top_heads_indices = np.unravel_index(np.argsort(avg_scores, axis=None)[::-1][:5], avg_scores.shape)
        top_heads = []
        for i in range(5):
            l = top_heads_indices[0][i]
            h = top_heads_indices[1][i]
            top_heads.append(f"L{l}H{h}: {avg_scores[l, h]:.4f}")
            
        return {
            "max_induction_score": float(np.max(avg_scores)),
            "avg_induction_score": float(np.mean(avg_scores)),
            "top_induction_heads": top_heads
        }
