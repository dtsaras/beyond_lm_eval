"""
Positional Attention Decay (RoPE Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates the geometric degradation of relative positional embeddings
(like RoPE) by computing the Spearman rank correlation between the
relative discrete distance of two tokens and the magnitude of their 
attention connection.

A structurally intact context window will show a strong negative correlation 
(closer tokens have higher attention in positional/local heads). If this 
correlation collapses, the model's positional geometry is degraded.

References:
- 2024-2025 Long-Context Extrapolation and RoPE literature.
"""

import torch
import numpy as np
from scipy.stats import spearmanr

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


@register_task("geometry_positional_decay")
class PositionalAttentionDecayTask(DiagnosticTask):
    """
    Computes the Spearman correlation between target-source token distance 
    and attention probability to measure the structural integrity of 
    positional encodings.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Positional Attention Decay Analysis...")
        num_samples = self.config.get("num_samples", 5)
        
        # We need reasonably long sequences to measure positional decay gracefully
        if dataset is None:
             text = "The quick brown fox jumps over the lazy dog. " * 10
             dataset = [{"text": text}] * num_samples
             
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        
        correlations = []
        
        # Run forward passes, requesting output_attentions so we don't have to hook
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                # Need sequence length > 4 to measure correlation properly
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                
                seq_len = inputs.input_ids.shape[1]
                if seq_len < 4:
                   continue
                   
                out = model(**inputs, output_attentions=True)
                if out.attentions is None or len(out.attentions) == 0:
                    return {"error": "Model did not return attentions. Cannot compute Positional Decay."}
                    
                # attentions shape: (num_layers, batch, num_heads, seq_len, seq_len)
                # We analyze the middle layer as a representative heuristic for structure
                mid_layer = len(out.attentions) // 2
                attn_entry = out.attentions[mid_layer]
                if attn_entry is None:
                    return {"error": "Attention weights at selected layer are None. Model may use SDPA."}
                attn_matrix = attn_entry[0] # (num_heads, seq_len, seq_len)
                
                # Average attention across heads to get the macro positional structure
                mean_attn = attn_matrix.mean(dim=0).cpu().numpy() # (seq_len, seq_len)
                
                # We want to correlate (i - j) distance with mean_attn[i, j] for all valid j < i
                distances = []
                attentions = []
                
                for i in range(1, seq_len):
                    for j in range(i):
                        distances.append(i - j)
                        attentions.append(mean_attn[i, j])
                        
                # Compute Spearman correlation
                # We expect a negative correlation (higher distance = lower attention)
                # so a perfectly structured local window is -1.0. 
                if len(distances) > 5:
                    corr, _ = spearmanr(distances, attentions)
                    if not np.isnan(corr):
                         correlations.append(corr)
                         
        if not correlations:
             return {"error": "Could not compute positional correlations (sequences too short or nan)."}
             
        mean_corr = float(np.mean(correlations))
        
        return {
            "mean_positional_decay_correlation": mean_corr,
            "interpretation": "Strong negative values (e.g. -0.5 to -1.0) indicate structurally sound local positional geometry."
        }
