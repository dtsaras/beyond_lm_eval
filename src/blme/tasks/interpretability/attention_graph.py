"""
Attention Graph Topology Task
──────────────────────────────────────────────────────────────────────
Treats the N×N attention matrix at each head as a directed graph where tokens
are nodes and attention weights are edges. We compute structural graph metrics 
to understand the macroscopic organization of attention:

1. PageRank Centrality: Identifies "attention sinks" (nodes that disproportionately 
   gather attention flow, e.g., the first token or punctuation).
2. Edge Density / Sparsity: How distributed vs focused the graph is.
3. Graph Assortativity (Optional): Tendency of nodes to connect to similar nodes.

References:
- "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023)
"""

import numpy as np
import torch
import warnings

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers
import logging
logger = logging.getLogger("blme")


def _power_iteration_pagerank(adj_matrix, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Computes PageRank centrality using power iteration.
    
    Args:
        adj_matrix: NumPy array (N, N) where A[i, j] is attention from token i to token j.
                    (Rows typically sum to 1 before any damping).
        alpha: Damping factor (probability of following an edge vs random teleporting).
        
    Returns:
        NumPy array (N,) of PageRank scores.
    """
    N = adj_matrix.shape[0]
    
    # Handle potentially all-zero rows (dead nodes) to prevent NaN
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    # If a row sums to zero, it should teleport uniformly
    transition_matrix = np.divide(adj_matrix, row_sums, out=np.ones_like(adj_matrix)/N, where=row_sums!=0)
    
    # Pagerank transition
    M = alpha * transition_matrix + (1 - alpha) / N * np.ones((N, N))
    
    # Initialize uniform distribution
    v = np.ones(N) / N
    
    for _ in range(max_iter):
        v_next = v @ M # left-multiply since transitions are from row to column
        if np.linalg.norm(v_next - v, ord=1) < tol:
            return v_next
        v = v_next
        
    return v


@register_task("interpretability_attention_graph")
class AttentionGraphTopologyTask(DiagnosticTask):
    """
    Analyzes the macroscopic graph structure of attention matrices.
    
    Metrics collected per head (averaged across samples):
    - Sink Centrality: The maximum PageRank score in the graph (how much does 
      the graph collapse into a sink token?)
    - Sink Token Bias: How often the absolute maximum PageRank belongs to token 0
      (the classic BOS attention sink).
    - Edge Gini: Using Gini coefficient on edge weights to measure sparsity.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Attention Graph Topology Analysis...")
        if dataset is None:
            dataset = [
                {"text": "Attention is a graph. In this graph, some tokens acts as sinks. This is a robust mechanism."}
                for _ in range(10)
            ]
        num_samples = self.config.get("num_samples", 10)
        samples = list(dataset)[:num_samples]
        
        device = next(model.parameters()).device
        
        layers = get_layers(model)
        num_layers = len(layers)
        
        agg_metrics = {
            "max_pageranks": [],
            "sink_is_bos": [],
            "edge_ginis": []
        }
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                
                # We need output_attentions=True to get the graph edges
                out = model(**inputs, output_attentions=True)
                attentions = out.attentions # Tuple of (batch, num_heads, seq_len, seq_len)
                
                if attentions is None:
                    return {"error": "Model does not return attention weights. Reload with attn_implementation='eager'."}
                
                seq_len = inputs['input_ids'].shape[1]
                if seq_len < 3:
                     continue
                     
                # Take the attention matrices from the last layer as representative,
                # or average across all layers for a global view. Let's compute globally.
                for l_idx, layer_attn in enumerate(attentions):
                    if layer_attn is None:
                        continue
                    # layer_attn shape: (1, num_heads, seq_len, seq_len)
                    layer_attn = layer_attn[0].cpu().numpy() # (num_heads, seq_len, seq_len)
                    num_heads = layer_attn.shape[0]
                    
                    for h_idx in range(num_heads):
                        adj = layer_attn[h_idx] # (seq_len, seq_len)
                        
                        # Compute PageRank
                        pr = _power_iteration_pagerank(adj)
                        max_pr = np.max(pr)
                        sink_idx = np.argmax(pr)
                        
                        agg_metrics["max_pageranks"].append(max_pr)
                        agg_metrics["sink_is_bos"].append(1 if sink_idx == 0 else 0)
                        
                        # Edge Gini (Sparsity of the complete graph)
                        flat_edges = np.sort(adj.flatten())
                        cum_edges = np.cumsum(flat_edges)
                        # Gini formula
                        n = len(flat_edges)
                        gini = (n + 1 - 2 * np.sum(cum_edges) / cum_edges[-1]) / n
                        agg_metrics["edge_ginis"].append(gini)
                        
        if not agg_metrics["max_pageranks"]:
             return {"error": "Sequence lengths too short or no samples."}
             
        # Aggregate the topology statistics
        return {
            "mean_sink_pagerank": float(np.mean(agg_metrics["max_pageranks"])),
            "max_sink_pagerank": float(np.max(agg_metrics["max_pageranks"])),
            "bos_sink_ratio": float(np.mean(agg_metrics["sink_is_bos"])),
            "mean_edge_gini": float(np.mean(agg_metrics["edge_ginis"])),
            "num_graphs_analyzed": len(agg_metrics["max_pageranks"])
        }
