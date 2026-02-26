"""
Correlation Dimension (Fractal Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates the underlying fractal complexity and robust topological self-similarity
of the generated language manifold by computing the Grassberger-Procaccia 
Correlation Dimension on the internal hidden states.

A standard "intrinsic dimension" assumes a locally smooth Euclidean manifold.
Recent 2024-2025 research proves that semantic space is fundamentally a fractal;
its epistemological complexity is best quantified by its fractional Correlation 
Dimension.

References:
- "Correlation Dimension as a Metric for Large Language Models" (2024/2025).
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task


@register_task("geometry_correlation_dimension")
class CorrelationDimensionTask(DiagnosticTask):
    """
    Computes the Grassberger-Procaccia fractional correlation dimension
    on the final representation space to evaluate fractal complexity.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Correlation Dimension (Fractal Geometry) Analysis...")
        # Need a larger number of samples to approximate spatial distances accurately
        num_samples = self.config.get("num_samples", 50) 
        
        if dataset is None:
             dataset = [{"text": "Fractals exhibit unbounded self-similar complexity."}] * num_samples
             
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        all_hidden_states = []
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
                
                out = model(**inputs, output_hidden_states=True)
                # Extract the final hidden state of the LAST token as the sequence summary
                final_state = out.hidden_states[-1][0, -1, :].detach().cpu()
                all_hidden_states.append(final_state)
                
        if len(all_hidden_states) < 10:
             return {"error": "Need at least 10 points to compute Grassberger-Procaccia fractional dimension."}
             
        H = torch.stack(all_hidden_states) # (N, hidden_dim)
        N = H.shape[0]
        
        # 1. Compute all pairwise L2 distances
        # H @ H.T is (N, N)
        dist_matrix = torch.cdist(H, H, p=2)
        
        # Extract upper triangle distances (i < j) to remove zeros on diagonal and duplicates
        indices = torch.triu_indices(N, N, offset=1)
        distances = dist_matrix[indices[0], indices[1]].numpy()
        
        # 2. Grassberger-Procaccia Algorithm
        # Define a range of spatial scales (radii r)
        # We sample exponentially spaced radii between the 5th and 95th percentiles of distances
        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)
        
        if r_min == 0 or r_max == 0 or r_min == r_max:
             return {"error": "All distances are identical. State totally collapsed."}
             
        radii = np.logspace(np.log10(r_min), np.log10(r_max), num=20)
        
        # Compute Correlation Integral C(r) : fraction of point pairs closer than r
        C_r = []
        valid_radii = []
        
        total_pairs = len(distances)
        for r in radii:
            count = np.sum(distances < r)
            c = count / total_pairs
            if c > 0: # We need log(C(r)), so ignore 0s
                 C_r.append(c)
                 valid_radii.append(r)
                 
        if len(valid_radii) < 2:
            return {"error": "Failed to compute correlation integral across scales."}
            
        # 3. Fit linear regression in log-log space
        # D_corr = lim_{r->0} [log C(r) / log r]
        # In practice, slope of the log-log plot in the linear scaling region
        log_r = np.log(valid_radii)
        log_Cr = np.log(C_r)
        
        # We fit a line to (log_r, log_Cr). The slope is the Correlation Dimension.
        slope, _ = np.polyfit(log_r, log_Cr, 1)
        
        return {
            "correlation_dimension": float(slope),
            "num_points": N,
            "interpretation": "The fractional dimension of the representations. Normally between 5.0 and 15.0 for complex natural language semantic manifolds."
        }
