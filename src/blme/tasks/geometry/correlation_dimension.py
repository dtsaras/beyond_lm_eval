"""
Correlation Dimension (Hidden-State Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates hidden-state point-cloud complexity by computing the
Grassberger-Procaccia correlation dimension on final-layer representations.

A standard "intrinsic dimension" assumes a locally smooth Euclidean manifold.
This task is explicitly a hidden-state GP estimator, not a log-probability
trajectory diagnostic.

References:
- Grassberger & Procaccia 1983 correlation integral estimator.
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


@register_task("geometry_correlation_dimension")
class CorrelationDimensionTask(DiagnosticTask):
    """
    Computes the Grassberger-Procaccia fractional correlation dimension
    on the final representation space to evaluate fractal complexity.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Hidden-State Correlation Dimension Analysis...")
        num_samples = self.config.get("num_samples", 100)
        max_length = self.config.get("max_length", 128)
        num_radii = self.config.get("num_radii", 30)
        # Use mean-pooled representations by default; set to "last" for
        # last-token only (original behaviour).
        pooling = self.config.get("pooling", "mean")
        # Cap total points for pairwise distance computation
        max_points = self.config.get("max_points", 2000)
        seed = int(self.config.get("seed", 42))

        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device
        all_hidden_states = []

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

                out = model(**inputs, output_hidden_states=True)
                # Use all token positions to get more data points
                last_hidden = out.hidden_states[-1][0].detach().cpu()  # (T, D)
                if pooling == "mean":
                    # Each sample contributes one mean-pooled vector
                    all_hidden_states.append(last_hidden.mean(dim=0))
                elif pooling == "all_tokens":
                    # Each token is a separate data point (many more points)
                    all_hidden_states.extend(list(last_hidden))
                else:  # "last"
                    all_hidden_states.append(last_hidden[-1])

        if len(all_hidden_states) < 20:
             return {"error": "Need at least 20 points for Grassberger-Procaccia correlation dimension."}

        H = torch.stack(all_hidden_states)  # (N, D)
        N = H.shape[0]

        # Subsample if too many points (pairwise distances are O(N^2))
        if N > max_points:
            rng = np.random.default_rng(seed)
            indices = rng.choice(N, max_points, replace=False)
            H = H[indices]
            N = max_points

        # 1. Compute all pairwise L2 distances
        dist_matrix = torch.cdist(H, H, p=2)

        # Extract upper triangle distances (i < j)
        tri_indices = torch.triu_indices(N, N, offset=1)
        # .float() so bf16 models (Gemma 4 etc.) don't crash on .numpy()
        distances = dist_matrix[tri_indices[0], tri_indices[1]].float().cpu().numpy()

        # 2. Grassberger-Procaccia Algorithm
        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)

        if r_min <= 0 or r_max <= 0 or r_min >= r_max:
             return {"error": "All distances are identical or degenerate. State totally collapsed."}

        radii = np.logspace(np.log10(r_min), np.log10(r_max), num=num_radii)

        # Compute Correlation Integral C(r): fraction of point pairs closer than r
        C_r = []
        valid_radii = []

        total_pairs = len(distances)
        for r in radii:
            count = np.sum(distances < r)
            c = count / total_pairs
            if c > 0:
                 C_r.append(c)
                 valid_radii.append(r)

        if len(valid_radii) < 3:
            return {"error": "Failed to compute correlation integral across scales."}

        # 3. Fit linear regression in log-log space
        log_r = np.log(valid_radii)
        log_Cr = np.log(C_r)

        slope, intercept = np.polyfit(log_r, log_Cr, 1)

        # Compute R^2 of the fit as a quality indicator
        predicted = slope * log_r + intercept
        ss_res = np.sum((log_Cr - predicted) ** 2)
        ss_tot = np.sum((log_Cr - np.mean(log_Cr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if pooling == "all_tokens":
            point_space = "token_hidden_states"
        elif pooling == "last":
            point_space = "last_token_hidden_states"
        else:
            point_space = "mean_pooled_hidden_states"

        return {
            "correlation_dimension": float(slope),
            "hidden_state_correlation_dimension": float(slope),
            "correlation_dimension_method": "hidden_state_grassberger_procaccia",
            "correlation_dimension_point_space": point_space,
            "num_points": N,
            "fit_r_squared": float(r_squared),
        }
