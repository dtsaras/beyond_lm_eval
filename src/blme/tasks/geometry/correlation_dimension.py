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


def _gp_correlation_dimension(distances, num_radii=30, rvals=None):
    """Grassberger-Procaccia (1983) correlation dimension from a 1-D array of
    pairwise distances (upper triangle, i<j).

    Builds the correlation integral C(r) = fraction of pairs with distance < r
    over a percentile-spaced (5th..95th) log radius grid, then returns the slope
    of log C(r) vs log r (the GP dimension estimate).

    Returns ``(slope, r_squared, status)`` where ``status`` is ``"ok"``,
    ``"degenerate"`` (non-positive / collapsed percentile range) or
    ``"insufficient"`` (<3 usable radii); ``slope``/``r_squared`` are ``None``
    unless ``"ok"``. ``rvals`` overrides the percentile grid (used by the
    reference-parity test to compare on an identical radius grid).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if rvals is None:
        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)
        if r_min <= 0 or r_max <= 0 or r_min >= r_max:
            return None, None, "degenerate"
        radii = np.logspace(np.log10(r_min), np.log10(r_max), num=num_radii)
    else:
        radii = np.asarray(rvals, dtype=np.float64)

    total_pairs = len(distances)
    C_r, valid_radii = [], []
    for r in radii:
        c = np.sum(distances < r) / total_pairs
        if c > 0:
            C_r.append(c)
            valid_radii.append(r)

    if len(valid_radii) < 3:
        return None, None, "insufficient"

    log_r = np.log(valid_radii)
    log_Cr = np.log(C_r)
    slope, intercept = np.polyfit(log_r, log_Cr, 1)
    predicted = slope * log_r + intercept
    ss_res = np.sum((log_Cr - predicted) ** 2)
    ss_tot = np.sum((log_Cr - np.mean(log_Cr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r_squared), "ok"


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

        # 2-3. Grassberger-Procaccia: correlation integral C(r) -> log-log slope.
        # Extracted into _gp_correlation_dimension so the reference-parity test
        # exercises BLME's real kernel (not a transcription).
        slope, r_squared, status = _gp_correlation_dimension(distances, num_radii=num_radii)
        if status == "degenerate":
            return {"error": "All distances are identical or degenerate. State totally collapsed."}
        if status == "insufficient":
            return {"error": "Failed to compute correlation integral across scales."}

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
