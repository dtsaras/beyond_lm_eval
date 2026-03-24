from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")

def _svd_metrics_for_layer(X):
    """Compute SVD-based isotropy metrics for a single (N, D) matrix."""
    X = X.float().numpy() if isinstance(X, torch.Tensor) else X
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    if len(X) < 10:
        return None
    X = X - np.mean(X, axis=0)

    try:
        _, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            _, S, _ = scipy_svd(X, full_matrices=False)
        except Exception:
            return None

    explained_variance = np.cumsum(S ** 2) / np.sum(S ** 2)
    auc = np.trapezoid(explained_variance) / max(1, len(explained_variance))

    p = S / (np.sum(S) + 1e-12)
    p = p[p > 1e-12]
    entropy_sv = -np.sum(p * np.log(p))
    effective_rank = float(np.exp(entropy_sv))

    eigenvalues = S ** 2
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    participation_ratio = float((sum_eig ** 2) / (sum_eig_sq + 1e-12))

    indices = np.random.choice(len(X), size=(min(1000, len(X)), 2), replace=True)
    vecs1 = X[indices[:, 0]]
    vecs2 = X[indices[:, 1]]
    norms1 = np.linalg.norm(vecs1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vecs2, axis=1, keepdims=True)
    vecs1 = vecs1 / (norms1 + 1e-9)
    vecs2 = vecs2 / (norms2 + 1e-9)
    cos_sims = np.sum(vecs1 * vecs2, axis=1)
    avg_cos_sim = float(np.mean(cos_sims))

    return {
        "svd_auc": float(auc),
        "cond_number": float(S[0] / S[-1]) if S[-1] > 0 else float("inf"),
        "avg_cosine_similarity": avg_cos_sim,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
    }


@register_task("geometry_svd")
class SVDIsotropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SVD Analysis...")
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(50)

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)
        layerwise = self.config.get("layerwise", False)

        # --- Per-layer mode (all layers) ---
        if layerwise and cache is not None and cache.is_populated and use_cache:
            all_layers = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
            if all_layers:
                per_layer = {}
                erank_per_layer = []
                pr_per_layer = []
                for li in sorted(all_layers.keys()):
                    m = _svd_metrics_for_layer(all_layers[li])
                    if m is not None:
                        per_layer[f"layer_{li}"] = m
                        erank_per_layer.append(m["effective_rank"])
                        pr_per_layer.append(m["participation_ratio"])
                    else:
                        erank_per_layer.append(float("nan"))
                        pr_per_layer.append(float("nan"))
                # Last-layer metrics as top-level for backward compat
                last_key = f"layer_{max(all_layers.keys())}"
                result = dict(per_layer.get(last_key, {}))
                result["layer_effective_rank"] = erank_per_layer
                result["layer_participation_ratio"] = pr_per_layer
                result["layer_metrics"] = per_layer
                return result

        # --- Single-layer mode (last layer, original behaviour) ---
        if cache is not None and cache.is_populated and use_cache:
            X = cache.get_hidden_states(layer_idx=-1, num_samples=num_samples)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=num_samples)

        m = _svd_metrics_for_layer(X)
        if m is None:
            return {"error": "Too few finite hidden states for SVD"}
        return m
