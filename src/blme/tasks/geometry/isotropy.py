from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
import logging
logger = logging.getLogger("blme")

@register_task("geometry_svd")
class SVDIsotropyTask(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running SVD Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        if cache is not None and cache.is_populated:
            X = cache.get_hidden_states(layer_idx=-1)
        else:
            X = collect_hidden_states(model, tokenizer, dataset, num_samples=self.config.get("num_samples", 100))
        X = X.float().numpy()
        # Filter NaN/Inf rows (can happen with fp16 models)
        finite_mask = np.all(np.isfinite(X), axis=1)
        if not np.all(finite_mask):
            logger.info(f"  Warning: Filtered {(~finite_mask).sum()} non-finite rows out of {len(X)}")
            X = X[finite_mask]
        if len(X) < 10:
            return {"error": "Too few finite hidden states for SVD"}
        X = X - np.mean(X, axis=0)
        
        try:
            U, S, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: try scipy's SVD which is often more robust
            try:
                from scipy.linalg import svd as scipy_svd
                U, S, Vh = scipy_svd(X, full_matrices=False)
            except Exception as e:
                return {"error": f"SVD failed: {e}"}
        singular_vals = S
        explained_variance = np.cumsum(singular_vals**2) / np.sum(singular_vals**2)
        
        # Calculate AUC of explained variance curve (lower = more isotropic)
        auc = np.trapezoid(explained_variance) / max(1, len(explained_variance))
        
        # Effective Rank (Roy & Vetterli, EUSIPCO 2007)
        # erank(X) = exp(H(p_1, ..., p_n)) where p_i = sigma_i / sum(sigma_j)
        p = singular_vals / (np.sum(singular_vals) + 1e-12)
        p = p[p > 1e-12]  # filter near-zero for log stability
        entropy_sv = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy_sv))
        
        # Participation Ratio (Gao et al., 2017)
        # PR = (sum lambda_i)^2 / sum(lambda_i^2), where lambda_i = sigma_i^2
        eigenvalues = singular_vals ** 2
        sum_eig = np.sum(eigenvalues)
        sum_eig_sq = np.sum(eigenvalues ** 2)
        participation_ratio = float((sum_eig ** 2) / (sum_eig_sq + 1e-12))
        
        # Cosine Anisotropy (Ethayarajh 2019)
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
