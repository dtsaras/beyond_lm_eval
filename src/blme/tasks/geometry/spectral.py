
from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")

try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    _HFConv1D = None


@register_task("geometry_spectral")
class WeightSpectralTask(DiagnosticTask):
    """
    Analyzes the spectral properties of weight matrices.
    Metrics:
    - Stable Rank: ||W||_F^2 / ||W||_2^2 (Bartlett et al., 2020)
    - Power Law Alpha: Fit to singular value distribution (Martin & Mahoney, 2021)
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Weight Spectral Analysis...")

        # Configurable tail fraction for Hill estimator (default 20%)
        tail_fraction = self.config.get("tail_fraction", 0.2)

        results = {}
        layer_stats = {}

        # GPT-2 uses transformers.pytorch_utils.Conv1D (NOT torch.nn.Conv1d)
        # for its QKV/MLP projections. Skipping it degrades the analysis to
        # a single nn.Linear (lm_head) and makes std_alpha = 0.
        TARGET_MODULES = [torch.nn.Linear, torch.nn.Conv1d]
        if _HFConv1D is not None:
            TARGET_MODULES.append(_HFConv1D)
        TARGET_MODULES = tuple(TARGET_MODULES)

        alphas = []
        stable_ranks = []

        modules_to_scan = []
        for name, module in model.named_modules():
            if isinstance(module, TARGET_MODULES):
                if "weight" in module._parameters and module.weight is not None:
                    modules_to_scan.append((name, module))

        logger.info(f"  Found {len(modules_to_scan)} linear modules.")

        for name, module in tqdm(modules_to_scan, desc="Analyzing Weights"):
            W = module.weight.detach().float()

            if W.dim() != 2:
                continue

            try:
                S = torch.linalg.svdvals(W)
                S_np = S.cpu().numpy()

                # 1. Stable Rank: ||W||_F^2 / ||W||_2^2
                if len(S_np) > 0 and S_np[0] > 0:
                    fro_sq = np.sum(S_np ** 2)
                    spec_sq = S_np[0] ** 2
                    stable_rank = fro_sq / spec_sq
                else:
                    stable_rank = 0.0

                # 2. Power Law Alpha (Martin & Mahoney 2021)
                # Hill estimator on top tail_fraction of singular values.
                k = max(2, int(tail_fraction * len(S_np)))
                top_k = S_np[:k]

                # Hill Estimator: alpha = 1 + k / sum(ln(x_i / x_min))
                if k > 0 and top_k[-1] > 1e-6:
                    x_min = top_k[-1]
                    log_sum = np.sum(np.log(top_k / x_min))
                    if log_sum > 0:
                        alpha = float(np.clip(1 + k / log_sum, 0, 20))
                    else:
                        alpha = 0.0
                else:
                    alpha = 0.0

                layer_stats[name] = {
                    "stable_rank": float(stable_rank),
                    "alpha": float(alpha),
                    "spectral_norm": float(S_np[0]) if len(S_np) > 0 else 0,
                }

                alphas.append(alpha)
                stable_ranks.append(stable_rank)

            except Exception as e:
                logger.info(f"Error analyzing {name}: {e}")
                continue

        results["avg_alpha"] = float(np.mean(alphas)) if alphas else 0.0
        results["avg_stable_rank"] = float(np.mean(stable_ranks)) if stable_ranks else 0.0
        results["min_alpha"] = float(np.min(alphas)) if alphas else 0.0
        results["max_alpha"] = float(np.max(alphas)) if alphas else 0.0
        results["std_alpha"] = float(np.std(alphas)) if alphas else 0.0
        results["median_alpha"] = float(np.median(alphas)) if alphas else 0.0
        results["tail_fraction"] = tail_fraction

        return results
