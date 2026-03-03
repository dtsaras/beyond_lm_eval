from ...tasks.base import DiagnosticTask
from ...registry import register_task
from .utils import collect_hidden_states
import numpy as np
import torch
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")


@register_task("geometry_collapse")
class RepresentationCollapseTask(DiagnosticTask):
    """
    Detects representation collapse by tracking Effective Rank across layers.
    A sharp drop in effective rank indicates dimensional collapse.
    Ref: Jing et al., "Understanding Dimensional Collapse in Contrastive
         Self-supervised Learning", ICLR 2021. arXiv:2011.09348
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Collapse Detection...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)

        # Collect hidden states from all layers
        if cache is not None and cache.is_populated:
            layer_activations = cache.get_hidden_states(layer_idx="all")
        else:
            layer_activations = collect_hidden_states(
            model, tokenizer, dataset, num_samples=num_samples, layer_idx="all"
        )

        layers = sorted(layer_activations.keys())
        erank_per_layer = []

        for idx in tqdm(layers, desc="Computing Effective Rank per Layer"):
            X = layer_activations[idx].float().numpy()
            # Filter NaN/Inf rows (fp16 models may produce extreme values)
            finite_mask = np.all(np.isfinite(X), axis=1)
            if not np.all(finite_mask):
                X = X[finite_mask]
            if len(X) < 5:
                erank_per_layer.append(0.0)
                continue
            X = X - np.mean(X, axis=0)

            try:
                S = np.linalg.svd(X, compute_uv=False)
            except np.linalg.LinAlgError:
                try:
                    from scipy.linalg import svdvals
                    S = svdvals(X)
                except Exception:
                    erank_per_layer.append(0.0)
                    continue

            # Effective Rank (Roy & Vetterli 2007)
            p = S / (np.sum(S) + 1e-12)
            p = p[p > 1e-12]
            entropy_sv = -np.sum(p * np.log(p))
            erank = float(np.exp(entropy_sv))
            erank_per_layer.append(erank)

        # Detect collapse: ratio of min erank to max erank
        erank_arr = np.array(erank_per_layer)
        max_erank = float(np.max(erank_arr))
        min_erank = float(np.min(erank_arr))
        collapse_ratio = min_erank / (max_erank + 1e-12)

        # Largest single-layer drop
        diffs = np.diff(erank_arr)
        max_drop = float(np.min(diffs)) if len(diffs) > 0 else 0.0
        max_drop_layer = int(np.argmin(diffs)) + 1 if len(diffs) > 0 else -1

        return {
            "erank_per_layer": erank_per_layer,
            "max_erank": max_erank,
            "min_erank": min_erank,
            "collapse_ratio": collapse_ratio,
            "max_drop": max_drop,
            "max_drop_layer": max_drop_layer,
        }
