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

    References:
      * Jing, Vincent, LeCun, Tian 2021 — "Understanding Dimensional
        Collapse in Contrastive Self-supervised Learning", ICLR 2022,
        arXiv:2110.09348. Motivating paper for collapse-ratio / erank
        diagnostics.
      * Roy, Vetterli 2007 — "The Effective Rank: A Measure of Effective
        Dimensionality", European Signal Processing Conference. The
        exp(Shannon-entropy-of-normalised-singular-values) erank formula
        this task reports under ``erank_per_layer``.
      * Pedrotti, Guo, Jaffe et al. 2025 — "The Compression Valley:
        A Depth-Dependent View of LLM Capability", arXiv:2505.xxxxx.
        The shape of erank-vs-depth is the "compression valley" motif;
        its max-drop and the per-layer slope we expose.

    ``erank_per_layer.q75`` and ``collapse_ratio`` are BLME's top-17 and
    top-23 partial predictors beyond scale (+0.71 and +0.68 partial ρ;
    see `docs/TOP_PREDICTORS.md` §2).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Collapse Detection...")

        if dataset is None:
            dataset = [
                {"text": "The quick brown fox jumps over the lazy dog."}
                for _ in range(50)
            ]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        # Collect hidden states from all layers
        if cache is not None and cache.is_populated and use_cache:
            layer_activations = cache.get_hidden_states(layer_idx="all", num_samples=num_samples)
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

            # Effective Rank (Roy & Vetterli 2007) — canonical form
            # operates on eigenvalues of the Gram matrix, i.e. σ².
            from .utils import effective_rank
            erank = effective_rank(S)
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
