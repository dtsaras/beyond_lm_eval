from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings
import torch
import numpy as np
from scipy.stats import skew
from tqdm import tqdm
import logging
logger = logging.getLogger("blme")

@register_task("geometry_hubness")
class GlobalHubnessTask(DiagnosticTask):
    """
    Analyzes the "hubness" of the embedding space: the skewness of the distribution
    of k-nearest neighbor occurrences.

    Implementation runs the k-NN search on GPU using ``torch.topk`` because
    the per-batch matmul (batch × vocab × d_model) is the bottleneck. The
    previous numpy-on-CPU implementation became infeasible for VLM-scale
    vocabs (e.g. Gemma 4's 262k tokens × 6k hidden = ~1200 TFLOPs of matmul,
    ~hours on CPU). The GPU version brings this back to seconds.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Global Hubness Analysis...")
        k_values = self.config.get("k_values", [10, 50, 100])
        batch_size = self.config.get("batch_size", 2000)

        E = get_embeddings(model)
        if E is None:
            return {"error": "Could not extract embeddings"}

        # Choose a CUDA device when available; the embedding's own device is
        # ideal because the weights are already there. Fall back to CPU.
        if E.is_cuda:
            device = E.device
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        # Cast to float32 for the cosine similarity (bf16 has too little
        # precision for the small differences between top-k similarities) and
        # normalize for cosine similarity in one shot. We materialise this on
        # the chosen device.
        E_norm = E.detach().to(device=device, dtype=torch.float32)
        E_norm = E_norm / (E_norm.norm(dim=1, keepdim=True) + 1e-10)
        n_vocab = E_norm.shape[0]
        max_k = max(k_values)

        # Per-token occurrence counters, one per k-value, kept on the same
        # device so the topk results can be scattered without round-trips.
        n_occ = {k: torch.zeros(n_vocab, dtype=torch.int64, device=device)
                 for k in k_values}

        with torch.no_grad():
            for i in tqdm(range(0, n_vocab, batch_size), desc="Hubness"):
                end = min(i + batch_size, n_vocab)
                # (B, V) cosine sim slice
                sims = E_norm[i:end] @ E_norm.T
                # Mask self-similarity (i+j is the diagonal entry of row j)
                rows = torch.arange(end - i, device=device)
                cols = torch.arange(i, end, device=device)
                sims[rows, cols] = float("-inf")
                # Top-(max_k) once per batch — k_values are nested
                _, top_idx = torch.topk(sims, max_k, dim=1, largest=True, sorted=False)
                # For each requested k, take the first k columns (any k rows
                # of an unsorted topk are still a valid top-k set, since the
                # set is what matters for occurrence counting). To make this
                # well-defined we take the k columns with the highest values
                # — equivalently, sort the max_k indices by their similarity.
                topk_sims = torch.gather(sims, 1, top_idx)
                sort_order = topk_sims.argsort(dim=1, descending=True)
                top_idx_sorted = torch.gather(top_idx, 1, sort_order)
                for k in k_values:
                    flat = top_idx_sorted[:, :k].reshape(-1)
                    n_occ[k].scatter_add_(0, flat, torch.ones_like(flat))

        results = {}
        for k in k_values:
            n_occ_cpu = n_occ[k].cpu().numpy().astype(np.int64)

            hub_skew = skew(n_occ_cpu)
            hub_max = int(n_occ_cpu.max())

            # Top 1% mass (concentration)
            top_1pct_threshold = np.percentile(n_occ_cpu, 99)
            denom = n_occ_cpu.sum()
            top_1pct_mass = float(n_occ_cpu[n_occ_cpu >= top_1pct_threshold].sum() / denom) if denom > 0 else 0.0

            # Gini coefficient — sorted ascending, 1-based ranks
            n_occ_sorted = np.sort(n_occ_cpu).astype(np.float64)
            n = len(n_occ_sorted)
            if n_occ_sorted.sum() > 0:
                gini = (2 * np.sum((np.arange(1, n + 1) * n_occ_sorted))) / (n * n_occ_sorted.sum()) - (n + 1) / n
            else:
                gini = 0.0

            results[f'hubness_k{k}_skew'] = float(hub_skew)
            results[f'hubness_k{k}_max'] = hub_max
            results[f'hubness_k{k}_top1pct'] = float(top_1pct_mass)
            results[f'hubness_k{k}_gini'] = float(gini)

        return results
