from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from .utils import collect_hidden_states
import logging
logger = logging.getLogger("blme")


@register_task("geometry_intrinsic_dim")
class IntrinsicDimensionTask(DiagnosticTask):
    """
    Estimates the Intrinsic Dimension (ID) of the embedding manifold
    using the Two-NN estimator.

    References:
      * Facco, d'Errico, Rodriguez, Laio 2017 — "Estimating the Intrinsic
        Dimension of Datasets by a Minimal Neighborhood Information",
        Scientific Reports 7, arXiv:1705.10933. The Two-NN estimator used
        here.
      * Ansuini, Laio, Macke, Zoccolan 2019 — "Intrinsic Dimension of
        Data Representations in Deep Neural Networks", NeurIPS 2019,
        arXiv:1905.12784. Characterises the per-layer ID profile that
        this task produces for language models.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Intrinsic Dimension Estimation (Two-NN)...")
        
        # Check mode: Embeddings (static) or Layer-wise Activations (dynamic)
        layerwise = self.config.get("layerwise", False)
        
        if layerwise:
            logger.info("  Mode: Layer-wise Activations")
            if dataset is None:
                # Mock dataset if missing
                dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            use_cache = self.config.get("use_cache", True)
                
            # Collect states from all layers
            logger.info("  Collecting hidden states...")
            # Use 'all' to get dict of {layer_idx: tensor}
            if cache is not None and cache.is_populated and use_cache:
                layer_activations = cache.get_hidden_states(
                    layer_idx="all",
                    num_samples=self.config.get("num_samples", 100),
                )
            else:
                layer_activations = collect_hidden_states(model, tokenizer, dataset, num_samples=self.config.get("num_samples", 100), layer_idx="all")
            
            results = {}
            # Compute ID for each layer
            sorted_layers = sorted(layer_activations.keys())
            id_trend = []
            
            for layer_idx in tqdm(sorted_layers, desc="Computing Layer IDs"):
                X = layer_activations[layer_idx].float().numpy()
                # Subsample if too large
                if len(X) > 20000:
                    indices = np.random.choice(len(X), 20000, replace=False)
                    X = X[indices]
                    
                lid_result = self._compute_id(X)
                lid = lid_result["intrinsic_dimension"]
                results[f"lid_layer_{layer_idx}"] = lid
                id_trend.append(lid)
                
            results["lid_trend"] = id_trend
            return results
            
        else:
             logger.info("  Mode: Static Embeddings")
             # 1. Get Embeddings
             from ..common import get_embeddings as _get_emb
             E = _get_emb(model)
             if E is None:
                 return {"error": "Could not extract embeddings"}
             
             E_np = E.float().cpu().numpy()
             return self._compute_id(E_np, sample_size=self.config.get("sample_size", None))

    def _compute_id(self, X, sample_size=None):
        """Two-NN intrinsic dimension of the point cloud ``X``.

        Uses GPU-batched ``torch.cdist + torch.topk`` when CUDA is
        available (100-1000× faster than sklearn's CPU kNN on
        vocab-scale embeddings). Falls back to sklearn on CPU-only
        hosts. Returns the same ``{intrinsic_dimension, sample_size}``
        dict either way.
        """
        # Accept torch.Tensor or numpy.ndarray.
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().float()
        else:
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))

        n_vocab = X_tensor.shape[0]
        if sample_size and sample_size < n_vocab:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_vocab, sample_size, replace=False)
            X_tensor = X_tensor[torch.from_numpy(indices)]

        N = int(X_tensor.shape[0])
        if N < 4:
            return {"intrinsic_dimension": float("nan"), "sample_size": N}

        r1, r2 = _two_nearest_neighbor_distances(X_tensor)
        if r1 is None:
            return {"intrinsic_dimension": float("nan"), "sample_size": N}

        # Drop exact duplicates (r1 = 0 → mu undefined). Near-duplicates
        # are kept; they're valid density samples.
        valid = r1 > 0
        r1, r2 = r1[valid], r2[valid]
        if r1.size < 4:
            return {"intrinsic_dimension": float("nan"), "sample_size": N}

        mus = r2 / r1
        # Filter numerical noise: mu must be ≥ 1 by construction.
        mus = mus[mus > 1.0]
        if mus.size < 4:
            return {"intrinsic_dimension": float("nan"), "sample_size": N}

        intrinsic_dim = _twonn_linear_fit(mus)
        return {
            "intrinsic_dimension": float(intrinsic_dim),
            "sample_size": N,
        }


def _two_nearest_neighbor_distances(
    X: torch.Tensor,
    chunk_rows: int = 4096,
) -> tuple:
    """Return ``(r1, r2)`` — first and second nearest-neighbor distances
    for every row of ``X`` — as 1-D numpy arrays.

    Uses a GPU ``torch.cdist`` + ``torch.topk`` implementation when
    CUDA is available; otherwise falls back to ``sklearn.neighbors``.
    Chunked along rows so the (Q, N) distance matrix doesn't blow up
    memory on 150 k-vocab embeddings.
    """
    # Prefer GPU (Tesla/RTX, ~1000× faster than sklearn on vocab-scale).
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = None

    if device is not None:
        try:
            X_dev = X.to(device)
            N = X_dev.shape[0]
            r1 = np.zeros(N, dtype=np.float64)
            r2 = np.zeros(N, dtype=np.float64)
            for s in range(0, N, chunk_rows):
                e = min(N, s + chunk_rows)
                d = torch.cdist(X_dev[s:e], X_dev, p=2)  # (chunk, N)
                # Take top-3 smallest; index 0 is self (distance 0).
                vals, _ = torch.topk(d, k=3, dim=1, largest=False, sorted=True)
                r1[s:e] = vals[:, 1].detach().cpu().double().numpy()
                r2[s:e] = vals[:, 2].detach().cpu().double().numpy()
                del d, vals
            return r1, r2
        except Exception as e:
            logger.info(
                f"GPU kNN failed ({type(e).__name__}: {e}); "
                "falling back to sklearn."
            )

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.info("sklearn not installed, skipping ID estimation")
        return None, None

    X_np = X.detach().float().cpu().numpy()
    nbrs = NearestNeighbors(
        n_neighbors=3, algorithm="auto", metric="euclidean", n_jobs=-1,
    ).fit(X_np)
    d, _ = nbrs.kneighbors(X_np)
    return d[:, 1], d[:, 2]


def _twonn_linear_fit(mus: np.ndarray, tail_trim: float = 0.1) -> float:
    """Two-NN estimator via the linear-regression form of Facco et al.
    2017, following the five-step algorithm on p. 3 of the paper and
    the reference implementation in ``scikit-dimension``.

    The method fits the empirical CDF
    ``log(1 − F(μ)) = −d · log(μ)`` by OLS through the origin,
    discarding the largest ``tail_trim`` fraction of μ values
    (the paper and skdim use 10 %) to dampen the influence of the
    heavy upper tail on the slope. We do NOT trim the lower tail —
    small μ's (μ ≈ 1) are the most informative points, not noise.

    We also do NOT floor the estimate at 1: the paper explicitly
    observes the estimate dipping below 1 on near-degenerate geometries
    (Fig. 3) and treats that as a diagnostic signal, not an error. If
    the downstream analysis wants a lower bound it can apply one
    explicitly.
    """
    if mus.size == 0:
        return float("nan")

    mus_sorted = np.sort(mus)
    n = mus_sorted.size
    # Empirical CDF F_i = i / N, matching Facco step 4 and
    # scikit-dimension's TwoNN implementation.
    F = np.arange(1, n + 1, dtype=np.float64) / n

    # Trim only the top `tail_trim` fraction (paper default: 10 %).
    # The last index has F = 1 → log(1 − F) = −∞; the trim also drops
    # that singularity.
    n_keep = int(n * (1.0 - tail_trim))
    if n_keep < 4:
        n_keep = n  # tiny sample — keep everything, accept noise
    mus_keep = mus_sorted[:n_keep]
    F_keep = F[:n_keep]

    # Guard against μ = 1 exactly (log μ = 0) — that's a valid (0, 0)
    # point on the regression line, but some samples may also hit
    # F = 1 at the top. We've already trimmed the top; for any
    # remaining F = 1 (can't happen with i/N and i < n), skip.
    valid = (mus_keep > 1.0) & (F_keep < 1.0)
    if valid.sum() < 4:
        return float("nan")

    x = np.log(mus_keep[valid])
    y = -np.log(1.0 - F_keep[valid])

    # OLS through the origin: d = Σ xy / Σ x²
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    d_est = float(np.sum(x * y) / denom)
    if not np.isfinite(d_est) or d_est <= 0:
        return float("nan")
    return d_est
