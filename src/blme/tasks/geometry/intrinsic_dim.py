from ...tasks.base import DiagnosticTask
from ...registry import register_task
import torch
import numpy as np
from tqdm import tqdm
from ._sequence_points import (DEFAULT_N_SEQUENCES, DEFAULT_TOKEN_DOCS, readout_for,
                               sequence_clouds)
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
        Scientific Reports 7, arXiv:1803.06992. The Two-NN estimator used
        here.
      * Ansuini, Laio, Macke, Zoccolan 2019 — "Intrinsic Dimension of
        Data Representations in Deep Neural Networks", NeurIPS 2019,
        arXiv:1905.12784. The per-layer ID profile (on vision CNNs).
    """
    # Layerwise mode: one point per sequence (Q1). The static mode measures the
    # input-embedding rows and is not token-based.
    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": "arXiv:1803.06992",
        "code": "AnsuiniAlessio/IntrinsicDimDeep@2c5140009eb3",
        "llm_precedent": ("Cheng et al. 2024 arXiv:2405.15471 (last token of each sequence, "
                          "every layer); Valeriani et al. 2023 arXiv:2302.00294 (one point "
                          "per sequence, TwoNN per layer)"),
    }
    ADAPTATION_NOTE = (
        "TwoNN (Facco et al. 2017) estimates intrinsic dimension from the ratio of each "
        "point's second- to first-nearest-neighbour distance, fitted through the origin "
        "with the top 10% of ratios discarded. In layerwise mode BLME applies it to one "
        "point per sequence at every block output: the last content token of each of 1000 "
        "documents, the same N for every model, Euclidean distance, as Cheng et al. 2024 do "
        "with the last token. A token cloud pooled across documents is avoided because 28-83% "
        "of a token's nearest neighbours lie in its own document; its value is kept as "
        "_meta_tokens_* with position 0 and massive rows excluded. TwoNN falls as N grows "
        "(Facco et al. report d against N), so the value is the ID at a fixed 1000-point "
        "scale. The static mode measures the input-embedding rows, whose count is the "
        "vocabulary size."
    )
    HEADLINE_KEYS = ("intrinsic_dimension", "lid_first_layer", "lid_mid_layer",
                     "lid_last_layer")
    _INV = "tests/tasks/invariance/test_geometry_intrinsic_dim.py::"
    # lid_mid_layer is an index-selected depth and moves under identity-block
    # insertion; intrinsic_dimension counts vocabulary rows (V-dependent).
    SIZE_INVARIANT = {
        "lid_first_layer": (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
                            _INV + "test_endpoints_invariant"),
        "lid_last_layer": (("zero_pad", "embed_isometric", "rescale", "insert_identity_blocks"),
                           _INV + "test_endpoints_invariant"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Intrinsic Dimension Estimation (Two-NN)...")

        # Static mode measures the input-embedding rows; layerwise mode the
        # hidden-state profile over block outputs.
        layerwise = self.config.get("layerwise", False)

        if layerwise:
            logger.info("  Mode: Layer-wise Activations")
            if dataset is None:
                dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]
            use_cache = self.config.get("use_cache", True)
            n_seq = int(self.config.get("num_samples", DEFAULT_N_SEQUENCES))
            seed = int(self.config.get("subsample_seed", 42))

            logger.info("  Collecting hidden states...")
            # One point per sequence (Q1): the last content token of each of
            # n_seq documents, at every block output (the deepest one the
            # hooked raw block output). A cache holding fewer documents, or no
            # hidden states, is bypassed (test_intrinsic_dim_protocol.py).
            sc = sequence_clouds(
                model, tokenizer, dataset, cache, n=n_seq, layer="all",
                replace_final_norm=True, use_cache=use_cache,
                token_docs=int(self.config.get("token_cloud_docs", DEFAULT_TOKEN_DOCS)),
                readout=readout_for(self.TOKEN_POLICY))
            if not sc.last:
                return {"error": "No hidden states available for intrinsic dimension"}

            sorted_layers = sorted(sc.last)
            id_trend = []
            results = {}
            for layer_idx in tqdm(sorted_layers, desc="Computing Layer IDs"):
                # float64: _compute_id rescales by a power of two and compares
                # radii exactly (test_intrinsic_dim_translation_invariance.py).
                lid = self._compute_id(sc.last[layer_idx].double().numpy())["intrinsic_dimension"]
                # Absolute-index names are _meta_: one index is a different
                # depth on models of different size.
                results[f"_meta_lid_layer_{layer_idx}"] = lid
                id_trend.append(lid)
            results["lid_trend"] = id_trend
            results.update(_profile_summary(id_trend))
            # P4: the OLS slope over layers moves when identity blocks are
            # inserted, so it is a _meta_ profile summary
            # (tests/tasks/invariance/test_geometry_intrinsic_dim.py).
            results["_meta_lid_depth_slope"] = results.pop("lid_depth_slope")

            # The token cloud (every token of the first token_cloud_docs
            # documents, sinks excluded, 20000-row seeded subsample): _meta_.
            if sc.tokens:
                tok_trend = []
                for layer_idx in sorted_layers:
                    X = sc.token_rows(layer_idx)[0].double().numpy()
                    if len(X) > 20000:
                        rng = np.random.default_rng(seed)
                        X = X[rng.choice(len(X), 20000, replace=False)]
                    tok_trend.append(self._compute_id(X)["intrinsic_dimension"])
                results["_meta_tokens_lid_trend"] = tok_trend
                results.update({f"_meta_tokens_{k}": v
                                for k, v in _profile_summary(tok_trend).items()})

            results["_meta_last_entry_is_block_output"] = bool(sc.info.get("last_is_block", False))
            # Why the cache was not used (the route is _meta_points_source).
            results["_meta_cache_rejected"] = sc.info.get("cache_rejected")
            results.update(sc.meta())
            return results

        else:
             logger.info("  Mode: Static Embeddings")
             from ..common import get_embeddings as _get_emb
             E = _get_emb(model)
             if E is None:
                 return {"error": "Could not extract embeddings"}

             # float64, as on the layerwise route.
             E_np = E.double().cpu().numpy()
             # `sample_size` subsamples the rows (None: every row).
             return self._compute_id(
                 E_np, sample_size=self.config.get("sample_size", None),
                 seed_offset=int(self.config.get("seed_offset", 0)))

    def _compute_id(self, X, sample_size=None, seed_offset=0):
        """Two-NN intrinsic dimension of the point cloud ``X``.

        Uses GPU-batched ``torch.cdist + torch.topk`` when CUDA is
        available; falls back to sklearn's ball tree on CPU-only hosts.
        Returns the same ``{intrinsic_dimension, sample_size}`` dict
        either way.
        """
        # Rescale by a POWER OF TWO to unit max. Two-NN depends only on r2/r1,
        # the CPU kNN underflows at extreme scales, and any other divisor
        # re-rounds the coordinates and changes which r1 == r2 / duplicate rows
        # are excluded (test_intrinsic_dim_translation_invariance.py).
        _g = float(np.max(np.abs(np.asarray(
            X.detach().cpu() if isinstance(X, torch.Tensor) else X, dtype=np.float64))))
        if np.isfinite(_g) and _g > 0:
            _s = float(2.0 ** np.ceil(np.log2(_g)))   # exact: mantissa preserved
            if np.isfinite(_s) and _s > 0:
                X = (X / _s) if isinstance(X, torch.Tensor) else (np.asarray(X) / _s)

        # float64 throughout: narrowing would re-round the rescaled coordinates.
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().double()
        else:
            X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float64))

        # Non-finite rows (an input that overflowed upstream) are dropped and
        # counted in _meta_n_nonfinite_rows, not silently estimated around.
        _finite_rows = torch.isfinite(X_tensor).all(dim=1)
        _n_nonfinite = int((~_finite_rows).sum().item())
        if _n_nonfinite:
            logger.warning(
                "intrinsic_dim: dropping %d/%d rows with non-finite entries",
                _n_nonfinite, int(X_tensor.shape[0]))
            X_tensor = X_tensor[_finite_rows]
            if X_tensor.shape[0] < 4:
                return {"intrinsic_dimension": float("nan"),
                        "_meta_sample_size": int(X_tensor.shape[0]),
                        "_meta_n_nonfinite_rows": _n_nonfinite}

        n_vocab = X_tensor.shape[0]
        if sample_size and sample_size < n_vocab:
            rng = np.random.default_rng(42 + seed_offset)
            indices = rng.choice(n_vocab, sample_size, replace=False)
            X_tensor = X_tensor[torch.from_numpy(indices)]

        N = int(X_tensor.shape[0])
        if N < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        r1, r2 = _two_nearest_neighbor_distances(X_tensor)
        if r1 is None:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        # Drop exact duplicates (r1 = 0 → mu undefined). Near-duplicates
        # are kept; they're valid density samples.
        valid = r1 > 0
        r1, r2 = r1[valid], r2[valid]
        if r1.size < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        mus = r2 / r1
        # Filter numerical noise: mu must be ≥ 1 by construction.
        mus = mus[mus > 1.0]
        if mus.size < 4:
            return {"intrinsic_dimension": float("nan"), "_meta_sample_size": N}

        intrinsic_dim = _twonn_linear_fit(mus)
        return {
            "intrinsic_dimension": float(intrinsic_dim),
            "_meta_sample_size": N,
            "_meta_n_nonfinite_rows": _n_nonfinite,
        }


def _profile_summary(trend):
    """first / mid / last entries and the OLS slope on relative depth."""
    if not trend:
        return {}
    n = len(trend)
    return {
        "lid_first_layer": float(trend[0]),
        "lid_mid_layer": float(trend[n // 2]),
        "lid_last_layer": float(trend[-1]),
        "lid_depth_slope": (
            float(np.polyfit(np.linspace(0.0, 1.0, n), np.asarray(trend, dtype=float), 1)[0])
            if n > 1 and np.all(np.isfinite(trend)) else float("nan")),
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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = None

    if device is not None:
        try:
            # float64 + direct differences: the mm-mode cdist (x^2+y^2-2xy)
            # cancels when r1 << ||x|| and inflates Two-NN.
            X_dev = X.to(device).double()
            N = X_dev.shape[0]
            r1 = np.zeros(N, dtype=np.float64)
            r2 = np.zeros(N, dtype=np.float64)
            for s in range(0, N, chunk_rows):
                e = min(N, s + chunk_rows)
                d = torch.cdist(X_dev[s:e], X_dev, p=2,
                                compute_mode="donot_use_mm_for_euclid_dist")
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

    # float64 + ball_tree, matching the GPU branch: brute force goes through
    # the Gram identity and keeps ~1e-12 relative error in r1 even in float64,
    # while a tree computes the true distance.
    X_np = X.detach().cpu().double().numpy()
    nbrs = NearestNeighbors(
        n_neighbors=3, algorithm="ball_tree", metric="euclidean",
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

    We also do NOT floor the estimate at 1: the Two-NN Pareto model has
    support d in (0, inf), so estimates below 1 are valid outputs on
    near-degenerate geometries and carry diagnostic signal. (The paper
    itself never discusses sub-1 estimates — no floor is implied by it.)
    If the downstream analysis wants a lower bound it can apply one
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

    # μ = 1 exactly (log μ = 0) and F = 1 carry no slope information.
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
