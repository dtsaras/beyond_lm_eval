"""
Persistent-Homology Dimension (PHD) — intrinsic dimension of the hidden-state
point cloud from the power-law scaling of total H0 persistence.

Reference:
    Tulchinskii, Kuznetsov, Kushnareva, Cherniavskii, Nikolenko, Burnaev,
    Barannikov, Piontkovskaya (2023). "Intrinsic Dimension Estimation for
    Robust Detection of AI-Generated Texts." NeurIPS 2023, arXiv:2306.04723.
    Official code: github.com/ArGintum/GPTID — IntrinsicDim.py, class ``PHD``
    (commit 8c8759e).

Definition (Tulchinskii et al. 2023, §3; Adams et al. 2020 / Schweinhart 2021
persistent-homology fractal dimension):
    For a metric point cloud, the total H0 persistence of a size-n subsample
    is the sum of edge weights of the Euclidean minimum spanning tree (the
    0-dimensional persistence diagram of the Vietoris-Rips filtration has
    death-birth values equal to the MST edge lengths). With an exponent
    parameter ``alpha`` the reference sums the alpha-th powers of those edge
    lengths:

        E_alpha(n) = sum_{e in MST(subsample_n)} len(e)^alpha .

    The persistent-homology dimension obeys a power law

        E_alpha(n) ~ n^{(d - alpha) / d}          (reference uses alpha = 1)

    so, fitting the exponent  m = slope of  log E vs log n  across several
    subsample sizes n, the intrinsic dimension is recovered as

        d = alpha / (alpha - m)   ==>  (alpha = 1)   d = 1 / (1 - m).

    The reference computes, for each candidate size n in ``range(min_points,
    max_points, point_jump)``, the *median* MST weight over ``n_points``
    (or ``n_points_min`` when n is large relative to the cloud) random
    subsamples, fits ``m`` by ordinary least squares on
    (log n, log median-length), averages ``m`` over ``n_reruns`` independent
    restarts, and returns ``1 / (1 - m_mean)``.

Implementation notes:
    * ``_phd_dimension`` reproduces GPTID's ``PHD.fit_transform`` /
      ``_calc_ph_dim_single`` / ``prim_tree`` EXACTLY (same Prim MST with the
      ``alpha`` power, same median-over-restarts, same OLS slope formula,
      same ``1/(1-m)`` inversion). The only deliberate difference is that the
      subsampling RNG is made explicit and deterministic: the reference draws
      from the global legacy ``np.random`` inside per-rerun threads, whose
      interleaving is non-reproducible. Here every ``np.random.choice`` is
      driven from a single seeded legacy ``np.random.RandomState`` in a fixed
      order (reruns outer, test-n middle, restarts inner), so the estimate is
      bit-for-bit reproducible and can be matched against the reference run
      single-threaded under the same seed. See the parity harness/test.
    * Point cloud: token hidden states at one layer (default: last), pooled
      per config (``all_tokens`` by default — PHD is defined on a *set* of
      token embeddings exactly as the reference feeds it text-token
      embeddings). Capped at ``max_points`` via a seeded draw so the O(n^2)
      distance matrices stay bounded.
    * float64 throughout; guards for too-few points return ``{"error": ...}``.
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from scipy.spatial.distance import cdist

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")

# Reference's floor for a usable cloud (GPTID: MINIMAL_CLOUD = 80).
MINIMAL_CLOUD = 80


def _prim_tree(adj_matrix: np.ndarray, alpha: float = 1.0) -> float:
    """Total alpha-weighted minimum-spanning-tree length via Prim's algorithm.

    Transcribed verbatim from GPTID/IntrinsicDim.py ``prim_tree`` (commit
    8c8759e). Returns ``sum_e len(e) ** alpha`` over the MST edges, which for
    a Euclidean (H0) filtration equals the total 0-dim persistence.
    """
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)

    return s.item()


def _phd_dimension(
    X,
    alpha: float = 1.0,
    metric: str = "euclidean",
    n_reruns: int = 3,
    n_points: int = 7,
    n_points_min: int = 3,
    min_points: int = 50,
    max_points: int = 512,
    point_jump: int = 40,
    seed: int = 42,
) -> float:
    """Persistent-homology intrinsic dimension of the point cloud ``X``.

    Faithful, deterministic reproduction of GPTID's ``PHD.fit_transform``
    (Tulchinskii et al. 2023). Parameters mirror the reference class /
    ``fit_transform`` signature:

    Args:
        X: point cloud, shape ``(n_samples, n_features)``.
        alpha: exponent in ``E_alpha(n) = sum len(e)^alpha`` (reference 1.0).
        metric: distance metric passed to ``scipy.spatial.distance.cdist``.
        n_reruns: number of independent restarts; the slope ``m`` is averaged.
        n_points: subsamples drawn per candidate size ``n``.
        n_points_min: subsamples drawn when ``n`` is large (cloud <= 2n).
        min_points, max_points, point_jump: define the candidate sizes
            ``range(min_points, max_points, point_jump)``.
        seed: RNG seed for the (otherwise non-deterministic) subsampling.

    Returns:
        Estimated intrinsic dimension ``d = 1 / (1 - m)`` as a Python float,
        or ``float('nan')`` on a degenerate fit.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3:
        return float("nan")

    n_cloud = X.shape[0]
    test_n = list(range(min_points, max_points, point_jump))
    if len(test_n) < 2:
        return float("nan")
    # Every candidate size must be drawable without replacement.
    if max(test_n) > n_cloud:
        return float("nan")

    # Deterministic subsampling: one seeded legacy RandomState driven in a
    # fixed nested order (reruns -> test_n -> restarts). Matching the
    # reference means seeding the global np.random the same way and running
    # it single-threaded; the parity harness does exactly that.
    rng = np.random.RandomState(seed)
    log_n = np.log(np.array(test_n, dtype=np.float64))
    N = len(test_n)

    ms = np.zeros(n_reruns, dtype=np.float64)
    for r in range(n_reruns):
        lengths = []
        for n in test_n:
            restarts = n_points_min if n_cloud <= 2 * n else n_points
            reruns = np.ones(restarts)
            for i in range(restarts):
                idx = rng.choice(n_cloud, size=n, replace=False)
                sub = X[idx]
                reruns[i] = _prim_tree(cdist(sub, sub, metric=metric), alpha)
            lengths.append(np.median(reruns))
        y = np.log(np.array(lengths, dtype=np.float64))
        # OLS slope, transcribed from GPTID _calc_ph_dim_single.
        ms[r] = (
            N * (log_n * y).sum() - log_n.sum() * y.sum()
        ) / (N * (log_n ** 2).sum() - log_n.sum() ** 2)

    m = float(np.mean(ms))
    if not np.isfinite(m) or m >= 1.0:
        return float("nan")
    d = 1.0 / (1.0 - m)
    return float(d) if np.isfinite(d) else float("nan")


def _gather_point_cloud(
    model, tokenizer, dataset, num_samples: int, max_length: int,
    layer_idx: int, pooling: str, cache, use_cache: bool,
) -> Optional[np.ndarray]:
    """Return an ``(N, D)`` float64 token-hidden-state cloud at ``layer_idx``.

    Prefers the shared cache; falls back to a private forward pass like the
    sibling geometry tasks. ``pooling``: ``all_tokens`` (default), ``mean``,
    or ``last``.
    """
    # Cache path — flat (N, D) token cloud at the chosen layer.
    if cache is not None and getattr(cache, "is_populated", False) and use_cache:
        tensor = cache.get_hidden_states(layer_idx=layer_idx, num_samples=num_samples)
        if tensor is not None and pooling == "all_tokens":
            return tensor.float().cpu().numpy().astype(np.float64)
        # For mean/last we need per-sample chunks.
        if tensor is not None and pooling in ("mean", "last"):
            chunks = cache.get_hidden_states(
                layer_idx=layer_idx, num_samples=num_samples, per_sample=True,
            )
            if chunks:
                vecs = []
                for ch in chunks:
                    if ch is None:
                        continue
                    ch = ch.float().cpu().numpy()
                    if ch.ndim != 2 or ch.shape[0] == 0:
                        continue
                    vecs.append(ch.mean(0) if pooling == "mean" else ch[-1])
                if vecs:
                    return np.asarray(vecs, dtype=np.float64)

    # Fallback: private forward pass.
    if dataset is None:
        from ...cache import load_default_corpus
        dataset = load_default_corpus(num_samples)
    samples = list(dataset)[:num_samples]
    if not samples:
        return None

    device = next(model.parameters()).device
    collected: List[np.ndarray] = []
    with torch.no_grad():
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length,
            ).to(device)
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # tuple length n_layers+1 (embedding + blocks)
            li = layer_idx if layer_idx >= 0 else len(hs) + layer_idx
            li = max(0, min(li, len(hs) - 1))
            h = hs[li][0].detach().float().cpu().numpy()  # (T, D)
            if h.ndim != 2 or h.shape[0] == 0:
                continue
            if pooling == "mean":
                collected.append(h.mean(0))
            elif pooling == "last":
                collected.append(h[-1])
            else:  # all_tokens
                collected.extend(list(h))
    if not collected:
        return None
    return np.asarray(collected, dtype=np.float64)


@register_task("geometry_phd_dimension")
class PHDimensionTask(DiagnosticTask):
    """
    Persistent-Homology Dimension of the hidden-state point cloud
    (Tulchinskii et al., NeurIPS 2023, arXiv:2306.04723; repo ArGintum/GPTID).

    Estimates intrinsic dimension ``d`` from the power-law scaling of total
    H0 (minimum-spanning-tree) persistence across random subsamples:
    ``E(n) ~ n^{(d-1)/d}`` (alpha=1), ``d = 1/(1-slope)``.

    Outputs (flat floats):
        phd_dimension            — estimated intrinsic dimension d
        _meta_num_points         — cloud size fed to the estimator
        _meta_layer_idx          — layer the cloud was drawn from
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistent-Homology Dimension (PHD) Estimation...")

        cfg = self.config
        num_samples = int(cfg.get("num_samples", 100))
        max_length = int(cfg.get("max_length", 128))
        layer_idx = int(cfg.get("layer_idx", -1))
        pooling = cfg.get("pooling", "all_tokens")
        max_points = int(cfg.get("max_points", 2000))
        seed = int(cfg.get("seed", 42))
        use_cache = cfg.get("use_cache", True)

        # PHD estimator params (mirror GPTID PHD.__init__ / fit_transform).
        alpha = float(cfg.get("alpha", 1.0))
        n_reruns = int(cfg.get("n_reruns", 3))
        n_points = int(cfg.get("n_points", 7))
        n_points_min = int(cfg.get("n_points_min", 3))
        min_points = int(cfg.get("min_points", 50))
        phd_max_points = int(cfg.get("phd_max_points", 512))
        point_jump = int(cfg.get("point_jump", 40))

        X = _gather_point_cloud(
            model, tokenizer, dataset, num_samples, max_length,
            layer_idx, pooling, cache, use_cache,
        )
        if X is None or X.shape[0] < MINIMAL_CLOUD:
            return {
                "error": (
                    f"Need at least {MINIMAL_CLOUD} points for PHD "
                    f"(got {0 if X is None else X.shape[0]})."
                )
            }

        # Cap points so the largest subsample's O(n^2) cdist stays bounded.
        if X.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            X = X[rng.choice(X.shape[0], max_points, replace=False)]

        n_cloud = X.shape[0]
        # The largest candidate size must be drawable; clamp if needed.
        eff_max = min(phd_max_points, n_cloud)
        if eff_max <= min_points + point_jump:
            return {
                "error": (
                    f"Cloud too small for PHD subsample schedule "
                    f"(n={n_cloud}, min_points={min_points})."
                )
            }

        d = _phd_dimension(
            X,
            alpha=alpha,
            metric="euclidean",
            n_reruns=n_reruns,
            n_points=n_points,
            n_points_min=n_points_min,
            min_points=min_points,
            max_points=eff_max,
            point_jump=point_jump,
            seed=seed,
        )

        return {
            "phd_dimension": float(d),
            "_meta_num_points": int(n_cloud),
            "_meta_layer_idx": int(layer_idx),
        }
