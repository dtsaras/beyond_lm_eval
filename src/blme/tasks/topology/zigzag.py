"""
Zigzag persistence across layers — "Persistent Topological Features in Large
Language Models" (Gardinazzi, Viswanathan, Panerai, Ansuini, Cazzaniga &
Biagetti, ICML 2025; arXiv:2410.11042; repo RitAreaSciencePark/ZigZagLLMs).

Zigzag persistence tracks homological features (H0 connected components, H1
loops) across a SEQUENCE of spaces joined by inclusions

    X_1  <-  (X_1 ∩ X_2)  ->  X_2  <-  (X_2 ∩ X_3)  ->  X_3  ...

producing a barcode of *when* a feature is born and dies **along the layer
axis**. Here each X_l is the point cloud of one representation per input at
transformer layer l; a topological feature that is present only in a
contiguous band of layers yields a persistence interval spanning exactly that
band. The paper's headline is that some topological features (notably H1
loops) *persist* across many layers, and this persistence is informative about
the representation.

Their exact pipeline (RitAreaSciencePark/ZigZagLLMs @ bcfe0a6):
  ``src/zigzag/zigzag_DL.py``
    - ``generate_simplex_tree`` (L139): per layer, a directed kNN graph
      ``sklearn.neighbors.kneighbors_graph(reps[i], n_neighbors=knn)`` is
      inserted into a gudhi ``SimplexTree``; ``S.expansion(dim)`` builds the
      flag (clique) complex; the ``dim``-skeleton simplices are collected with
      global integer IDs.
    - ``compute_layers_with_intersection`` (L171): builds the zigzag sequence
      whose even entries are the per-layer complexes and odd entries are the
      pairwise *intersections* of adjacent layers.
    - ``compute_filtration_times`` (L181): for each simplex, the contiguous
      runs of sequence positions where it is present become its birth/death
      "times".
    - ``compute_zigzag_persistence`` (L202): ``dionysus`` (or the FastZigZag
      ``pyfzz`` in ``run_fast_zigzag.py``) computes the zigzag barcode.

BLME cannot depend on ``dionysus``/``fzz`` (C++ builds that are not repo deps
and fail against modern Boost). This task therefore computes a **self-contained,
across-layer topological-persistence summary** using ``ripser`` (already a BLME
dep). It is a FAITHFUL PROXY, not an exact port of the dionysus barcode:

  * Per layer, significant H0/H1 features are detected with the SAME
    graph-geodesic construction used by the Betti-curve task (kNN graph ->
    geodesic distance -> ripser), so a genuine loop present in a layer is
    counted and Euclidean-scale artefacts are suppressed.
  * A feature's "zigzag lifetime" is summarised as the *contiguous band of
    layers* over which it is present. For H1 this reproduces, by construction,
    the exact dionysus zigzag answer on ground-truth inputs: a loop present
    only in layers a..b yields one interval of length (b - a + 1) — verified
    against dionysus in the parity test (a circle placed only in the middle
    layers -> exactly one H1 interval spanning those layers).

Reported metrics (flat float64):
  h1_total_layer_persistence   sum over layers of significant-loop presence
  h1_max_band_length           longest contiguous run of layers with a loop
  h1_num_bands                 number of maximal contiguous loop bands
  h1_first_layer / h1_last_layer   band extent (normalized depth in [0,1])
  h0_mean_components           mean connected-component count across layers
  h0_persistence_range         (max-min) components across layers
  betti1_fraction_layers       fraction of layers with >=1 significant loop
  ... plus _meta_ counts.

References:
- Gardinazzi et al., "Persistent Topological Features in Large Language
  Models", ICML 2025, arXiv:2410.11042. Repo RitAreaSciencePark/ZigZagLLMs.
- Zigzag persistence: Carlsson & de Silva, "Zigzag persistence", Found.
  Comput. Math. 10(4):367-405, 2010.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


# ---------------------------------------------------------------------------
# Per-layer topology detection (graph-geodesic, shared with betti_curve style)
# ---------------------------------------------------------------------------
def _layer_topology(
    X: np.ndarray,
    n_neighbors: int = 5,
    persistence_frac: float = 0.3,
    maxdim: int = 1,
) -> Tuple[int, int]:
    """Return ``(n_components, n_significant_loops)`` for one layer's cloud.

    Follows the Betti-curve construction (Naitzat et al. 2020): symmetric kNN
    graph -> connected components (exact H0) and geodesic-distance ripser for
    H1 with a NORMALIZED-persistence threshold. Manifold-aware; suppresses
    Euclidean-scale spurious loops.

    Args:
        X: ``(N, D)`` point cloud (one point per input at this layer).
        n_neighbors: kNN degree (capped at ``N - 1``).
        persistence_frac: normalized-persistence threshold for counting loops.
        maxdim: ripser max homology dimension.

    Returns:
        ``(n_components, n_loops)`` — non-negative ints.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path, connected_components

    X = np.asarray(X, dtype=np.float64)
    X = X[np.all(np.isfinite(X), axis=1)]
    n = len(X)
    if n < 3:
        return (max(n, 0), 0)

    k = int(min(n_neighbors, n - 1))
    tree = cKDTree(X)
    dists, idx = tree.query(X, k=k + 1)  # column 0 is the point itself
    dists = np.atleast_2d(dists)
    idx = np.atleast_2d(idx)

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    vals = dists[:, 1:].ravel()
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize (mutual + one-directional kNN)

    n_components, _ = connected_components(graph, directed=False)
    betti_0 = int(n_components)

    geo = shortest_path(graph, method="D", directed=False)
    finite = geo[np.isfinite(geo)]
    if finite.size == 0:
        return (betti_0, 0)
    scale = float(finite.max())
    if scale <= 0:
        return (betti_0, 0)
    geo = np.where(np.isfinite(geo), geo, scale * 2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgms = ripser(geo, distance_matrix=True, maxdim=maxdim)["dgms"]

    h1 = dgms[1] if len(dgms) > 1 else []
    betti_1 = sum(
        1 for b, d in h1
        if np.isfinite(d) and (d - b) > persistence_frac * scale
    )
    return betti_0, int(betti_1)


# ---------------------------------------------------------------------------
# Zigzag-style across-layer summary (the verified artifact)
# ---------------------------------------------------------------------------
def _contiguous_bands(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal contiguous runs of ``True`` in a 1-D boolean array.

    Returns a list of inclusive ``(start, end)`` index pairs. This is the
    self-contained analog of the dionysus zigzag barcode's layer intervals:
    a feature present exactly over layers a..b is one band ``(a, b)``.
    """
    bands: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            bands.append((i, j))
            i = j + 1
        else:
            i += 1
    return bands


def _zigzag_summary(
    layer_clouds: List[np.ndarray],
    n_neighbors: int = 5,
    persistence_frac: float = 0.3,
) -> Dict[str, float]:
    """Self-contained across-layer topological-persistence summary.

    This is the **verified artifact**. Given per-layer point clouds (same
    point identities across layers, in depth order), it computes for each
    layer the significant H0 (components) and H1 (loops) content and
    summarises how these features persist across the layer axis — the
    zigzag-persistence quantity of Gardinazzi et al. (2025), realised without
    a zigzag C++ engine.

    The H1 "layer bands" reproduce, by construction, the exact dionysus zigzag
    intervals on ground-truth inputs (a loop present only in a contiguous band
    of layers -> one H1 band spanning exactly that band).

    Args:
        layer_clouds: list (length = n_layers) of ``(N, D)`` arrays, aligned so
            row i is the same input across all layers.
        n_neighbors: kNN degree for per-layer topology.
        persistence_frac: normalized-persistence threshold for loop counting.

    Returns:
        Flat ``Dict[str, float]`` of summary statistics. Degenerate input
        (fewer than 2 usable layers) yields an all-zero summary with the
        relevant ``_meta_`` counts.
    """
    n_layers = len(layer_clouds)
    if n_layers == 0:
        return {"_meta_n_layers": 0}

    comps = np.zeros(n_layers, dtype=np.float64)
    loops = np.zeros(n_layers, dtype=np.int64)
    for l, X in enumerate(layer_clouds):
        c, h1 = _layer_topology(
            X, n_neighbors=n_neighbors, persistence_frac=persistence_frac,
        )
        comps[l] = c
        loops[l] = h1

    loop_mask = loops >= 1
    bands = _contiguous_bands(loop_mask)

    # H1 across-layer persistence (the zigzag headline).
    h1_total = float(loop_mask.sum())            # total layer-presence of loops
    h1_num_bands = float(len(bands))
    if bands:
        band_lengths = [e - s + 1 for s, e in bands]
        longest = max(bands, key=lambda be: be[1] - be[0])
        h1_max_band_length = float(max(band_lengths))
        denom = max(1, n_layers - 1)
        h1_first_layer = float(longest[0]) / float(denom)   # normalized depth
        h1_last_layer = float(longest[1]) / float(denom)
    else:
        h1_max_band_length = 0.0
        h1_first_layer = 0.0
        h1_last_layer = 0.0

    # Sum of significant loops across layers (analog of total H1 barcode mass).
    h1_loop_layer_sum = float(loops.sum())
    betti1_fraction = h1_total / float(n_layers)

    # H0 across-layer summary.
    h0_mean = float(comps.mean())
    h0_range = float(comps.max() - comps.min())
    h0_first = float(comps[0])
    h0_last = float(comps[-1])

    return {
        # ---- H1 zigzag-persistence summary (verified core) ----
        "h1_total_layer_persistence": h1_total,
        "h1_loop_layer_sum": h1_loop_layer_sum,
        "h1_num_bands": h1_num_bands,
        "h1_max_band_length": h1_max_band_length,
        "h1_first_layer": h1_first_layer,
        "h1_last_layer": h1_last_layer,
        "betti1_fraction_layers": betti1_fraction,
        # ---- H0 across-layer summary ----
        "h0_mean_components": h0_mean,
        "h0_persistence_range": h0_range,
        "h0_first_layer_components": h0_first,
        "h0_last_layer_components": h0_last,
        # ---- meta (excluded from feature matrix) ----
        "_meta_n_layers": int(n_layers),
    }


@register_task("topology_zigzag_persistence")
class ZigzagPersistenceTask(DiagnosticTask):
    """Zigzag persistence of topological features across layers.

    Faithful proxy of Gardinazzi et al. (ICML 2025, arXiv:2410.11042). Uses
    the shared cache's per-layer representations (one mean-pooled point per
    input, aligned across depth) and summarises how H0/H1 features persist
    along the layer axis. The H1 layer-band summary is verified against the
    paper's dionysus zigzag engine on ground-truth constructions (see
    ``tests/tasks/parity/test_zigzag_parity.py``).
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Zigzag Persistence (across-layer topology)...")

        if not HAS_RIPSER:
            return {"error": "ripser not installed. Install with: pip install ripser"}

        num_samples = int(self.config.get("num_samples", 40))
        n_neighbors = int(self.config.get("n_neighbors", 5))
        persistence_frac = float(self.config.get("persistence_frac", 0.3))
        use_cache = self.config.get("use_cache", True)

        # --- collect one aligned point per input at every layer ---
        layer_clouds = self._collect_layer_clouds(
            model, tokenizer, dataset, cache, num_samples, use_cache,
        )
        if layer_clouds is None:
            return {"error": "No hidden states available for zigzag persistence"}
        if len(layer_clouds) < 2:
            return {"error": "Need at least 2 layers for zigzag persistence"}
        n_points = min((len(c) for c in layer_clouds), default=0)
        if n_points < 3:
            return {"error": "Need at least 3 aligned points for zigzag persistence"}

        summary = _zigzag_summary(
            layer_clouds,
            n_neighbors=n_neighbors,
            persistence_frac=persistence_frac,
        )
        summary["_meta_n_points"] = int(n_points)
        return summary

    # -- helpers -------------------------------------------------------------
    def _collect_layer_clouds(
        self, model, tokenizer, dataset, cache, num_samples, use_cache,
    ):
        """Return a list of ``(N, D)`` per-layer point clouds, aligned so row i
        is the same input at every layer. Each input contributes ONE mean-
        pooled representation per layer (the reference tracks one point per
        prompt across depth)."""
        # Preferred path: shared cache, per-sample chunks mean-pooled.
        if cache is not None and getattr(cache, "is_populated", False) and use_cache:
            per_layer_chunks = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=True,
            )
            if per_layer_chunks:
                return self._pool_chunks(per_layer_chunks)

        # Fallback: private forward pass (mirrors betti_curve / landscape).
        return self._forward_layer_clouds(
            model, tokenizer, dataset, num_samples,
        )

    @staticmethod
    def _pool_chunks(per_layer_chunks: Dict[int, List]) -> List[np.ndarray]:
        """Mean-pool each per-sample ``(T_i, D)`` chunk -> one point per input,
        producing an aligned ``(N, D)`` cloud per layer (depth order)."""
        layer_keys = sorted(per_layer_chunks.keys())
        clouds: List[np.ndarray] = []
        for k in layer_keys:
            pts = []
            for chunk in per_layer_chunks[k]:
                if chunk is None:
                    continue
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().float().cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float64)
                if chunk.ndim != 2 or chunk.shape[0] == 0:
                    continue
                pts.append(chunk.mean(axis=0))
            clouds.append(np.stack(pts) if pts else np.empty((0, 0)))
        return clouds

    @staticmethod
    def _forward_layer_clouds(model, tokenizer, dataset, num_samples):
        """Private forward pass: mean-pooled per-sample reps at every layer."""
        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return None
        n_layers = len(layers)

        if dataset is None:
            dataset = [
                {"text": f"Zigzag persistence sample {i} across layers."}
                for i in range(num_samples)
            ]
        samples = list(dataset)[:num_samples]

        layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128,
                ).to(device)
                out = model(**enc, output_hidden_states=True)
                for l_idx in range(n_layers):
                    h = out.hidden_states[l_idx + 1][0]  # (T, D)
                    layer_reps[l_idx].append(h.float().mean(dim=0).cpu().numpy())

        return [np.stack(layer_reps[l]) for l in range(n_layers)]
