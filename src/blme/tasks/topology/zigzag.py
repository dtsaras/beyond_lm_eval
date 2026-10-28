"""
Zigzag persistence across layers (Gardinazzi, Viswanathan, Panerai, Ansuini,
Cazzaniga & Biagetti, ICML 2025; arXiv:2410.11042; RitAreaSciencePark/ZigZagLLMs).

Each layer contributes one point cloud (one point per prompt, rows aligned
across layers). Per layer a kNN graph is built and filled to its flag complex;
the zigzag X_1 <- X_1 n X_2 -> X_2 <- ... (``zigzag_DL.py``
``compute_layers_with_intersection``) is computed with dionysus, and the bars
are read in layer units (the paper's effective barcode, Eq. 4).

* ``zz_true_*``: the paper's construction. ``_true_zigzag_barcode`` gives a
  simplex present over layers a..b the interval [a, b + 1), which gives the
  reference's effective barcode in layer units.
  ``Z_p(l1, l2)`` is Eq. 7 (with footnote 6: 0 when beta_p(l1) = 0).
* ``h1_*`` / ``h0_*`` / ``betti1_fraction_layers``: a BLME proxy. Per layer it
  counts significant loops on the graph-geodesic metric (ripser) and reports
  contiguous bands of layers that carry one; it does not track identity.
* ``zz_h1_*``: one adjacent-layer zigzag step over GF(2) on the shared vertex
  set (``_h1_persisting_edges``); pairwise, so it cannot see a bar's death.

Zigzag persistence: Carlsson & de Silva, Found. Comput. Math. 10(4), 2010.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import math
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
    # Floor zero distances (duplicate points): csgraph reads an explicit zero
    # as "no edge", which would inflate beta_0.
    vals = np.maximum(dists[:, 1:].ravel(), 1e-12)
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

    # ripser casts the distance matrix to float32: a power-of-two rescale
    # (exact) keeps out-of-range magnitudes finite, and `scale` moves with it so
    # the threshold test is unchanged (tests/tasks/test_topology.py).
    _gm = float(np.max(np.abs(geo))) if geo.size else 0.0
    if _gm > 0 and np.isfinite(_gm) and not (1e-6 < _gm < 1e6):
        _e = int(max(-126, min(127, round(math.log2(_gm)))))
        geo = np.ldexp(geo, -_e)
        scale = float(np.ldexp(scale, -_e))

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


# ---------------------------------------------------------------------------
# One adjacent-layer zigzag step over GF(2). The band summary does not track
# feature identity (five different loops in five layers look like one loop
# persisting five layers); this step does, on the shared vertex set.
#
# For the zigzag step  H1(X_i) --f--> H1(X_i u X_{i+1}) <--g-- H1(X_{i+1}),
# work in the F2 edge space of the union U:
#     rank f = dim(Z1(X_i) + B1(U)) - dim B1(U)
# (the kernel of f is (Z1(X_i) n B1(U))/B1(X_i), and B1(X_i) is contained in
# Z1(X_i) n B1(U), so the rank is the codimension computation above), and the
# number of classes present in BOTH layers is
#     dim(im f n im g) = rank f + rank g - dim(im f + im g),
#     dim(im f + im g) = dim(Z1(X_i) + Z1(X_{i+1}) + B1(U)) - dim B1(U).
# Everything reduces to F2 ranks of explicit matrices.
# ---------------------------------------------------------------------------

def _f2_rank(cols) -> int:
    """Rank over GF(2) of a list of integer bitmasks (each a column vector)."""
    pivots = {}
    rank = 0
    for v in cols:
        v = int(v)
        while v:
            top = v.bit_length() - 1
            if top in pivots:
                v ^= pivots[top]
            else:
                pivots[top] = v
                rank += 1
                break
    return rank


def _graph_edges(D: np.ndarray, eps: float):
    """Undirected edge list of the eps-Rips graph, as sorted (i, j) pairs."""
    iu = np.triu_indices(D.shape[0], k=1)
    keep = D[iu] <= eps
    return list(zip(iu[0][keep].tolist(), iu[1][keep].tolist()))


def _fundamental_cycles(edges, n_vertices, edge_index):
    """Cycle-space basis as GF(2) bitmasks over the UNION edge indexing.

    Spanning forest by union-find; each non-forest edge closes exactly one
    fundamental cycle. Returns |E| - |V| + c independent cycles.
    """
    parent = list(range(n_vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    adj = {i: [] for i in range(n_vertices)}
    tree_edges, extra = [], []
    for (u, v) in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree_edges.append((u, v))
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))
        else:
            extra.append((u, v))

    # Root each forest component and record parent pointers for path lookup.
    depth = {}
    par = {}
    for start in range(n_vertices):
        if start in depth:
            continue
        depth[start] = 0
        par[start] = None
        stack = [start]
        while stack:
            x = stack.pop()
            for (y, e) in adj[x]:
                if y not in depth:
                    depth[y] = depth[x] + 1
                    par[y] = (x, e)
                    stack.append(y)

    cycles = []
    for (u, v) in extra:
        mask = 1 << edge_index[(u, v)]
        a, b = u, v
        # Walk both endpoints up to their common ancestor.
        while depth[a] > depth[b]:
            pa, pe = par[a]
            mask ^= 1 << edge_index[pe]
            a = pa
        while depth[b] > depth[a]:
            pb, pe = par[b]
            mask ^= 1 << edge_index[pe]
            b = pb
        while a != b:
            pa, pe = par[a]
            mask ^= 1 << edge_index[pe]
            a = pa
            pb, pe2 = par[b]
            mask ^= 1 << edge_index[pe2]
            b = pb
        cycles.append(mask)
    return cycles


def _triangle_boundaries(edge_set, edge_index):
    """GF(2) boundary vectors of every 2-simplex of the flag complex."""
    nbr = {}
    for (u, v) in edge_set:
        nbr.setdefault(u, set()).add(v)
        nbr.setdefault(v, set()).add(u)
    cols = []
    for (u, v) in edge_set:
        for w in nbr.get(u, ()) & nbr.get(v, ()):
            if w > v:                      # each triangle once, u < v < w
                cols.append((1 << edge_index[(u, v)])
                            ^ (1 << edge_index[(min(u, w), max(u, w))])
                            ^ (1 << edge_index[(min(v, w), max(v, w))]))
    return cols


def _f2_intersect(U, W, n_bits: int):
    """Basis of U n W for two GF(2) subspaces given by spanning sets of
    bitmasks, via the Zassenhaus algorithm.

    Stack rows [u | u] for u in U and [w | 0] for w in W in a 2n-bit space and
    row-reduce; the rows whose HIGH half vanished lie in both spans, and their
    low halves are a basis of the intersection.
    """
    rows = [(int(u) << n_bits) | int(u) for u in U]
    rows += [(int(w) << n_bits) for w in W]
    low_mask = (1 << n_bits) - 1
    pivots = {}
    reduced = []
    for v in rows:
        while v:
            top = v.bit_length() - 1
            if top in pivots:
                v ^= pivots[top]
            else:
                pivots[top] = v
                reduced.append(v)
                break
    return [v & low_mask for v in reduced
            if v != 0 and (v >> n_bits) == 0]


def _knn_edges(X: np.ndarray, k: int):
    """Symmetric kNN edge list. Scale-free, so layers with very different
    activation norms are compared on the same footing -- a fixed metric radius
    would make deep layers (large norms) look artificially disconnected."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 3:
        return []
    k = int(min(k, n - 1))
    _, idx = cKDTree(X).query(X, k=k + 1)
    idx = np.atleast_2d(idx)
    out = set()
    for i in range(n):
        for j in idx[i, 1:]:
            j = int(j)
            if i != j:
                out.add((min(i, j), max(i, j)))
    return sorted(out)


def _h1_persisting_edges(e_a, e_b, n: int):
    """One zigzag step between two layers on a shared vertex set.

    Gardinazzi et al. build the zigzag through INTERSECTIONS
    (``compute_layers_with_intersection``):

        X_i  <--f--  X_i n X_{i+1}  --g-->  X_{i+1}

    (the union would fill in classes present in both layers: a hexagon with
    spokes from a centre to alternate vertices in A and B persists in the
    intersection zigzag but dies in the union).

    For a type-A3 quiver V_A <-f- V_M -g-> V_B, the multiplicity of the
    full-support bar [1,3] -- the classes present in the middle AND surviving
    into both ends -- is

        m[1,3] = dim V_M - dim(ker f + ker g).

    With V_M = Z1(M)/B1(M), ker f = (Z1(M) n B1(A))/B1(M) and likewise for g,
    the B1(M) terms cancel and this reduces to

        m[1,3] = dim Z1(M) - dim( (Z1(M) n B1(A)) + (Z1(M) n B1(B)) ),

    which needs a genuine subspace INTERSECTION, not just ranks -- hence
    _f2_intersect above.

    Returns ``(dim_h1_a, dim_h1_b, n_shared)``, the layers' own H1 dimensions
    and the shared classes, so died = dim_h1_a - n_shared and
    born = dim_h1_b - n_shared.
    """
    e_a = list(e_a)
    e_b = list(e_b)
    e_m = sorted(set(e_a) & set(e_b))
    e_all = sorted(set(e_a) | set(e_b))
    if not e_all:
        return 0, 0, 0
    idx = {e: i for i, e in enumerate(e_all)}
    nb = len(e_all)

    b_a = _triangle_boundaries(e_a, idx)
    b_b = _triangle_boundaries(e_b, idx)
    z_a = _fundamental_cycles(e_a, n, idx)
    z_b = _fundamental_cycles(e_b, n, idx)

    dim_h1_a = len(z_a) - _f2_rank(b_a)
    dim_h1_b = len(z_b) - _f2_rank(b_b)

    if not e_m:
        return max(0, dim_h1_a), max(0, dim_h1_b), 0

    z_m = _fundamental_cycles(e_m, n, idx)
    if not z_m:
        return max(0, dim_h1_a), max(0, dim_h1_b), 0

    k_f = _f2_intersect(z_m, b_a, nb) if b_a else []
    k_g = _f2_intersect(z_m, b_b, nb) if b_b else []
    shared = len(z_m) - _f2_rank(list(k_f) + list(k_g))
    return max(0, dim_h1_a), max(0, dim_h1_b), max(0, shared)


def _h1_persisting_across(D_a: np.ndarray, D_b: np.ndarray, eps: float):
    """Distance-matrix entry point (used by the ground-truth tests)."""
    return _h1_persisting_edges(_graph_edges(D_a, eps),
                                _graph_edges(D_b, eps), D_a.shape[0])


def _zigzag_summary(
    layer_clouds: List[np.ndarray],
    n_neighbors: int = 5,
    persistence_frac: float = 0.3,
) -> Dict[str, float]:
    """Band and GF(2) step summaries of per-layer clouds (BLME proxies).

    Given per-layer point clouds (same point identities across layers, in
    depth order), it counts each layer's significant H0/H1 content and
    summarises it along depth. The H1 "layer bands" are a PRESENCE summary:
    contiguous runs of layers in which some loop exists, not feature identity.
    The identity-aware step quantities come from ``_h1_persisting_edges``.

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

    # Identity-aware step: does the SAME loop survive from layer l to l+1
    # (test_same_loop_persists_and_shuffled_loop_does_not).
    n_pts = min((len(c) for c in layer_clouds if c is not None), default=0)
    persist_steps, born_steps, died_steps, dim_a_steps = [], [], [], []
    if n_layers >= 2 and n_pts >= 8:
        try:
            edge_sets = [_knn_edges(np.asarray(X, dtype=np.float64)[:n_pts],
                                    n_neighbors)
                         for X in layer_clouds]
            for l in range(n_layers - 1):
                h1_a, h1_b, shared = _h1_persisting_edges(
                    edge_sets[l], edge_sets[l + 1], n_pts)
                persist_steps.append(shared)
                # Against the layers' own H1 dimensions (rank f / rank g would
                # drop the ker f / ker g bars).
                born_steps.append(max(0, h1_b - shared))   # new in layer l+1
                died_steps.append(max(0, h1_a - shared))   # gone after layer l
                dim_a_steps.append(h1_a)
        except Exception as exc:                          # pragma: no cover
            logger.info(f"  induced-rank H1 matching skipped: "
                        f"{type(exc).__name__}: {exc}")
            persist_steps, born_steps, died_steps, dim_a_steps = [], [], [], []

    # The paper's multi-layer zigzag barcode (dionysus).
    true_zz = {}
    if n_layers >= 2 and n_pts >= 8:
        try:
            true_zz = _zigzag_paper_metric(
                [np.asarray(c, dtype=np.float64)[:n_pts] for c in layer_clouds],
                n_neighbors=n_neighbors, dim=1, max_points=n_pts)
        except Exception as exc:
            logger.info(f"  true zigzag unavailable ({type(exc).__name__}: "
                        f"{exc}); falling back to the GF(2) pairwise family")
            true_zz = {}

    def _m(v):
        return float(np.mean(v)) if v else float("nan")

    # survival = shared / dim H1(layer l). Raw counts are a kNN flag complex's
    # cycle rank, which grows with N, so only ratios are features.
    surv = [sh / dim_a_steps[i] for i, sh in enumerate(persist_steps)
            if i < len(dim_a_steps) and dim_a_steps[i] > 0]

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
        # A dead H1 family is reported as a condition.
        "_meta_h1_family_is_identically_zero": bool(betti1_fraction == 0.0),
        **true_zz,
        # survival_rate drops zero-Betti steps (the aggregation footnote 6
        # exists to prevent), so it is _meta_; zz_true_* counts every step.
        "_meta_zz_h1_survival_rate": _m(surv),
        # Turnover: a Jaccard distance on the two layers' H1 classes,
        # (born + died) / (dim_a + born), bounded in [0, 1]; steps where both
        # layers are trivial are dropped (test_turnover_rate_is_bounded_and_scale_free).
        "zz_h1_turnover_rate": _m([
            (born_steps[i] + died_steps[i]) / (dim_a_steps[i] + born_steps[i])
            for i in range(min(len(born_steps), len(dim_a_steps)))
            if (dim_a_steps[i] + born_steps[i]) > 0
        ]),
        # Raw counts grow with N.
        "_meta_zz_h1_persisting_mean": _m(persist_steps),
        "_meta_zz_h1_born_mean": _m(born_steps),
        "_meta_zz_h1_died_mean": _m(died_steps),
        "_meta_zz_h1_dim_mean": _m(dim_a_steps),
        "_meta_zz_n_steps": int(len(persist_steps)),
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

    One mean-pooled point per input at every block output (the shared cache's
    per-sample states when available, else a private forward), aligned across
    depth. See the module docstring for the three output families.
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
        # A profile of block outputs: the deepest entry is the raw last block.
        if cache is not None and getattr(cache, "is_populated", False) and use_cache:
            per_layer_chunks = cache.get_hidden_states(
                layer_idx="all", num_samples=num_samples, per_sample=True,
                replace_final_norm=True,
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

        # The hook is removed in `finally` (tests/tasks/test_forward_hook_leaks.py).
        try:
            layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
            # Same estimand as the cache route: the raw last block replaces ln_f.
            _sink = {}
            _hook = layers[-1].register_forward_hook(
                lambda mod, inp, outp: _sink.__setitem__(
                    "h", (outp[0] if isinstance(outp, tuple) else outp).detach()))
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    enc = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=512,
                    ).to(device)
                    _sink.pop("h", None)
                    out = model(**enc, output_hidden_states=True)
                    hs = list(out.hidden_states)
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None and len(hs) >= 2
                            and _h_raw.shape == hs[-1].shape):
                        hs[-1] = _h_raw
                    for l_idx in range(n_layers):
                        h = hs[l_idx + 1][0]  # (T, D)
                        # Pooled in float64 and kept there.
                        layer_reps[l_idx].append(h.double().mean(dim=0).cpu().numpy())
        finally:
            _hook.remove()

        return [np.stack(layer_reps[l]) for l in range(n_layers)]


# ---------------------------------------------------------------------------
# The paper's multi-layer zigzag barcode (dionysus). Z_p(l1, l2) needs each
# bar's death, a property of the whole layer sequence that no pairwise step
# can recover. dionysus==2.1.1: an unpinned install can fall through to an
# sdist that does not build against Boost 1.74.
# ---------------------------------------------------------------------------

def _layer_simplices(X: np.ndarray, k: int):
    """Vertices, edges and triangles of one layer's kNN flag complex."""
    edges = _knn_edges(X, k)
    n = len(X)
    nbr = {}
    for (u, v) in edges:
        nbr.setdefault(u, set()).add(v)
        nbr.setdefault(v, set()).add(u)
    tris = []
    for (u, v) in edges:
        for w in nbr.get(u, ()) & nbr.get(v, ()):
            if w > v:
                tris.append((u, v, w))
    return ([(i,) for i in range(n)], [tuple(e) for e in edges], tris)


def _true_zigzag_barcode(layer_clouds, n_neighbors=5, max_points=None):
    """Full zigzag barcode over the layer sequence, via dionysus fast_zigzag.

    Returns ``(dgms, n_layers)`` where ``dgms[p]`` is the dimension-p barcode.
    Bar coordinates are in dionysus's zigzag time units: a simplex present in
    layers [a, b] is given the interval [a, b + 1).
    """
    import dionysus as dio

    clouds = [np.asarray(c, dtype=np.float64) for c in layer_clouds]
    if len(clouds) < 2:
        return None, 0
    n_pts = min(len(c) for c in clouds)
    if max_points:
        n_pts = min(n_pts, int(max_points))
    if n_pts < 8:
        return None, 0
    clouds = [c[:n_pts] for c in clouds]
    L = len(clouds)

    # Presence set per simplex across layers.
    present = {}
    for li, X in enumerate(clouds):
        verts, edges, tris = _layer_simplices(X, n_neighbors)
        for s in verts + edges + tris:
            present.setdefault(s, []).append(li)

    simplices, times = [], []
    for s, layers in present.items():
        layers.sort()
        # Contiguous runs -> [birth, death) intervals.
        iv, start, prev = [], layers[0], layers[0]
        for x in layers[1:]:
            if x == prev + 1:
                prev = x
            else:
                iv += [float(start), float(prev + 1)]
                start = prev = x
        iv += [float(start), float(prev + 1)]
        simplices.append(dio.Simplex(list(s)))
        times.append(iv)

    f = dio.Filtration(simplices)
    # Filtration reorders; re-align times to the filtration's own order.
    order = {tuple(sorted(sx)): i for i, sx in enumerate(present.keys())}
    times_sorted = [times[order[tuple(sorted(list(sx)))]] for sx in f]
    _, dgms, _ = dio.zigzag_homology_persistence(f, times_sorted)
    return dgms, L


def _zigzag_paper_metric(layer_clouds, n_neighbors=5, dim=1, max_points=None):
    """Gardinazzi et al. Z_p: the share of dimension-p bars that span [l1, l2].

    Z_p(l1, l2) = #{bars : birth <= min(l1,l2) and death > max(l1,l2)} / #bars_p

    Reported here for ADJACENT pairs (mean and min) and for the whole depth
    (first layer to last), which is the quantity the paper's layer-band analysis
    reads off.
    """
    dgms, L = _true_zigzag_barcode(layer_clouds, n_neighbors, max_points)
    if dgms is None or len(dgms) <= dim or L < 2:
        return {}
    bars = [(float(p.birth), float(p.death)) for p in dgms[dim]]
    if not bars:
        return {"zz_true_n_bars": 0}

    def _betti(l):
        """beta_p at layer l = bars alive there."""
        return sum(1 for b, d in bars if b <= l < d)

    def span(l1, l2):
        """Z_p(l1, l2) (Eq. 7), normalised by beta_p(l1), so Z_p(l, l) = 1."""
        lo, hi = min(l1, l2), max(l1, l2)
        denom = _betti(lo)
        if denom == 0:
            # Footnote 6 / post_process.py:30-32: div(num, 0) = 0.
            return 0.0
        return sum(1 for b, d in bars if b <= lo and d > hi) / denom

    # Every step counts (footnote 6).
    adj = [x for x in (span(l, l + 1) for l in range(L - 1)) if not np.isnan(x)]
    return {
        "zz_true_adjacent_mean": float(np.mean(adj)) if adj else float("nan"),
        "zz_true_adjacent_min": float(np.min(adj)) if adj else float("nan"),
        "zz_true_full_depth": span(0, L - 1),
        "zz_true_n_bars": int(len(bars)),
        "zz_true_mean_bar_len": float(np.mean([d - b for b, d in bars])),
    }
