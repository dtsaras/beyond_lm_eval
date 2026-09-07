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


DISCLOSED DEVIATIONS (audit 2026-07-16): Gardinazzi et al. extract the
LAST-TOKEN representation per prompt at each layer; BLME mean-pools
tokens per prompt (consistent with the rest of the topology family).
This changes the point cloud, not the zigzag construction.
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
    # Floor zero distances (duplicate points): explicit-zero csr entries are
    # treated as NO edge by csgraph, inflating beta_0 (same fix as
    # betti_curve.py; audit 2026-07-16 demonstrated h0=7.0 vs 1.0 on a
    # corpus with one repeated line).
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

    # RESCALE TO UNIT SCALE BEFORE ripser.
    #
    # ripser casts the distance matrix to float32 internally
    # (ripser.py:324, `np.array(dm[...], dtype=np.float32)`), so entries above
    # float32's 3.4e38 overflow to +inf and the filtration silently loses its
    # top. On a 40-point unit circle -- betti_1 is exactly 1 by construction --
    # this reads 1 at every scale from 1e-30 through 1e38 and then **0** at
    # 2.406159650500376e38, collapsing the entire H1 family. With the rescale
    # it reads 1 at all of them.
    #
    # A POWER-OF-TWO rescale, matching `persistence_entropy.py`, and gated to
    # the out-of-range case. `np.ldexp` only shifts exponents, so it is exact
    # in binary floating point: `scale` is shifted by the same factor, both
    # sides of the `(d - b) > persistence_frac * scale` test move together, and
    # inside 1e-6..1e6 nothing happens at all. This provably cannot move any
    # shipped number -- unlike dividing by `scale`, which is a real division
    # and could flip a borderline count.
    #
    # `topology_homology` was given a rescale on 2026-09-07 and the sweep
    # stopped there; this site and `betti_curve.py` were both left behind
    # (Opus round 9, 2026-09-07). Population of ripser call sites: 5 --
    # homology, persistence_landscape and persistence_entropy already rescale.
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
# Genuine cross-layer H1 feature matching (F2 linear algebra).
#
# The band summary below counts, per layer, WHETHER a loop exists, and calls a
# contiguous run of such layers a "band". That does not track feature IDENTITY:
# five DIFFERENT loops in five consecutive layers produce exactly the same band
# as ONE loop that persists for five layers. Those are opposite topological
# stories and the zigzag barcode of Gardinazzi et al. separates them.
#
# No zigzag engine is installable here (dionysus needs boost and fails to
# build; pyfzz is not on PyPI; gudhi 3.12 exposes no zigzag binding). But the
# quantity the metric actually needs -- how many H1 classes SURVIVE from one
# layer to the next -- is computable exactly, without a full barcode, because
# every layer shares the same vertex set.
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

    so we do the same. (An earlier version of this code used the UNION, which
    is a different -- and worse -- object: adversarial review 2026-09-03 gave
    the counterexample of a hexagon 0-5 with centre 6, where layer A carries
    spokes {6-0,6-1,6-2} and layer B carries {6-3,6-4,6-5}. The hexagon is a
    nonzero H1 class in BOTH layers, but the flag complex of the union fills
    it in, so the union construction reports 0 persisting. The intersection
    construction correctly reports 1.)

    For a type-A3 quiver V_A <-f- V_M -g-> V_B, the multiplicity of the
    full-support bar [1,3] -- the classes present in the middle AND surviving
    into both ends -- is

        m[1,3] = dim V_M - dim(ker f + ker g).

    With V_M = Z1(M)/B1(M), ker f = (Z1(M) n B1(A))/B1(M) and likewise for g,
    the B1(M) terms cancel and this reduces to

        m[1,3] = dim Z1(M) - dim( (Z1(M) n B1(A)) + (Z1(M) n B1(B)) ),

    which needs a genuine subspace INTERSECTION, not just ranks -- hence
    _f2_intersect above.

    Returns ``(dim_h1_a, dim_h1_b, n_shared)``. Note the first two are now the
    layers' own H1 dimensions, so a caller can form the honest
        died = dim_h1_a - n_shared,  born = dim_h1_b - n_shared
    (the previous version used rank f and rank g, which silently DROP the
    ker f / ker g bars: on gpt2 at k=10 that made survival_rate exactly
    1.0000 at every step -- a constant feature).
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
    """Self-contained across-layer topological-persistence summary.

    This is the **verified artifact**. Given per-layer point clouds (same
    point identities across layers, in depth order), it computes for each
    layer the significant H0 (components) and H1 (loops) content and
    summarises how these features persist across the layer axis — the
    zigzag-persistence quantity of Gardinazzi et al. (2025), realised without
    a zigzag C++ engine.

    The H1 "layer bands" are a PRESENCE summary, not a barcode: they record
    the contiguous runs of layers in which SOME loop exists. They do not track
    feature identity, so N different loops in N consecutive layers give the
    same band as one loop persisting N layers. The identity-aware quantity is
    computed separately by ``_h1_persisting_across`` and reported as the
    ``zz_h1_persisting_*`` family; prefer those for any claim about loops
    surviving across depth.

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

    # ---- Identity-aware cross-layer H1 persistence ----
    # Unlike the band summary above, this asks whether the SAME loop survives
    # from layer l to layer l+1, via the rank of the induced map into the union
    # complex. See _h1_persisting_edges. Ground truth (24-point circle, kNN
    # k=5): the same circle in both layers gives persisting=1; a different
    # cycle on the same vertices gives persisting=0 -- while the band summary
    # reports an identical length-2 band in both cases.
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
                # Honest accounting against the layers' OWN H1 dimensions.
                # Using rank f / rank g here instead would drop the ker f /
                # ker g bars and pin survival_rate at exactly 1.0.
                born_steps.append(max(0, h1_b - shared))   # new in layer l+1
                died_steps.append(max(0, h1_a - shared))   # gone after layer l
                dim_a_steps.append(h1_a)
        except Exception as exc:                          # pragma: no cover
            logger.info(f"  induced-rank H1 matching skipped: "
                        f"{type(exc).__name__}: {exc}")
            persist_steps, born_steps, died_steps, dim_a_steps = [], [], [], []

    # TRUE multi-layer zigzag barcode (dionysus fast_zigzag), which is the
    # paper's actual construction. The GF(2) pairwise family below is retained
    # as a fallback for environments where dionysus will not build.
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

    # Fraction of a layer's loops that survive to the next layer. This is the
    # quantity the band summary was standing in for.
    # survival = shared / dim H1(layer l). The RAW counts are the cycle rank of
    # a kNN flag complex at one scale, which grows with the number of points
    # (measured on gpt2: 4.0 at num_samples=40 -> 22.1 at 100), so they are not
    # comparable across models and are demoted to _meta. The RATIO is the
    # reportable quantity.
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
        # A dead H1 family is a REPORTABLE CONDITION, not a value. At the
        # old `persistence_frac: 0.3` every H1 column was exactly 0.0000
        # on every model at every N tested -- seven constant feature
        # columns that looked like measurements.
        "_meta_h1_family_is_identically_zero": bool(betti1_fraction == 0.0),
        # ---- identity-aware zigzag quantities (prefer these) ----
        **true_zz,
        # DEMOTED to _meta_ (2026-09-05), and the reason is NOT what an earlier
        # version of this comment said.
        #
        # It claimed the two keys were BIT-IDENTICAL on real models. That was
        # measured BEFORE the footnote-6 fix, which made a zero-Betti step
        # contribute 0 rather than be dropped. They now differ on gpt2:
        # 0.5916666666666666 (= 71/120, over 10 retained steps) against
        # 0.5378787878787878 (= 71/132, over all 11), 9.1% apart -- while
        # remaining identical on distilgpt2 and pythia-70m (red-team review,
        # 2026-09-05).
        #
        # The demotion still stands, on the better ground: `survival_rate`
        # DROPS zero-Betti steps, which is exactly the biased aggregation
        # footnote 6 exists to prevent. The identity-aware `zz_true_*` family
        # counts every step and is the quantity to use.
        "_meta_zz_h1_survival_rate": _m(surv),
        # Steps where the layer has NO H1 class are dropped, exactly as
        # survival_rate drops them. The previous `/max(1, dim_a)` turned such a
        # step into an UNNORMALIZED count, reintroducing the scale dependence
        # the _meta_ demotion removed: the mean went 12 -> 57 -> 165 -> 385 as N
        # went 40 -> 400, and it fired on 1 of 11 gpt2 steps at the study config
        # (13.4% inflation) while survival_rate excluded those same steps --
        # the two exported metrics disagreed about which steps count.
        # Turnover as a Jaccard distance on the two layers' H1 classes:
        #     (born + died) / (dim_a + dim_b - shared) = 1 - shared/union
        # Bounded in [0, 1] and symmetric in the two layers. Dividing by dim_a
        # alone is NOT scale-free -- a step from dim_a = 1 to dim_b = 40 yields
        # 39 -- and the previous `/max(1, dim_a)` was worse still, turning
        # dim_a = 0 steps into unnormalized counts (mean 12 -> 385 as N went
        # 40 -> 400). Steps where both layers are trivial are dropped, exactly
        # as survival_rate drops them, so the two metrics agree on which steps
        # count.
        # Turnover as a Jaccard distance on the two layers' H1 classes:
        #     (born + died) / |union|,   |union| = dim_a + dim_b - shared
        #                                        = dim_a + born
        # Bounded in [0, 1] and symmetric in the two layers. Dividing by dim_a
        # alone is NOT scale-free -- a step from dim_a = 1 to dim_b = 40 gives
        # 39 -- and the previous `/max(1, dim_a)` was worse, turning dim_a = 0
        # steps into unnormalized counts (mean 12 -> 385 as N went 40 -> 400).
        # Steps where both layers are trivial are dropped, exactly as
        # survival_rate drops them, so the two metrics agree on which steps
        # count (adversarial review, 2026-09-03).
        "zz_h1_turnover_rate": _m([
            (born_steps[i] + died_steps[i]) / (dim_a_steps[i] + born_steps[i])
            for i in range(min(len(born_steps), len(dim_a_steps)))
            if (dim_a_steps[i] + born_steps[i]) > 0
        ]),
        # Raw counts are scale-dependent (cycle rank grows with N); kept for
        # auditing only, excluded from the feature matrix by the _meta prefix.
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

    Cross-layer topological persistence after Gardinazzi et al. (ICML 2025,
    arXiv:2410.11042). The identity-aware ``zz_h1_*`` outputs implement the
    paper's INTERSECTION zigzag, X_i <- X_i n X_{i+1} -> X_{i+1}, over GF(2)
    on the shared vertex set; ``_meta_zz_h1_survival_rate`` is the multiplicity of
    the full-support bar as a fraction of the layer's own H1 dimension.
    Reported as a RATIO because the raw dimension is the cycle rank of a kNN
    flag complex and grows with the point count (gpt2: 2.4 -> 65.5 as
    num_samples goes 40 -> 200, while the ratio holds at 0.59 / 0.54 / 0.59).

    The legacy band statistics below are a proxy. Uses
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
        # POST-FINAL-NORM: `layer_idx="all"` builds a PER-LAYER PROFILE, and
        # HF's last entry is `ln_f(b_{L-1})`. One LayerNorm output inside a
        # profile of block outputs is a category error under either
        # convention -- the profile's slope, extremes and quantiles all
        # compare it against entries of a different kind. Repo-wide sweep,
        # 2026-09-05: 24 of 39 call sites were missing this opt-in.
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

        # post-final-norm / hook hygiene: a task that fails must leave the model exactly as it found it. A leaked forward hook is paid by every LATER task in the run and pins the last activation tensor alive (discovery-based leak test, 2026-09-05).
        try:
            layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in range(n_layers)}
            # POST-FINAL-NORM, fallback route: `hidden_states[n_layers]` is
            # `ln_f(b_{L-1})`. The cache route now opts in, so without this the two
            # routes would compute different estimands -- which is how several of
            # these defects survived: the tested path and the shipped path differed.
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
                    # post-final-norm: SUBSTITUTED -- the deepest sampled layer must be a
                    # block output like every other entry in this profile.
                    hs = list(out.hidden_states)
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None and len(hs) >= 2
                            and _h_raw.shape == hs[-1].shape):
                        hs[-1] = _h_raw
                    for l_idx in range(n_layers):
                        h = hs[l_idx + 1][0]  # (T, D)
                        layer_reps[l_idx].append(h.float().mean(dim=0).cpu().numpy())
    
        finally:
            _hook.remove()

        return [np.stack(layer_reps[l]) for l in range(n_layers)]


# ---------------------------------------------------------------------------
# TRUE multi-layer zigzag persistence (dionysus fast_zigzag).
#
# The GF(2) machinery above computes, for ONE adjacent pair, the multiplicity of
# the full-support bar in a 3-term zigzag. That is a real quantity, but it is
# NOT the paper's metric, and the difference is not a technicality:
#
#     Z_p(l1, l2) counts bars with birth <= min(l1,l2) AND death > max(l1,l2)
#
# -- bars spanning the WHOLE interval, "and have existed throughout the layers
# in between". A bar's death index is a global property of the interval
# decomposition of the ENTIRE layer sequence and cannot be recovered by
# comparing two layers, so no pairwise construction can compute it.
#
# `dionysus==2.1.1` bundles fast_zigzag (Dey-Hou) under BSD-3-Clause and does
# compute the full barcode. The version pin matters: on glibc 2.34-2.38 an
# unpinned install falls through to an sdist that fails against Boost 1.74.
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
        """Z_p(l1, l2), normalized by beta_p(l1) -- NOT by the total bar count.

        The denominator matters and getting it wrong is not subtle: the paper's
        defining identity is Z_p(l, l) = 1, since every bar alive at l trivially
        spans [l, l]. Dividing by the total bar count instead gave **0.0705**
        against the authors' own shipped barcode
        (`llama_3_8b_dim_4_knn_4_last_token.csv`) where the identity requires
        exactly 1.0 -- off by 14.3x (adversarial review, 2026-09-03).
        """
        lo, hi = min(l1, l2), max(l1, l2)
        denom = _betti(lo)
        if denom == 0:
            # ZERO, not NaN. The paper's footnote 6 says verbatim
            # "Z_p(l1, l2) should be 0 by definition" when beta_p(l1) = 0, and
            # the released `post_process.py:30-32` implements exactly that with
            # `div(num, 0) = 0`. Returning NaN and DROPPING the step inverted
            # the summary: on [line, circle, circle] BLME reported adjacent
            # mean 1.000000 and min 1.000000 where the reference gives 0.500000
            # and 0.000000, and full-depth NaN where it gives 0.0 (independent
            # Opus review, 2026-09-05). A layer with no p-cycles has none
            # surviving -- that is a measurement, not a missing value.
            return 0.0
        return sum(1 for b, d in bars if b <= lo and d > hi) / denom

    # Every step counts now; `span` no longer returns NaN for a zero-Betti
    # layer, so dropping steps would re-introduce the same bias.
    adj = [x for x in (span(l, l + 1) for l in range(L - 1)) if not np.isnan(x)]
    return {
        "zz_true_adjacent_mean": float(np.mean(adj)) if adj else float("nan"),
        "zz_true_adjacent_min": float(np.min(adj)) if adj else float("nan"),
        "zz_true_full_depth": span(0, L - 1),
        "zz_true_n_bars": int(len(bars)),
        "zz_true_mean_bar_len": float(np.mean([d - b for b, d in bars])),
        # NOTE: a Z_p(l, l) == 1 self-check is TAUTOLOGICAL under this
        # implementation -- at l1 == l2 the numerator's condition
        # (birth <= l and death > l) is exactly the definition of beta_p(l),
        # which is the denominator, so it can never fail and proves nothing
        # (confirmed against the authors' own shipped barcode; adversarial
        # review, 2026-09-03). Removed rather than left as false assurance.
        #
        # The meaningful check is external: BLME's `_betti`/`span` reproduce the
        # authors' published beta_1 vector (14176, 15828, ..., 16321) and their
        # Z_1(l, l+1) row (0.0926, 0.2015, ..., 0.4335) EXACTLY when run on the
        # barcode CSVs shipped with RitAreaSciencePark/ZigZagLLMs.
    }
