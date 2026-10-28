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


# ---------------------------------------------------------------------------
# The paper's point clouds: the last token of each prompt, tokenised without
# special tokens, at each block's post-attention normalisation and at the final
# norm (extract.py:56-62,73; helpers.py:5-22; dataset_utils.py:14-22).
# ---------------------------------------------------------------------------
_MLP_ATTRS = ("mlp", "feed_forward", "ffn", "block_sparse_moe")


def _block_mlp(block):
    """The block's MLP submodule; its input is the post-attention norm output."""
    for name in _MLP_ATTRS:
        mod = getattr(block, name, None)
        if isinstance(mod, torch.nn.Module):
            return name, mod
    return None, None


def _first_tensor(args, kwargs):
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _zigzag_window(model, max_length: int) -> int:
    cfg = getattr(model, "config", None)
    limit = None
    for attr in ("max_position_embeddings", "n_positions"):
        v = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            limit = v
            break
    return int(min(max_length, limit)) if limit else int(max_length)


def _paper_clouds(model, tokenizer, texts, window: int, mean_rows=None):
    """Last-token rows at the L post-attention-norm sites and the final norm.

    The norm2 site is read as the input of the block's MLP: that is exactly the
    ``post_attention_layernorm`` output the reference hooks on Llama / Mistral /
    Pythia (``ln_2`` on GPT-2, ``pre_feedforward_layernorm`` on Gemma 2/3), and
    whether it really is a normalisation output is checked on the first forward.

    Args:
        texts: prompts (stripped, as ``dataset_utils.encode_func`` does).
        window: tokenisation window (truncation).
        mean_rows: optional set of row indices for which the mean over all tokens
            (the reference's ``use_last_token=False`` pooling) is kept as well.

    Returns ``(last, mean, info)``: ``last`` is a list of L+1 ``(N, d)`` float64
    arrays (sites 0..L-1, then the final norm); ``mean`` maps a row index to its
    list of L+1 pooled vectors; ``info`` describes what was read.
    """
    from ..common import get_final_norm
    from .._tokens import strip_trailing_special

    blocks = get_layers(model)
    final_norm = get_final_norm(model)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("zigzag: could not locate the transformer blocks")
    if final_norm is None:
        raise RuntimeError("zigzag: could not locate the final normalisation layer")
    mlps = []
    for i, b in enumerate(blocks):
        _, mlp = _block_mlp(b)
        if mlp is None:
            raise RuntimeError(f"zigzag: block {i} has no MLP submodule ({_MLP_ATTRS})")
        mlps.append(mlp)
    n_sites = len(blocks) + 1
    device = next(model.parameters()).device
    mean_rows = set(mean_rows or ())

    cur = {}
    probe = {}            # first forward only: norm child outputs per block
    norm_src = [None] * len(blocks)
    handles = []

    def _pre(i):
        def fn(mod, args, kwargs):
            x = _first_tensor(args, kwargs)
            if x is None:
                return
            cur[i] = x
            outs = probe.get(i)
            if outs is not None:
                for name, o in outs.items():
                    if o is x or (o.data_ptr() == x.data_ptr() and o.shape == x.shape):
                        norm_src[i] = name
        return fn

    def _norm_probe(i, name):
        def fn(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            if isinstance(o, torch.Tensor) and i in probe:
                probe[i][name] = o
        return fn

    def _final(mod, args, out):
        cur["final"] = out[0] if isinstance(out, tuple) else out

    last = [[] for _ in range(n_sites)]
    mean = {}
    n_at_window = n_bos = 0
    probe_handles = []
    try:
        for i, mlp in enumerate(mlps):
            handles.append(mlp.register_forward_pre_hook(_pre(i), with_kwargs=True))
            probe[i] = {}
            for name, child in blocks[i].named_children():
                if "norm" in type(child).__name__.lower():
                    probe_handles.append(child.register_forward_hook(_norm_probe(i, name)))
        handles.append(final_norm.register_forward_hook(_final))
        bos = getattr(tokenizer, "bos_token_id", None)
        with torch.no_grad():
            for r, text in enumerate(texts):
                enc = tokenizer(text.strip(), add_special_tokens=False, return_tensors="pt",
                                truncation=True, max_length=window)
                enc, _ = strip_trailing_special(enc, tokenizer)
                T = int(enc["input_ids"].shape[1])
                n_at_window += int(T >= window)
                n_bos += int(bos is not None and T > 0 and int(enc["input_ids"][0, 0]) == bos)
                cur.clear()
                model(**{k: v.to(device) for k, v in enc.items()})
                if probe:
                    probe.clear()
                    for h in probe_handles:
                        h.remove()
                    probe_handles = []
                missing = [i for i in range(len(blocks)) if i not in cur] + (
                    [] if "final" in cur else ["final"])
                if missing:
                    raise RuntimeError(f"zigzag: no activation captured at sites {missing}")
                acts = [cur[i] for i in range(len(blocks))] + [cur["final"]]
                for s, a in enumerate(acts):
                    last[s].append(a[0, T - 1].detach().to("cpu", torch.float64))
                if r in mean_rows:
                    mean[r] = [a[0].detach().to("cpu", torch.float64).mean(dim=0) for a in acts]
    finally:
        for h in handles + probe_handles:
            h.remove()
    clouds = [torch.stack(rows).numpy() if rows else np.empty((0, 0)) for rows in last]
    info = {
        "n_sites": n_sites,
        "site_norm_module": norm_src[0],
        "sites_are_norm_outputs": bool(all(n is not None for n in norm_src)),
        "n_at_window": int(n_at_window),
        "n_docs_starting_with_bos": int(n_bos),
    }
    return clouds, mean, info


def _prompts(dataset, n: int, tokenizer, window: int):
    """The first ``n`` non-empty prompts of ``dataset`` (stripped text)."""
    out = []
    for s in list(dataset or [])[:n]:
        text = (s.get("text", "") if isinstance(s, dict) else str(s)).strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=window)["input_ids"]
        if len(ids) >= 1:
            out.append(text)
    return out


@register_task("topology_zigzag_persistence")
class ZigzagPersistenceTask(DiagnosticTask):
    """Zigzag persistence of topological features across layers.

    The paper's point clouds (``_paper_clouds``): one last-token point per
    prompt at each of the L post-attention-norm sites and the final norm. See
    the module docstring for the three output families; the mean over each
    prompt's tokens (the reference's other pooling option) is kept as _meta_.
    """

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Zigzag Persistence (across-layer topology)...")

        if not HAS_RIPSER:
            return {"error": "ripser not installed. Install with: pip install ripser"}

        num_samples = int(self.config.get("num_samples", 40))
        n_neighbors = int(self.config.get("n_neighbors", 5))
        persistence_frac = float(self.config.get("persistence_frac", 0.3))
        window = _zigzag_window(model, int(self.config.get("max_length", 2048)))

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        texts = _prompts(dataset, num_samples, tokenizer, window)
        if len(texts) < 3:
            return {"error": "Need at least 3 non-empty prompts for zigzag persistence"}
        rows = list(range(len(texts)))
        layer_clouds, pooled, info = _paper_clouds(
            model, tokenizer, texts, window, mean_rows=rows)
        if len(layer_clouds) < 2:
            return {"error": "Need at least 2 layers for zigzag persistence"}
        n_points = min((len(c) for c in layer_clouds), default=0)

        summary = _zigzag_summary(
            layer_clouds,
            n_neighbors=n_neighbors,
            persistence_frac=persistence_frac,
        )
        summary["_meta_n_points"] = int(n_points)
        summary.update(_protocol_meta(info, window))
        # The reference's other pooling (use_last_token=False), on the same sites.
        mean_clouds = [np.stack([pooled[r][s] for r in rows]) for s in range(info["n_sites"])]
        try:
            mp = _zigzag_paper_metric(mean_clouds, n_neighbors=n_neighbors, dim=1,
                                      max_points=len(rows))
        except ImportError:
            mp = {}
        for k in ("zz_true_adjacent_mean", "zz_true_mean_bar_len", "zz_true_n_bars"):
            if k in mp:
                summary[f"_meta_meanpool_{k}"] = mp[k]
        return summary


def _protocol_meta(info, window):
    return {
        "_meta_zz_n_clouds": int(info["n_sites"]),
        "_meta_zz_site_norm_module": info["site_norm_module"],
        "_meta_zz_sites_are_norm_outputs": bool(info["sites_are_norm_outputs"]),
        "_meta_zz_window": int(window),
        "_meta_zz_n_at_window": int(info["n_at_window"]),
        "_meta_zz_n_docs_starting_with_bos": int(info["n_docs_starting_with_bos"]),
    }


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
