"""
Betti Curve / Topological Complexity Trajectory — tracks how Betti numbers
change across ALL layers of the network.

The key insight from Naitzat et al. (JMLR 2020) is that well-generalized
networks progressively simplify topology: Betti numbers (connected components,
loops) decrease with depth. The rate of this decrease and the final complexity
are both informative structural metrics.

Method (refined adaptation of Naitzat et al. and topnn/topnn_framework):
  Betti numbers are computed on the kNN graph, not on raw Euclidean distance.
  * beta_0 = number of connected components of the symmetric kNN graph.
  * beta_1 = H1 classes alive at a radius of ``hop_radius`` hops on the
    unit-edge (hop) metric of each component, read with ripser -- the paper's
    Eirene read at minrad=1, maxrad=2, numrad=1.
  The homology engine differs from the paper's Eirene; the construction is
  validated on the paper's own examples (tests/tasks/test_topology.py).

References:
- "Topology of Deep Neural Networks" (Naitzat, Zhitnikov & Lim, JMLR 21(184):1-40, 2020, arXiv:2004.06093)
"""

import torch
import math
import numpy as np
import warnings

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ._points import fallback_dataset, last_token_points, point_meta, rows_policy
import logging
logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _knn_k(n: int, n_neighbors="auto") -> int:
    """The kNN degree for an n-point cloud. k COUNTS THE POINT ITSELF, as the
    released loader's NearestNeighbors(n_neighbors=k).kneighbors_graph(X) does
    (X passed explicitly, betti_calc_parallel.py:60-66): k - 1 true neighbours,
    k capped at n. "auto" is max(14, ceil(log2 N)); 14 is the paper's
    calibrated k for its hollow-torus example (Fig. 3(e))."""
    if isinstance(n_neighbors, str) and n_neighbors == "auto":
        k = int(max(14, math.ceil(math.log2(max(n, 2)))))
    else:
        k = int(n_neighbors)
    return int(max(1, min(k, n)))


def _count_betti(data, maxdim=1, n_neighbors="auto", persistence_frac=0.3,
                 return_persistence=False, hop_metric=True,
                 hop_radius=2.0):
    """Count Betti numbers from a point cloud via graph-geodesic homology.
    Raises a clear ImportError (not a bare NameError on ``ripser``) when the
    optional ripser dependency is missing.

    Follows Naitzat et al. (JMLR 2020) and the reference topnn_framework:
    build a kNN graph, take graph-geodesic (shortest-path) distances, and
    read topology from that — manifold-aware, unlike raw Euclidean distance.

      * ``beta_0`` = number of connected components of the symmetric kNN
        graph (scipy ``connected_components``). Parameter-light and robust;
        recovers the true cluster count.
      * ``beta_1`` = with ``hop_metric`` (default), the H1 classes alive at
        ``hop_radius`` hops; otherwise the H1 loops whose NORMALIZED
        persistence ``(death - birth) / geodesic_diameter`` exceeds
        ``persistence_frac``.

    Args:
        data: (N, D) point cloud.
        maxdim: max homology dimension for ripser (1 = up to loops).
        n_neighbors: kNN graph degree (capped at N-1).
        persistence_frac: normalized-persistence threshold for counting loops.

    Returns:
        (betti_0, betti_1): connected components and significant loops.
    """
    if not HAS_RIPSER:
        raise ImportError(
            "geometry/topology Betti-curve analysis requires the optional "
            "'ripser' package. Install it with: pip install ripser"
        )
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path, connected_components

    X = np.asarray(data, dtype=np.float64)
    X = X[np.all(np.isfinite(X), axis=1)]

    # Duplicate points are quotiented, as the released loader does
    # (topnn_framework betti_calc_parallel.py read_csv: np.unique(..., axis=0)).
    X = np.unique(X, axis=0)

    n = len(X)
    if n < 3:
        # beta_0 is the component count of the graph: with k >= 1 the
        # surviving points are joined, so it is 1 (0 for an empty cloud).
        _b0 = 1 if n >= 1 else 0
        return ((_b0, 0, 0.0, 0.0) if return_persistence else (_b0, 0))

    k = _knn_k(n, n_neighbors)
    tree = cKDTree(X)
    # Column 0 is the point itself (with exact duplicates removed above).
    dists, idx = tree.query(X, k=k)
    dists = np.asarray(dists).reshape(n, k)
    idx = np.asarray(idx).reshape(n, k)

    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].ravel()
    # csgraph reads an explicit zero as "no edge", so distances are floored.
    vals = np.maximum(dists[:, 1:].ravel(), 1e-12)
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize (mutual + one-directional kNN)

    # beta_0 is read from the kNN graph as built; no connectivity repair.
    n_components, comp_labels = connected_components(graph, directed=False)
    betti_0 = int(n_components)

    # H1 per connected component (H1 of a disjoint union is the direct sum),
    # so every geodesic is finite and no infinity cap is needed.
    lifetimes = []
    alive = []
    for comp in range(n_components):
        idx = np.flatnonzero(comp_labels == comp)
        if idx.size < 4:
            continue                       # too small to carry a 1-cycle
        sub = graph[idx][:, idx]
        if hop_metric:
            # Unit-edge (hop) metric, as Naitzat et al. use (Sec. 6.3:
            # "Each edge, regardless of its Euclidean length, has the same
            # length of one").
            sub = sub.copy()
            sub.data = np.ones_like(sub.data)
        geo_c = shortest_path(sub, method="D", directed=False)
        finite_c = geo_c[np.isfinite(geo_c)]
        if finite_c.size == 0:
            continue
        scale_c = float(finite_c.max())
        if scale_c <= 0:
            continue
        # ripser casts to float32: a power-of-two rescale (exact) keeps
        # out-of-range magnitudes finite; hop counts are never rescaled.
        _gm = float(np.max(np.abs(geo_c))) if geo_c.size else 0.0
        if _gm > 0 and np.isfinite(_gm) and not (1e-6 < _gm < 1e6):
            _e = int(max(-126, min(127, round(math.log2(_gm)))))
            geo_c = np.ldexp(geo_c, -_e)
            scale_c = float(np.ldexp(scale_c, -_e))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dgms_c = ripser(geo_c, distance_matrix=True, maxdim=maxdim)["dgms"]
        h1_c = dgms_c[1] if len(dgms_c) > 1 else []
        if hop_metric:
            # Fixed-radius read: classes alive at `hop_radius` hops (Eirene at
            # minrad=1, maxrad=2, numrad=1 in topnn_framework).
            alive.extend(1 for b, d in h1_c
                         if np.isfinite(d) and b <= hop_radius < d)
        # Normalised by this component's geodesic diameter.
        lifetimes.extend((d - b) / scale_c for b, d in h1_c if np.isfinite(d))

    betti_1 = (int(sum(alive)) if hop_metric
               else sum(1 for lt in lifetimes if lt > persistence_frac))

    # Threshold-free H1 summaries.
    if not return_persistence:
        return betti_0, int(betti_1)
    max_pers = float(max(lifetimes)) if lifetimes else 0.0
    total_pers = float(sum(lifetimes)) if lifetimes else 0.0
    return betti_0, int(betti_1), max_pers, total_pers


@register_task("topology_betti_curve")
class BettiCurveTask(DiagnosticTask):
    """
    Traces the Betti number trajectory (β0, β1) across all layers of the model.

    Following Naitzat et al. (JMLR 2020), tracks how topological complexity
    changes with depth. Reports:
    - β0 at each layer (connected components)
    - β1 at each layer (loops/holes)
    - Rate of topological simplification
    """
    TOKEN_POLICY = "last_per_sequence"

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Betti Curve (Topological Complexity Trajectory)...")
        # Q1: one point per sequence (its last token), N = 1000 for every model.
        num_samples = int(self.config.get("num_samples", 1000))
        n_neighbors = self.config.get("n_neighbors", "auto")
        if n_neighbors != "auto":
            n_neighbors = int(n_neighbors)
        persistence_frac = float(self.config.get("persistence_frac", 0.3))
        use_cache = self.config.get("use_cache", True)

        if not HAS_RIPSER:
            msg = "Ripser library not installed. Install with: pip install ripser"
            logger.info(msg)
            return {"error": msg}

        if dataset is None:
            dataset = fallback_dataset(num_samples, "Betti curve")
        pts, pinfo = last_token_points(
            model, tokenizer, dataset, cache, num_samples=num_samples,
            layers=lambda L: range(L), policy=rows_policy(self.TOKEN_POLICY),
            use_cache=use_cache)
        if not pts or min(len(X) for X in pts.values()) < 3:
            return {"error": "Need at least 3 samples for Betti curve"}
        num_layers = len(pts)
        layer_reps = {l: pts[l] for l in range(num_layers)}
        _last_is_block = bool(pinfo["last_is_block"])

        betti_0_curve = []
        betti_1_curve = []
        h1_maxpers_curve = []
        h1_totpers_curve = []

        for l_idx in range(num_layers):
            data = np.array(layer_reps[l_idx])
            b0, b1, mp, tp = _count_betti(
                data, maxdim=1,
                n_neighbors=n_neighbors, persistence_frac=persistence_frac,
                return_persistence=True,
            )
            betti_0_curve.append(b0)
            betti_1_curve.append(b1)
            h1_maxpers_curve.append(mp)
            h1_totpers_curve.append(tp)

        results = {
            "betti_0_curve": betti_0_curve,
            "betti_1_curve": betti_1_curve,
            "betti_0_first": betti_0_curve[0],
            "betti_0_last": betti_0_curve[-1],
        }

        # Topological simplification ratio
        if betti_0_curve[0] > 0:
            results["simplification_ratio"] = float(betti_0_curve[-1] / betti_0_curve[0])
        else:
            results["simplification_ratio"] = 1.0

        # Slope of beta_0 against normalised depth x in [0, 1].
        if len(betti_0_curve) > 1:
            denom = max(1, num_layers - 1)
            x = np.arange(num_layers, dtype=np.float64) / float(denom)
            slope = np.polyfit(x, betti_0_curve, 1)[0]
            results["betti_0_decay_rate"] = float(slope)
        else:
            results["betti_0_decay_rate"] = 0.0

        # Layer with the most loops, absolute and as a depth fraction.
        results["max_betti_1_layer"] = int(np.argmax(betti_1_curve))
        results["max_betti_1_layer_frac"] = (
            float(np.argmax(betti_1_curve)) / float(max(1, num_layers - 1)))

        # Under the hop metric every H1 bar has lifetime 1, so the normalised
        # maximum is 1 / (hop diameter): a reciprocal diameter, not a spread.
        results["_meta_last_entry_is_block_output"] = bool(_last_is_block)
        results["h1_max_persistence_curve"] = h1_maxpers_curve
        results["h1_max_persistence"] = float(np.max(h1_maxpers_curve))
        results["h1_max_persistence_mean"] = float(np.mean(h1_maxpers_curve))
        results["h1_total_persistence_mean"] = float(np.mean(h1_totpers_curve))
        results["h1_max_persistence_layer_frac"] = (
            float(np.argmax(h1_maxpers_curve)) / float(max(1, num_layers - 1)))
        results["max_betti_1"] = int(np.max(betti_1_curve))
        results.update(point_meta(pinfo, pts[num_layers // 2]))
        _n_mid = int(len(np.unique(np.asarray(pts[num_layers // 2]), axis=0)))
        results["_meta_knn_k"] = _knn_k(_n_mid, n_neighbors)
        results["_meta_knn_true_neighbours"] = results["_meta_knn_k"] - 1
        return results
