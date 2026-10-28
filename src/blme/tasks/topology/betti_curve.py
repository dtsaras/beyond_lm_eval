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
from ..common import get_layers
import logging
logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


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

    # "auto": max(14, ceil(log2 N)); 14 is the paper's calibrated k for its
    # hollow-torus example (Fig. 3(e)), which reads beta_1 = 2 there.
    if isinstance(n_neighbors, str) and n_neighbors == "auto":
        k = int(max(14, math.ceil(math.log2(max(n, 2)))))
    else:
        k = int(n_neighbors)
    k = int(min(k, n - 1))
    tree = cKDTree(X)
    # Column 0 is the point itself (with exact duplicates removed above).
    dists, idx = tree.query(X, k=k + 1)
    dists = np.atleast_2d(dists)
    idx = np.atleast_2d(idx)

    rows = np.repeat(np.arange(n), k)
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
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Betti Curve (Topological Complexity Trajectory)...")
        # The topology-of-representations literature works at ~10^3 points.
        num_samples = self.config.get("num_samples", 1000)
        n_neighbors = self.config.get("n_neighbors", "auto")
        if n_neighbors != "auto":
            n_neighbors = int(n_neighbors)
        persistence_frac = float(self.config.get("persistence_frac", 0.3))

        if not HAS_RIPSER:
            msg = "Ripser library not installed. Install with: pip install ripser"
            logger.info(msg)
            return {"error": msg}

        device = next(model.parameters()).device

        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    dataset.append({"text": dset[i]["text"]})
            except Exception as e:
                logger.info(
                    f"Warning: could not load lambada_openai ({type(e).__name__}: "
                    "missing `datasets`, offline, or download error); falling back "
                    "to default examples."
                )
                dataset = [{"text": f"Topological sample {i} for Betti curve calculation."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for Betti curve"}

        layers = get_layers(model)
        num_layers = len(layers)

        # Collect mean-pooled representations at EVERY layer
        layer_reps = {l: [] for l in range(num_layers)}

        # post-final-norm: SUBSTITUTED -- the deepest cloud is the raw last block (HF's last entry is
        # ln_f(block L-1)); the hook is removed in `finally`.
        _sink = {}
        _hook = layers[-1].register_forward_hook(
            lambda mod, inp, outp: _sink.__setitem__(
                "h", (outp[0] if isinstance(outp, tuple) else outp).detach()))
        _n_fwd = _n_sub = 0
        try:
            with torch.no_grad():
                for s in samples:
                    text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    _sink.pop("h", None)
                    out = model(**inputs, output_hidden_states=True)
                    _hs = list(out.hidden_states)
                    _n_fwd += 1
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None
                            and tuple(_h_raw.shape) == tuple(_hs[-1].shape)):
                        _hs[-1] = _h_raw
                        _n_sub += 1

                    for l_idx in range(num_layers):
                        hidden = _hs[l_idx + 1][0]
                        # One mean-pooled vector per document, pooled in float64.
                        rep = hidden.double().mean(dim=0).cpu().numpy()
                        layer_reps[l_idx].append(rep)
        finally:
            _hook.remove()
        _last_is_block = bool(_n_fwd > 0 and _n_sub == _n_fwd)

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

        return results
