"""
Betti Curve / Topological Complexity Trajectory — tracks how Betti numbers
change across ALL layers of the network.

The key insight from Naitzat et al. (JMLR 2020) is that well-generalized
networks progressively simplify topology: Betti numbers (connected components,
loops) decrease with depth. The rate of this decrease and the final complexity
are both informative structural metrics.

Method (2026-06-22 redesign — refined adaptation of Naitzat et al.):
  Following the paper and its reference code (topnn/topnn_framework), Betti
  numbers are computed on a **graph-geodesic** distance (kNN graph + shortest
  paths), NOT raw Euclidean distance, so manifold structure is respected.
  * beta_0 = number of connected components of the symmetric kNN graph — robust
    and parameter-light; it recovers the true cluster count (validated: K
    separated blobs -> beta_0 = K), whereas the previous per-layer
    median-Euclidean threshold collapsed beta_0 to ~1 regardless of structure.
  * beta_1 = number of H1 loops with NORMALIZED persistence (death-birth)/
    geodesic-diameter > ``persistence_frac`` (default 0.3), from ripser on the
    geodesic distance matrix. Validated: noisy circle -> 1, figure-8 -> 2,
    high-dim gaussian noise -> 0 (no spurious loops).
  This is NOT exact parity with the paper's Eirene backend (different homology
  engine and a single-fixed-scale read); it is a faithful, validated adaptation.

References:
- "Topology of Deep Neural Networks" (Naitzat, Zhitnikov & Lim, JMLR 21(184):1-40, 2020, arXiv:2004.06093)
- "Topological Data Analysis of Large Language Models' Hidden Representations"
  (General TDA on Transformers literature)
"""

import torch
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


def _count_betti(data, maxdim=1, n_neighbors=5, persistence_frac=0.3):
    """Count Betti numbers from a point cloud via graph-geodesic homology.

    Follows Naitzat et al. (JMLR 2020) and the reference topnn_framework:
    build a kNN graph, take graph-geodesic (shortest-path) distances, and
    read topology from that — manifold-aware, unlike raw Euclidean distance.

      * ``beta_0`` = number of connected components of the symmetric kNN
        graph (scipy ``connected_components``). Parameter-light and robust;
        recovers the true cluster count.
      * ``beta_1`` = number of H1 loops whose NORMALIZED persistence
        ``(death - birth) / geodesic_diameter`` exceeds ``persistence_frac``,
        from ripser on the geodesic distance matrix (disconnected pairs are
        capped at twice the finite geodesic diameter).

    Args:
        data: (N, D) point cloud.
        maxdim: max homology dimension for ripser (1 = up to loops).
        n_neighbors: kNN graph degree (capped at N-1).
        persistence_frac: normalized-persistence threshold for counting loops.

    Returns:
        (betti_0, betti_1): connected components and significant loops.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path, connected_components

    X = np.asarray(data, dtype=np.float64)
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
        num_samples = self.config.get("num_samples", 20)
        n_neighbors = int(self.config.get("n_neighbors", 5))
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
            except ImportError:
                logger.info("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [{"text": f"Topological sample {i} for Betti curve calculation."} for i in range(num_samples)]
        samples = list(dataset)[:num_samples]
        if len(samples) < 3:
            return {"error": "Need at least 3 samples for Betti curve"}
        
        layers = get_layers(model)
        num_layers = len(layers)
        
        # Collect mean-pooled representations at EVERY layer
        layer_reps = {l: [] for l in range(num_layers)}
        
        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                out = model(**inputs, output_hidden_states=True)
                
                for l_idx in range(num_layers):
                    hidden = out.hidden_states[l_idx + 1][0]
                    # .float() first: numpy doesn't accept bf16 tensors.
                    rep = hidden.mean(dim=0).float().cpu().numpy()
                    layer_reps[l_idx].append(rep)
        
        betti_0_curve = []
        betti_1_curve = []
        
        for l_idx in range(num_layers):
            data = np.array(layer_reps[l_idx])
            b0, b1 = _count_betti(
                data, maxdim=1,
                n_neighbors=n_neighbors, persistence_frac=persistence_frac,
            )
            betti_0_curve.append(b0)
            betti_1_curve.append(b1)
        
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
        
        # Rate of decay: linear regression slope of β0 vs **normalised
        # depth** (x in [0, 1]) so the slope is comparable across
        # models with different layer counts. Using raw layer index
        # instead makes a 12-layer and 80-layer model's slopes scale
        # differently — same issue as the fix applied to
        # dynamics/gradient_flow.
        if len(betti_0_curve) > 1:
            denom = max(1, num_layers - 1)
            x = np.arange(num_layers, dtype=np.float64) / float(denom)
            slope = np.polyfit(x, betti_0_curve, 1)[0]
            results["betti_0_decay_rate"] = float(slope)
        else:
            results["betti_0_decay_rate"] = 0.0
            
        # Max β1 layer (most topological loops)
        results["max_betti_1_layer"] = int(np.argmax(betti_1_curve))
        results["max_betti_1"] = int(np.max(betti_1_curve))
            
        return results
