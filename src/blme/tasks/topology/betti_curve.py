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

    # DUPLICATES ARE NOT QUOTIENTED. Reverted 2026-09-07, see below.
    #
    # Round 9 reported that the paper's point-set X and the released loader
    # treat the input as a SET, so adding 5 exact copies to a 42-point circle
    # should not change the answer. I implemented the quotient. Two existing
    # tests failed, both encoding a deliberate campaign-3 decision:
    #
    #   `test_betti_zero_distance_edges_do_not_inflate_beta0` builds 8 exact
    #   duplicates plus a far blob and requires beta_0 == 2. Quotienting leaves
    #   ONE point at the origin, which at k=4 must connect to the far blob, so
    #   beta_0 collapses to 1. The duplicate cluster stops being a cluster.
    #
    # So the quotient does not merely remove redundancy here -- it destroys
    # multiplicity that the kNN construction is using as density. Whether the
    # released loader really quotients is exactly the sort of claim I took on
    # trust for `geometry_magnitude` the same day and had to withdraw after
    # running the actual reference (magnipy does NOT de-duplicate; it perturbs).
    #
    # Not reinstated until someone runs the released `topnn_framework` loader
    # and shows what it does. Filed in PENDING_FIXES.md item 18.
    n = len(X)
    if n < 3:
        return ((max(n, 0), 0, 0.0, 0.0) if return_persistence
                else (max(n, 0), 0))

    # Neighbourhood size. A FIXED k (was 5) is wrong for two reasons: it does
    # not scale with N, and it routinely leaves the kNN graph DISCONNECTED --
    # at which point the geodesic matrix has infinities that get capped at an
    # arbitrary "twice the finite diameter", and that arbitrary constant then
    # sets the H1 persistence values the Betti count thresholds on. So beta_1
    # depended on a fallback constant rather than on the data.
    #
    # ``n_neighbors="auto"`` (now the default) starts at max(5, ceil(log2 N))
    # and grows k until the graph is connected, so the infinity cap never
    # fires. The cap is retained only as a last resort.
    if isinstance(n_neighbors, str) and n_neighbors == "auto":
        # Floor at 14, the paper's own calibrated k. `ceil(log2 N)` gives 10 at
        # N = 1000, where Naitzat Fig. 3(e)'s hollow torus reads beta_1 = 8
        # against a published 2; at k = 14 it reads 2 (adversarial review,
        # 2026-09-03).
        k = int(max(14, math.ceil(math.log2(max(n, 2)))))
    else:
        k = int(n_neighbors)
    k = int(min(k, n - 1))
    tree = cKDTree(X)
    # Column 0 is usually the point itself, but with exact-duplicate points
    # the self-entry and the duplicate (both distance 0) may swap columns.
    # Dropping column 0 is still safe: the surviving zero-distance entry is
    # either the duplicate edge (kept, floored below) or a harmless (i, i)
    # self-loop that csgraph ignores.
    dists, idx = tree.query(X, k=k + 1)
    dists = np.atleast_2d(dists)
    idx = np.atleast_2d(idx)

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    # Keep zero-distance edges (duplicate points) as real edges: scipy's csgraph
    # treats an explicit-zero entry as "no edge", so duplicates would otherwise
    # be disconnected and inflate beta_0. Floor at a tiny epsilon.
    vals = np.maximum(dists[:, 1:].ravel(), 1e-12)
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize (mutual + one-directional kNN)

    # beta_0 is read from the HONEST kNN graph, before any connectivity repair.
    # Growing k until the graph connects would merge genuinely separate
    # clusters: on c well-separated Gaussians at N=100 the k-growing variant
    # returned beta_0 = 1 for c = 2, 3, 5 and 10 (adversarial review
    # 2026-09-03). Since the paper's headline is the simplification ratio
    # beta_0_late / beta_0_early, that bias drives the ratio to 1 and the
    # slope to 0 -- i.e. it manufactures the paper's claimed effect.
    n_components, comp_labels = connected_components(graph, directed=False)
    betti_0 = int(n_components)

    # For the GEODESIC step the graph must still be connected, or the distance
    # matrix carries infinities that get capped at an arbitrary constant which
    # then sets the H1 persistences. Bridge components with their
    # minimum-weight connecting edges instead of inflating k, so beta_0 above
    # is untouched and the geodesic metric is finite.
    # H1 is computed PER CONNECTED COMPONENT and combined.
    #
    # H_1 of a disjoint union is the direct sum of the components' H_1, so this
    # is the topologically correct decomposition -- and it removes the need to
    # connect components at all. Forcing connectivity distorted the geodesic
    # metric enough to destroy the loops being counted: with c separated noisy
    # circles (truth beta_1 = c) the bridged single-graph computation returned
    # beta_1 = 0 for every c >= 2 (adversarial review, 2026-09-03).
    #
    # Working inside each component also makes every geodesic finite by
    # construction, so the old "cap infinities at twice the finite diameter"
    # fallback -- an arbitrary constant that then set the H1 persistences the
    # Betti count thresholds on -- is gone entirely.
    lifetimes = []
    alive = []
    for comp in range(n_components):
        idx = np.flatnonzero(comp_labels == comp)
        if idx.size < 4:
            continue                       # too small to carry a 1-cycle
        sub = graph[idx][:, idx]
        if hop_metric:
            # UNIT-EDGE (hop) metric, which is what Naitzat et al. actually use:
            # JMLR Sec. 6.3 -- "Each edge, regardless of its Euclidean length,
            # has the same length of one when measured in delta_k" -- and their
            # code confirms it (kneighbors_graph with no mode= is binary, and
            # Eirene is run at minrad=1, maxrad=2, numrad=1, i.e. a read at two
            # hops).
            #
            # It changes the answer on the paper's OWN example: for the hollow
            # torus of their Figure 3(e), published beta = (1, 2, 1), the
            # weighted-geodesic version returns beta_1 = 1 while the hop metric
            # returns exactly 2 (adversarial review, 2026-09-03).
            sub = sub.copy()
            sub.data = np.ones_like(sub.data)
        geo_c = shortest_path(sub, method="D", directed=False)
        finite_c = geo_c[np.isfinite(geo_c)]
        if finite_c.size == 0:
            continue
        scale_c = float(finite_c.max())
        if scale_c <= 0:
            continue
        # POWER-OF-TWO rescale before ripser -- see the long note in
        # `zigzag.py`. ripser casts the distance matrix to float32
        # (ripser.py:324), so entries past 3.4e38 overflow to +inf and the
        # filtration loses its top: a 40-point circle whose betti_1 is 1 by
        # construction reads 1 from 1e-30 through 1e38 and then 0 at
        # 2.406159650500376e38.
        #
        # Gated to out-of-range magnitudes, which is what keeps the two
        # branches below consistent. `hop_metric` compares an ABSOLUTE radius
        # (`b <= hop_radius < d`) against a hop count -- a small integer, always
        # inside 1e-6..1e6 -- so no rescale ever fires on that path and the
        # fixed-radius read is untouched. The geodesic path normalises
        # lifetimes by `scale_c`, which is shifted by the same exact factor.
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
            # FIXED-RADIUS read, which is what Naitzat et al. do: Eirene is run
            # at minrad=1, maxrad=2, numrad=1, i.e. beta_p is the number of
            # classes ALIVE at a radius of `hop_radius` hops -- not a count of
            # bars that survive a persistence threshold.
            #
            # Keeping the persistence threshold on a hop metric was still wrong:
            # hop diameters are tiny (a handful of hops), so a 0.3-of-diameter
            # cut removes genuine classes. On the paper's Figure 3(e) hollow
            # torus (published beta = (1, 2, 1)) the thresholded read returns
            # beta_1 = 1 while the fixed-radius read can return 2.
            #
            # The radius is settled at 2 hops: Eirene's
            # `minrad=1, maxrad=2, numrad=1` collapses to a SINGLE radius of
            # r = 2 with no persistence threshold, which is exactly this read
            # (verified against `topnn/topnn_framework`).
            #
            # My earlier note here said "the radius is not settled". That was a
            # MISDIAGNOSIS: the uncalibrated parameter is **k**, not the radius.
            # At the paper's own calibrated k = 14 (and k = 20), radius 2
            # reproduces the published beta_1 = 2 for Naitzat Fig. 3(e)'s hollow
            # torus at both N = 1000 and N = 2000. At the shipped auto-k = 10
            # the same torus reads 8 (adversarial review, 2026-09-03).
            alive.extend(1 for b, d in h1_c
                         if np.isfinite(d) and b <= hop_radius < d)
        # Normalized by THIS component's geodesic diameter, so a large and a
        # small copy of the same shape score alike.
        lifetimes.extend((d - b) / scale_c for b, d in h1_c if np.isfinite(d))

    betti_1 = (int(sum(alive)) if hop_metric
               else sum(1 for lt in lifetimes if lt > persistence_frac))

    # Continuous H1 summaries. The hard count above is ZERO for every real
    # model tested: on gpt2 the maximum normalized H1 persistence per layer is
    # 0.106 0.122 0.092 0.106 0.106 0.110 0.140 0.132 0.165 0.164 0.150 0.049
    # -- never within a factor of two of the 0.3 threshold -- so beta_1, and
    # therefore max_betti_1 and max_betti_1_layer, are constants across the
    # whole model zoo (adversarial review 2026-09-03). Those same numbers are
    # a clear depth profile, so the signal is real and it is the THRESHOLD
    # that discards it. These threshold-free summaries are the ones to use.
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
        # 20 points is not a point cloud: persistent homology on 20 mean-pooled
        # vectors returns integers driven by sampling noise, and the reported
        # simplification_ratio was then a ratio of two single-digit counts.
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
        
        # RECOVER THE DEEPEST BLOCK'S RAW OUTPUT with a forward hook.
        #
        # THE TENTH instance of this defect, and the last one to be found: HF's
        # final `hidden_states` entry is `ln_f(b_{L-1})`, so the curve mixed
        # 11 block outputs with ONE LayerNorm output. On gpt2 the deepest
        # layer's H1 total persistence read 5.2727 against 27.0000 for the true
        # block output (+412%), and `h1_max_persistence_layer_frac` FLIPPED
        # 0.0 -> 1.0 (independent Opus review, 2026-09-05). Model-dependent --
        # pythia-70m is unaffected -- which is exactly what makes it dangerous
        # in a cross-model comparison.
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
                    # post-final-norm: SUBSTITUTED -- the deepest sampled layer must
                    # be a block output like every other entry in this profile.
                    _hs = list(out.hidden_states)
                    _n_fwd += 1
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None
                            and tuple(_h_raw.shape) == tuple(_hs[-1].shape)):
                        _hs[-1] = _h_raw
                        _n_sub += 1

                    for l_idx in range(num_layers):
                        hidden = _hs[l_idx + 1][0]
                        # .float() first: numpy doesn't accept bf16 tensors.
                        rep = hidden.mean(dim=0).float().cpu().numpy()
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
        # Absolute layer index leaks depth (and therefore model size) into the
        # feature matrix; report normalized depth alongside it.
        results["max_betti_1_layer"] = int(np.argmax(betti_1_curve))
        results["max_betti_1_layer_frac"] = (
            float(np.argmax(betti_1_curve)) / float(max(1, num_layers - 1)))

        # Threshold-free H1 summaries: these vary across models and depth
        # where the thresholded counts above are identically zero.
        # WHAT `h1_max_persistence` ACTUALLY IS, measured.
        #
        # Under the HOP metric every H1 bar has lifetime exactly 1 -- verified
        # on five clouds carrying 38 to 3,032 bars -- so this summary is
        # algebraically 1/(hop diameter), and all 36 panel layer values across
        # four models are exactly 1/integer (independent Opus review,
        # 2026-09-05). It is a reciprocal DIAMETER, not a persistence spread;
        # the manuscript presents it as the carrier of the H1 signal, which it
        # is only in that restricted sense. Recorded here so no reader infers
        # more from it than the metric can express.
        results["_meta_last_entry_is_block_output"] = bool(_last_is_block)
        results["h1_max_persistence_curve"] = h1_maxpers_curve
        results["h1_max_persistence"] = float(np.max(h1_maxpers_curve))
        results["h1_max_persistence_mean"] = float(np.mean(h1_maxpers_curve))
        results["h1_total_persistence_mean"] = float(np.mean(h1_totpers_curve))
        results["h1_max_persistence_layer_frac"] = (
            float(np.argmax(h1_maxpers_curve)) / float(max(1, num_layers - 1)))
        results["max_betti_1"] = int(np.max(betti_1_curve))
            
        return results
