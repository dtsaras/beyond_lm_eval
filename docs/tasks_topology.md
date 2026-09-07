# Topology Tasks

This module contains metrics that utilize Computational Topological Data Analysis (TDA) to map the continuous, complex shape of the language manifold across model layers.

**Current registry coverage (5 tasks)**: `topology_betti_curve`,
`topology_homology`, `topology_persistence_entropy`,
`topology_persistence_landscape`, and `topology_zigzag_persistence`.

**Paper-faithful vs. BLME proxy notes**: The topology tasks use standard
Vietoris-Rips persistent-homology summaries on sampled hidden-state point
clouds. Persistence landscapes cite Bubenik's original JMLR/arXiv paper
(`arXiv:1207.6437`); `arXiv:1501.00179` is the later Bubenik-Dlotko toolbox
paper, useful as an implementation reference but not the original landscape
definition. Zigzag persistence is a documented *faithful proxy* of the
Gardinazzi et al. pipeline (ripser-based layer-band summary rather than the
paper's dionysus zigzag engine).

**Verification campaign (2026-07, Batch 1)**: every task below carries a
`Verification` field recording its evidence tier per the campaign standard —
official-code parity, analytic anchors, and/or independent reimplementation.

---

## 1. Persistent Homology & Betti-0
* **What are we measuring**: The fundamental topological connectivity of the semantic latent space.
* **How are we measuring**: By constructing a Vietoris-Rips filtration over point clouds of context embeddings. We compute the number of connected components (Betti-0) as a function of the neighborhood radius.
* **Hypothesis**: Random noise text will form a massive disconnected point cloud. Mathematically coherent language will collapse into deeply connected low-dimensional sub-manifolds, reflecting rule-based syntax.
* **Citation/Paper**: `Zomorodian, A., & Carlsson, G. (2005). Computing Persistent Homology.` [Discrete & Computational Geometry 33(2):249-274]; `Edelsbrunner, H., & Harer, J. (2008). Persistent homology — a survey.` [Contemporary Mathematics, 453, 257-282]; motivation for applying TDA to learned representations: `Naitzat, Zhitnikov & Lim (2020)` [JMLR 21(184):1-40].
* **Paper summary**: Zomorodian & Carlsson give the founding algorithm of computational persistent homology: a filtered simplicial complex decomposes into birth-death intervals (the barcode/persistence diagram), computable by matrix reduction over a field — this is what `ripser` implements. Edelsbrunner & Harer survey the theory. The practical upshot for BLME: lifespans (death − birth) of H0/H1 bars quantify how robustly clusters and loops exist in a representation cloud across all scales at once, with no threshold to tune.
* **File & Function**: `src/blme/tasks/topology/homology.py` -> `PersistentHomologyTask`
* **Critical Info**: TDA scales exponentially with the number of points. Computations must be restricted to small semantic trajectories or requires massive linear algebra optimizations (like Ripser).
* **Verification** (2026-07, campaign B1): backend is ripser 0.6.14 called directly; cross-backend parity vs GUDHI 3.12 RipsComplex (6 H1 bars, bar-by-bar ≤ 6.2e-8) and vs scipy minimum-spanning-tree for H0 (elder rule, ≤ 9.6e-8 at float32); analytic anchors (collinear points, regular hexagon, unit square) exact. Duplicate-target-layer bug (≤2-layer models) fixed in `d4daa1a`; venue of the Naitzat citation corrected (JMLR, not ICLR).
* **B8 extension** (2026-07): scale-stratified persistence summaries — H0/H1 bars split into fine vs coarse by death threshold 0.5×(mean pairwise distance), reporting counts and the fine fraction of persistence mass per layer. Motivated by Fay et al. (ArXiv: 2505.20435): degenerate/adversarial inputs COMPRESS latent topology (fewer fine-scale features, amplified coarse ones), so the fine/coarse split separates regimes the raw lifespan means conflate. Anchored on constructed diagrams with known fine/coarse membership.

## 2. Persistence Entropy
* **What are we measuring**: The topological disorder and structural noise of the representation manifold.
* **How are we measuring**: Extracting the 'birth' and 'death' parameters from the persistent homology diagram, treating them as a probability distribution of feature lifespans, and computing the Shannon entropy.
* **Hypothesis**: High persistence entropy implies the space contains many small, chaotic, short-lived topological artifacts. Low persistence entropy means a few massive, dominant, globally robust semantic features structure the space.
* **Citation/Paper**: `Rucco, M., Castiglione, F., Merelli, E., & Pettini, M. (2016). Characterisation of the Idiotypic Immune Network Through Persistent Entropy.` [Proc. ECCS 2014, Springer]; `Chintakunta, H., Gentimis, T., Gonzalez-Diaz, R., Jimenez, M.-J., & Krim, H. (2015). An entropy-based persistence barcode.` [Pattern Recognition 48(2):391-401]; stability + canonical log2 definition: `Atienza, N., Gonzalez-Diaz, R., & Soriano-Trigueros, M. (2020). On the stability of persistent entropy and new summary functions for topological data analysis.` [Pattern Recognition 107:107509; arXiv:1803.08304].
* **Paper summary**: Rucco et al. introduce persistent entropy — the Shannon entropy of normalized bar lifespans — to give a single scalar for the "disorder" of a barcode, applied there to distinguish dynamical regimes of an immune-network simulation. Chintakunta et al. use the same entropy idea to select significant features within a barcode. Atienza et al. prove stability (small diagram perturbations produce bounded entropy changes) and fix the canonical definition (Def. 3.1, log base 2), which is exactly the formulation BLME implements.
* **File & Function**: `src/blme/tasks/topology/persistence_entropy.py` -> `PersistenceEntropyTask`
* **Critical Info**: Directly translates the visual scatter-plot of persistence diagrams into a single rigorous scalar summarizing topological complexity. Edge conventions: single bar → 0.0; empty diagram → 0.0 (giotto-tda would emit nan/-1 fill there — BLME's guard is deliberate and documented).
* **Verification** (2026-07, campaign B1): official-package parity — executed giotto-tda 0.6.2 `PersistenceEntropy().fit_transform` on 203 cases (random lifespan sets, a real ripser H0 diagram, analytic cases): max |BLME − gtda| = 2.7e-15; analytic anchors exact (8 equal bars → 3.0; lifespans [1,3] → 0.8112781). Docstring citations corrected (wrong author set on the 2016 paper; "Entropy 2019" venue was wrong — Pattern Recognition 2020).

## 3. Layer-Wise Topological Complexity (Betti Curves)
* **What are we measuring**: How the shape of language data transforms mathematically from syntax (shallow layers) to abstraction (deep layers).
* **How are we measuring**: kNN graph per layer → graph-geodesic distances → β0 = connected components; β1 = H1 loops with normalized persistence above `persistence_frac` (ripser on the geodesic matrix). Simplification/decay summaries across depth.
* **Hypothesis**: Input layers contain highly disconnected, messy word tokens (high Betti curves). Deep conceptual layers merge these into unified structural representations, causing Betti curves to collapse and simplify.
* **Citation/Paper**: `Naitzat, G., Zhitnikov, A., & Lim, L. (2020). Topology of Deep Neural Networks.` [JMLR 21(184):1-40, 2020; arXiv:2004.06093]
* **Paper summary**: Naitzat et al. pass topologically entangled binary datasets through trained ReLU networks and track Betti numbers of each class's point cloud layer by layer, computed on graph-geodesic distances (kNN graph), not raw Euclidean ones. Key findings: well-generalizing networks progressively *simplify* topology — Betti numbers fall with depth until the classes are topologically trivial; the simplification is concentrated in deeper layers; and non-homeomorphic activations (ReLU) enable topology change where smooth activations (tanh) preserve it. BLME's per-layer β0/β1 curves and decay-rate summary operationalize exactly this simplification signature for LLM hidden states.
* **File & Function**: `src/blme/tasks/topology/betti_curve.py` -> `BettiCurveTask`
* **Critical Info**: The task is a sampled hidden-state topology proxy; it is not a full topological theorem about language manifolds. It is a **refined adaptation**: same geodesic construction as the paper, but ripser instead of the paper's Julia/Eirene engine and a normalized-persistence read for β1 instead of a single fixed scale (disclosed in the module docstring).
* **Verification** (2026-07, campaign B1): triple-engine cross-check — β0/β1 vs networkx components, from-scratch GF(2) boundary reduction, and GUDHI all agree (0 mismatches on 12 randomized blob/ring mixtures; full H1 diagram endpoint diff ≤ 9.0e-8); ground-truth anchors (4 blobs → β0=4, noisy circle → (1,1), figure-8 → 2, high-dim noise → 0). Zero-weight-edge bug (duplicate points inflated β0) fixed in `d4daa1a`; derived `evaluate()` stats (decay slope on normalized depth, simplification ratio, argmax) now numerically pinned in `tests/tasks/test_topology.py::test_betti_curve`; a placeholder pseudo-citation was removed from the docstring.

## 4. Persistence Landscapes
* **What are we measuring**: Functional summaries of persistence diagrams that retain more structure than scalar lifetime statistics.
* **How are we measuring**: Converting each birth/death pair into a tent function and reporting integrals, maxima, and norms for the first few landscape functions.
* **Hypothesis**: Larger landscape mass indicates stronger or longer-lived topological features in the sampled representation cloud.
* **Citation/Paper**: `Bubenik, P. (2015). Statistical Topological Data Analysis using Persistence Landscapes.` [JMLR 16:77-102, ArXiv: 1207.6437]. Implementation reference: `Bubenik, P. & Dlotko, P. (2017). A persistence landscapes toolbox for topological statistics.` [ArXiv: 1501.00179]
* **Paper summary**: Bubenik's landscapes map a persistence diagram to a sequence of piecewise-linear functions L_k(t) — the k-th largest tent value at each t — embedding barcodes into a Banach space where means, variances, and hypothesis tests are well-defined (a central limit theorem holds). Landscapes are stable (small diagram changes produce small landscape changes) and, unlike scalar summaries, preserve enough structure to distinguish diagrams with equal total persistence. The toolbox paper supplies the standard discretization algorithms that `persim` (and BLME) follow.
* **File & Function**: `src/blme/tasks/topology/persistence_landscape.py` -> `PersistenceLandscapeTask`
* **Critical Info**: Requires `ripser`. BLME evaluates early/mid/late hidden-state point clouds; it is a standard landscape summary, not a bespoke LLM topology theorem. Reported keys are `layer_{l}_{h0|h1}_landscape_integrals` (list over k), `_landscape_max_1`, `_landscape_norm_1`, `_mean_landscape_integral`.
* **Verification** (2026-07, campaign B1): official-package parity vs persim 0.3.8 `PersLandscapeApprox` — single-bar case and a 5-bar overlapping diagram at K=4 (max diff 8.9e-16; the multi-bar case exercises the k-selection sort, added in B1); brute-force Bubenik-definition reimplementation diff 0.0; analytic single-bar integrals/max/L2 exact. Docstring metric list corrected to the actually-emitted keys; `target_layers` deduped for ≤2-layer models.

## 5. Zigzag Persistence (Across-Layer Feature Tracking)
* **What are we measuring**: How individual topological features (loops, components) *persist across the layer axis* — the full evolutionary path of representation topology through the network, rather than one summary per layer aggregated afterwards.
* **How are we measuring**: Per layer, significant H0/H1 features are detected with the same graph-geodesic kNN + ripser construction as the Betti-curve task; a feature's "lifetime" is then the contiguous band of layers where it is present. Summaries: number of bands, max band length, normalized first/last layer of the longest band, per-layer loop-count sum vs. layers-with-loops, H0 component statistics.
* **Hypothesis**: Coherent semantic structure produces topological features that persist across many consecutive layers (long bands); noise produces short-lived features. Band structure is a model signature that shifts under fine-tuning.
* **Citation/Paper**: `Gardinazzi, Y., Viswanathan, K., Panerai, G., Ansuini, A., Cazzaniga, A., & Biagetti, M. (2025). Persistent Topological Features in Large Language Models.` [ICML 2025; arXiv:2410.11042]. Zigzag persistence theory: `Carlsson, G., & de Silva, V. (2010). Zigzag Persistence.` [Foundations of Computational Mathematics 10:367-405].
* **Paper summary**: Carlsson & de Silva extend persistence to sequences of spaces connected by maps in alternating directions (X₁ → X₁∩X₂ ← X₂ …), which is exactly the structure of "the same prompts, re-embedded at each successive layer". Gardinazzi et al. apply this to LLMs: build a kNN complex per layer plus pairwise intersections, run zigzag homology (dionysus), and introduce descriptors measuring how p-dimensional holes persist and evolve across layers — tracking the full evolutionary path instead of aggregating per-layer summaries. As a downstream application they derive a layer-pruning criterion from these descriptors that matches state-of-the-art pruning while retaining a system-level view. BLME's layer-band summary is a faithful proxy for their descriptors.
* **File & Function**: `src/blme/tasks/topology/zigzag.py` -> `ZigzagPersistenceTask`
* **Critical Info**: **Faithful proxy, not a bit-exact port**: the paper's dionysus/FastZigZag engines are C++ builds that cannot be BLME dependencies. BLME detects per-layer features and summarizes contiguous layer-bands; on ground-truth constructions this matches the dionysus zigzag barcode exactly (bar ↔ band, floor(dionysus birth layer) = band onset), recorded in the parity fixture.
* **Verification** (2026-07, campaign B1): three-way anchoring — (A) ground-truth constructions with known lifetimes (circle in specific layers, two disjoint circles, two clusters) recovered exactly; (B) recorded dionysus barcodes from the official pipeline match BLME's bands on all scenarios; (C) independent from-scratch reimplementation bit-exact (< 1e-12) on all scenarios, including B1's new non-degenerate ones that separate `h1_loop_layer_sum` from `h1_total_layer_persistence` and exercise non-constant H0.
