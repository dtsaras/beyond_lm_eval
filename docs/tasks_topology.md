# Topology Tasks

This module contains metrics that utilize Computational Topological Data Analysis (TDA) to map the continuous, complex shape of the language manifold across model layers.

**Current registry coverage (4 tasks)**: `topology_betti_curve`,
`topology_homology`, `topology_persistence_entropy`, and
`topology_persistence_landscape`.

**Paper-faithful vs. BLME proxy notes**: The topology tasks use standard
Vietoris-Rips persistent-homology summaries on sampled hidden-state point
clouds. Persistence landscapes cite Bubenik's original JMLR/arXiv paper
(`arXiv:1207.6437`); `arXiv:1501.00179` is the later Bubenik-Dlotko toolbox
paper, useful as an implementation reference but not the original landscape
definition.

---

## 1. Persistent Homology & Betti-0
* **What are we measuring**: The fundamental topological connectivity of the semantic latent space.
* **How are we measuring**: By constructing a Vietoris-Rips filtration over point clouds of context embeddings. We compute the number of connected components (Betti-0) as a function of the neighborhood radius.
* **Hypothesis**: Random noise text will form a massive disconnected point cloud. Mathematically coherent language will collapse into deeply connected low-dimensional sub-manifolds, reflecting rule-based syntax.
* **Citation/Paper**: `Edelsbrunner, H., & Harer, J. (2008). Persistent homology-a survey.` [Contemporary Mathematics, 453, 257-282].
* **File & Function**: `src/blme/tasks/topology/homology.py` -> `PersistentHomologyTask`
* **Critical Info**: TDA scales exponentially with the number of points. Computations must be restricted to small semantic trajectories or requires massive linear algebra optimizations (like Ripser).

## 2. Persistence Entropy
* **What are we measuring**: The topological disorder and structural noise of the representation manifold.
* **How are we measuring**: Extracting the 'birth' and 'death' parameters from the persistent homology diagram, treating them as a probability distribution of feature lifespans, and computing the Shannon entropy. 
* **Hypothesis**: High persistence entropy implies the space contains many small, chaotic, short-lived topological artifacts. Low persistence entropy means a few massive, dominant, globally robust semantic features structure the space.
* **Citation/Paper**: `Rucco, M., Castiglione, F., Merelli, E., & Pettini, M. (2016). Characterisation of the Idiotypic Immune Network Through Persistent Entropy.` Related barcode-entropy framing: `Chintakunta et al. (2015). An entropy-based persistence barcode.`
* **File & Function**: `src/blme/tasks/topology/persistence_entropy.py` -> `PersistenceEntropyTask`
* **Critical Info**: Directly translates the visual scatter-plot of persistence diagrams into a single rigorous scalar summarizing topological complexity.

## 3. Layer-Wise Topological Complexity (Betti Curves)
* **What are we measuring**: How the shape of language data transforms mathematically from syntax (shallow layers) to abstraction (deep layers).
* **How are we measuring**: Computing Betti counts at a data-dependent threshold across selected transformer layers, then reporting simplification and decay summaries.
* **Hypothesis**: Input layers contain highly disconnected, messy word tokens (high Betti curves). Deep conceptual layers merge these into unified structural representations, causing Betti curves to collapse and simplify.
* **Citation/Paper**: `Naitzat, G., Zhitnikov, A., & Lim, L. (2020). Topology of Deep Neural Networks.` [JMLR 21(184):1-40, 2020; arXiv:2004.06093]
* **File & Function**: `src/blme/tasks/topology/betti_curve.py` -> `BettiCurveTask`
* **Critical Info**: The task is a sampled hidden-state topology proxy; it is not a full topological theorem about language manifolds.

## 4. Persistence Landscapes
* **What are we measuring**: Functional summaries of persistence diagrams that retain more structure than scalar lifetime statistics.
* **How are we measuring**: Converting each birth/death pair into a tent function and reporting integrals, maxima, and norms for the first few landscape functions.
* **Hypothesis**: Larger landscape mass indicates stronger or longer-lived topological features in the sampled representation cloud.
* **Citation/Paper**: `Bubenik, P. (2015). Statistical Topological Data Analysis using Persistence Landscapes.` [JMLR 16:77-102, ArXiv: 1207.6437]. Implementation reference: `Bubenik, P. & Dlotko, P. (2017). A persistence landscapes toolbox for topological statistics.` [ArXiv: 1501.00179]
* **File & Function**: `src/blme/tasks/topology/persistence_landscape.py` -> `PersistenceLandscapeTask`
* **Critical Info**: Requires `ripser`. BLME evaluates early/mid/late hidden-state point clouds; it is a standard landscape summary, not a bespoke LLM topology theorem.
