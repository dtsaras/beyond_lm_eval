# Topology Tasks

This module contains metrics that utilize Computational Topological Data Analysis (TDA) to map the continuous, complex shape of the language manifold across model layers.

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
* **Citation/Paper**: `Chintakunta, H., Gentimis, T., Gonzalez-Diaz, R., Jimenez, M.-J., & Krim, H. (2015). An entropy-based persistence barcode.` [Pattern Recognition, 48(2), 391-401] 
* **File & Function**: `src/blme/tasks/topology/persistence_entropy.py` -> `PersistenceEntropyTask`
* **Critical Info**: Directly translates the visual scatter-plot of persistence diagrams into a single rigorous scalar summarizing topological complexity.

## 3. Layer-Wise Topological Complexity (Betti Curves)
* **What are we measuring**: How the shape of language data transforms mathematically from syntax (shallow layers) to abstraction (deep layers).
* **How are we measuring**: Tracking the evolution of the Betti-0 curve area across sequential transformer layers.
* **Hypothesis**: Input layers contain highly disconnected, messy word tokens (high Betti curves). Deep conceptual layers merge these into unified structural representations, causing Betti curves to collapse and simplify.
* **Citation/Paper**: Derived from general topological data analysis applied layer-wise. (No single conference paper; related to persistent homology layer analysis literature).
* **File & Function**: `src/blme/tasks/topology/betti_curve.py` -> `BettiCurveSimplificationTask`
* **Critical Info**: The task compares the area-under-the-curve (AUC) of the Betti plot across layers. A negative delta indicates topological simplification as information flows through the network.
