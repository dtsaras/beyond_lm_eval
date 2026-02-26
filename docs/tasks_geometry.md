# Geometry Tasks

This module contains metrics that evaluate the high-dimensional spatial geometry of the LLM's latent representation manifold.

---

## 1. Local Intrinsic Dimensionality (LID)
* **What are we measuring**: The local degrees of freedom of the representation manifold around a specific point.
* **How are we measuring**: Using Maximum Likelihood Estimation (MLE) on the nearest neighbor distances (typically k=10 or k=20) to compute the local non-integer dimensionality.
* **Hypothesis**: Models with excessively high LID might suffer from the curse of dimensionality and overfitting, while very low LID implies over-compression.
* **Citation/Paper**: `Amsaleg, L., et al. (2015). Estimating local intrinsic dimensionality.` [ACM KDD 2015, DOI: 10.1145/2783258.2783405] (ArXiv equivalent: 1905.12784 for related work)
* **File & Function**: `src/blme/tasks/geometry/lid.py` -> `LocalIntrinsicDimensionTask`
* **Critical Info**: LID changes drastically from shallow layers to deep layers, often forming an "intrinsic dimension bottleneck."

## 2. Lipschitz Continuity Analysis
* **What are we measuring**: The local smoothness and sensitivity of the model to small perturbations in the input space.
* **How are we measuring**: Practically estimated by computing the ratio of the distance between output representations to the distance between input representations for closely neighbored points.
* **Hypothesis**: High Lipschitz constants indicate an unstable, highly chaotic representation space vulnerable to adversarial perturbations. Low constants indicate smooth, stable generalization.
* **Citation/Paper**: `Anil, C., Lucas, J., & Grosse, R. (2019). Sorting out Lipschitz constant estimation.` [ArXiv: 1811.05381]
* **File & Function**: `src/blme/tasks/geometry/lipschitz.py` -> `LipschitzContinuityTask`
* **Critical Info**: Extremely hard to measure analytically; this task uses an empirical local approximation based on sampled neighbors.

## 3. Representational Similarity Analysis (RSA)
* **What are we measuring**: The structural isomorphism between the representation spaces of two different models, or two different layers.
* **How are we measuring**: By computing a Representational Dissimilarity Matrix (RDM) of pairwise distances for a set of inputs, and then finding the Spearman rank correlation between the upper triangles of two RDMs.
* **Hypothesis**: Two networks might have different exact geometries but computationally identical relative similarity structures. RSA allows comparison across models with different hidden dimensions.
* **Citation/Paper**: `Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis-connecting the branches of systems neuroscience.` [Frontiers in Systems Neuroscience]
* **File & Function**: `src/blme/tasks/geometry/rsa.py` -> `RSATask`
* **Critical Info**: Because RSA is $O(N^2)$, the `max_tokens` parameter controls the computational cost.

## 4. Latent Mahalanobis OOD Distance
* **What are we measuring**: How far Out-Of-Distribution (OOD) a sample is, accounting for the natural covariance of the in-distribution manifold.
* **How are we measuring**: By modeling a reference dataset's representations as a multivariate Gaussian (computing empirical mean and covariance matrix), and then measuring the Mahalanobis distance of new test points relative to this Gaussian.
* **Hypothesis**: Simple Euclidean distance is flawed in highly anisotropic spaces. Mahalanobis distance correctly scales by the principal axes of variance, providing a true measure of semantic anomaly.
* **Citation/Paper**: `Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks.` [ArXiv: 1807.03888]
* **File & Function**: `src/blme/tasks/geometry/mahalanobis.py` -> `MahalanobisOODTask`
* **Critical Info**: The covariance matrix must be inverted or pseudo-inverted. High-dimensional spaces ($d > N$) require Tikhonov regularization (adding $\epsilon I$) to prevent singularity.

## 5. Trace of the Empirical Fisher Information Matrix
* **What are we measuring**: The local curvature and sharpness of the representation manifold.
* **How are we measuring**: By computing the Trace of the Empirical Fisher Information Matrix (FIM) of the token representations with respect to the output logits/probabilities.
* **Hypothesis**: A "sharp" minimum (high trace) often correlates with poor generalization out-of-distribution, while a "flat" minimum (low trace) suggests robust generalization.
* **Citation/Paper**: `Information Geometry of Large Language Models` (2024/2025). [Ongoing literature, no single unified conference paper].
* **File & Function**: `src/blme/tasks/geometry/information_geometry.py` -> `FisherInformationTraceTask`
* **Critical Info**: FIM is computationally intractable to store entirely; the trace is an efficient scalar summary of total curvature.

## 6. Matrix Entropy (Information Bottleneck)
* **What are we measuring**: The data compression capabilities of the LLM layers over inference.
* **How are we measuring**: By computing the von Neumann spectral entropy over the internal covariance matrix of the hidden states at each layer.
* **Hypothesis**: As information passes through an LLM, the model actively filters out noise. A decreasing or low layer-wise matrix entropy indicates the model is actively forming a tighter semantic "Information Bottleneck".
* **Citation/Paper**: `Wei, L., Tan, Z., Li, C., Wang, J., & Huang, W. (2024). Large Language Model Evaluation via Matrix Entropy.` [ArXiv: 2401.17139]
* **File & Function**: `src/blme/tasks/geometry/matrix_entropy.py` -> `MatrixEntropyTask`
* **Critical Info**: Values typically decrease monotonically in deeper layers as the network compresses raw syntax into refined semantic logic.

## 7. Correlation Dimension (Fractal Geometry)
* **What are we measuring**: The underlying fractal complexity and self-similarity of the generated language manifold.
* **How are we measuring**: Using the Grassberger-Procaccia algorithm. Measures the fraction of points within a radius $r$ and computes the log-log scaling coefficient.
* **Hypothesis**: Standard intrinsic dimensions incorrectly assume the text space is locally flat (Euclidean). Correlation dimension proves language lies on a highly complex fractal attractor.
* **Citation/Paper**: `Du, X., & Tanaka-Ishii, K. (2024/2025). Correlation Dimension of Autoregressive Large Language Models.` [NeurIPS 2025 / ArXiv]
* **File & Function**: `src/blme/tasks/geometry/correlation_dimension.py` -> `CorrelationDimensionTask`
* **Critical Info**: Requires larger sample sizes to compute pairwise distances effectively. Normal text generally exhibits a non-integer structural dimension around ~6-7.

## 8. Positional Attention Decay (RoPE Geometry)
* **What are we measuring**: The structural integrity and geometric degradation of context windows.
* **How are we measuring**: Computing the Spearman rank correlation between absolute positional discrete token distance and the attention magnitude allocated to those past tokens.
* **Hypothesis**: To extrapolate well to long sequences, the attention matrix should exhibit a structurally sound, smooth geometric decay relative to distance. Breakdown (random correlations) indicates failure of the positional embeddings (e.g., RoPE).
* **Citation/Paper**: Derived from general 2024 Long-Context Extrapolation literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/positional_decay.py` -> `PositionalAttentionDecayTask`
* **Critical Info**: Requires sequences longer than a few tokens to establish a valid distance/attention correlation pattern.

## 9. SVD Isotropy (geometry_svd)
* **What are we measuring**: The isotropy (roundness) of the representation space.
* **How are we measuring**: Decomposing the hidden state matrix with SVD and calculating the ratio of the top singular value to the sum of all singular values, or looking at the variance drop-off.
* **Hypothesis**: Highly anisotropic spaces (e.g., dominating outlier dimensions) collapse semantics into a narrow cone, degrading similarity metrics. Isotropic spaces utilize capacity more uniformly.
* **Citation/Paper**: `Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?` [ArXiv: 1909.00512]
* **File & Function**: `src/blme/tasks/geometry/isotropy.py` -> `SvdIsotropyTask`
* **Critical Info**: Language models almost always suffer from an "anisotropy cone" unless explicitly regularized or normalized.

## 10. Hubness
* **What are we measuring**: The tendency of certain tokens to be the "nearest neighbor" of an unusually high number of other tokens in latent space.
* **How are we measuring**: Computing pairwise cosine similarities and tracking the skewed distribution of incoming Nearest Neighbor (1-NN) edges. High skew/max indicates severe hubness.
* **Hypothesis**: The "Curse of Dimensionality" leads to spatial hubs in high dimensions. These hubs crowd semantic spaces and degrade zero-shot retrieval and generation.
* **Citation/Paper**: `Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data.` [Journal of Machine Learning Research (JMLR) Vol 11]
* **File & Function**: `src/blme/tasks/geometry/hubness.py` -> `HubnessTask`
* **Critical Info**: Highly sensitive to the choice of similarity metric (L2 distance vs Cosine). Usually worse under Euclidean distance.

## 11. Category Separation
* **What are we measuring**: How well conceptually related words group together organically.
* **How are we measuring**: Comparing the average intra-category distance vs inter-category distance without labeled supervision.
* **Hypothesis**: A model with a rich geometric understanding of language will organically cluster related concepts (e.g., animals, colors) far from unrelated ones.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/categories.py` -> `CategoryGeometryTask`
* **Critical Info**: Also computes category Purity and generates coordinates for UMAP/t-SNE visualization if installed.

## 12. Geometry Consistency
* **What are we measuring**: The temporal stability of representational distances across generation steps.
* **How are we measuring**: Computing the difference in successive output hidden states during auto-regressive decoding.
* **Hypothesis**: Abrupt erratic jumps in successive generation steps often precede hallucinatory behavior, while stable steps indicate confident predictions.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/consistency.py` -> `GeometryConsistencyTask`
* **Critical Info**: Heavily dependent on generation parameters like temperature and top-p.

## 13. Representation Collapse
* **What are we measuring**: Severe rank collapse or dimension degeneration in the hidden states of identical repeated input tokens or near-duplicated data.
* **How are we measuring**: Computing the cosine similarity between outputs that should have been distinguishable but degenerate to the same vector due to depth.
* **Hypothesis**: Specifically in deep transformers, representations can over-smooth and lose their distinct individual token identities.
* **Citation/Paper**: `Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.` [ArXiv: 2103.03404]
* **File & Function**: `src/blme/tasks/geometry/collapse.py` -> `RepresentationCollapseTask`
* **Critical Info**: More pronounced in deep networks lacking robust layernorms or residual pathway scaling.

## 14. Spectral Decay
* **What are we measuring**: The rate at which the capacity of the model's intermediate dimensions is effectively utilized.
* **How are we measuring**: By fitting a power-law exponent (alpha) to the spectrum of eigenvalues from the hidden space covariance.
* **Hypothesis**: If alpha is too low, the manifold is overly noisy. If alpha is highly peaked, the manifold is collapsed. The "1/f" spectral decay is considered optimal for learning architectures.
* **Citation/Paper**: `Stringer, C., Pachitariu, M., Steinmetz, N., Carandini, M., & Harris, K. D. (2019). High-dimensional geometry of population responses in visual cortex.` [ArXiv: 1808.03612]
* **File & Function**: `src/blme/tasks/geometry/spectral.py` -> `SpectralAnalysisTask`
* **Critical Info**: Shows strong parallels between biological neural networks and artificial models.

## 15. Mutual Information (Geometric)
* **What are we measuring**: The spatial mutual information or dependency captured via kernel density estimations across layers.
* **How are we measuring**: Analyzing kernel similarity matrices in hidden spaces across layers to detect whether the information content is preserved or transformed.
* **Hypothesis**: Information Bottleneck theory posits that models first maximize MI with inputs, then progressively forget (minimize MI) extraneous details, compressing semantics.
* **Citation/Paper**: `Shwartz-Ziv, R., & Tishby, N. (2017). Opening the black box of deep neural networks via information.` [ArXiv: 1703.00810]
* **File & Function**: `src/blme/tasks/geometry/mutual_info.py` -> `MutualInfoTask`
* **Critical Info**: Computationally demanding. Requires careful bandwidth tuning for the KDE calculation.

## 16. CKA (Centered Kernel Alignment)
* **What are we measuring**: The similarity between the underlying structures of two sets of representations without requiring them to have the same features.
* **How are we measuring**: Computing the Frobenius norm of cross-covariance matrices. Equivalent to computing the correlation of dot-product similarity matrices.
* **Hypothesis**: Permits diagnosing whether two distinct layers (or models) are learning structurally analogous concepts, ignoring rotations or isotropic scalings. 
* **Citation/Paper**: `Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited.` [ArXiv: 1905.00414]
* **File & Function**: `src/blme/tasks/geometry/cka.py` -> `CKATask`
* **Critical Info**: Less prone to the scaling artifacts that affect normal canonical correlation analysis (CCA).

## 17. Unembedding Geometry
* **What are we measuring**: The relationship between the hidden language representations in the final layer and the static unembedding parameters located in the LM head.
* **How are we measuring**: Looking at the angles (cosine similarity) and norms between the highest logit token vectors and the actual dynamic context state vectors. 
* **Hypothesis**: The final language modeling head forces representations into distinct regions of the LM head space. The "Unembedding" operation exhibits severe bias due to token frequency in the pre-training set.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/unembedding.py` -> `UnembeddingGeometryTask`
* **Critical Info**: Typically reveals that frequent tokens dominate the manifold geometry by pushing less frequent tokens away from the origin computationally.

## 18. Perplexity (Baseline Geometry)
* **What are we measuring**: Baseline auto-regressive predictability.
* **How are we measuring**: The exponentiated average negative log-likelihood of a sequence.
* **Hypothesis**: As the foundational sanity check, it proves the model can actually model text. Used purely as a baseline correlate for other intrinsic measures.
* **Citation/Paper**: Canonical language modeling metric. 
* **File & Function**: `src/blme/tasks/geometry/perplexity.py` -> `PerplexityTask`
* **Critical Info**: Lower is better. Included primarily to compute correlations with the more advanced geometric variables.

## 19. Backward-Compatible Alignment (geometry_alignment)
* **What are we measuring**: (Legacy) Measuring alignment with golden expert models.
* **How are we measuring**: Routing to GEM metrics.
* **Hypothesis**: Retained only for API compatibility with legacy systems.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/alignment.py` -> `AlignmentResidualTask`
* **Critical Info**: Alias task that simply routes arguments to GEM.

## 20. Backward-Compatible Substitution (geometry_substitution)
* **What are we measuring**: (Legacy) Measuring word-level substitutions.
* **How are we measuring**: Routing to GEM substitutions module.
* **Hypothesis**: Retained only for API compatibility.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/alignment.py` -> `SubstitutionConsistencyTask`
* **Critical Info**: Alias task.

## 21. Global Intrinsic Dimension (geometry_intrinsic_dim / PDE)
* **What are we measuring**: The global effective dimensionality of the dataset within the model's space.
* **How are we measuring**: Using TwoNN (Two Nearest Neighbors algorithm) across the entire manifold.
* **Hypothesis**: Models that operate heavily on distinct subspaces lower the overall effective dimension. A model perfectly memorizing data tends to have very sparse high dimensions.
* **Citation/Paper**: `Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information.` [Scientific Reports]
* **File & Function**: `src/blme/tasks/geometry/intrinsic_dim.py` -> `IntrinsicDimensionTask`
* **Critical Info**: Returns a single global scalar, contrasting with Local Intrinsic Dimensionality (LID) which assesses local point neighborhoods.

*(Note: There is an index 22, but `geometry/` also contains files like `information_geometry.py`, `matrix_entropy.py` which are already listed above).*
