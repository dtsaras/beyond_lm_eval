# Geometry Tasks

This module contains metrics that evaluate the high-dimensional spatial geometry of the LLM's latent representation manifold.

**Current registry coverage (27 tasks)**: `geometry_categories`,
`geometry_cka`, `geometry_collapse`, `geometry_contextualization`,
`geometry_correlation_dimension`, `geometry_hsic`, `geometry_hubness`,
`geometry_intrinsic_dim`, `geometry_isoscore`, `geometry_lid`,
`geometry_lipschitz`, `geometry_mahalanobis`, `geometry_matrix_entropy`,
`geometry_mp_bulk_deviation`, `geometry_neural_collapse`,
`geometry_perplexity`, `geometry_positional_decay`,
`geometry_prediction_alignment`, `geometry_representation_sensitivity`,
`geometry_rsa`, `geometry_schatten`, `geometry_spectral`, `geometry_svd`,
`geometry_tokenizer_efficiency`, `geometry_trajectory_curvature`,
`geometry_unembedding`, and `geometry_weight_norms`.

**Paper-faithful vs. BLME proxy notes**: IsoScore uses the Rudman et al.
covariance-isotropy score (`arXiv:2108.07344`). `geometry_hsic` implements
HSIC dependence, not a KDE mutual-information estimator. `geometry_prediction_alignment`,
`geometry_categories`, and `geometry_weight_norms` are BLME diagnostics without a
single canonical paper. `geometry_tokenizer_efficiency` reports tokenizer
properties (fertility/compression/vocabulary usage); these are confounds and
efficiency diagnostics, not standalone proof of downstream quality.

---

## 1. Local Intrinsic Dimensionality (LID)
* **What are we measuring**: The local degrees of freedom of the representation manifold around a specific point.
* **How are we measuring**: Using Maximum Likelihood Estimation (MLE) on the nearest neighbor distances (typically k=10 or k=20) to compute the local non-integer dimensionality.
* **Hypothesis**: Models with excessively high LID might suffer from the curse of dimensionality and overfitting, while very low LID implies over-compression.
* **Citation/Paper**: `Levina, E. & Bickel, P. J. (2004). Maximum Likelihood Estimation of Intrinsic Dimension.` [NeurIPS 2004]. Secondary: `Ma, X., et al. (2018). Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality.` [ICLR 2018]
* **File & Function**: `src/blme/tasks/geometry/lid.py` -> `LocalIntrinsicDimensionalityTask`
* **Critical Info**: LID changes drastically from shallow layers to deep layers, often forming an "intrinsic dimension bottleneck."

## 2. Lipschitz Continuity Analysis
* **What are we measuring**: The local smoothness and sensitivity of the model to small perturbations in the input space.
* **How are we measuring**: Practically estimated by computing the ratio of the distance between output representations to the distance between input representations for closely neighbored points.
* **Hypothesis**: High Lipschitz constants indicate an unstable, highly chaotic representation space vulnerable to adversarial perturbations. Low constants indicate smooth, stable generalization.
* **Citation/Paper**: `Anil, C., Lucas, J., & Grosse, R. (2019). Sorting out Lipschitz function approximation.` [ICML 2019, ArXiv: 1811.05381]. Related estimation work: `Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks.` and `Scaman, K. & Virmaux, A. (2018). Lipschitz regularity of deep neural networks.` [NeurIPS 2018, ArXiv: 1805.10965]
* **File & Function**: `src/blme/tasks/geometry/lipschitz.py` -> `LipschitzContinuityTask`
* **Critical Info**: Extremely hard to measure analytically; this task uses an empirical local approximation based on sampled neighbors.

## 3. Representational Similarity Analysis (RSA)
* **What are we measuring**: The structural isomorphism between the representation spaces of two different models, or two different layers.
* **How are we measuring**: By computing a Representational Dissimilarity Matrix (RDM) of pairwise distances for a set of inputs, and then finding the Spearman rank correlation between the upper triangles of two RDMs.
* **Hypothesis**: Two networks might have different exact geometries but computationally identical relative similarity structures. RSA allows comparison across models with different hidden dimensions.
* **Citation/Paper**: `Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis-connecting the branches of systems neuroscience.` [Frontiers in Systems Neuroscience]
* **File & Function**: `src/blme/tasks/geometry/rsa.py` -> `RepresentationalSimilarityTask`
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
* **Citation/Paper**: `Amari, S. (1998). Natural gradient works efficiently in learning.` [Neural Computation, Vol 10(2)] — foundational reference for information geometry of neural networks. Application to LLMs is an active area of research.
* **File & Function**: `src/blme/tasks/geometry/information_geometry.py` -> `RepresentationSensitivityTask`
* **Critical Info**: FIM is computationally intractable to store entirely; the trace is an efficient scalar summary of total curvature.

## 6. Matrix Entropy (Information Bottleneck)
* **What are we measuring**: The data compression capabilities of the LLM layers over inference.
* **How are we measuring**: By computing the von Neumann spectral entropy over the internal covariance matrix of the hidden states at each layer.
* **Hypothesis**: As information passes through an LLM, the model actively filters out noise. A decreasing or low layer-wise matrix entropy indicates the model is actively forming a tighter semantic "Information Bottleneck".
* **Citation/Paper**: `Wei, L., Tan, Z., Li, C., Wang, J., & Huang, W. (2024). Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models.` [NeurIPS 2024, ArXiv: 2401.17139]
* **File & Function**: `src/blme/tasks/geometry/matrix_entropy.py` -> `MatrixEntropyTask`
* **Critical Info**: Values typically decrease monotonically in deeper layers as the network compresses raw syntax into refined semantic logic.

## 7. Correlation Dimension (Fractal Geometry)
* **What are we measuring**: The underlying fractal complexity and self-similarity of the generated language manifold.
* **How are we measuring**: Using the Grassberger-Procaccia algorithm. Measures the fraction of points within a radius $r$ and computes the log-log scaling coefficient.
* **Hypothesis**: Correlation dimension can reveal non-integer scaling structure in the sampled representation cloud. It is evidence about fractal-like scaling, not proof that language globally lies on a fractal attractor.
* **Citation/Paper**: `Grassberger, P. & Procaccia, I. (1983). Characterization of strange attractors.` [Phys. Rev. Lett. 50(5)]. BLME uses the classical estimator; it does not implement a paper-specific LLM correlation-dimension pipeline.
* **File & Function**: `src/blme/tasks/geometry/correlation_dimension.py` -> `CorrelationDimensionTask`
* **Critical Info**: Requires larger sample sizes to compute pairwise distances effectively. Normal text generally exhibits a non-integer structural dimension around ~6-7.

## 8. Positional Attention Decay (RoPE Geometry)
* **What are we measuring**: The structural integrity and geometric degradation of context windows.
* **How are we measuring**: Computing the Spearman rank correlation between absolute positional discrete token distance and the attention magnitude allocated to those past tokens.
* **Hypothesis**: To extrapolate well to long sequences, the attention matrix should exhibit a structurally sound, smooth geometric decay relative to distance. Breakdown (random correlations) indicates failure of the positional embeddings (e.g., RoPE).
* **Citation/Paper**: `Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.` and long-context extrapolation methods like `Chen, Q., et al. (2023). Extending Context Window of Large Language Models via Position Interpolation.`
* **File & Function**: `src/blme/tasks/geometry/positional_decay.py` -> `PositionalAttentionDecayTask`
* **Critical Info**: Requires sequences longer than a few tokens to establish a valid distance/attention correlation pattern.

## 9. SVD Isotropy (geometry_svd)
* **What are we measuring**: The isotropy (roundness) of the representation space.
* **How are we measuring**: Decomposing the hidden state matrix with SVD and calculating the ratio of the top singular value to the sum of all singular values, or looking at the variance drop-off.
* **Hypothesis**: Highly anisotropic spaces (e.g., dominating outlier dimensions) collapse semantics into a narrow cone, degrading similarity metrics. Isotropic spaces utilize capacity more uniformly.
* **Citation/Paper**: `Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?` [EMNLP 2019, ArXiv: 1909.00512]
* **File & Function**: `src/blme/tasks/geometry/isotropy.py` -> `SVDIsotropyTask`
* **Critical Info**: Language models almost always suffer from an "anisotropy cone" unless explicitly regularized or normalized.

## 10. Hubness
* **What are we measuring**: The tendency of certain tokens to be the "nearest neighbor" of an unusually high number of other tokens in latent space.
* **How are we measuring**: Computing pairwise cosine similarities and tracking the skewed distribution of incoming Nearest Neighbor (1-NN) edges. High skew/max indicates severe hubness.
* **Hypothesis**: The "Curse of Dimensionality" leads to spatial hubs in high dimensions. These hubs crowd semantic spaces and degrade zero-shot retrieval and generation.
* **Citation/Paper**: `Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data.` [Journal of Machine Learning Research (JMLR) Vol 11]
* **File & Function**: `src/blme/tasks/geometry/hubness.py` -> `GlobalHubnessTask`
* **Critical Info**: Highly sensitive to the choice of similarity metric (L2 distance vs Cosine). Usually worse under Euclidean distance.

## 11. Category Separation
* **What are we measuring**: How well conceptually related words group together organically.
* **How are we measuring**: Comparing the average intra-category distance vs inter-category distance without labeled supervision.
* **Hypothesis**: A model with a rich geometric understanding of language will organically cluster related concepts (e.g., animals, colors) far from unrelated ones.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/categories.py` -> `CategoryGeometryTask`
* **Critical Info**: Also computes category Purity and generates coordinates for UMAP/t-SNE visualization if installed.

## 12. Prediction Alignment (`geometry_prediction_alignment`)
* **What are we measuring**: How closely the final hidden state points toward the output-projection vector for the next token.
* **How are we measuring**: Computing cosine similarity between each final hidden state and the `lm_head.weight` row for its target token. For tied-output models this is a normalized-logit proxy; for untied-output models BLME uses the actual output projection rather than the input embedding table.
* **Hypothesis**: Higher alignment indicates that the representation is geometrically arranged to support the next-token prediction. This is a BLME diagnostic, not a canonical paper metric.
* **Citation/Paper**: Derived from standard output-embedding geometry and logit-lens style analyses; no single canonical paper.
* **File & Function**: `src/blme/tasks/geometry/consistency.py` -> `PredictionAlignmentTask`
* **Critical Info**: Heavily dependent on generation parameters like temperature and top-p.

## 13. Representation Collapse
* **What are we measuring**: Severe rank collapse or dimension degeneration in the hidden states of identical repeated input tokens or near-duplicated data.
* **How are we measuring**: Computing the cosine similarity between outputs that should have been distinguishable but degenerate to the same vector due to depth.
* **Hypothesis**: Specifically in deep transformers, representations can over-smooth and lose their distinct individual token identities.
* **Citation/Paper**: `Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.` [ICML 2021, ArXiv: 2103.03404]
* **File & Function**: `src/blme/tasks/geometry/collapse.py` -> `RepresentationCollapseTask`
* **Critical Info**: More pronounced in deep networks lacking robust layernorms or residual pathway scaling.

## 14. Weight Spectral Decay
* **What are we measuring**: Heavy-tailed spectral structure and effective utilization of learned weight matrices.
* **How are we measuring**: Scanning linear/Conv1D weight matrices, computing stable rank and a Hill-estimator tail exponent over top singular values.
* **Hypothesis**: Heavy-tailed spectra can indicate learned correlation structure and implicit self-regularization; extreme collapse or noise-like spectra are warnings.
* **Citation/Paper**: `Martin, C. H. & Mahoney, M. W. (2019/2021). Traditional and Heavy-Tailed Self Regularization in Neural Network Models / Heavy-Tailed Universality Predicts Trends in Test Accuracies.` BLME's `avg_alpha` is a singular-value Hill proxy, not the exact WeightWatcher ESD alpha.
* **File & Function**: `src/blme/tasks/geometry/spectral.py` -> `WeightSpectralTask`
* **Critical Info**: Shows strong parallels between biological neural networks and artificial models.

## 15. HSIC Dependence (`geometry_hsic`)
* **What are we measuring**: Kernel dependence between representation sets, usually across layers or between input and layer representations.
* **How are we measuring**: Computing Hilbert-Schmidt Independence Criterion (HSIC) from centered kernel matrices.
* **Hypothesis**: High HSIC indicates strong statistical dependence between two representation views; low HSIC suggests a stronger transformation or decorrelation.
* **Citation/Paper**: `Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms.` Related representation-similarity use in `Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.`
* **File & Function**: `src/blme/tasks/geometry/mutual_info.py` -> `HSICDependenceTask`
* **Critical Info**: Registered task name is `geometry_hsic`; stale docs/recipes should not refer to `geometry_mutual_info`.

## 16. CKA (Centered Kernel Alignment)
* **What are we measuring**: The similarity between the underlying structures of two sets of representations without requiring them to have the same features.
* **How are we measuring**: Computing the Frobenius norm of cross-covariance matrices. Equivalent to computing the correlation of dot-product similarity matrices.
* **Hypothesis**: Permits diagnosing whether two distinct layers (or models) are learning structurally analogous concepts, ignoring rotations or isotropic scalings. 
* **Citation/Paper**: `Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited.` [ICML 2019, ArXiv: 1905.00414]
* **File & Function**: `src/blme/tasks/geometry/cka.py` -> `CKATask`
* **Critical Info**: Less prone to the scaling artifacts that affect normal canonical correlation analysis (CCA).

## 17. Unembedding Geometry
* **What are we measuring**: The relationship between the hidden language representations in the final layer and the static unembedding parameters located in the LM head.
* **How are we measuring**: Looking at the angles (cosine similarity) and norms between the highest logit token vectors and the actual dynamic context state vectors. 
* **Hypothesis**: The final language modeling head forces representations into distinct regions of the LM head space. The "Unembedding" operation exhibits severe bias due to token frequency in the pre-training set.
* **Citation/Paper**: Derived from unembedding-geometry and vocabulary-bias analyses; BLME's alignment and purity summaries are diagnostics inspired by unembedding-geometry and vocabulary-bias analyses (effective rank: Roy & Vetterli 2007, EUSIPCO).
* **File & Function**: `src/blme/tasks/geometry/unembedding.py` -> `UnembeddingDiagnosticsTask`
* **Critical Info**: Typically reveals that frequent tokens dominate the manifold geometry by pushing less frequent tokens away from the origin computationally.

## 18. Perplexity (Baseline / Y-variable Geometry)
* **What are we measuring**: Baseline auto-regressive predictability.
* **How are we measuring**: The exponentiated average negative log-likelihood of a sequence.
* **Hypothesis**: As the foundational sanity check, it proves the model can actually model text. Used purely as a baseline correlate for other intrinsic measures.
* **Citation/Paper**: Canonical language modeling metric. 
* **File & Function**: `src/blme/tasks/geometry/perplexity.py` -> `RarePPLTask`
* **Critical Info**: Lower is better. This is a performance-like baseline and is excluded from the primary intrinsic-predictor feature set.

## 19. Global Intrinsic Dimension (geometry_intrinsic_dim / PDE)
* **What are we measuring**: The global effective dimensionality of the dataset within the model's space.
* **How are we measuring**: Using TwoNN (Two Nearest Neighbors algorithm) across the entire manifold.
* **Hypothesis**: Models that operate heavily on distinct subspaces lower the overall effective dimension. A model perfectly memorizing data tends to have very sparse high dimensions.
* **Citation/Paper**: `Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information.` [Scientific Reports]
* **File & Function**: `src/blme/tasks/geometry/intrinsic_dim.py` -> `IntrinsicDimensionTask`
* **Critical Info**: Returns a single global scalar, contrasting with Local Intrinsic Dimensionality (LID) which assesses local point neighborhoods.

## 20. Trajectory Curvature (geometry_trajectory_curvature)
* **What are we measuring**: How straight the token-position trajectory of a sentence is at each layer, and how much deeper layers *straighten* it.
* **How are we measuring**: For per-sample hidden states $x_1 \dots x_T$ at a layer, form difference vectors $v_t = x_{t+1} - x_t$ and compute the discrete curvature $c_t = \arccos\!\big(\langle v_t, v_{t+1}\rangle / (\lVert v_t\rVert\,\lVert v_{t+1}\rVert)\big)$ at each interior position; the per-layer curvature is the mean over positions and samples (radians). Summary outputs: `curvature_mean_first_layer` / `_mid_layer` / `_last_layer`, `curvature_overall_mean`, `straightening_ratio` $= (c_\text{first} - c_\text{last})/c_\text{first}$, `curvature_slope` (OLS slope on normalized depth $l/(L-1)$), and quartiles across layers.
* **Hypothesis**: Trained LMs progressively straighten sentence trajectories with depth — straighter paths permit prediction by linear extrapolation — and models with better next-word prediction straighten more. A positive `straightening_ratio` / negative `curvature_slope` is the expected signature of a well-trained model.
* **Citation/Paper**: `Hosseini, E. A. & Fedorenko, E. (2023). Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language.` [NeurIPS 2023, ArXiv: 2311.04930]
* **File & Function**: `src/blme/tasks/geometry/trajectory_curvature.py` -> `TrajectoryCurvatureTask`
* **Critical Info**: Trajectory order matters — the task consumes per-sample `(T, D)` chunks (`per_sample=True`), never the flattened token cloud. The BOS token is skipped (`skip_first_tokens`, default 1) because its hidden state is an extreme outlier; cosines are clamped to $[-1, 1]$ and zero-norm difference vectors are skipped.

## 21. Marchenko-Pastur Bulk Deviation (geometry_mp_bulk_deviation)
* **What are we measuring**: How strongly per-layer activation spectra deviate from the random-matrix-theory iid null, i.e. how much *structure* (spiked directions) the representation carries beyond noise.
* **How are we measuring**: At first/mid/last layers, z-score each dimension of the flattened token cloud $X \in \mathbb{R}^{N \times D}$ (dropping zero-variance dims), form the correlation matrix $C = Z^\top Z / N$ and its eigenvalues. Under the iid null the spectrum follows the Marchenko-Pastur law with ratio $\gamma = D/N$ and bulk support $[(1-\sqrt{\gamma})^2, (1+\sqrt{\gamma})^2]$. We report the fraction of eigenvalues above the buffered upper edge (`mp_outlier_frac`, `edge_tol` = 0.05 absorbs Tracy-Widom $O(N^{-2/3})$ edge fluctuations), the variance fraction carried by those spikes (`mp_spike_energy`), and the KS distance between the empirical eigenvalue CDF and the $\gamma$-matched MP CDF (`mp_ks_distance`), plus fixed-aspect-ratio variants (`*_g25`) on a seeded token subsample with $\gamma = 0.25$.
* **Hypothesis**: Real activations are highly structured: a substantial outlier fraction and spike energy indicate many learned signal directions standing above the noise bulk (BBP-supercritical spikes). Models whose spectra hug the MP bulk encode little linear structure at that layer.
* **Citation/Paper**: `Marchenko, V. A. & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.` [Mat. Sb. 72(114):4; Math. USSR-Sbornik 1(4), 457-483] and `Baik, J., Ben Arous, G., & Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices.` [Annals of Probability 33(5), 1643-1697, ArXiv: math/0403022]
* **File & Function**: `src/blme/tasks/geometry/rmt_bulk.py` -> `MPBulkDeviationTask`
* **Critical Info**: Uses the *unordered* flattened token cloud (`per_sample=False` is correct here — RMT statements concern a population of rows). $\gamma$ varies across models, but the MP reference is $\gamma$-matched per layer so the comparison stays fair; the `*_g25` variants additionally pin $\gamma = 0.25$ for strict cross-model comparability (NaN when fewer than $\lceil D/0.25 \rceil$ tokens exist).

## 22. Contextualization
* **What are we measuring**: How much token representations change across contexts and how anisotropic the contextualized representation space is.
* **How are we measuring**: Computing Ethayarajh-style self-similarity, intra-sentence similarity, and maximum explainable variance summaries over hidden states.
* **Hypothesis**: Stronger contextualization means token representations depend meaningfully on context rather than collapsing to static lexical identity.
* **Citation/Paper**: `Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?` [EMNLP 2019, ArXiv: 1909.00512]
* **File & Function**: `src/blme/tasks/geometry/contextualization.py` -> `ContextualizationTask`
* **Critical Info**: MEV and similarity profiles are descriptive diagnostics; primary predictor analysis excludes sample-count/configuration fields.

## 23. IsoScore
* **What are we measuring**: Uniformity of embedding-space dimension utilization.
* **How are we measuring**: Applying Rudman et al.'s covariance-isotropy IsoScore to hidden-state clouds.
* **Hypothesis**: More isotropic representations use dimensional capacity more evenly and avoid narrow anisotropy cones.
* **Citation/Paper**: `Rudman, W., Gillman, N., Rayne, S., & Eickhoff, C. (2022). IsoScore: Measuring the Uniformity of Embedding Space Utilization.` [Findings of ACL 2022, ArXiv: 2108.07344]
* **File & Function**: `src/blme/tasks/geometry/isotropy.py` -> `IsoScoreTask`
* **Critical Info**: Distinct from `geometry_svd`, which reports generic SVD/effective-rank diagnostics.

## 24. Neural Collapse
* **What are we measuring**: Topic-label neural-collapse proxies on a bundled labelled corpus.
* **How are we measuring**: Computing NC1 within-class collapse, NC2 equinorm coefficient of variation, and an ETF cosine-deviation proxy over class means.
* **Hypothesis**: Lower within-class scatter and more ETF-like class means indicate a cleaner labelled representation geometry.
* **Citation/Paper**: `Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training.` [PNAS, ArXiv: 2008.08186]
* **File & Function**: `src/blme/tasks/geometry/neural_collapse.py` -> `NeuralCollapseTask`
* **Critical Info**: BLME reports `nc2_etf_cosine_deviation_proxy`; it does not implement full NC2 normalized-Gram Frobenius distance or NC3 self-duality.

## 25. Schatten / MNN / RankMe
* **What are we measuring**: Spectral geometry of row-normalized per-sentence hidden-state matrices.
* **How are we measuring**: Centering columns, row-L2 normalizing, then computing row-normalized Schatten-p norms (p=1,4,∞), Matrix Nuclear-Norm, and RankMe.
* **Hypothesis**: Spectral concentration and nuclear-norm summaries can proxy representation compression and rank utilization.
* **Citation/Paper**: `Yusupov et al. (2025). From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation.` [ArXiv: 2509.25359]; `Li, Xia, Chang, & Wu (2024). Large Language Model Evaluation via Matrix Nuclear-Norm.` [ArXiv: 2410.10672]; `Garrido et al. (2023). RankMe.` [ICML 2023, ArXiv: 2210.02885]
* **File & Function**: `src/blme/tasks/geometry/schatten.py` -> `SchattenNormTask`
* **Critical Info**: `schatten_2` is intentionally not exposed because it equals a content-free function of token count and width after row-L2 normalization.

## 26. Tokenizer Efficiency
* **What are we measuring**: Tokenizer-dependent fertility and compression properties for a fixed text sample.
* **How are we measuring**: Counting tokens per word/character, compression ratio, vocabulary size, and related tokenizer statistics.
* **Hypothesis**: Tokenizer efficiency affects compute and can correlate with model family maturity, but it is a confound rather than a representation-quality proof.
* **Citation/Paper**: Related to tokenizer fertility and tokenization-efficiency literature; BLME treats this as a tokenizer diagnostic.
* **File & Function**: `src/blme/tasks/geometry/tokenizer_efficiency.py` -> `TokenizerEfficiencyTask`
* **Critical Info**: Marked non-primary for intrinsic-predictor analyses because it is tokenizer/configuration dependent.

## 27. Weight Norm Profiles
* **What are we measuring**: Layerwise norms and stable-rank style summaries of model weight matrices.
* **How are we measuring**: Scanning weight matrices and reporting Frobenius norm, spectral norm, and derived uniformity/profile summaries.
* **Hypothesis**: Extreme layer-norm concentration can reveal architectural or training instabilities.
* **Citation/Paper**: BLME diagnostic related to standard spectral/norm analyses; no single canonical paper.
* **File & Function**: `src/blme/tasks/geometry/weight_norms.py` -> `WeightNormProfileTask`
* **Critical Info**: Use alongside `geometry_spectral`; do not interpret raw norm size without architecture context.

