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
* **Paper summary**: Levina & Bickel derive the MLE of intrinsic dimension from nearest-neighbor distances by modeling them as a Poisson process (their canonical form sums to k−1). Ma et al. adapt it (the −k variant, their Eq. 4) to characterize adversarial subspaces, showing LID rises sharply for adversarial examples. BLME implements Ma et al.'s −k variant (per-point), reporting mean/std/median/min/max.
* **Verification** (2026-07, campaign B5): `_lid_mle` bit-exact (diff 0.0) vs the official xingjunm `mle_batch` transcription; full-pipeline ≤4.3e-7 on 200×32. Docstring citations corrected: the Levina-Bickel title was fabricated ("Local Intrinsic Dimensionality Estimation via Maximum Likelihood" → real title) and the −k formula is Ma et al. 2018's, not Levina-Bickel's (metadata note was already accurate).

## 2. Lipschitz Continuity Analysis
* **What are we measuring**: The local smoothness and sensitivity of the model to small perturbations in the input space.
* **How are we measuring**: Per-layer relative-change ratios of the residual stream: for each token, ‖h^(l+1) − h^(l)‖ / ‖h^(l)‖, averaged over tokens per layer (`lipschitz_max` = max over layers of these per-layer token-means; the mean-aggregation choice is documented in the module docstring).
* **Hypothesis**: High Lipschitz constants indicate an unstable, highly chaotic representation space vulnerable to adversarial perturbations. Low constants indicate smooth, stable generalization.
* **Citation/Paper**: `Anil, C., Lucas, J., & Grosse, R. (2019). Sorting out Lipschitz function approximation.` [ICML 2019, ArXiv: 1811.05381]. Related estimation work: `Miyato, T., et al. (2018). Spectral Normalization for Generative Adversarial Networks.` and `Scaman, K. & Virmaux, A. (2018). Lipschitz regularity of deep neural networks.` [NeurIPS 2018, ArXiv: 1805.10965]
* **File & Function**: `src/blme/tasks/geometry/lipschitz.py` -> `LipschitzContinuityTask`
* **Critical Info**: Proxy-only — an empirical hidden-state relative-change diagnostic, NOT the operator-norm Lipschitz bound of the cited papers (disclaimed in the docstring). The exact σ_max kernel is verified separately against the official avirmaux/lipEstimation power method.
* **Verification** (2026-07, campaign B4): hand-derived 4-layer/3-pair analytic case matches all five outputs to 1e-6; distilgpt2 end-to-end matches a from-scratch NumPy reference; official lipEstimation clone diffed identical to upstream and its power method run live against BLME's svdvals kernel. The mean-vs-max aggregation ambiguity (campaign-3 DISCREPANCY) is resolved and documented.

## 3. Representational Similarity Analysis (RSA)
* **What are we measuring**: The structural isomorphism between the representation spaces of two different models, or two different layers.
* **How are we measuring**: By computing a Representational Dissimilarity Matrix (RDM) of pairwise distances for a set of inputs, and then finding the Spearman rank correlation between the upper triangles of two RDMs.
* **Hypothesis**: Two networks might have different exact geometries but computationally identical relative similarity structures. RSA allows comparison across models with different hidden dimensions.
* **Citation/Paper**: `Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis-connecting the branches of systems neuroscience.` [Frontiers in Systems Neuroscience]
* **File & Function**: `src/blme/tasks/geometry/rsa.py` -> `RepresentationalSimilarityTask`
* **Critical Info**: Because RSA is $O(N^2)$, the `max_tokens` parameter controls the computational cost.
* **Verification** (2026-07, campaign B6/C3): CONFIRMED_CORRECT — all 6 outputs match a from-scratch Spearman (rank + Pearson-on-ranks, no scipy) AND rsatoolbox 0.3.2 to 1.1e-16 on controlled 5-layer activations; citations verified (Kriegeskorte, Mur & Bandettini 2008, Front. Syst. Neurosci. 2:4).

## 4. Latent Mahalanobis OOD Distance
* **What are we measuring**: How far Out-Of-Distribution (OOD) a sample is, accounting for the natural covariance of the in-distribution manifold.
* **How are we measuring**: By modeling a reference dataset's representations as a multivariate Gaussian (computing empirical mean and covariance matrix), and then measuring the Mahalanobis distance of new test points relative to this Gaussian.
* **Hypothesis**: Simple Euclidean distance is flawed in highly anisotropic spaces. Mahalanobis distance correctly scales by the principal axes of variance, providing a true measure of semantic anomaly.
* **Citation/Paper**: `Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks.` [ArXiv: 1807.03888]
* **File & Function**: `src/blme/tasks/geometry/mahalanobis.py` -> `MahalanobisOODTask`
* **Critical Info**: The covariance matrix must be inverted or pseudo-inverted. High-dimensional spaces ($d > N$) require Tikhonov regularization (adding $\epsilon I$) to prevent singularity.
* **Verification** (2026-07, campaign B7): distance == official deep_Mahalanobis_detector kernel (sqrt(−2·term_gau), diff 0.0 at matched precision); Ledoit-Wolf-vs-empirical proxy deviation quantified & documented; the dataset=None held-out bug (scored duplicates of fit texts, inverting the OOD conclusion) fixed with a distinct ID pool + regression test; NEW analytic pin for the sklearn-free pinv fallback (closed-form diagonal-covariance case); paper's Eq. 2 (not Eq. 3) reference fixed in the parity comments.

## 5. Trace of the Empirical Fisher Information Matrix
* **What are we measuring**: The local curvature and sharpness of the representation manifold.
* **How are we measuring**: By computing the Trace of the Empirical Fisher Information Matrix (FIM) of the token representations with respect to the output logits/probabilities.
* **Hypothesis**: A "sharp" minimum (high trace) often correlates with poor generalization out-of-distribution, while a "flat" minimum (low trace) suggests robust generalization.
* **Citation/Paper**: `Amari, S. (1998). Natural gradient works efficiently in learning.` [Neural Computation, Vol 10(2)] — foundational reference for information geometry of neural networks. Application to LLMs is an active area of research.
* **File & Function**: `src/blme/tasks/geometry/information_geometry.py` -> `RepresentationSensitivityTask`
* **Critical Info**: FIM is computationally intractable to store entirely; the trace is an efficient scalar summary of total curvature.
* **Verification** (2026-07, campaign B7): closed-form gradient W_y − E_p[W] == float64 autograd and finite differences (≤1.1e-6 rel, pure fp32 round-off) on real gpt2; NEW bias-ful-head cross-check (the `logits += bias` branch had zero coverage — flipping its sign passed the suite). Citation fixed: the docstring cited a non-existent (title, year) pairing — now Amari 1998, "Natural Gradient Works Efficiently in Learning", Neural Computation 10(2).

## 6. Matrix Entropy (Information Bottleneck)
* **What are we measuring**: The data compression capabilities of the LLM layers over inference.
* **How are we measuring**: By computing the von Neumann spectral entropy over the internal covariance matrix of the hidden states at each layer.
* **Hypothesis**: As information passes through an LLM, the model actively filters out noise. A decreasing or low layer-wise matrix entropy indicates the model is actively forming a tighter semantic "Information Bottleneck".
* **Citation/Paper**: `Wei, L., Tan, Z., Li, C., Wang, J., & Huang, W. (2024). Diff-eRank: A Novel Rank-Based Metric for Evaluating Large Language Models.` [NeurIPS 2024, ArXiv: 2401.17139]
* **Paper summary**: Wei et al. define per-sentence matrix entropy as the von Neumann entropy of the trace-normalized covariance of centered, row-normalized hidden states (Def. 4.1/4.3, averaged over sentences and divided by log d), and show it separates trained from random models and tracks quality; v2 of the paper (Diff-eRank, NeurIPS 2024) recasts the same quantity as an effective-rank difference between the untrained and trained model.
* **File & Function**: `src/blme/tasks/geometry/matrix_entropy.py` -> `MatrixEntropyTask`
* **Critical Info**: Values typically decrease monotonically in deeper layers as the network compresses raw syntax into refined semantic logic. Cache and fresh paths both truncate at 512 tokens (the official protocol used 2048 — documented in the code).
* **Verification** (2026-07, campaign B4): NEW dedicated parity file `tests/tasks/parity/test_matrix_entropy_parity.py` — verbatim float64 transcription of the official Diff-eRank normalize/cal_cov/cal_entropy vs BLME's kernel on three shapes incl. N<d and low-rank (≤1e-5, fp32 kernel), an exact ln 2 analytic anchor, and a mutant-discrimination check (previously NO numeric parity test existed and centering/normalization mutants passed the suite). The 512-vs-128 cache/fresh truncation mismatch (campaign-3 DISCREPANCY) is fixed.

## 7. Correlation Dimension (Fractal Geometry)
* **What are we measuring**: The underlying fractal complexity and self-similarity of the generated language manifold.
* **How are we measuring**: Using the Grassberger-Procaccia algorithm. Measures the fraction of points within a radius $r$ and computes the log-log scaling coefficient.
* **Hypothesis**: Correlation dimension can reveal non-integer scaling structure in the sampled representation cloud. It is evidence about fractal-like scaling, not proof that language globally lies on a fractal attractor.
* **Citation/Paper**: `Grassberger, P. & Procaccia, I. (1983). Characterization of strange attractors.` [Phys. Rev. Lett. 50(5)]. BLME uses the classical estimator; it does not implement a paper-specific LLM correlation-dimension pipeline.
* **File & Function**: `src/blme/tasks/geometry/correlation_dimension.py` -> `CorrelationDimensionTask`
* **Critical Info**: Requires larger sample sizes to compute pairwise distances effectively. Normal text generally exhibits a non-integer structural dimension around ~6-7.
* **Paper summary**: Grassberger & Procaccia define the correlation integral C(r) — the fraction of point pairs within distance r — and show its log-log slope ν (the correlation dimension) measures an attractor's fractal dimensionality from a finite sample; it lower-bounds the Hausdorff dimension and became the standard practical estimator.
* **Verification** (2026-07, campaign B5): kernel matches official `nolds.corr_dim` (run live in the parity suite) after removing nolds' diagonal self-match artifact; slope-invariant normalization difference vs the paper's 1/N² documented. Stale references to the pre-extraction inline kernel updated (helper `_gp_correlation_dimension` since 754e06b); REPOSITORIES.md pointer fixed (giotto-tda has no CorrelationDimension — nolds/skdim are the real references).

## 8. Positional Attention Decay (RoPE Geometry)
* **What are we measuring**: The structural integrity and geometric degradation of context windows.
* **How are we measuring**: Computing the Spearman rank correlation between absolute positional discrete token distance and the attention magnitude allocated to those past tokens.
* **Hypothesis**: To extrapolate well to long sequences, the attention matrix should exhibit a structurally sound, smooth geometric decay relative to distance. Breakdown (random correlations) indicates failure of the positional embeddings (e.g., RoPE).
* **Citation/Paper**: `Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.` and long-context extrapolation methods like `Chen, Q., et al. (2023). Extending Context Window of Large Language Models via Position Interpolation.`
* **File & Function**: `src/blme/tasks/geometry/positional_decay.py` -> `PositionalAttentionDecayTask`
* **Critical Info**: Requires sequences longer than a few tokens to establish a valid distance/attention correlation pattern.
* **Verification** (2026-07, campaign B7): proxy-only (no defining equation); Spearman kernel == fully hand-written tie-averaged reference to 1.1e-16 over 200 randomized rows; RoFormer decay premise and ALiBi citation verified real and topical; a stale ledger warning flag (claimed-but-unrecorded citation correction) removed.

## 9. SVD Isotropy (geometry_svd)
* **What are we measuring**: The isotropy (roundness) of the representation space.
* **How are we measuring**: Centered SVD of the hidden-state matrix → Roy-Vetterli effective rank, participation ratio, and spectral AUC; plus the Ethayarajh Eq. 4 anisotropy baseline — the mean cosine of ~1000 sampled pairs of RAW (uncentered) states, self-pairs excluded.
* **Hypothesis**: Highly anisotropic spaces (e.g., dominating outlier dimensions) collapse semantics into a narrow cone, degrading similarity metrics. Isotropic spaces utilize capacity more uniformly.
* **Citation/Paper**: `Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?` [EMNLP 2019, ArXiv: 1909.00512]
* **File & Function**: `src/blme/tasks/geometry/isotropy.py` -> `SVDIsotropyTask`
* **Critical Info**: Language models almost always suffer from an "anisotropy cone" unless explicitly regularized or normalized.
* **Paper summary**: Ethayarajh shows contextual representations are highly anisotropic — random word pairs have near-1 cosine similarity in upper layers of GPT-2/BERT/ELMo — so isotropy-corrected measures are needed; the anisotropy baseline (his Eq. 4) is the expected cosine between uniformly sampled occurrence pairs on raw representations.
* **Verification** (2026-07, campaign B5): erank/PR/AUC pinned analytically (exact singular values, ≤1e-13) and vs the official IsoScore package; `avg_cosine_similarity` historically deviated — it was computed on CENTERED states (reads ~0 always, −0.047 on gpt2 where Eq. 4 gives 0.973) — now computed on raw states with self-pair rejection per the cited definition, and exact-pinned in the comprehensive parity test (was bound-checked only).

## 10. Hubness
* **What are we measuring**: The tendency of certain tokens to be the "nearest neighbor" of an unusually high number of other tokens in latent space.
* **How are we measuring**: Computing pairwise cosine similarities and, for each k in `k_values`, the k-occurrence distribution N_k (how many times each token appears in other tokens' top-k lists, self excluded). Reported: skewness of N_k (the canonical hubness statistic), Gini, max, and top-1% mass.
* **Hypothesis**: The "Curse of Dimensionality" leads to spatial hubs in high dimensions. These hubs crowd semantic spaces and degrade zero-shot retrieval and generation.
* **Citation/Paper**: `Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data.` [JMLR 11:2487-2531]; `Tomašev, Radovanović, Mladenić, Ivanović (2014). The Role of Hubness in Clustering High-Dimensional Data.` [IEEE TKDE 26(3):739-751]
* **Paper summary**: Radovanović et al. show that in high dimensions the k-occurrence distribution becomes heavily right-skewed: a few points ("hubs") appear in a disproportionate share of neighbor lists, an intrinsic-dimensionality effect that degrades nearest-neighbor methods; S_Nk (skewness of N_k) is their canonical hubness measure — exactly what BLME reports. Tomašev et al. extend the analysis to clustering.
* **File & Function**: `src/blme/tasks/geometry/hubness.py` -> `GlobalHubnessTask`
* **Critical Info**: Highly sensitive to the choice of similarity metric (L2 distance vs Cosine). Usually worse under Euclidean distance.
* **Verification** (2026-07, campaign B4): kernel matches scikit-hubness (bincount + scipy.stats.skew, self-hit filter, Gini) line-for-line; NEW end-to-end pin in `test_geometry.py` — evaluate() on a seeded 257×24 fake model vs an independent full-argsort numpy reference (skew/gini/max ≤1e-6; a dropped self-mask now fails loudly). PAPERS.md Tomašev title corrected.

## 11. Category Separation
* **What are we measuring**: How well conceptually related words group together organically.
* **How are we measuring**: Comparing the average intra-category distance vs inter-category distance without labeled supervision.
* **Hypothesis**: A model with a rich geometric understanding of language will organically cluster related concepts (e.g., animals, colors) far from unrelated ones.
* **Citation/Paper**: Derived from general geometric alignment literature [No specific conference paper].
* **File & Function**: `src/blme/tasks/geometry/categories.py` -> `CategoryGeometryTask`
* **Critical Info**: Also computes category Purity and generates coordinates for UMAP/t-SNE visualization if installed.
* **Verification** (2026-07, campaign B7): separation+purity were already pinned; NEW coverage for the previously never-executed kernels — relation consistency and analogy exact on a constructed constant-relation embedding (both 1.0 analytically), hub_skew/hub_max vs an independent k-occurrence recomputation. BLME diagnostic (proxy-only, tokenizer/category dependent).

## 12. Prediction Alignment (`geometry_prediction_alignment`)
* **What are we measuring**: How closely the final hidden state points toward the output-projection vector for the next token.
* **How are we measuring**: Computing cosine similarity between each final hidden state and the `lm_head.weight` row for its target token. For tied-output models this is a normalized-logit proxy; for untied-output models BLME uses the actual output projection rather than the input embedding table.
* **Hypothesis**: Higher alignment indicates that the representation is geometrically arranged to support the next-token prediction. This is a BLME diagnostic, not a canonical paper metric.
* **Citation/Paper**: Derived from standard output-embedding geometry and logit-lens style analyses; no single canonical paper.
* **File & Function**: `src/blme/tasks/geometry/consistency.py` -> `PredictionAlignmentTask`
* **Critical Info**: Heavily dependent on generation parameters like temperature and top-p.
* **Verification** (2026-07, campaign B7): tied model == pure-numpy cosine reference (≤7e-9); the central untied-head claim is NOW numerically pinned (GPT2 with tie_word_embeddings=False and a re-randomized head: the projection tensor must be lm_head.weight, not the embedding table — previously only a source-grep guarded this).

## 13. Representation Collapse
* **What are we measuring**: Severe rank collapse or dimension degeneration in the hidden states of identical repeated input tokens or near-duplicated data.
* **How are we measuring**: Computing the cosine similarity between outputs that should have been distinguishable but degenerate to the same vector due to depth.
* **Hypothesis**: Specifically in deep transformers, representations can over-smooth and lose their distinct individual token identities.
* **Citation/Paper**: `Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.` [ICML 2021, ArXiv: 2103.03404]
* **File & Function**: `src/blme/tasks/geometry/collapse.py` -> `RepresentationCollapseTask`
* **Critical Info**: More pronounced in deep networks lacking robust layernorms or residual pathway scaling.
* **Verification** (2026-07, campaign B6): erank == verbatim Roy-Vetterli Eq(1)/Def. 1 + scipy svdvals (analytic anchors exact: sv=[4,2,2] → 2^1.5 to 4.4e-16); B6 pinned the previously untested `max_drop`/`max_drop_layer` (a sign flip passed the suite before). Citations fixed: first author Queipo-de-Llano restored on arXiv:2510.06477 (docstring/metadata/PAPERS.md); a fabricated PAPERS.md row (wrong authors AND a physics arXiv id) deleted; the 'top-23' predictor rank corrected to 22.

## 14. Weight Spectral Decay
* **What are we measuring**: Heavy-tailed spectral structure and effective utilization of learned weight matrices.
* **How are we measuring**: Scanning linear/Conv1D weight matrices, computing stable rank and a Hill-estimator tail exponent over top singular values.
* **Hypothesis**: Heavy-tailed spectra can indicate learned correlation structure and implicit self-regularization; extreme collapse or noise-like spectra are warnings.
* **Citation/Paper**: `Martin, C. H. & Mahoney, M. W. (2019/2021). Traditional and Heavy-Tailed Self Regularization in Neural Network Models / Heavy-Tailed Universality Predicts Trends in Test Accuracies.` BLME's `avg_alpha` is a singular-value Hill proxy, not the exact WeightWatcher ESD alpha.
* **Paper summary**: Martin & Mahoney's Heavy-Tailed Self-Regularization theory fits power laws to the eigenvalue spectral density of weight matrices and finds that well-trained networks develop heavy tails (small α) without explicit regularization; across families, smaller/more uniform α predicts better test accuracy — with the follow-up (Martin, Peng & Mahoney, Nat. Comms. 2021) showing α-based metrics predict quality WITHOUT any training/test data. BLME's `avg_alpha` is a documented proxy: a Hill estimator on the fixed top-20% singular-value tail; at matched x_min the identity α_σ = 2·α_MM − 1 to the official eigenvalue fit is exact.
* **File & Function**: `src/blme/tasks/geometry/spectral.py` -> `WeightSpectralTask`
* **Critical Info**: Shows strong parallels between biological neural networks and artificial models. Stable rank ‖W‖_F²/‖W‖_2² definition: Rudelson & Vershynin 2007 (previously mis-attributed to Bartlett et al. 2017, which never uses it — fixed in B4).
* **Verification** (2026-07, campaign B4): Hill kernel equals official weightwatcher-0.7.7 WW_powerlaw at forced x_min (identity exact at 0.0); NEW anti-drift test drives the REAL WeightSpectralTask on a planted-spectrum Linear and pins avg_alpha/avg_stable_rank against the transcribed kernel (the parity file previously never imported blme).
* **B8 extension** (2026-07): the task now ALSO emits the official HT-SR estimator — Clauset power-law MLE α = 1 + n/Σln(λ/x_min) with KS-optimal x_min scan on the correlation-matrix EIGENVALUES (`alpha_official`, per-layer + avg/median) and `alpha_hat` = α·log₁₀ λ_max (Martin-Mahoney quality metric). Verified: planted Pareto α=3 recovered as 2.994. The 2026-07-16 audit found the KS scan used the right-continuous empirical CDF (Clauset/plfit/WW use left-continuous) — invisible on smooth planted spectra but real weight spectra landed in different KS minima (mean |Δα| 0.41, max 3.64 vs live WW over gpt2's modules). Fixed (+ scan cap raised): now equals live weightwatcher 0.7.7 across ALL 49 gpt2 modules to mean |Δα| 3e-4, max 2.1e-3. The Hill-proxy `avg_alpha` is retained unchanged.

## 15. HSIC Dependence (`geometry_hsic`)
* **What are we measuring**: Kernel dependence between representation sets, usually across layers or between input and layer representations.
* **How are we measuring**: Computing Hilbert-Schmidt Independence Criterion (HSIC) from centered kernel matrices.
* **Hypothesis**: High HSIC indicates strong statistical dependence between two representation views; low HSIC suggests a stronger transformation or decorrelation.
* **Citation/Paper**: `Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005). Measuring Statistical Dependence with Hilbert-Schmidt Norms.` Related representation-similarity use in `Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.`
* **File & Function**: `src/blme/tasks/geometry/mutual_info.py` -> `HSICDependenceTask`
* **Critical Info**: Registered task name is `geometry_hsic`; stale docs/recipes should not refer to `geometry_mutual_info`.
* **Verification** (2026-07, campaign B6): normalized HSIC == verbatim Kornblith cka() + explicit Gretton Def. 2 (fixture 0.450953881199808, diff 3.3e-15); B6 refactored evaluate() to call the parity-pinned `_nhsic_from_grams` kernel (previously the pinned wrapper was never used by production and only key presence was asserted). Docstring corrected: the anchor layer is the FIRST TRANSFORMER BLOCK's output, not the input embedding (the cache drops hs[0]).

## 16. CKA (Centered Kernel Alignment)
* **What are we measuring**: The similarity between the underlying structures of two sets of representations without requiring them to have the same features.
* **How are we measuring**: Computing the Frobenius norm of cross-covariance matrices. Equivalent to computing the correlation of dot-product similarity matrices.
* **Hypothesis**: Permits diagnosing whether two distinct layers (or models) are learning structurally analogous concepts, ignoring rotations or isotropic scalings. 
* **Citation/Paper**: `Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited.` [ICML 2019, ArXiv: 1905.00414]
* **File & Function**: `src/blme/tasks/geometry/cka.py` -> `CKATask`
* **Critical Info**: Less prone to the scaling artifacts that affect normal canonical correlation analysis (CCA).
* **Verification** (2026-07, campaign B6/C3): CONFIRMED_CORRECT — executed the ACTUAL official google-research Demo.ipynb code: feature_space_linear_cka = 0.450953881199808 bit-identical to the pinned fixture; BLME ≤1.8e-8 (float32 cast). Citations verified (Kornblith et al. ICML 2019; Cortes et al. 2012 centered alignment).

## 17. Unembedding Geometry
* **What are we measuring**: Static geometry of the LM head (unembedding) matrix: how many directions it effectively uses, whether semantically related tokens cluster, and how it relates to the input embeddings.
* **How are we measuring**: Four static diagnostics — (1) tied-weight check (W_out ≈ E_in); (2) Roy-Vetterli effective rank of the MEAN-CENTERED unembedding matrix (GPU SVD); (3) top-k category purity over a bundled concept taxonomy (fraction of each labelled token's k nearest unembedding rows sharing its category); (4) per-token cosine alignment between the final-layer-norm-transformed input embedding and the unembedding row. No hidden states, logits, or norms are involved.
* **Hypothesis**: Higher effective rank = richer prediction geometry; higher purity = semantically organized unembedding space; alignment characterizes how far apart the input/output vocabularies drift in untied models.
* **Citation/Paper**: Effective rank: `Roy, O. & Vetterli, M. (2007). The Effective Rank: A Measure of Effective Dimensionality.` [EUSIPCO 2007]. The purity and alignment summaries are BLME diagnostics (proxy-only).
* **Paper summary**: Roy & Vetterli define the effective rank as exp of the Shannon entropy of the L1-normalized singular values — a smooth, threshold-free count of "how many dimensions the matrix really uses"; BLME applies it to the mean-centered unembedding matrix (centering changes the value by ~4% and is part of the definition used here).
* **File & Function**: `src/blme/tasks/geometry/unembedding.py` -> `UnembeddingDiagnosticsTask`
* **Critical Info**: Tier-1 (weights only). Purity iterates the labelled tokens directly (a historic bug sampled the whole vocab and purity collapsed to 0); the final layer norm is applied before alignment so tied models don't trivially score 1.0.
* **Verification** (2026-07, campaign B4): effective rank matches a float64 numpy re-derivation of the EUSIPCO definition (5.4e-6 rel, fp32-vs-fp64); purity matched an independently rebuilt label map + full-argsort reference to 1e-10 on distilgpt2; NEW exact analytic anchor in `test_geometry.py` — two orthogonal category clusters → purity exactly 1.0 (previously only bound-checked).

## 18. Perplexity (Baseline / Y-variable Geometry)
* **What are we measuring**: Baseline auto-regressive predictability.
* **How are we measuring**: The exponentiated average negative log-likelihood of a sequence.
* **Hypothesis**: As the foundational sanity check, it proves the model can actually model text. Used purely as a baseline correlate for other intrinsic measures.
* **Citation/Paper**: Canonical language modeling metric. 
* **File & Function**: `src/blme/tasks/geometry/perplexity.py` -> `RarePPLTask`
* **Critical Info**: Lower is better. This is a performance-like baseline and is excluded from the primary intrinsic-predictor feature set.
* **Verification** (2026-07, campaign B7): NEW dedicated parity file — ppl_overall/ppl_rare/ppl_freq/BPC reconstructed from scratch (float64 log-softmax, independent (count,id) bucket sort) on gpt2 (≤1e-4 rel); BOS-adding-tokenizer BPC regression pinned (the historic recount/scoring desync under-reported BPC ~13%); empty-text rows skip cleanly (previously crashed). Both campaign-3 BUGs fixed.

## 19. Global Intrinsic Dimension (geometry_intrinsic_dim / PDE)
* **What are we measuring**: The global effective dimensionality of the dataset within the model's space.
* **How are we measuring**: Using TwoNN (Two Nearest Neighbors algorithm) across the entire manifold.
* **Hypothesis**: Models that operate heavily on distinct subspaces lower the overall effective dimension. A model perfectly memorizing data tends to have very sparse high dimensions.
* **Citation/Paper**: `Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information.` [Scientific Reports]
* **File & Function**: `src/blme/tasks/geometry/intrinsic_dim.py` -> `IntrinsicDimensionTask`
* **Critical Info**: Returns a single global scalar, contrasting with Local Intrinsic Dimensionality (LID) which assesses local point neighborhoods.
* **Paper summary**: Facco et al.'s Two-NN estimates global intrinsic dimension from only the two nearest neighbors of each point: the ratio μ = r2/r1 follows a Pareto distribution with exponent d, so a straight-line fit through the origin of the empirical CDF recovers d — minimizing the effect of density variation and curvature; they discard the top 10% of μ values for robustness (BLME matches, tail_trim=0.1).
* **Verification** (2026-07, campaign B5): bit-identical (0.0) to a verbatim skdim 0.3.4 `_TwoNN` fit transcription on analytic Pareto samples; formula-faithful label kept deliberately conservative (μ≤1 pre-filter deviates from skdim on duplicate-heavy data). Docstring corrected: the paper never discusses sub-1 estimates (a "Fig. 3" claim was fabricated) — the honest rationale is the Pareto support d ∈ (0, ∞); ledger cert label reconciled.

## 20. Trajectory Curvature (geometry_trajectory_curvature)
* **What are we measuring**: How straight the token-position trajectory of a sentence is at each layer, and how much deeper layers *straighten* it.
* **How are we measuring**: For per-sample hidden states $x_1 \dots x_T$ at a layer, form difference vectors $v_t = x_{t+1} - x_t$ and compute the discrete curvature $c_t = \arccos\!\big(\langle v_t, v_{t+1}\rangle / (\lVert v_t\rVert\,\lVert v_{t+1}\rVert)\big)$ at each interior position; the per-layer curvature is the mean over positions and samples (radians). Summary outputs: `curvature_mean_first_layer` / `_mid_layer` / `_last_layer`, `curvature_overall_mean`, `straightening_ratio` $= (c_\text{first} - c_\text{last})/c_\text{first}$, `curvature_slope` (OLS slope on normalized depth $l/(L-1)$), and quartiles across layers.
* **Hypothesis**: Trained LMs progressively straighten sentence trajectories with depth — straighter paths permit prediction by linear extrapolation — and models with better next-word prediction straighten more. A positive `straightening_ratio` / negative `curvature_slope` is the expected signature of a well-trained model.
* **Citation/Paper**: `Hosseini, E. A. & Fedorenko, E. (2023). Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language.` [NeurIPS 2023, ArXiv: 2311.04930]
* **File & Function**: `src/blme/tasks/geometry/trajectory_curvature.py` -> `TrajectoryCurvatureTask`
* **Critical Info**: Trajectory order matters — the task consumes per-sample `(T, D)` chunks (`per_sample=True`), never the flattened token cloud. The BOS token is skipped (`skip_first_tokens`, default 1) because its hidden state is an extreme outlier; cosines are clamped to $[-1, 1]$ and zero-norm difference vectors are skipped.
* **Paper summary**: Hosseini & Fedorenko show that trained LMs "straighten" sentence trajectories with depth — the discrete curvature (angle between consecutive difference vectors of the hidden-state trajectory) decreases relative to the first contextualized layer — and that models with better next-word prediction straighten more; curvature is averaged per sentence, then across sentences.
* **Verification** (2026-07, campaign B6): angle kernel == verbatim official curvature_utils.py to 8.9e-16 (f64); the layer aggregation now matches the paper's mean-of-per-sentence-means (the pooled-angles deviation was fixed and is pinned by the NEW parity file `test_trajectory_curvature_parity.py`, whose variable-length trajectories discriminate the two aggregations). Docstring corrected: the shared cache's first layer (hs[1]) EQUALS the paper's L0 baseline — an earlier note claiming a baseline mismatch was wrong.

## 21. Marchenko-Pastur Bulk Deviation (geometry_mp_bulk_deviation)
* **What are we measuring**: How strongly per-layer activation spectra deviate from the random-matrix-theory iid null, i.e. how much *structure* (spiked directions) the representation carries beyond noise.
* **How are we measuring**: At first/mid/last layers, z-score each dimension of the flattened token cloud $X \in \mathbb{R}^{N \times D}$ (dropping zero-variance dims), form the correlation matrix $C = Z^\top Z / N$ and its eigenvalues. Under the iid null the spectrum follows the Marchenko-Pastur law with ratio $\gamma = D/N$ and bulk support $[(1-\sqrt{\gamma})^2, (1+\sqrt{\gamma})^2]$. We report the fraction of eigenvalues above the buffered upper edge (`mp_outlier_frac`, `edge_tol` = 0.05 absorbs Tracy-Widom $O(N^{-2/3})$ edge fluctuations), the variance fraction carried by those spikes (`mp_spike_energy`), and the KS distance between the empirical eigenvalue CDF and the $\gamma$-matched MP CDF (`mp_ks_distance`), plus fixed-aspect-ratio variants (`*_g25`) on a seeded token subsample with $\gamma = 0.25$.
* **Hypothesis**: Real activations are highly structured: a substantial outlier fraction and spike energy indicate many learned signal directions standing above the noise bulk (BBP-supercritical spikes). Models whose spectra hug the MP bulk encode little linear structure at that layer.
* **Citation/Paper**: `Marchenko, V. A. & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.` [Mat. Sb. 72(114):4; Math. USSR-Sbornik 1(4), 457-483] and `Baik, J., Ben Arous, G., & Péché, S. (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices.` [Annals of Probability 33(5), 1643-1697, ArXiv: math/0403022]
* **Paper summary**: Marchenko & Pastur derive the limiting eigenvalue distribution of sample covariance matrices of iid noise — the "MP bulk" with support [(1±√γ)²] and, for γ>1, a point mass at 0 from rank deficiency. Baik, Ben Arous & Péché characterize when a true signal direction escapes the bulk (the BBP phase transition): spikes above the threshold detach as outlier eigenvalues. BLME uses both as the null model: eigenvalues above the buffered bulk edge are learned structure, and the KS distance to the γ-matched MP CDF summarizes overall deviation from noise.
* **File & Function**: `src/blme/tasks/geometry/rmt_bulk.py` -> `MPBulkDeviationTask`
* **Verification** (2026-07, campaign B4): _mp_cdf vs transcribed scikit-rmt closed form and scipy.quad (≤2.3e-5 away from γ=1); BBP detachment threshold and outlier location confirmed analytically. Both campaign-3 BUGs fixed with regression tests: the γ>1 KS distance now handles the zero-eigenvalue atom (was returning ~1−1/γ regardless of fit) and _mp_cdf is accurate at γ=1 (was off by up to 0.59 at the density singularity); the previously untested γ≥1 branch is now covered.
* **B8 extension** (2026-07): lower-edge outlier statistics for γ<1 — `lower_outlier_frac`/`lower_energy` count eigenvalues below (1−tol)·λ₋ of the fitted MP bulk, following Staats et al. 2024 (ArXiv: 2410.17770), who show SMALL eigenvalues below the MP bulk carry learned structure (removing them hurts more than removing large outliers). NaN for γ≥1 (λ₋=0, no lower edge). Anchors: iid Gaussian → ≈0; planted correlated dims → ≥0.04.
* **Critical Info**: Uses the *unordered* flattened token cloud (`per_sample=False` is correct here — RMT statements concern a population of rows). $\gamma$ varies across models, but the MP reference is $\gamma$-matched per layer so the comparison stays fair; the `*_g25` variants additionally pin $\gamma = 0.25$ for strict cross-model comparability (NaN when fewer than $\lceil D/0.25 \rceil$ tokens exist).

## 22. Contextualization
* **What are we measuring**: How much token representations change across contexts and how anisotropic the contextualized representation space is.
* **How are we measuring**: Computing Ethayarajh-style self-similarity, intra-sentence similarity, and maximum explainable variance summaries over hidden states.
* **Hypothesis**: Stronger contextualization means token representations depend meaningfully on context rather than collapsing to static lexical identity.
* **Citation/Paper**: `Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?` [EMNLP 2019, ArXiv: 1909.00512]
* **File & Function**: `src/blme/tasks/geometry/contextualization.py` -> `ContextualizationTask`
* **Critical Info**: MEV and similarity profiles are descriptive diagnostics; primary predictor analysis excludes sample-count/configuration fields.
* **Verification** (2026-07, campaign B7): SelfSim/IntraSim/MEV kernels == official kawine/contextual formulas (≤3.3e-16); NEW value pins for the 3 previously uncovered kernels (MEV vs sklearn PCA with a centering-mutant check; pairwise mean vs full-triu; anisotropy baseline exact ray anchor). Docstring now states MEV is reported UNADJUSTED (the paper's headline MEV is anisotropy-adjusted and not comparable); ledger cert label reconciled to refined-adaptation.

## 23. IsoScore
* **What are we measuring**: Uniformity of embedding-space dimension utilization.
* **How are we measuring**: Applying Rudman et al.'s covariance-isotropy IsoScore to hidden-state clouds.
* **Hypothesis**: More isotropic representations use dimensional capacity more evenly and avoid narrow anisotropy cones.
* **Citation/Paper**: `Rudman, W., Gillman, N., Rayne, T., & Eickhoff, C. (2022). IsoScore: Measuring the Uniformity of Embedding Space Utilization.` [Findings of ACL 2022, ArXiv: 2108.07344]
* **File & Function**: `src/blme/tasks/geometry/isotropy.py` -> `IsoScoreTask`
* **Critical Info**: Distinct from `geometry_svd`, which reports generic SVD/effective-rank diagnostics.
* **Paper summary**: Rudman et al.'s IsoScore measures what fraction of dimensions a point cloud uses uniformly: PCA-align, L2-normalize the covariance spectrum, measure its distance from the identity, and rescale so an isotropic cloud scores 1 and a 1-dimensional cloud scores ~0 (Algorithm 1 / Eq. 4.1); unlike partition-function isotropy measures it is rotation-invariant and has an exact k-of-n interpretation.
* **Verification** (2026-07, campaign B5): exact analytic anchors — k-of-n isotropic dims → (k−1)/(n−1) to 2.2e-16; ≤4e-8 vs the official IsoScore 2.0.1 package in all regimes; NEW live-package cases added including the previously untested N<d zero-padding branch (agrees to 9e-9). Author-initial typo fixed (Taylor Rayne).

## 24. Neural Collapse
* **What are we measuring**: Topic-label neural-collapse proxies on a bundled labelled corpus.
* **How are we measuring**: Computing NC1 within-class collapse, NC2 equinorm coefficient of variation, and an ETF cosine-deviation proxy over class means.
* **Hypothesis**: Lower within-class scatter and more ETF-like class means indicate a cleaner labelled representation geometry.
* **Citation/Paper**: `Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training.` [PNAS, ArXiv: 2008.08186]
* **File & Function**: `src/blme/tasks/geometry/neural_collapse.py` -> `NeuralCollapseTask`
* **Critical Info**: BLME reports `nc2_etf_cosine_deviation_proxy`; it does not implement full NC2 normalized-Gram Frobenius distance or NC3 self-duality.
* **Verification** (2026-07, campaign B6): NC1 == the officially-RUN rhubarbwu/neural-collapse library (pinv/svd diff 0.0); NC2-ETF proxy correctly described as the Fig. 4 mean-absolute-deviation measure (docstring said "standard deviation" — Fig. 3's different statistic); comment pinpoint fixed (Eq. 3-4 are §2.9, not §2.3); dead import removed; NEW behavioral test drives the real evaluate() (previously the double-norm guard was a string match only).

## 25. Schatten / MNN / RankMe
* **What are we measuring**: Spectral geometry of row-normalized per-sentence hidden-state matrices.
* **How are we measuring**: Centering columns, row-L2 normalizing, then computing row-normalized Schatten-p norms (p=1,4,∞), Matrix Nuclear-Norm, and RankMe.
* **Hypothesis**: Spectral concentration and nuclear-norm summaries can proxy representation compression and rank utilization.
* **Citation/Paper**: `Yusupov et al. (2025). From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation.` [ArXiv: 2509.25359]; `Li, Xia, Chang, & Wu (2024). Large Language Model Evaluation via Matrix Nuclear-Norm.` [ArXiv: 2410.10672]; `Garrido et al. (2023). RankMe.` [ICML 2023, ArXiv: 2210.02885]
* **Paper summary**: Li et al. propose the Matrix Nuclear-Norm — an O(n²) L1,2 column-norm upper bound on the nuclear norm of the (centered, row-normalized) hidden-state matrix — as a single capability/compression scalar that avoids O(n³) SVDs. Yusupov et al. stress-test Schatten norms and related rank/geometry metrics of internal representations as reference-free text-quality signals (noting Schatten norms partly track output length — one reason BLME reports dimension-normalized, per-sentence-averaged forms). Garrido et al.'s RankMe (exp of the entropy of normalized singular values) predicts downstream quality of self-supervised representations without labels.
* **File & Function**: `src/blme/tasks/geometry/schatten.py` -> `SchattenNormTask`
* **Critical Info**: `schatten_2` is intentionally not exposed because it equals a content-free function of token count and width after row-L2 normalization. Schatten-p norms are reported divided by d^(1/p) (see the paper appendix's scaling derivation).
* **Verification** (2026-07, campaign B4): _normalise_matrix and _matrix_nuclear_norm_fast match the official MLGroupJLU/MatrixNuclearNorm MNN.py verbatim (diff 0.0); the per-sentence pipeline including the 2026-06-22 mnn/T fix is NOW pinned end-to-end in `test_geometry.py` against an in-test transcription of the official pipeline (previously reverting /T to /d passed all 31 tests); RankMe epsilon-placement deviation (O(1e-5)) documented in the kernel docstring.

## 26. Tokenizer Efficiency
* **What are we measuring**: Tokenizer-dependent fertility and compression properties for a fixed text sample.
* **How are we measuring**: Counting tokens per word/character, compression ratio, vocabulary size, and related tokenizer statistics.
* **Hypothesis**: Tokenizer efficiency affects compute and can correlate with model family maturity, but it is a confound rather than a representation-quality proof.
* **Citation/Paper**: Related to tokenizer fertility and tokenization-efficiency literature; BLME treats this as a tokenizer diagnostic.
* **File & Function**: `src/blme/tasks/geometry/tokenizer_efficiency.py` -> `TokenizerEfficiencyTask`
* **Critical Info**: Marked non-primary for intrinsic-predictor analyses because it is tokenizer/configuration dependent.
* **Verification** (2026-07, campaign B7): analytic fake-tokenizer anchors exact (fertility 8/3, hand-computed Shannon entropy, vocab utilisation); proxy-only label reconciled in the ledger. Citations fixed: a FABRICATED author list ("Rajput, Chamberlain, Reese" for arXiv:2310.08754 — real authors Ali, Fromm, Thellmann et al.) and Rust et al. credited as popularizing (not introducing) fertility, which the paper itself attributes to Ács 2019; partial-ρ range corrected (+0.71 to +0.77).

## 27. Weight Norm Profiles
* **What are we measuring**: Layerwise norms and stable-rank style summaries of model weight matrices.
* **How are we measuring**: Scanning weight matrices and reporting Frobenius norm, spectral norm, and derived uniformity/profile summaries.
* **Hypothesis**: Extreme layer-norm concentration can reveal architectural or training instabilities.
* **Citation/Paper**: Standard definitions — Frobenius norm, spectral norm (σ_max), stable rank ‖W‖_F²/‖W‖_2² (Rudelson & Vershynin 2007). Certified formula-faithful (upgraded from proxy-only in B4: the formulas are textbook-standard and independently parity-tested).
* **File & Function**: `src/blme/tasks/geometry/weight_norms.py` -> `WeightNormProfileTask`
* **Critical Info**: Use alongside `geometry_spectral`; do not interpret raw norm size without architecture context. `norm_uniformity` = max(0, 1 − CV) — clamped at zero (now documented). 3-D fused-MoE expert tensors are handled by per-expert slicing (campaign-3 BUG: they silently got spectral_norm=Frobenius and stable_rank=1.0).
* **Verification** (2026-07, campaign B4): analytic planted-singular-value parity (frob/spec/srank/uniformity ≤1.2e-6); stable rank agrees with the actual pip weightwatcher `stable_rank` on all distilgpt2 layers (9.2e-5 rel); ledger/metadata certification reconciled to formula-faithful.


## 28. Metric-Space Magnitude (geometry_magnitude)
* **What are we measuring**: The "effective number of distinct points" of a representation cloud across geometric scales — a diversity/structure summary from category theory.
* **How are we measuring**: Magnitude |tX| = 1ᵀ ζ⁻¹ 1 with ζ_ij = exp(−t·d_ij), evaluated on a geometric ladder of scales around the magnipy median-heuristic scale; summaries: magnitude at t_med (computed directly at t_med), log-log AUC (magnitude function area), and the magnitude dimension (max log-log slope).
* **Hypothesis**: Magnitude profiles capture multi-scale diversity: higher magnitude at moderate scales = more effectively distinct representations; the magnitude dimension proxies intrinsic dimensionality.
* **Citation/Paper**: `Limbeck, K., Andreeva, R., Sarkar, R., & Rieck, B. (2024). Metric Space Magnitude for Evaluating the Diversity of Latent Representations.` [NeurIPS 2024, ArXiv: 2311.16054]
* **Paper summary**: Limbeck et al. bring Leinster's metric-space magnitude to representation learning: magnitude counts the effective number of distinct points as a function of scale, and its area/dimension summaries quantify diversity robustly. They propose magnitude-based diversity measures for embeddings and validate against known diversity manipulations. BLME's scale ladder + log-log AUC are documented BLME choices (the paper uses its own convergence-scale selection).
* **File & Function**: `src/blme/tasks/geometry/magnitude.py` -> `MagnitudeTask`
* **Critical Info**: O(n³) solve per scale — point counts are capped. Duplicate points make ζ singular; a perturbation fallback matching magnipy is used.
* **Verification** (2026-07, campaign B5): `_magnitude` bit-exact vs FIVE official magnipy paths (≤2.8e-14) + Leinster line-subset closed form (1+Σtanh(t·gap/2), 8.9e-16); B5 pinned the median-heuristic scale against live magnipy INCLUDING duplicate clouds (the historic zero-distance filter deviation is fixed), `magnitude_med_scale` is now |tX| at t_med exactly, and the numpy-1.x `np.trapezoid` crash is fixed with a source-level guard test.

## 29. Vendi Score (geometry_vendi_score)
* **What are we measuring**: The effective number of distinct samples in a representation cloud — exp of the Shannon entropy of the similarity-kernel eigenvalues.
* **How are we measuring**: Cosine (or RBF) kernel K with unit diagonal over (subsampled) hidden states; VS = exp(−Σ λ_i log λ_i) for eigenvalues of K/n.
* **Hypothesis**: Higher Vendi score = more diverse representations; unlike effective rank / RankMe it uses a NONLINEAR kernel, so it measures sample diversity rather than linear-subspace utilization.
* **Citation/Paper**: `Friedman, D. & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine Learning.` [TMLR 2023, ArXiv: 2210.02410]
* **Paper summary**: Friedman & Dieng define the Vendi Score as the exponential of the von Neumann entropy of a similarity kernel — a reference-free diversity metric requiring no labels or reference distribution, interpretable as the effective number of dissimilar samples. They apply it across modalities to diagnose mode collapse and diversity trade-offs.
* **File & Function**: `src/blme/tasks/geometry/vendi.py` -> `VendiScoreTask`
* **Critical Info**: Subsampling to `max_points` gives a deterministic size-CAPPED estimate (VS ≤ max_points) — not unbiased (docstring corrected in B5); the cap is exposed via `_meta_n_points_per_layer` and the normalized variant.
* **Verification** (2026-07, campaign B5/C2): bit-exact vs official `vendi_score.vendi.score_K` on the same kernel (live pip package); independent eigvalsh Definition-1 re-derivation 0.0–1.4e-14; official `score_X(normalize=True)` path matches to 8.9e-16.

## 30. Persistent-Homology Dimension (geometry_phd_dimension)
* **What are we measuring**: A topologically-motivated intrinsic dimension: how total minimum-spanning-tree edge length scales with sample size.
* **How are we measuring**: The GPTID estimator: for subsample sizes n, the total H0 persistence (MST length) E(n) scales as n^((d−1)/d) (α=1); d is recovered from the log-log slope m as d = 1/(1−m), with medians over subsamples and averaged restarts.
* **Hypothesis**: Lower PHD = more compressed, lower-dimensional representation manifolds; Tulchinskii et al. show PHD separates human from AI-generated text.
* **Citation/Paper**: `Tulchinskii, E., Kuznetsov, K., Kushnareva, L., Cherniavskii, D., Nikolenko, S., Burnaev, E., Barannikov, S., & Piontkovskaya, I. (2023). Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts.` [NeurIPS 2023, ArXiv: 2306.04723]
* **Paper summary**: The paper shows text embeddings of human-written text have a characteristically higher persistent-homology dimension (~9 for human vs ~7 for generated English), giving a robust AI-text detector that survives domain shift. The PHD estimator itself (MST-length power law) goes back to Birdal et al.'s topological generalization bounds; Tulchinskii et al.'s GPTID implementation is the official reference.
* **File & Function**: `src/blme/tasks/geometry/phd_dimension.py` -> `PHDimensionTask`
* **Critical Info**: Stochastic estimator — RNG-pinned for determinism. The general-α inversion in the docstring is d = α/(1−m) (a wrong formula α/(α−m) was fixed in B5); GPTID returns 1/(1−m) regardless of α, correct only at α=1 (documented).
* **Verification** (2026-07, campaign B5/C2): seed-matched bit-exact (diff 0.0) vs official ArGintum/GPTID on default AND non-default schedules including α=0.5; reference checkout at `~/.cache/blme-refrepos/GPTID` (B1 infra) so the parity tests run rather than skip.

## 31. CKNNA — Mutual-kNN Conditional CKA (geometry_cknna)
* **What are we measuring**: LOCAL representational alignment between layers: CKA restricted to pairs that are mutual k-nearest neighbors, so only local-neighborhood structure contributes.
* **How are we measuring**: Row-L2-normalize the features (reference preprocessing), build kernels K, L; CKNNA = HSIC(mask⊙K, mask⊙L) / (√(HSIC(mask_K⊙K,·)·HSIC(mask_L⊙L,·)) + 1e-6) with the mutual-kNN mask, unbiased HSIC (Song et al. 2012 Eq. 5) by default.
* **Hypothesis**: Per the Platonic Representation Hypothesis, alignment between representations (especially local-neighborhood alignment) increases with model capability and across modalities.
* **Citation/Paper**: `Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). Position: The Platonic Representation Hypothesis.` [ICML 2024, PMLR 235:20617-20642, ArXiv: 2405.07987]
* **Paper summary**: Huh et al. argue that as models scale, their representations converge toward a shared statistical model of reality across architectures and modalities; they measure this convergence with mutual-kNN alignment metrics (CKNNA is their kernel-alignment variant defined in the appendix) and show alignment increases with scale and downstream competence.
* **File & Function**: `src/blme/tasks/geometry/cknna.py` -> `CKNNATask`
* **Critical Info**: Identical inputs give ~1.0 (exactly 1.0 only with biased HSIC — the unbiased reference default gives ~0.9999 via the +1e-6 regularizer, documented). `distance_agnostic=True` is NON-FUNCTIONAL (crashes identically in the official code at the pinned commit) and now raises NotImplementedError.
* **Verification** (2026-07, campaign B6/C2/C3): bit-exact vs official platonic-rep `metrics.cknna` (unbiased + biased HSIC, both orderings) at the pinned commit dcd76ba; B6 added coverage for the `_l2_normalize` preprocessing (was untested — the fixture tensors were pre-normalized) with a full-pipeline equality check on unnormalized input.

## 32. Procrustes Linearity (geometry_procrustes_linearity)
* **What are we measuring**: How close each adjacent-layer transformation is to a LINEAR map — the "transformers are secretly linear" signature.
* **How are we measuring**: Per adjacent layer pair, fit the unconstrained least-squares map A = X⁺Y (NOT orthogonal Procrustes) between normalized representations and report the resulting similarity, per the official LLM-Microscope procedure; the across-depth profile is the signal.
* **Hypothesis**: Adjacent-layer maps in trained LMs are nearly linear (similarity ≈ 0.99); departures localize where nonlinear computation concentrates.
* **Citation/Paper**: `Razzhigaev, A., Mikhalchuk, M., Goncharova, E., Gerasimenko, N., Oseledets, I., Dimitrov, D., & Kuznetsov, A. (2024). Your Transformer is Secretly Linear.` [ACL 2024 (long), ArXiv: 2405.12250]
* **Paper summary**: The paper finds a near-perfect linear relationship (Procrustes similarity ~0.99) between sequential transformer layers' embeddings, shows the linearity is connected to the residual stream's dominance, and demonstrates that pruning the most-linear layers (or regularizing linearity away) barely hurts — evidence that much of depth is close to linear.
* **File & Function**: `src/blme/tasks/geometry/procrustes_linearity.py` -> `ProcrustesLinearityTask`
* **Critical Info**: The absolute value is conditioning-dependent (the official code's unguarded 1/S amplifies ~1e-17 singular values) — interpret the across-depth PROFILE, not any single value (documented in the parity test and metadata).
* **Verification** (2026-07, campaign B6/C2/C3): CONFIRMED_CORRECT — |diff| = 0 vs the ACTUAL pip wheel llm_microscope 0.0.7 on 4 fresh cases including a rank-deficient one and an exact-linear anchor (1.0); real-model values 0.9988–0.99999 consistent with the paper's ~0.99 headline.

## 33. Superposition Strength (geometry_superposition_strength)
* **What are we measuring**: How tightly the unembedding packs V vocabulary directions into m ≪ V dimensions — the degree of representation superposition, normalized by the information-theoretic packing optimum.
* **How are we measuring**: Mean squared cosine overlap E_{i≠j}[cos²(W_i, W_j)] over row pairs of the RAW row-normalized LM head — the official-repo protocol (overlap-0.py: no centering, all pairs; BLME uses exact all-pairs when feasible, else a seeded 200k-pair sample) — divided by the Welch bound κ² = (V−m)/(m(V−1)). Also reported: a MEAN-CENTERED variant (disclosed BLME auxiliary that removes the shared anisotropy direction), the vs-random-Gaussian ratios (mso·m), and row-norm dispersion stats.
* **Hypothesis**: Per Liu-Liu-Gore, LLMs operate in the STRONG-superposition regime — overlap ≈ the Welch/1/m optimum (ratio ≈ 1–2) — and this packing regime underlies robust neural scaling; departures flag under-utilized representation capacity.
* **Citation/Paper**: `Liu, Y., Liu, Z., & Gore, J. (2025). Superposition Yields Robust Neural Scaling.` [NeurIPS 2025, ArXiv: 2505.10465]
* **Paper summary**: Liu et al. build toy models spanning weak→strong superposition and show the neural-scaling exponent is governed by the superposition regime: under strong superposition, loss scales robustly as ~1/m regardless of the feature-frequency distribution, because representation vectors pack near the geometric overlap floor. They verify that open-weight LLMs (GPT-2, Pythia, OPT, Qwen) sit in the strong regime: the mean squared overlap of their LM-head vectors follows the ~1/m Welch scaling across model widths, and connect this to the observed ~1/m loss-scaling of real LLM families.
* **File & Function**: `src/blme/tasks/geometry/superposition_strength.py` -> `SuperpositionStrengthTask`
* **Critical Info**: Weights-only (no forward pass). PROTOCOL (corrected in the 2026-07-16 audit): the headline uses RAW rows exactly as the official code does — the paper's ~1/m claim is a log-log scaling TREND of the raw overlap across hidden sizes, not a coefficient-1 statement. On real LLMs the shared anisotropy direction dominates the raw value (gpt2: raw 0.075 = 58× Welch vs centered 0.0025 = 1.9× Welch), so BLME also emits `mean_squared_overlap_centered` as a disclosed auxiliary isolating packing net of the common mean.
* **Verification** (2026-07, campaign B8 + 2026-07-16 audit): analytic anchors exact — orthonormal rows → 0, simplex ETF achieves the Welch bound to rel 1e-6, iid Gaussian rows → 1/m (rel 0.1); sampled ≈ exact (rel 0.05). The 2026-07-16 adversarial audit read the OFFICIAL repo (overlap-0.py) and established it never centers — the headline was switched to the raw protocol and the centered variant relabeled a BLME auxiliary, pinned by `test_task_headline_follows_official_raw_protocol`.

## 34. Effective Depth / Block Influence (geometry_effective_depth)
* **What are we measuring**: How much each transformer layer actually transforms the residual stream, and how much of the network's depth does real work.
* **How are we measuring**: Block Influence BI_l = 1 − E_t[cos(h_l,t, h_{l+1},t)] per layer (ShortGPT's hidden-state form), plus per-layer contribution ratios and `effective_depth_90` — the fraction of depth needed to accumulate 90% of the total transformation (a BLME summary INSPIRED BY Csordás et al.'s depth-utilization analysis; their own statistic is an importance-weighted depth score, not this). Also: mean/min/max BI and the fraction of near-identity layers (BI < 0.05).
* **Hypothesis**: Trained LMs contain redundant, near-identity layers (especially late-middle ones); low effective depth signals wasted capacity — such layers can be pruned with little performance loss (ShortGPT) and models "use their depth" far from fully (Csordás).
* **Citation/Paper**: `Men, X., et al. (2024). ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.` [ArXiv: 2403.03853]; `Csordás, R., Manning, C. D., & Potts, C. (2025). Do Language Models Use Their Depth Efficiently?` [NeurIPS 2025, ArXiv: 2505.13898]
* **Paper summary**: Men et al. define Block Influence — one minus the expected token-wise cosine between a block's input and output — rank layers by it, and show that simply deleting the lowest-BI ~25% of layers of Llama-2/Baichuan barely hurts benchmarks, beating contemporary pruning methods; redundancy concentrates in later-middle layers. Csordás et al. analyze residual-stream contributions and find the second half of layers in modern LLMs contributes far less and mostly refines existing features rather than computing new ones — i.e., models under-utilize their depth; they quantify this with cumulative-contribution profiles across depth (the shape BLME's `effective_depth_90` summarizes).
* **File & Function**: `src/blme/tasks/geometry/effective_depth.py` -> `EffectiveDepthTask`
* **Critical Info**: Runs its OWN forward pass (the shared cache drops the embedding output and its final entry is post-final-norm). A forward hook captures the last block's PRE-norm output — the 2026-07-16 audit showed ~90% of the old last-layer BI on gpt2 was the ln_f artifact (BI 0.737→0.166 after the fix; effective_depth_90 0.917→0.25). The `frac_low_influence` 0.05 threshold is a BLME choice (ShortGPT ranks and prunes to a ratio; no threshold).
* **Verification** (2026-07, campaign B8): analytic anchors exact — identity layer → BI = 0, sign-flip layer → BI = 2, random orthogonal rotation → BI ≈ 1, and a stack with all transformation concentrated in the first quarter → effective_depth_90 = 0.25. `tests/tasks/parity/test_b8_new_metrics_parity.py::TestEffectiveDepth`.
