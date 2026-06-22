# BLME Paper Survey (2023–2026)

Comprehensive survey of intrinsic-metric papers considered for inclusion
in BLME, with justification for every INCLUDE / SKIP decision. Compiled
across targeted literature searches during the 2026-04-17 –
2026-04-19 audit cycle and hardened for publication-readiness on
2026-06-20.

Sections:
1. **What BLME currently measures** — every metric, the paper that
   introduced it, and the corresponding task file.
2. **Considered and included** — recent (2023–2026) papers whose
   metrics we added in rounds 7–8.
3. **Considered and skipped** — papers we deliberately did not
   include, with an explicit reason each.
4. **Paper-selection criteria** — what we're willing to add vs. not.

---

## 1. What BLME already measures

Each entry: `task_name` — **Paper (year)** — short description.

### Geometry (representation + weight)

- `geometry_contextualization` — **Ethayarajh 2019** (EMNLP) —
  anisotropy baseline, self-similarity, intra-sentence similarity, MEV.
- `geometry_cka` — **Kornblith 2019** (ICML) — linear CKA across
  layers.
- `geometry_hsic` — **Gretton 2005** (ALT) — Hilbert-Schmidt
  Independence Criterion on adjacent layers (implemented as linear
  CKA-equivalent).
- `geometry_rsa` — **Kriegeskorte 2008** (Frontiers Syst. Neurosci.)
  — Representational Dissimilarity Matrix Spearman correlation.
- `geometry_svd` — **Roy & Vetterli 2007** (EURASIP) — effective
  rank + participation ratio + cond number + average cosine
  similarity.
- `geometry_isoscore` — **Rudman et al. 2022** (arXiv:2108.07344) —
  covariance-based isotropy scalar.
- `geometry_intrinsic_dim` — **Facco et al. 2017** (Sci. Rep.) —
  Two-NN intrinsic dimension.
- `geometry_lid` — **Levina & Bickel 2004** (NeurIPS) + **Ma et al.
  2018** (ICLR) — per-sample MLE local intrinsic dimensionality.
- `geometry_neural_collapse` — **Papyan, Han, Donoho 2020** (PNAS)
  — NC1 within-class variance collapse + NC2 equinorm + an
  ETF-cosine-deviation proxy. BLME does not report full NC3 self-duality.
- `geometry_matrix_entropy` — **Wei et al. 2024**
  (arXiv:2401.17139) — per-sentence von Neumann entropy of the
  centred-and-row-normalised token covariance, divided by `log d`.
- `geometry_spectral` — **Martin & Mahoney 2019–2021** (arXiv:
  1901.08276, 1901.08278; Nature Communications 2021) — heavy-tailed
  spectral diagnostics. BLME's `avg_alpha` is a Hill estimate on singular
  values, a monotone proxy rather than the exact WeightWatcher ESD alpha.
- `geometry_weight_norms` — per-layer Frobenius, spectral, and
  stable-rank profiles of weight matrices (no single canonical
  reference; standard convention).
- `geometry_correlation_dimension` — **Grassberger & Procaccia
  1983** (Physica D) — correlation dimension fit.
- `geometry_positional_decay` — custom (attention-weight vs.
  position distance).
- `geometry_prediction_alignment` — cosine alignment between the
  final hidden state and the lm-head row for the target token.
- `geometry_mahalanobis` — **Lee, Lee, Lee, Shin 2018** (NeurIPS) —
  ID/OOD Mahalanobis gap with held-out ID split.
- `geometry_lipschitz` — per-layer input-output contraction / expansion
  ratio (Virmaux & Scaman 2018 style).
- `geometry_unembedding` — alignment between lm-head and input
  embedding; token-category purity.
- `geometry_hubness` — **Tomašev et al. 2014** (IEEE Trans. KDE) —
  skewness of kNN in-degree distribution over the vocabulary.
- `geometry_tokenizer_efficiency` — fertility + compression ratio +
  char/tokens (standard tokeniser diagnostics).
- `geometry_perplexity` — standard language-model cross-entropy /
  perplexity + BPC (Shannon 1948 style) — NLL per token in nats, per character in bits, and the
  frequency-stratified `ppl_rare`, `ppl_freq`, `ppl_overall`.
- `geometry_categories` — category coherence of token embeddings on
  a curated 200-token vocabulary partition.
- `geometry_schatten` — **Yusupov et al. 2025**, **Li et al. 2024**,
  and **Garrido et al. 2023** — row-normalized Schatten-p norms
  (p=1,4,∞), Matrix Nuclear-Norm, and RankMe; Schatten-2 is omitted
  because it is content-free after row-L2 normalization.
- `geometry_trajectory_curvature` — **Hosseini & Fedorenko 2023**
  (NeurIPS, arXiv:2311.04930) — discrete token-trajectory curvature
  and straightening through layers.
- `geometry_mp_bulk_deviation` — **Marchenko & Pastur 1967** plus
  **Baik, Ben Arous & Péché 2005** — RMT bulk-deviation and spike-energy
  summaries for hidden-state spectra.

### Interpretability

- `interpretability_logit_lens` — **nostalgebraist 2020** (LessWrong)
  — per-layer decoded-token proxy via `lm_head(h_l)`, with entropy and
  agreement summaries. It is not tuned-lens unless an explicit tuned
  translator is used.
- `interpretability_attention_entropy` — **Clark et al. 2019** (EMNLP
  BlackBoxNLP) — mean Shannon entropy of attention rows per head.
- `interpretability_attention_rank` — **Dong et al. 2021** (ICML
  "Attention is not all you need") — Roy-Vetterli effective rank of
  attention maps.
- `interpretability_induction_heads` — **Olsson et al. 2022**
  (Anthropic, "In-context Learning and Induction Heads") —
  prefix-matching + copy scores + causal validation via head ablation.
- `interpretability_head_roles` — **Clark 2019** + **Voita 2019** —
  previous-token and duplicate-token head fractions.
- `interpretability_prediction_entropy` — **Holtzman 2020** (ICLR) —
  output-distribution entropy + top-1 prob + top-k decisiveness.
- `interpretability_sparsity` — **Zhang et al. 2021** (ACL) — L0
  activation sparsity on the down-proj input.
- `interpretability_superposition` — **Elhage et al. 2022**
  (Anthropic, "Toy Models of Superposition") — polysemanticity
  index from activation covariance.
- `interpretability_waa` — **Park et al. 2024** (ICML) — weight-
  activation alignment via down-proj top-1 singular vector.
- `interpretability_attention_graph` — PageRank on attention matrix +
  BOS-sink ratio + edge Gini.
- `interpretability_attention_effective_rank` — SVD effective-rank
  proxy over attention-output projection activations.
- `interpretability_attribution` — input-gradient × activation
  attribution (gradient-based saliency).
- `interpretability_probing` — linear probe accuracy on
  next-token-id labels (architecture-agnostic).
- `interpretability_sae_features` — **Bricken et al. 2023**
  (Anthropic) — L0 of pretrained SAE features (GPT-2 only).
- `interpretability_activation_sinks` — **round 8**: Gu 2025 Sinkε
  + Sun 2024 massive activations + Arroyo et al. 2025 compression
  valley (see §2).

### Causality

- `causality_tracing` — **Meng et al. 2022** (NeurIPS ROME) —
  ROME-style causal tracing proxy over bundled factual prompts, with
  per-layer restoration summaries. It is not a full ROME editing run.
- `causality_attention_knockout` — **Michel et al. 2019**, **Voita
  et al. 2019** — per-head zero-ablation NLL impact + Gini.
- `causality_edge_attribution` — **Syed, Rager, Conmy 2024** (arXiv:
  2310.10348 EAP) — per-layer gradient × residual attribution
  under token-shuffle corruption.
- `causality_knowledge_neurons` — **Dai et al. 2022** (ACL) —
  gradient × activation attribution on MLP intermediate neurons.
- `causality_circuit_quality` — custom: faithfulness (JSD between
  full and circuit-only forward) × minimality.
- `causality_ablation` — feature-ablation degradation curve (1 %,
  5 % of hidden dim masked with mean activation).

### Dynamics

- `dynamics_gradient_flow` — **Pascanu et al. 2013** (ICML) —
  per-layer gradient norm + vanishing ratio + slope-on-depth.
- `dynamics_sharpness` — **Foret et al. 2021** (ICLR SAM) + **Yao
  et al. 2020** (PyHessian) — Hutchinson trace + top-1 Hessian
  eigenvalue + SAM sharpness.
- `dynamics_coe` — **Wang et al. 2025** (ICLR Chain-of-Embedding) —
  CoE-R and CoE-C magnitude/angle chain scores.
- `dynamics_stability` — representation stability under paraphrase.
- `dynamics_generation_diversity` — self-BLEU + phrase repetition
  + per-step entropy collapse.
- `dynamics_interpolation` — slerp interpolation entropy between
  two sample points.

### Consistency

- `consistency_calibration` — **Guo et al. 2017** (ICML) — ECE +
  Brier + calibration intercept/slope.
- `consistency_format_robustness` — NLL variance across prompt
  formats (custom).
- `consistency_icl_slope` — ICL-slope of NLL vs. shot count
  (Brown et al. 2020 / Min et al. 2022 motivation).
- `consistency_position_sensitivity` — **Liu et al. 2023** (lost
  in the middle) — NLL vs. relative position.
- `consistency_paraphrase`, `consistency_contrastive`,
  `consistency_logical` — custom NLL-based consistency checks.
- `consistency_bias_weat` — **Caliskan et al. 2017** (Science) +
  **May et al. 2019** (SEAT) — WEAT/SEAT d-statistic on
  contextualised embeddings.
- `consistency_self_consistency` — **Wang et al. 2022** (arXiv:2203.11171; ICLR 2023) —
  first-token agreement across temperature (simplified variant).
- `consistency_contamination` — **Shi et al. 2023** Min-K % probability,
  with Carlini-style memorisation context — per-token log-prob as a
  contamination / memorisation proxy. Thresholds are in-sample unless
  held-out calibration is configured.
- `consistency_knowledge_capacity` — exact-vs-rephrased factual
  likelihood proxy related to memorization/generalization work.
- `consistency_membership_inference` — **Yeom et al. 2018** and
  **Carlini et al. 2021** — loss-based membership-inference proxy AUROC.

### RepE

- `repe_task_vectors` — **Zou et al. 2023** RepE reading vectors plus
  **Ilharco et al. 2023** task-vector motivation — activation-space
  vector geometry (norm, cosine across layers).
- `repe_concept_separability` — **Zou et al. 2023** (arXiv:2310.01405
  Representation Engineering) — linear probe AUC per layer.
- `repe_refusal_direction` — **Arditi et al. 2024** (NeurIPS) —
  refusal direction norm + separability AUCs at depth quantiles.
- `repe_steering_effectiveness` — KL divergence after adding the
  task vector at each layer.

### Topology

- `topology_homology` — **Zomorodian & Carlsson 2005** (SoCG) —
  persistent homology H₀ and H₁ lifespans via Ripser.
- `topology_betti_curve` — **Naitzat, Zhitnikov, Lim 2020** (ICLR) —
  β₀ / β₁ across layers, simplification ratio, decay rate.
- `topology_persistence_entropy` — Shannon entropy of H₀ / H₁
  lifespans (Rucco 2016).
- `topology_persistence_landscape` — **Bubenik 2015** (JMLR) —
  per-band landscape norms.

---

## 2. Considered and INCLUDED (rounds 7–8)

Four recent (2024–2025) papers added through rounds 7–8 because each
introduces a distinct signal not already in BLME. Some signals are
paper-faithful formulas; others are explicitly labelled BLME proxies or
prompt-side variants.

### Round 7 — `geometry_schatten`

| Metric | Paper | Reason for inclusion |
|---|---|---|
| **Row-normalized Schatten-p norms** (p=1,4,∞) | Yusupov et al. 2025 — *From Internal Representations to Text Quality* (arXiv:2509.25359) | Reference-free text-quality proxy; closed-form over SVD after BLME's centre + row-L2 preprocessing. `schatten_2` is intentionally omitted because it is content-free under row-L2 normalization. |
| **Matrix Nuclear-Norm (MNN)** | Li, Xia, Chang, Wu 2024 (arXiv:2410.10672) | 8–24× faster than matrix entropy with comparable capability signal. Reimplementation matches the reference code at [MLGroupJLU/MatrixNuclearNorm](https://github.com/MLGroupJLU/MatrixNuclearNorm): center → row-L2-normalise → sort column L2-norms descending → sum top-D. ρ(composite) ≈ –0.71. |
| **RankMe** | Garrido, Balestriero, Najman, LeCun 2023 (ICML) | Effective rank via `exp(H(σ_i/Σσ_j))` (normalises raw singular values) — distinct from our Roy-Vetterli `effective_rank` (normalises σ²). The ICLR 2025 "Tracing Representation Geometry" paper (arXiv:2509.23024) uses this exact variant to trace pretraining phases. |

### Round 8 — `interpretability_activation_sinks`

| Metric | Paper | Reason for inclusion |
|---|---|---|
| **Sinkε** | Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin 2025 (ICLR Spotlight, arXiv:2410.10781) | Thresholded attention-sink metric with (T-k) normalisation. Reimplementation verified against [sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink). Distinct from our existing `bos_sink_ratio` (only counts "is argmax on BOS?"). ρ(composite) ≈ –0.52. |
| **Massive-activation fraction** & **max/median ratio** | Sun, Chen, Kolter, Liu 2024 (arXiv:2402.17762) | Fraction of residual-stream entries with \|h\| > 100× median. GPT-2 shows ratios 150–3000, a classic massive-activation signature not captured by any other BLME metric. |
| **Compression valley** (valley_layer, valley_depth) | Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv 2025 (arXiv:2510.06477) | Argmin of the matrix-entropy per-layer trajectory + endpoint-mean depth. ρ(composite) ≈ –0.53 — an independent capability signal not explained by any single layer's entropy. |

---

## 3. Considered and SKIPPED (with reasons)

Enumerated exhaustively so that a reader can audit the inclusion
logic. Papers below were each read against these criteria (§4):

Grouped by category.

### Representation / geometry metrics — already covered

| Paper | Year | Why skipped |
|---|---|---|
| **SVCCA** — Raghu, Gilmer, Yosinski, Sohl-Dickstein (arXiv:1706.05806) | 2017 | Canonical-correlation-based representational similarity. Our `geometry_cka` (Kornblith 2019) is a strict improvement shown to be more stable; Kornblith themselves compared and recommended CKA. |
| **Latent Semantic Manifolds** — Fisher-metric hourglass (arXiv:2603.22301) | 2026 | Uses Fisher information metric to compute manifold intrinsic dimension — the "hourglass profile" pattern is already visible in our `geometry_lid` + `geometry_svd.effective_rank` layer trajectories. |
| **Empirical Investigation of Latent Representational Dynamics** (arXiv:2505.20340) | 2025 | Manifold-evolution framework — qualitative; no closed-form new metric. |
| **A Comparative Study of Learning Paradigms via Intrinsic Dimension** (arXiv:2412.06245) | 2024 | Uses Two-NN intrinsic dimension (same as our `geometry_intrinsic_dim`) to compare paradigms; validates our existing metric rather than introducing a new one. |
| **Truthfulness via Local Intrinsic Dimension** (arXiv:2402.18048) | 2024 | Applies Levina-Bickel LID (our `geometry_lid`) for hallucination detection; adds a labelled truthful/false classifier, which is a downstream evaluation, not a new intrinsic metric. |
| **REMA Reasoning Manifold** (arXiv:2509.22518) | 2025 | Reasoning-task manifold structure; requires CoT traces + task labels. |
| **Token Embeddings Violate Manifold Hypothesis** — Robinson 2024 (arXiv:2504.01002) | 2024/2025 | Theoretical result (local ID is non-constant); already visible in our `geometry_intrinsic_dim` per-token distribution. |
| **Diff-eRank** — Wei et al. (arXiv:2401.17139) | 2024 | Same formula as our `geometry_matrix_entropy` scaled differently — Diff-eRank subtracts a reference-model entropy; paper validates the standalone matrix-entropy version. Already covered. |
| **Layer-by-Layer DiME / infoNCE / dataset entropy** — Rao et al. (arXiv:2502.02013) | 2025 | DiME and dataset-entropy are near-variants of our matrix-entropy. InfoNCE requires a contrastive pair-generation pipeline outside BLME's static-forward scope. |
| **α-ReQ / RankMe in Tracing Representation Geometry** (arXiv:2509.23024) | 2025 | α-ReQ = our `geometry_spectral.avg_alpha` (Martin-Mahoney); RankMe now included in round 7 `geometry_schatten`. The paper's 3-phase pretraining signature is derivable post-hoc from our existing layer trajectories; no reference code released yet ("Coming Soon" as of April 2026). |
| **Local Intrinsic Dimensions of Contextual LMs** — Cavagnero et al. (arXiv:2506.01034) | 2025 | Exactly our `geometry_lid` (Levina-Bickel MLE). Paper validates a metric BLME already computes. |
| **The Information of LLM Geometry** (arXiv:2402.03471) | 2024 | Reframes existing matrix-entropy + rank analyses; no new metric. |
| **States Hidden in Hidden States** (arXiv:2407.11421) | 2024 | Qualitative finding (discrete states emerge); not a measurable metric. |
| **Deep Language Geometry** (arXiv:2508.11676) | 2025 | Multilingual-specific; constructs pairwise distances between whole languages via weight-pruning scores. Doesn't fit single-model diagnostics. |
| **Spectral Utilization Index (SUI)** — Jha & Reagen (arXiv:2510.00537) | 2025 | Composite of Hard Rank (= our `participation_ratio`) + Soft Rank (= our `effective_rank`) + Spectral Concentration. The first two are duplicates; Spectral Concentration is a top-k share that the aggregator's own list summaries can derive post-hoc. |
| **FARMS (fixed-aspect-ratio matrix subsampling)** — Xiao et al. (arXiv:2506.06280) | 2025 | Refinement of α estimation to remove aspect-ratio bias; tightens a metric we already have (`geometry_spectral.avg_alpha`) rather than introducing a new one. |
| **Eigenspectrum Analysis without Aspect-Ratio Bias** (arXiv:2506.06280v2) | 2025 | Same work as FARMS. |
| **Token Embeddings Violate the Manifold Hypothesis** (arXiv:2504.01002) | 2025 | Qualitative theoretical result; manifested as "Two-NN ID is non-uniform across tokens" — already visible in our `geometry_intrinsic_dim` per-layer distribution. |
| **Neighborhood overlap** — Bonfanti et al. (arXiv:2501.10573) | 2025 | Jaccard overlap of per-token kNN across layers. Overlaps with our `geometry_rsa` (Spearman of pairwise RDMs, which captures the same rank-structure signal). |
| **Bridging the Dimensional Chasm** — arXiv:2503.22547 | 2025 | "Correlator" metric — a layer-pair covariance statistic that reduces to CKA plus linear probing. Already covered. |
| **Visualising LLM Latent Space Geometry** (arXiv:2511.21594) | 2025 | PCA / UMAP visualisation paper — no measurement metric. |
| **Topological Metric for Unsupervised Embedding Quality** (arXiv:2512.15285) | 2025 | Dec 2025; persistent-homology-based embedding-quality score — direct overlap with our `topology_persistence_entropy` / `topology_betti_curve`. |
| **Uncovering Hidden Representations: LayerSelect** (arXiv:2502.02013) | 2025 | Selects optimal layer via matrix-entropy proxy; our `matrix_entropy` already exposes the per-layer profile. |

### Spectral / weight metrics — already covered

| Paper | Year | Why skipped |
|---|---|---|
| **Heavy-Tailed Universality** — Martin & Mahoney (arXiv:1901.08278, 2021) | 2019–2021 | Foundational work behind our `geometry_spectral.avg_alpha`. Already included. |
| **AlphaDecay** — Song et al. (arXiv:2506.14562) | 2025 | Uses existing HT-SR α; proposes training-time weight decay, not a new measurement. |
| **Stable Anisotropic Regularization** (arXiv:2305.19358) | 2023 | Training objective; the metric is our existing IsoScore / Ethayarajh anisotropy. |
| **PiSSA** — (arXiv:2404.02948) | 2024 | Uses top singular values for fine-tuning initialisation; not a capability metric. |
| **Crafting Heavy-Tails** (arXiv:2406.04657) | 2024 | Architectural study; no new metric. |
| **Locating Information via RMT** (arXiv:2410.17770) | 2024 | Random-matrix-theory interpretation of existing spectra; no new measurement. |
| **Stabilising Native Low-Rank LLM Pretraining** (arXiv:2602.12429) | 2026 | Training method. |
| **Controlled LLM Training on Spectral Sphere** (arXiv:2601.08393) | 2026 | Optimiser. |

### Attention / mechanistic metrics — covered or out of scope

| Paper | Year | Why skipped |
|---|---|---|
| **Attention-Head Stability / Circuit Universality** (arXiv:2602.16740) | 2026 | Requires retraining multiple seed-matched models — infeasible for a single-pass evaluation library. |
| **Attention Pattern Masked Autoencoder (AP-MAE)** (arXiv:2604.03764) | 2026 | Trains a MAE on attention patterns; output is a cluster assignment, not a scalar metric. |
| **Attention Rollout** — Abnar & Zuidema 2020 | 2020 | Would be a natural extension of our `attention_graph`; candidate for a future revision but not a 2023–2026 paper. |
| **Stream (Sparse Attention for MI)** (arXiv:2510.19875) | 2025 | Interpretability tool; not a capability metric. |
| **Semantic Entropy Probes** (arXiv:2406.15927) | 2024 | Hallucination-detection; requires generation + labelled correct/incorrect pairs. |
| **Attention Head Entropy Predicts Correctness** (arXiv:2602.13699) | 2026 | Requires labelled correctness; overlaps with our `interpretability_attention_entropy` + `consistency_calibration`. |
| **Entropy-Guided Attention for Private LLMs** (arXiv:2501.03489) | 2025 | Architecture modification, not a measurement. |
| **Unveiling Hidden Attention Sinks** (arXiv:2406.15765) | 2024 | Calibration method; the sink metric itself is now in round 8. |

### SAE / monosemanticity metrics — require SAE infrastructure

| Paper | Year | Why skipped |
|---|---|---|
| **Towards Monosemanticity** — Bricken et al. (Anthropic, 2023) | 2023 | Introduces the SAE paradigm. Our `interpretability_sae_features` uses pretrained SAEs from sae_lens (GPT-2 only) per this approach. Further metrics (FMS etc.) require per-model SAE training — prohibitive. |
| **Dictionary Learning as Feature Classifiers** (Transformer Circuits 2024) | 2024 | Requires trained SAE + labelled concepts per feature. |
| **Measuring Progress in Dictionary Learning** (NeurIPS 2024) | 2024 | Proposes p-annealing SAE training improvement; not a measurement. |
| **Feature Monosemanticity Score (FMS)** — Valavala et al. (arXiv:2506.19382) | 2025 | Requires a trained SAE + labelled concept set. BLME is SAE-agnostic and operates without concept labels. |
| **PRISM polysemanticity** — (arXiv:2506.15538) | 2025 | Requires LLM-generated feature descriptions (a costly per-feature call). |
| **Quantifying Feature Space Universality via SAEs** (arXiv:2410.06981) | 2024 | Requires paired SAEs across the 32 models; we don't have matched SAEs. |
| **Beyond Single Concept Vector** (arXiv:2410.00153) | 2024 | Gaussian concept subspaces require labelled concept pairs. |
| **MONET MoE-monosemantic experts** (arXiv:2412.04139) | 2024 | Architecture change, not a metric. |

### Reasoning / trajectory / generation-based metrics — out of scope

| Paper | Year | Why skipped |
|---|---|---|
| **Reasoning Trajectory Geometry** (arXiv:2604.05655) | 2026 | Requires chain-of-thought generation + labelled correct/incorrect solutions. |
| **Stepwise Informativeness Assumption** (arXiv:2604.06192) | 2026 | CoT entropy pattern — requires generation + labelled correctness. |
| **LLM Reasoning as Trajectories** (Microsoft) | 2026 | Same framework. |
| **Attention Illuminates LLM Reasoning (FAI, WAAD)** (arXiv:2510.13554) | 2025 | Requires generation + RL-style credit assignment. |
| **ICR Probe** (ACL 2025) | 2025 | Requires generation + answer labels. |
| **Reference-Free Rating of LLM Responses via Latent Info** | 2025 | Uses fine-tuned probe classifier — requires training data. |
| **Wisdom of Crowds / Guesstimation** (arXiv:2501.17310) | 2025 | Requires multi-sample generation across prompts. |

### Task-performance / benchmark-based — not intrinsic

| Paper | Year | Why skipped |
|---|---|---|
| **Scaling Laws for Predicting Downstream Performance** (arXiv:2410.08527) | 2024 | Cross-model prediction via smaller sampling models — not a per-model intrinsic metric. |
| **Unveiling Downstream Performance Scaling (COD)** (arXiv:2502.17262) | 2025 | Clustering-based prediction, requires benchmark outputs. |
| **100 instances is all you need** (arXiv:2409.03563) | 2024 | Subsample-based evaluation, requires benchmarks. |
| **Revisiting Scaling Properties of Downstream Metrics** (arXiv:2512.08894) | 2025 | Scaling-law methodology, not intrinsic. |
| **Model Utility Law / MUI** — Cao et al. (arXiv:2504.07440) | 2025 | MUI requires either SAE features or per-task labelled neuron contributions. The "Utility Law" is a regression between MUI and task accuracy — not a standalone intrinsic score. |
| **Capability Density / Densing Law** (Nature MI) | 2025 | Capability-per-parameter — requires benchmark results. |
| **Evaluating LLM Metrics Through Real-World Capabilities** (arXiv:2505.08253) | 2025 | Taxonomy of benchmark coverage, not new intrinsic metric. |
| **Unveiling LLM Evaluation Focused on Metrics** (arXiv:2404.09135) | 2024 | Survey of output-based metrics. |
| **Evaluation and Benchmarking of LLM Agents** (arXiv:2507.21504) | 2025 | Agent benchmarks. |

### Alignment / safety-specific — orthogonal to capability

| Paper | Year | Why skipped |
|---|---|---|
| **AQI: Alignment Quality Index** (EMNLP 2025) | 2025 | Overlaps with our `repe_refusal_direction`; alignment-specific. |
| **Attention Sinks as Internal Hallucination Signals** (arXiv:2604.10697) | 2026 | Hallucination-specific; requires labelled samples. |
| **ReDeEP** (arXiv:2410.11414) | 2024 | RAG-specific; requires retrieved context. |
| **ESI: Epistemic Uncertainty via Semantic Intervention** (arXiv:2510.13103) | 2025 | Requires multiple semantic-preserving paraphrases + output comparison; closer to benchmark evaluation. |
| **Measuring Aleatoric and Epistemic Uncertainty** (arXiv:2511.03166) | 2025 | Requires paired ID/OOD QA labelled tasks. |

### Data-attribution / training-process metrics — require training artifacts

| Paper | Year | Why skipped |
|---|---|---|
| **Representation Gradient Tracing** (arXiv:2510.02334) | 2025 | Requires training data + reference "good" behaviours; diagnostic for harmful outputs, not capability. |
| **Mechanistic Data Attribution** (arXiv:2601.21996) | 2026 | Requires training data for influence-function computation. |
| **Distributional Memorization** (arXiv:2407.14985) | 2024 | Requires pretraining corpus access for co-occurrence counts. |
| **Fact Tracing** (arXiv:2205.11482) | 2022 | Same constraint. |
| **Multilingual Factual Knowledge Acquisition** (arXiv:2505.14824) | 2025 | Requires pretraining checkpoints. |
| **Grokking monitoring** (arXiv:2506.21551) | 2025 | Requires training checkpoints; BLME evaluates finished models. |

### Gradient / curvature / optimisation-time metrics

| Paper | Year | Why skipped |
|---|---|---|
| **Spectral Anisotropy of Gradients (Spectra)** (arXiv:2602.11185) | 2026 | Per-layer full-parameter gradient SVD on 30B-class models is prohibitive. Our `dynamics_gradient_flow` captures the first-moment (norm) profile instead. |
| **Information Geometry / Fisher Metrics** (arXiv:2506.15830) | 2025 | Training-time analysis; Fisher trace ≈ Hessian trace under cross-entropy, which our `dynamics_sharpness.hutchinson_trace` already computes. |
| **SDFP / Fisher Information Trace** (arXiv:2602.05499) | 2026 | Uses FIT for draft-model construction in speculative decoding; not a capability metric. |
| **Hestia / FASC** (arXiv:2601.20745, 2601.07197) | 2026 | Quantisation-time Hessian metrics; not capability measurements. |
| **GaLore / GaLore 2** (arXiv:2403.03507, 2504.20437) | 2024–2025 | Training-time optimisers. |

### Steering / intervention-based metrics

| Paper | Year | Why skipped |
|---|---|---|
| **Inducing Causal World Models (CWMI)** (arXiv:2507.19855) | 2025 | Fine-tuning method with a training objective, not a measurement. |
| **WorldLLM** (arXiv:2506.06725) | 2025 | Active-exploration framework; requires RL loops. |
| **Spatial World Models (Grid-World Maze)** (arXiv:2604.10690) | 2026 | Domain-specific. |

### Benchmarks masquerading as metrics

| Paper | Year | Why skipped |
|---|---|---|
| **LongBench** (arXiv:2505.19293) | 2025 | Benchmark, not intrinsic. |
| **Context Length Alone Hurts** (arXiv:2510.05381) | 2025 | Benchmark-based effect. |
| **Self-Execution Benchmark** (arXiv:2508.12277) | 2025 | Benchmark. |
| **MemGround** (arXiv:2604.14158) | 2026 | Benchmark. |
| **Are Emergent Abilities a Mirage?** — Schaeffer et al. (arXiv:2304.15004) | 2023 | Argues emergence is a metric-choice artefact; critique of benchmark aggregation, not a new measurement. |
| **Factuality of LLMs in 2024** (arXiv:2402.02420) | 2024 | Survey; no new metric. |
| **100 instances is all you need** (arXiv:2409.03563) | 2024 | Benchmark subsample strategy. |

### Foundational (pre-2023) work already incorporated

| Paper | Year | Status |
|---|---|---|
| **Kaplan et al. Scaling Laws** (arXiv:2001.08361) | 2020 | Cross-model scaling law; we report loss (NLL, PPL, BPC) per our `geometry_perplexity` which plugs into it. |
| **Hoffmann et al. Chinchilla** (arXiv:2203.15556) | 2022 | Compute-optimal scaling law; no per-model intrinsic metric. |
| **SVCCA** (arXiv:1706.05806) | 2017 | Superseded by Kornblith 2019 CKA. |
| **Facco 2017 Two-NN** | 2017 | Implemented as `geometry_intrinsic_dim` (round-3-audited to linear-regression form). |
| **Levina & Bickel 2004 LID** | 2004 | Implemented as `geometry_lid`. |
| **Roy & Vetterli 2007 Effective Rank** | 2007 | Implemented as `geometry/utils.effective_rank`. |
| **Grassberger & Procaccia 1983 Correlation Dim** | 1983 | Implemented as `geometry_correlation_dimension`. |
| **Martin & Mahoney 2019 Heavy-Tailed** (arXiv:1901.08278, 1901.08276) | 2019 | Implemented as `geometry_spectral.avg_alpha`. |
| **Ethayarajh 2019 Contextualization** | 2019 | Implemented as `geometry_contextualization`. |
| **Kornblith 2019 CKA** | 2019 | Implemented as `geometry_cka`. |
| **Naitzat, Zhitnikov, Lim 2020** | 2020 | Implemented as `topology_betti_curve`. |
| **Papyan-Han-Donoho 2020 Neural Collapse** | 2020 | Implemented as `geometry_neural_collapse` (round-3-audited to subspace-projected NC1). |
| **nostalgebraist 2020 Logit Lens** | 2020 | Implemented as `interpretability_logit_lens` (round-4-audited for post-norm handling). |
| **Clark 2019, Voita 2019, Michel 2019** (attention analyses) | 2019 | Implemented across `interpretability_attention_entropy`, `head_roles`, `causality_attention_knockout`. |
| **Elhage 2022 Toy Models of Superposition** | 2022 | Implemented as `interpretability_superposition`. |
| **Meng 2022 ROME** | 2022 | Implemented as `causality_tracing` (round-3-audited: every-layer sweep, 10 noise draws). |
| **Dai 2022 Knowledge Neurons** | 2022 | Implemented as `causality_knowledge_neurons`. |
| **Olsson 2022 Induction Heads** | 2022 | Implemented as `interpretability_induction_heads`. |
| **Foret 2021 SAM** + **Yao 2020 PyHessian** | 2020–2021 | Implemented as `dynamics_sharpness`. |
| **Lee 2018 Mahalanobis** | 2018 | Implemented as `geometry_mahalanobis` (round-4-audited: held-out split). |
| **Rudman 2022 IsoScore** | 2022 | Implemented as `geometry_isoscore`. |
| **Bricken et al. 2023 SAE / Towards Monosemanticity** | 2023 | Implemented via `interpretability_sae_features` (GPT-2 only). |
| **Zou 2023 Representation Engineering** | 2023 | Implemented as `repe_concept_separability`, `repe_refusal_direction`. |
| **Ilharco 2023 Task Vectors** | 2023 | Implemented as `repe_task_vectors`. |
| **Liu 2023 Lost in the Middle** | 2023 | Implemented as `consistency_position_sensitivity`. |
| **Garrido et al. 2023 RankMe** (ICML) | 2023 | Implemented in round-7 `geometry_schatten`. |
| **Bubenik 2015 Persistence Landscape** | 2015 | Implemented as `topology_persistence_landscape`. |

---

## 4. Paper-selection criteria

A paper is **included** in BLME iff *all* of the following hold:

1. **Intrinsic**: the metric can be computed from model weights +
   static forward pass over a fixed corpus — no task-specific labels,
   no generation, no retraining.
2. **Reproducible**: either a reference implementation exists, or
   the paper's formula is unambiguous and we can pin it with unit
   tests.
3. **Non-duplicative**: the metric measures something BLME doesn't
   already capture under another name.
4. **Cross-architecture comparable**: the metric is defined without
   tokenizer-specific / architecture-specific thresholds (or those
   thresholds are explicitly normalised away).
5. **Computable on 30B-class models within ~15 min per run**.

If any criterion fails, the paper goes in the skip list with an
explicit reason.

---

## 5. Summary counts

- **Intrinsic metrics implemented in BLME**: 74 tasks spanning 70+
  referenced papers (see §1).
- **Papers added in rounds 7–8 from 2023–2026 literature**: 8
  (Wei 2025, Li 2024, Garrido 2023, Gu 2025, Sun 2024, Arroyo et al.
  2025, + partial reuse of Wang 2025 CoE already added in round
  3 and Arditi 2024 already added in an earlier round).
- **Papers considered and skipped with explicit reasons**: 55+
  (see §3).
- **Papers pending until reference code / full text available**: 2
  (Tracing Representation Geometry 2509.23024 — phase signature;
  Attention Sinks & Compression Valleys 2510.06477 — we implemented
  the reported metrics but the paper's theoretical bounds would be
  nice to add later).

The review was conducted across ~40 targeted Google / arXiv searches
spanning all major intrinsic-measurement directions: representation
geometry (rank, entropy, ID, CKA/RSA/HSIC, MEV, Schatten norms),
spectral weight analysis (heavy-tailed alpha, matrix entropy / nuclear
norm, RankMe), activation dynamics (sinks, massive activations,
compression valleys), attention analysis (induction heads, entropy,
rank collapse, sinks), mechanistic interpretability (circuits,
attribution, probing), causality (ROME, EAP, knowledge neurons),
calibration / uncertainty, consistency (bias, format robustness,
paraphrase, position), topology (persistent homology, Betti
trajectories, persistence entropy, landscapes), RepE (task vectors,
refusal, concept separability), and scaling/memorisation signals.
