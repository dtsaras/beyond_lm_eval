# Interpreting BLME Results

This guide explains what each BLME metric measures, what values to expect, and what they imply about your model.

> [!TIP]
> Metrics are most informative when **compared across models** (e.g., different checkpoints, scales, or architectures). Absolute values depend on model family, dataset, and sample size.

---

## Geometry Metrics

### SVD Isotropy (`geometry_svd`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `svd_auc` | 0–1 | AUC of cumulative explained variance. **Lower = more isotropic** (richer representation space). |
| `effective_rank` | 1–D | Exponential of singular value entropy. How many dimensions are "active." Higher is generally better. |
| `participation_ratio` | 1–D | (Σλ)²/Σλ². Similar to effective rank but more sensitive to dominant eigenvalues. |
| `avg_cosine_similarity` | −1 to 1 | Mean pairwise cosine between random hidden states. **Near 0 = isotropic** (ideal); near 1 = collapsed/anisotropic. |
| `cond_number` | ≥1 | Ratio of largest to smallest singular value. Very large values (>10⁴) indicate near-degenerate geometry. |

**Guidance**: Well-trained models typically have `effective_rank` ≫ 1 and `avg_cosine_similarity` < 0.5. A low effective rank with high cosine similarity signals **representation collapse**.

---

### Representation Collapse (`geometry_collapse`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `erank_per_layer` | list | Effective rank at each layer. |
| `max_erank` / `min_erank` | 1–D | Highest and lowest effective rank across layers. |
| `collapse_ratio` | 0–1 | Min/max effective-rank ratio. Lower = more collapse. |

**Guidance**: If `min_erank` drops close to 1 in later layers, those layers may be collapsing representations into a low-dimensional manifold. Compare early vs. late layers.

---

### Local Intrinsic Dimensionality (`geometry_lid`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `lid_mean` | >0 | Average LID across sampled neighborhoods. Estimates local manifold dimension. |
| `lid_std` | ≥0 | Variance in LID. High std means the representation space is geometrically heterogeneous. |

**Guidance**: Models with higher LID are storing information in more dimensions locally. Very low LID (< 5) in higher layers may indicate over-compression.

---

### CKA Similarity (`geometry_cka`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `avg_adjacent_cka` | 0–1 | Mean CKA between consecutive layers. **Higher = layers are more similar.** |
| `cka_matrix` | 0–1 matrix | Full pairwise CKA between all layers. |

**Guidance**: A block-diagonal CKA matrix reveals that the model has distinct "phases" of computation. Uniformly high CKA across all layers may indicate redundancy.

---

### Hubness (`geometry_hubness`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `hubness_k{N}_skew` | any | Skewness of k-occurrence distribution. **High skewness = hubness problem** (few points are neighbors of many). |
| `hubness_k{N}_gini` | 0–1 | Gini coefficient of neighbor counts. Higher = more inequality. |
| `hubness_k{N}_top1pct` | 0–1 | Fraction of neighbor mass in the top 1% of tokens. Higher = more hub concentration. |
| `hubness_k{N}_max` | ≥0 | Max neighbor count for any token. |

**Guidance**: Hubness (skewness > 2) is a known pathology of high-dimensional spaces. It corrupts nearest-neighbor-based downstream tasks.

---

### Lipschitz Constants (`geometry_lipschitz`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `lipschitz_mean` | >0 | Average inter-layer Lipschitz constant. How much the representation changes per layer. |
| `lipschitz_max` | >0 | Worst-case expansion factor across all layers. |

**Guidance**: Very large Lipschitz constants (>10) suggest unstable layers where small input perturbations cause large representation shifts.

---

### Matrix Entropy (`geometry_matrix_entropy`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `mean_matrix_entropy` | ≥0 | Average von Neumann entropy of per-layer covariance matrices. |
| `layer_matrix_entropies` | dict | Entropy per layer. |

**Guidance**: Decreasing entropy from early to late layers indicates an **information bottleneck** — the model is compressing input information into more structured representations. This is generally desirable.

---

### Other Geometry Metrics

| Task | Key Metric | What It Tells You |
|------|-----------|-------------------|
| `geometry_rsa` | `rsa_adjacent_mean` | How similar the representational geometry is between consecutive layers (higher = more stable). |
| `geometry_hsic` | `avg_adjacent_hsic` | HSIC dependence between adjacent layers. Higher = more linearly dependent layer transitions. |
| `geometry_intrinsic_dim` | `intrinsic_dimension` | Global intrinsic dimensionality via Two-NN. (Layer-wise mode yields `lid_layer_*` keys.) |
| `geometry_prediction_alignment` | `prediction_alignment_mean` | Cosine alignment between final hidden states and the output-projection row for the target next token. |
| `geometry_perplexity` | `ppl_rare` / `ppl_freq` | Perplexity on rare vs. frequent tokens. Large gaps signal poor tail performance. |
| `geometry_positional_decay` | `mean_positional_decay_correlation` | How attention weight decays with distance. Strong negative (< −0.5) = healthy local structure. |
| `geometry_spectral` | `avg_alpha`, `avg_stable_rank` | Power-law exponent and stable rank of weight matrices. Extreme values indicate brittle spectra. |
| `geometry_mahalanobis` | `ood_separation_gap` | Mahalanobis distance gap between in-distribution and OOD data. Larger = better OOD detection. |
| `geometry_representation_sensitivity` | `representation_sensitivity` | Closed-form sensitivity proxy `||∇_h log P(y|h)||²`. Higher = the output distribution is more locally sensitive to hidden-state perturbations. |
| `geometry_correlation_dimension` | `correlation_dimension` | Fractal complexity of the representation manifold. Non-integer values indicate self-similar structure. |
| `geometry_categories` | `*_separation`, `*_purity` | Per-category separation and purity scores in embedding space. |
| `geometry_unembedding` | `unembedding_eff_rank`, `unembedding_purity_mean` | Structure and tokenizer-dependent category purity of the unembedding space (plus tied-weight flag). |
| `geometry_isoscore` | `isoscore` | Rudman et al. covariance-isotropy score. Higher = more uniform use of embedding dimensions. |
| `geometry_neural_collapse` | `nc1_within_class_collapse`, `nc2_etf_cosine_deviation_proxy` | Topic-label neural-collapse proxies; NC2 is an ETF cosine-deviation proxy, not full NC2/NC3. |
| `geometry_schatten` | `row_normalized_schatten_1_last`, `row_normalized_matrix_nuclear_norm_last` | Row-normalized spectral geometry summaries. Schatten-2 is intentionally omitted as content-free after row-L2 normalization. |
| `geometry_mp_bulk_deviation` | `mp_outlier_frac_*`, `mp_spike_energy_*` | Deviation from a Marchenko-Pastur random-matrix null; larger spike energy means more structured directions. |

---

## Interpretability Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `interpretability_logit_lens` | `layer{i}_acc` | 0–1 | Per-layer agreement with final-token prediction (e.g., `layer0_acc`, `layer1_acc`). |
| `interpretability_attention_entropy` | `avg_entropy_total` | ≥0 | Average attention entropy. **Higher = more diffuse attention**; lower = sharper focus. |
| `interpretability_prediction_entropy` | `mean_entropy` | ≥0 | Output distribution entropy. Higher = less confident predictions. |
| `interpretability_induction_heads` | `avg_induction_score` | 0–1 | Average induction-head strength across layers/heads. |
| `interpretability_sparsity` | `global_mean_l0` | 0–1 | Fraction of active neurons. Lower = sparser activation. |
| `interpretability_probing` | `max_probing_accuracy` | 0–1 | Best linear probe accuracy across layers. |
| `interpretability_attribution` | `mean_gradient_x_activation`, `attribution_gini` | ≥0 / 0–1 | Input-embedding gradient × activation attribution magnitude and concentration. |
| `interpretability_attention_graph` | `mean_sink_pagerank` | 0–1 | Degree to which attention collapses onto a sink token. |
| `interpretability_superposition` | `mean_polysemanticity_index` | 0–1 | Bimodality coefficient of neuron activations. Higher = more superposition (neurons encode multiple features). |
| `interpretability_waa` | `mean_waa_alignment` | 0–1 | Alignment between weight SVD vectors and activation PCA vectors. Higher = more efficient capacity utilization. |
| `interpretability_attention_effective_rank` | `mean_attention_output_effective_rank_entropy` | ≥0 | Effective-rank proxy over attention output projections. Higher = broader subspace usage; not a direct concept-level polysemanticity score. |
| `interpretability_sae_features` | `mean_active_features_l0` | ≥0 | Mean number of active SAE features per token. Lower = sparser, more disentangled representations. |
| `interpretability_activation_sinks` | `sink_epsilon_fraction`, `massive_activation_fraction`, `valley_depth` | 0–1 / ≥0 | Sinkε, massive-activation, and compression-valley diagnostics. |
| `interpretability_attention_rank` | `mean_effective_rank` | ≥0 | Effective rank of attention maps; lower ranks can indicate attention-rank collapse. |
| `interpretability_head_roles` | `mean_previous_token_score`, `frac_duplicate_token_heads` | 0–1 | Fraction and strength of simple previous-token / duplicate-token head roles. |

---

## Topology Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `topology_homology` | `layer_*_mean_persistence_h0` | ≥0 | Per-layer persistence lifespans for connected components and loops. |
| `topology_persistence_entropy` | `layer_*_pe_h0`, `pe_simplification_ratio` | ≥0 | Per-layer persistence entropy and simplification across depth. |
| `topology_betti_curve` | `betti_0_curve`, `simplification_ratio` | ≥0 | Betti trajectory across layers and its simplification ratio. |
| `topology_persistence_landscape` | `layer_*_h0_mean_landscape_integral` | ≥0 | Persistence-landscape functional summaries of H0/H1 diagrams. |

**Guidance**: These metrics characterize the *shape* of the representation space. More complex topology (higher Betti numbers, higher persistence entropy) often correlates with richer learned representations.

---

## Causality Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `causality_tracing` | `max_aie` | 0–1 | Strongest average indirect effect from layer-wise restoration. |
| `causality_ablation` | `area_under_degradation_curve` | ≥0 | How much loss increases when ablating residual-stream feature dimensions. Larger AUC = more brittle. |
| `causality_attention_knockout` | `head_impact_gini_coefficient` | 0–1 | Concentration of head importance. Higher = few heads dominate. |
| `causality_circuit_quality` | `circuit_quality_score` | 0–1 | Harmonic mean of circuit faithfulness and minimality. Higher = compact, faithful circuit. |
| `causality_edge_attribution` | `attribution_gini`, `top1_layer_share` | 0–1 | Concentration of first-order attribution mass across layers; a proxy, not recovered circuit edges. |
| `causality_knowledge_neurons` | `mean_saliency_gini`, `saliency_layer_mean` | 0–1 / layer | Concentration and layer location of MLP saliency for target logits; not validated fact editing. |

**Guidance**: Compare `causality_ablation` across models — more robust models degrade gracefully. Large `max_knockout_impact` or high `head_impact_gini_coefficient` indicates a few critical heads. A high `circuit_quality_score` means the model's behavior can be reproduced by a small subset of its layers.

---

## Consistency Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `consistency_calibration` | `ece` | 0–1 | Expected Calibration Error. **Lower = better calibrated** (predicted confidence matches accuracy). |
| `consistency_paraphrase` | `representation_distance_ratio_l2` | ≥0 | Ratio of paraphrase vs unrelated representation distance. **Lower = better semantic invariance.** |
| `consistency_logical` | `premise_decreases_conclusion_likelihood_rate` | 0–1 | Fraction of cases where the premise lowers conclusion likelihood. **Lower = more internally consistent.** |
| `consistency_contrastive` | `mean_rejection_ratio` | ≥0 | Ratio of P(false/exclusive) to P(true/factual). **Lower = better rejection of false alternatives.** |
| `consistency_contamination` | `min_k_score` | log-prob | Shi et al. Min-k% mean log probability. Higher (less negative) can indicate memorization-like text. |
| `consistency_knowledge_capacity` | `paraphrase_probability_ratio` | 0–1+ | Ratio of rephrased to exact completion probability; a memorization/generalization proxy, not capacity scaling. |
| `consistency_format_robustness` | `format_nll_sensitivity` | ≥0 | NLL spread across prompt formats. Lower = less format sensitivity. |
| `consistency_position_sensitivity` | `lost_in_middle_nll_depth` | ≥0 | NLL penalty for middle-position facts vs beginning/end. Higher = stronger lost-in-the-middle effect. |
| `consistency_self_consistency` | `sampling_stability_mean_first_token_agreement` | 0–1 | First-token agreement across sampled completions; proxy, not CoT majority-vote accuracy. |
| `consistency_icl_slope` | `icl_slope`, `icl_gain` | any | NLL change as in-context demonstrations are added. |
| `consistency_membership_inference` | `separability_auroc` | 0–1 | Loss-based separability proxy; default data are not verified training members. |

---

## Dynamics Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `dynamics_stability` | `stability_mean` | 0–1 | Jaccard overlap of k-NN neighborhoods between model embeddings. Higher = more stable. |
| `dynamics_interpolation` | `convexity_gap` | any | Entropy bump at the midpoint of interpolation. Higher gap = less convex latent space. |
| `dynamics_coe` | `coe_r`, `coe_c` | any / ≥0 | Wang et al. CoE magnitude/angle chain scores on the prompt-side hidden-state chain. |
| `dynamics_gradient_flow` | `gradient_flow_entropy`, `gradient_flow_slope` | ≥0 / any | Distribution and depth trend of per-layer gradient norms under next-token CE. |
| `dynamics_generation_diversity` | `mean_distinct_1`, `mean_self_bleu`, `entropy_collapse_delta` | 0–1 / any | Diversity, self-overlap, and entropy drift of sampled completions. |
| `dynamics_sharpness` | `hutchinson_trace_estimate`, `top_eigenvalue_estimate`, `sam_sharpness` | ≥0 | Loss-landscape curvature and SAM-style local sharpness estimates. |

---

## Representation Engineering Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `repe_task_vectors` | `mean_vector_norm` | ≥0 | Average magnitude of contrastive task vectors across layers. Larger = stronger task direction. |
| `repe_concept_separability` | `max_auc` | 0–1 | Peak linear separability of concept pairs across layers. Higher = more linearly decodable concepts. |
| `repe_steering_effectiveness` | `steering_success_rate` | 0–1 | Fraction of layers where steering vectors produce a measurable output shift (KL > threshold). Higher = model is more steerable. |
| `repe_refusal_direction` | `separability_auc`, `mean_projection_gap` | 0–1 / any | Linear separability and projection gap between harmful/harmless prompts along the refusal direction. |

---

## Common Patterns to Watch For

### 🟢 Healthy Model
- `effective_rank` > 50, `avg_cosine_similarity` < 0.3
- `ece` < 0.1
- Decreasing `matrix_entropy` through layers
- `avg_induction_score` > 0.3

### 🔴 Warning Signs
- `effective_rank` < 10 → representation collapse
- `avg_cosine_similarity` > 0.8 → severe anisotropy
- `lipschitz_max` > 100 → unstable layers
- `ece` > 0.3 → poor calibration
- `logical_violation_rate` > 0.5 → incoherent reasoning
- `ppl_rare` / `ppl_freq` ratio > 10 → poor tail token modeling
