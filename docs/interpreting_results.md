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
| `geometry_mutual_info` | `avg_adjacent_mi` | Mutual information between layers. Higher = more information preserved across layers. |
| `geometry_intrinsic_dim` | `intrinsic_dimension` | Global intrinsic dimensionality via Two-NN. (Layer-wise mode yields `lid_layer_*` keys.) |
| `geometry_consistency` | `cosine_consistency_mean` | How well the embedding space predicts next-token logits (alignment of representation and output spaces). |
| `geometry_perplexity` | `ppl_rare` / `ppl_freq` | Perplexity on rare vs. frequent tokens. Large gaps signal poor tail performance. |
| `geometry_positional_decay` | `mean_positional_decay_correlation` | How attention weight decays with distance. Strong negative (< −0.5) = healthy local structure. |
| `geometry_spectral` | `avg_alpha`, `avg_stable_rank` | Power-law exponent and stable rank of weight matrices. Extreme values indicate brittle spectra. |
| `geometry_mahalanobis` | `ood_separation_gap` | Mahalanobis distance gap between in-distribution and OOD data. Larger = better OOD detection. |
| `geometry_information_fisher` | `empirical_fisher_trace` | Trace of empirical Fisher information. Higher = model is more sensitive to input perturbations. |
| `geometry_correlation_dimension` | `correlation_dimension` | Fractal complexity of the representation manifold. Non-integer values indicate self-similar structure. |
| `geometry_categories` | `*_separation`, `*_purity` | Per-category separation and purity scores in embedding space. |
| `geometry_unembedding` | `unembedding_eff_rank`, `unembedding_purity_mean` | Structure and category purity of the unembedding space (plus tied-weight flag). |

---

## Interpretability Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `interpretability_logit_lens` | `layer{i}_acc` | 0–1 | Per-layer agreement with final-token prediction (e.g., `layer0_acc`, `layer1_acc`). |
| `interpretability_attention_entropy` | `avg_entropy_total` | ≥0 | Average attention entropy. **Higher = more diffuse attention**; lower = sharper focus. |
| `interpretability_prediction_entropy` | `mean_entropy` | ≥0 | Output distribution entropy. Higher = less confident predictions. |
| `interpretability_induction_heads` | `avg_induction_score` | 0–1 | Average induction-head strength across layers/heads. |
| `interpretability_sparsity` | `global_mean_l0` | 0–1 | Fraction of active neurons. Higher sparsity = more selective activation. |
| `interpretability_probing` | `max_probing_accuracy` | 0–1 | Best linear probe accuracy across layers. |
| `interpretability_attribution` | `component_coherence_mean` | −1 to 1 | Coherence of layer update directions in token space. Higher = more consistent attribution. |
| `interpretability_attention_graph` | `mean_sink_pagerank` | 0–1 | Degree to which attention collapses onto a sink token. |
| `interpretability_superposition` | `mean_polysemanticity_index` | 0–1 | Bimodality coefficient of neuron activations. Higher = more superposition (neurons encode multiple features). |
| `interpretability_waa` | `mean_waa_alignment` | 0–1 | Alignment between weight SVD vectors and activation PCA vectors. Higher = more efficient capacity utilization. |
| `interpretability_attention_polysemanticity` | `mean_attention_svd_entropy` | ≥0 | SVD entropy of attention head outputs. Higher = more polysemantic (superposed) attention heads. |
| `interpretability_sae_features` | `mean_active_features_l0` | ≥0 | Mean number of active SAE features per token. Lower = sparser, more disentangled representations. |

---

## Topology Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `topology_homology` | `layer_*_mean_persistance_h0` | ≥0 | Per-layer persistence lifespans for connected components and loops. |
| `topology_persistence_entropy` | `layer_*_pe_h0`, `pe_simplification_ratio` | ≥0 | Per-layer persistence entropy and simplification across depth. |
| `topology_betti_curve` | `betti_0_curve`, `simplification_ratio` | ≥0 | Betti trajectory across layers and its simplification ratio. |

**Guidance**: These metrics characterize the *shape* of the representation space. More complex topology (higher Betti numbers, higher persistence entropy) often correlates with richer learned representations.

---

## Causality Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `causality_tracing` | `max_aie` | 0–1 | Strongest average indirect effect from layer-wise restoration. |
| `causality_ablation` | `area_under_degradation_curve` | ≥0 | How much loss increases when ablating neurons. Larger AUC = more brittle. |
| `causality_attention_knockout` | `head_impact_gini_coefficient` | 0–1 | Concentration of head importance. Higher = few heads dominate. |
| `causality_circuit_quality` | `circuit_quality_score` | 0–1 | Harmonic mean of circuit faithfulness and minimality. Higher = compact, faithful circuit. |

**Guidance**: Compare `causality_ablation` across models — more robust models degrade gracefully. Large `max_knockout_impact` or high `head_impact_gini_coefficient` indicates a few critical heads. A high `circuit_quality_score` means the model's behavior can be reproduced by a small subset of its layers.

---

## Consistency Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `consistency_calibration` | `ece` | 0–1 | Expected Calibration Error. **Lower = better calibrated** (predicted confidence matches accuracy). |
| `consistency_paraphrase` | `isometry_ratio_l2` | ≥0 | Ratio of paraphrase vs unrelated distance. **Lower = better semantic invariance.** |
| `consistency_logical` | `logical_violation_rate` | 0–1 | Fraction of cases where P(conclusion) > P(premise). **Lower = more logically consistent.** |
| `consistency_contrastive` | `mean_rejection_ratio` | ≥0 | Ratio of P(factual) to P(contradictory). **Higher = better discriminates facts from fiction.** |
| `consistency_contamination` | `contamination_score` | 0–1 | Min-k% probability ratio. **Closer to 1.0 = more likely memorized** (uniform token probabilities). |
| `consistency_knowledge_capacity` | `generalization_ratio` | 0–1+ | Ratio of rephrased to exact completion log probs. **Closer to 1.0 = better generalization** vs memorization. |

---

## Dynamics Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `dynamics_stability` | `stability_mean` | 0–1 | Jaccard overlap of k-NN neighborhoods between model embeddings. Higher = more stable. |
| `dynamics_interpolation` | `convexity_gap` | any | Entropy bump at the midpoint of interpolation. Higher gap = less convex latent space. |
| `dynamics_coe` | `mean_magnitude_change` | ≥0 | Average drift between successive generation steps. Lower = more stable trajectories. |

---

## Representation Engineering Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `repe_task_vectors` | `mean_vector_norm` | ≥0 | Average magnitude of contrastive task vectors across layers. Larger = stronger task direction. |
| `repe_concept_separability` | `max_auc` | 0–1 | Peak linear separability of concept pairs across layers. Higher = more linearly decodable concepts. |
| `repe_steering_effectiveness` | `steering_success_rate` | 0–1 | Fraction of layers where steering vectors produce a measurable output shift (KL > threshold). Higher = model is more steerable. |

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
