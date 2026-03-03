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
| `skewness_k{N}` | any | Skewness of k-occurrence distribution. **High skewness = hubness problem** (few points are neighbors of many). |
| `gini_k{N}` | 0–1 | Gini coefficient of neighbor counts. Higher = more inequality. |

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
| `geometry_intrinsic_dim` | `intrinsic_dimension` | Global intrinsic dimensionality via MLE estimator. |
| `geometry_consistency` | `cosine_consistency_mean` | How well the embedding space predicts next-token logits (alignment of representation and output spaces). |
| `geometry_perplexity` | `ppl_rare` / `ppl_freq` | Perplexity on rare vs. frequent tokens. Large gaps signal poor tail performance. |
| `geometry_positional_decay` | `mean_positional_decay_correlation` | How attention weight decays with distance. Strong negative (< −0.5) = healthy local structure. |
| `geometry_spectral` | `spectral_norm_*` | Weight matrix spectral norms. Very large norms can cause training instability. |
| `geometry_mahalanobis` | `ood_separation_gap` | Mahalanobis distance gap between in-distribution and OOD data. Larger = better OOD detection. |
| `geometry_information_fisher` | `empirical_fisher_trace` | Trace of empirical Fisher information. Higher = model is more sensitive to input perturbations. |

---

## Interpretability Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `interpretability_logit_lens` | `convergence_layer` | 0–L | Earliest layer where logit lens correctly predicts the final output token. Earlier = faster convergence. |
| `interpretability_attention_entropy` | `mean_entropy` | ≥0 | Average attention entropy. **Higher = more diffuse attention** (attending everywhere); lower = sharper, more focused attention. |
| `interpretability_prediction_entropy` | `mean_predictive_entropy` | ≥0 | Output distribution entropy. Higher = less confident predictions. |
| `interpretability_induction_heads` | `induction_score` | 0–1 | Fraction of identified induction heads. Higher = stronger in-context learning capability. |
| `interpretability_sparsity` | `mean_activation_sparsity` | 0–1 | Fraction of near-zero activations. Higher sparsity = more efficient representation. |
| `interpretability_probing` | `probe_accuracy` | 0–1 | Linear probing accuracy for syntactic features. Higher = more linearly decodable information. |
| `interpretability_attribution` | `mean_attribution_entropy` | ≥0 | Entropy of component attribution scores. Higher = more distributed computation. |
| `interpretability_attention_graph` | `graph_density` | 0–1 | Fraction of significant attention edges. Dense graphs = less structured attention patterns. |

---

## Topology Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `topology_homology` | `betti_0`, `betti_1` | ≥0 | Betti numbers measuring connected components and loops in the representation manifold. |
| `topology_persistence_entropy` | `persistence_entropy` | ≥0 | Shannon entropy of the persistence diagram. **Higher = more complex topology**. |
| `topology_betti_curve` | `betti_curve_auc` | ≥0 | Area under the Betti curve. Summarizes topological complexity across all scales. |

**Guidance**: These metrics characterize the *shape* of the representation space. More complex topology (higher Betti numbers, higher persistence entropy) often correlates with richer learned representations.

---

## Causality Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `causality_tracing` | `avg_indirect_effect` | any | Mean causal effect of hidden states on output. Larger magnitude = stronger causal role. |
| `causality_ablation` | `area_under_degradation_curve` | 0–1 | How much performance degrades when ablating neurons. **Lower AUC = more robust** (less reliance on individual neurons). |
| `causality_attention_knockout` | `mean_kl_divergence` | ≥0 | KL divergence after knocking out attention heads. Higher = that head matters more. |

**Guidance**: Compare `causality_ablation` across models — more robust models degrade gracefully. High `mean_kl_divergence` on knockout identifies critical attention heads.

---

## Consistency Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `consistency_calibration` | `ece` | 0–1 | Expected Calibration Error. **Lower = better calibrated** (predicted confidence matches accuracy). |
| `consistency_paraphrase` | `invariance_score` | 0–1 | How stable representation are across paraphrases. Higher = more semantically consistent. |
| `consistency_logical` | `logical_violation_rate` | 0–1 | Fraction of cases where P(conclusion) > P(premise). **Lower = more logically consistent.** |
| `consistency_contrastive` | `mean_rejection_ratio` | ≥0 | Ratio of P(factual) to P(contradictory). **Higher = better discriminates facts from fiction.** |

---

## Dynamics Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `dynamics_stability` | `neighborhood_overlap` | 0–1 | Fraction of k-nearest neighbors shared between two model checkpoints. Higher = more stable representations during training. |
| `dynamics_interpolation` | `interpolation_smoothness` | ≥0 | How smoothly the model transitions between sequential hidden states. Lower = smoother trajectories. |
| `dynamics_coe` | `chain_divergence` | ≥0 | How much embeddings diverge during multi-step generation. Lower = more stable generation. |

---

## Representation Engineering Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `repe_task_vectors` | `mean_cosine_similarity` | −1 to 1 | Alignment of contrastive task vectors across layers. Higher = more consistent concept encoding. |
| `repe_concept_separability` | `max_auc` | 0–1 | Peak linear separability of concept pairs across layers. Higher = more linearly decodable concepts. |

---

## Common Patterns to Watch For

### 🟢 Healthy Model
- `effective_rank` > 50, `avg_cosine_similarity` < 0.3
- `ece` < 0.1
- Decreasing `matrix_entropy` through layers
- `induction_score` > 0.3

### 🔴 Warning Signs
- `effective_rank` < 10 → representation collapse
- `avg_cosine_similarity` > 0.8 → severe anisotropy
- `lipschitz_max` > 100 → unstable layers
- `ece` > 0.3 → poor calibration
- `logical_violation_rate` > 0.5 → incoherent reasoning
- `ppl_rare` / `ppl_freq` ratio > 10 → poor tail token modeling
