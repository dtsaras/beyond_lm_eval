# Interpreting BLME Results

This guide explains what each BLME metric measures, what values to expect, and what they imply about your model.

> [!TIP]
> Metrics are most informative when **compared across models** (e.g., different checkpoints, scales, or architectures). Absolute values depend on model family, dataset, and sample size.

---

## Geometry Metrics

### SVD Isotropy (`geometry_svd`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `_meta_svd_auc` | 0–1 | Diagnostic: AUC of cumulative explained variance. Lower = more isotropic. |
| `effective_rank` | 1–D | Exponential of singular value entropy. How many dimensions are "active." Higher is generally better. |
| `participation_ratio` | 1–D | (Σλ)²/Σλ². Similar to effective rank but more sensitive to dominant eigenvalues. |
| `avg_cosine_similarity` | −1 to 1 | Mean pairwise cosine between random hidden states. **Near 0 = isotropic** (ideal); near 1 = collapsed/anisotropic. |
| `_meta_cond_number` | ≥1 | Diagnostic: ratio of largest to smallest singular value. |

**Guidance**: Well-trained models typically have `effective_rank` ≫ 1 and `avg_cosine_similarity` < 0.5. A low effective rank with high cosine similarity signals **representation collapse**.

---

### Representation Collapse (`geometry_collapse`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `qdl_entropy_min` | ≥0 | Minimum over blocks of the mean per-passage spectral entropy H = −Σ p log p, p_j = σ_j²/‖X‖_F² (Queipo-de-Llano et al. 2025, Eq. 1; first 64 tokens, uncentred). The minimum ranges over every block, the last pre-norm block included. |
| `_meta_erank_per_layer` | list | Effective rank at each layer. |
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
| `min_adjacent_cka` | 0–1 | Lowest CKA between consecutive blocks: the sharpest single-step change. |
| `early_late_cka`, `min_offdiag_cka` | 0–1 | CKA between the first and last block, and the lowest CKA over all block pairs. |
| `*_debiased` | ≈0–1 | The same summaries with the debiased (unbiased-HSIC) CKA estimator. |

**Guidance**: A low `min_adjacent_cka` marks a block that rewrites the representation; uniformly high CKA across all blocks may indicate redundancy. The full per-pair matrix is not returned.

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
| `relative_change_max` | ≥0 | Largest mean relative block-to-block change ‖b_{l+1} − b_l‖ / ‖b_l‖ over the raw block outputs (pre-final-norm; sinks excluded). |
| `total_angular_path` | ≥0 | Sum over block transitions of the mean angle arccos cos(b_l, b_{l+1}). |
| `e2e_log_norm_growth` | any | Mean log(‖b_{L−1}‖ / ‖b_0‖) from the first to the last block output. Negative when the residual norm shrinks (Gemma's √d-scaled embeddings). |
| `embedding_relative_change` | ≥0 | Mean ‖b_0 − e‖ / ‖e‖: block 0's own update to the embedding e, which the two path keys above do not include. |

**Guidance**: These are representation-path statistics, not Lipschitz constants of the network; the per-layer ratios `_meta_lipschitz_mean` / `_meta_lipschitz_max` are diagnostics. Compare models with the depth- and scale-free keys above.

---

### Matrix Entropy (`geometry_matrix_entropy`)

| Metric | Range | Meaning |
|--------|-------|---------|
| `matrix_entropy` | 0–1 | Matrix entropy H / ln d of each text's last-layer (post-final-norm) token matrix, centred and row-normalised, averaged over texts (Wei et al. 2024, Def. 4.1–4.3). |
| `matrix_entropy_raw` | ≥0 | The same entropy before the 1 / ln d normalisation (nats). The per-layer values are `_meta_layer_matrix_entropies`. |

**Guidance**: In the per-layer profile (`_meta_`), decreasing entropy from early to late layers indicates an **information bottleneck** — the model is compressing input information into more structured representations. This is generally desirable.

---

### Other Geometry Metrics

| Task | Key Metric | What It Tells You |
|------|-----------|-------------------|
| `geometry_rsa` | `rsa_adjacent_min_corrdist`, `rsa_early_late_corrdist` | RSA (correlation-distance RDMs) between consecutive blocks at their least similar step, and between the first and last block. |
| `geometry_hsic` | `min_adjacent_hsic`, `hsic_compression_ratio` (and `*_debiased`) | Normalised HSIC between consecutive blocks at their weakest step, and the last-to-first dependence ratio. |
| `geometry_intrinsic_dim` | `lid_first_layer`, `lid_mid_layer`, `lid_last_layer` | Two-NN intrinsic dimension at the first, middle and last block (`layerwise: true`, the default and the study's setting). Static mode (`layerwise: false`) emits only `intrinsic_dimension` (input-embedding rows, not a headline key). |
| `geometry_prediction_alignment` | `prediction_alignment_mean` | Cosine alignment between final hidden states and the output-projection row for the target next token. |
| `geometry_perplexity` | `bits_per_byte` | Strided next-token NLL in bits per source byte (tokenizer-independent). The perplexities, including the rare/frequent split `_meta_ppl_rare` / `_meta_ppl_freq`, are `_meta_`. |
| `geometry_positional_decay` | `positional_decay_nosink_write_weighted` | Spearman correlation between query–key distance and attention over keys 1..i−1 (self and position 0 excluded), each head-row weighted by the norm of that head's write at the query; flat rows count as 0. More negative = more local attention. |
| `geometry_spectral` | `avg_alpha_official`, `median_alpha_official`, `avg_alpha_hat`, `avg_stable_rank` | WeightWatcher power-law exponent of the language-model weight matrices (vision/audio towers excluded), its scale-weighted form, and the stable rank. |
| `geometry_mahalanobis` | `auroc`, `ood_separation_ratio`, `mahalanobis_sq_per_dim_id`, `mahalanobis_sq_per_dim_ood`, `ood_separation_gap_per_dim` | Single-Gaussian (Ledoit–Wolf) Mahalanobis scores of held-out texts against their own token-shuffled versions; whole articles go to the fit or the held-out half. |
| `geometry_representation_sensitivity` | `S_unit` | Mean squared norm of the gradient of log P(y) with respect to the unit-gain normalised final state. Higher = the output distribution is more locally sensitive to hidden-state perturbations. It does not depend on how a model splits scale between the final-norm gain and the unembedding, and it carries a factor of the width. The raw `||∇_h log P(y|h)||²` and its std are `_meta_representation_sensitivity_raw` and `_meta_representation_sensitivity_raw_std`. |
| `geometry_correlation_dimension` | `correlation_dimension` | Fractal complexity of the representation manifold. Non-integer values indicate self-similar structure. |
| `geometry_categories` | `*_separation`, `*_purity` | Per-category separation and purity scores in embedding space. |
| `geometry_unembedding` | `unembedding_eff_rank`, `unembedding_eff_rank_centered` | Roy-Vetterli effective rank of the unembedding rows the tokenizer can emit (ids below `len(tokenizer)`), raw and mean-centred. The values over every row are `_meta_unembedding_eff_rank_all_rows` and `_meta_unembedding_eff_rank_centered_all_rows`, and `_meta_n_unused_rows` counts the rows no token reaches. Category purity and the embedding alignment are tokenizer-dependent `_meta_` diagnostics. |
| `geometry_isoscore` | `isoscore` | Rudman et al. covariance-isotropy score. Higher = more uniform use of embedding dimensions. |
| `geometry_neural_collapse` | `nc1_cdnv_galanti` | Galanti et al. class-distance-normalised variance of the post-final-norm states, every position labelled by its next token (Wu & Papyan 2024). Lower = more collapsed. |
| `geometry_schatten` | `rankme_last`, `row_normalized_rankme_last`, `row_normalized_matrix_nuclear_norm_last`, `row_normalized_schatten_inf_last` | Per-text spectral summaries of the last layer, on the first 256 tokens of long WikiText articles. The raw MLP-activation Schatten norms are `_meta_schatten_1_mlp` and `_meta_schatten_4_mlp` (they carry the activation scale and d_ff). Schatten-2 is intentionally omitted as content-free after row-L2 normalization. |
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
| `topology_homology` | `{first,mid,last}_layer_mean_persistence_h0_norm`, `*_max_persistence_h0_norm`, `*_mean_persistence_h1_norm`, `*_num_loops_h1` | ≥0 | Persistence lifetimes of connected components and loops divided by the cloud's mean pairwise distance, and the number of loops, at the first, middle and last block. The lifetimes in the representation's own units are `_meta_{first,mid,last}_layer_*`. |
| `topology_persistence_entropy` | `{first,mid,last}_layer_pe_h1` (and `*_norm`) | ≥0 | H1 persistence entropy at the first, middle and last block, raw and normalised by log of the bar count. NaN when a layer has no H1 bars. |
| `topology_betti_curve` | `betti_1_curve_mean` | ≥0 | Mean over layers of β1 at 2 hops on the kNN graph (Naitzat et al.'s released Eirene read). At the study's N it is 0 or near 0 on every model checked; the curves themselves are `betti_*_curve`. |
| `topology_persistence_landscape` | `{first,mid,last}_layer_h1_*_mpd*` | ≥0 | H1 persistence-landscape summaries divided by the cloud's mean pairwise distance to the matching power. The raw-unit summaries and the H0 closed forms are `_meta_`. |

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
| `dynamics_gradient_flow` | `gradient_flow_log_ratio_in_out` | any | log of the next-token-CE gradient norm at the first block's input over that at the final-norm input (first 64 tokens; sinks excluded). Negative = the input-side gradient is smaller; co-varies with residual-norm growth. |
| `dynamics_generation_diversity` | `mean_distinct_1`, `mean_self_bleu`, `sampling_entropy_collapse_delta` | 0–1 / any | Diversity, self-overlap, and entropy drift of sampled completions. |
| `dynamics_sharpness` | `hutchinson_trace_estimate`, `top_eigenvalue_estimate`, `sam_sharpness` | ≥0 | Loss-landscape curvature and SAM-style local sharpness estimates. |

---

## Representation Engineering Metrics

| Task | Key Metric | Range | What It Tells You |
|------|-----------|-------|-------------------|
| `repe_task_vectors` | `consensus_energy_rel`, `pair_direction_cos_mean`, `max_norm_layer_fraction` | 0–1 | How much of the task vectors' energy lies in their consensus direction, how aligned the per-pair directions are, and the depth fraction of the largest-norm layer. |
| `repe_concept_separability` | `lat_test_acc`, `lat_layer_frac` | 0–1 | Held-out accuracy of the LAT concept direction at its selected layer, and that layer's depth fraction. |
| `repe_steering_effectiveness` | `steering_concept_over_random`, `steering_beats_random_frac`, `steering_semantic_effect`, `steering_semantic_success_rate` | ≥0 / 0–1 / any / 0–1 | Median over tested layers of the concept direction's effect over a random direction's, the share of layers where it wins, and the effect on data-defined true/false anchor tokens. `directional_consistency`, `directional_success_rate` and `steering_sign_reversal` summarise projections onto the steering direction and are near 1 by construction; they check that the estimand is directional, not that steering works. The KL success rate is `_meta_steering_success_rate`. |
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
- `relative_change_max` far above the panel → one block rewrites the residual stream
- `ece` > 0.3 → poor calibration
- `logical_violation_rate` > 0.5 → incoherent reasoning
- `ppl_rare` / `ppl_freq` ratio > 10 → poor tail token modeling
