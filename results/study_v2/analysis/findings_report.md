# BLME Study — Findings Report

*Generated from results/study_v2/aggregated.csv (32 models, 812 columns) and analysis/*.csv outputs.*

*Low-power filter: features with n < 20 excluded from top correlates to prevent spurious ±1.0 ρ values.*

## Q1. Benchmark Y distribution

### Model-size range
- 32 models, parameter range **70M – 31.0B** (median 2000M).
- Log-range spans 2.6 decades.

### Composite benchmark distribution
- n=32 models with finite composite Y.
- Range: **0.065 – 0.990**, median 0.422.

**Top 5 by composite score:**
- qwen3.5-27b-it         (qwen3.5, 27000M) → 0.990
- gemma4-31b             (gemma4, 31000M) → 0.954
- qwen3.5-4b             (qwen3.5, 4000M) → 0.797
- qwen3.5-9b             (qwen3.5, 9000M) → 0.773
- gemma4-e4b             (gemma4, 4500M) → 0.758

**Bottom 5 by composite score:**
- pythia-1b              (pythia, 1011M) → 0.092
- gpt2-small             (gpt2, 124M) → 0.092
- pythia-160m            (pythia, 162M) → 0.086
- pythia-70m             (pythia, 70M) → 0.085
- olmo-1b                (olmo, 1180M) → 0.065

### Individual benchmark Spearman vs. log(N_params)
- hellaswag_acc                    ρ(log N, Y) = +0.847  (n=32)
- piqa_acc                         ρ(log N, Y) = +0.825  (n=32)
- arc_easy_acc                     ρ(log N, Y) = +0.755  (n=32)
- arc_challenge_acc                ρ(log N, Y) = +0.788  (n=32)
- winogrande_acc                   ρ(log N, Y) = +0.789  (n=32)
- mmlu_acc                         ρ(log N, Y) = +0.698  (n=32)
- composite_benchmark              ρ(log N, Y) = +0.785  (n=32)


## Q2/Q3. Size baseline & top predictors beyond scale

### Size-only baseline on composite (n=32)
- Linear R² = **0.498**
- Spearman ρ(log N, composite) = **+0.785** (p = 1.0e-07)

### Univariate Spearman with composite (n≥20)
- 731 features survive the n≥20 power filter (out of 731 tested).
- 356 / 731 of those are FDR-significant at q<0.05.

**Top 20 by |ρ| (univariate, n≥20):**
- `repe_task_vectors.layer_task_vector_cosine_sim.min     ` -0.92*** (n=32)
- `repe_task_vectors.layer_task_vector_cosine_sim.std     ` +0.91*** (n=32)
- `geometry_perplexity.ppl_freq                           ` -0.89*** (n=31)
- `causality_ablation.loss_ablate_1pct                    ` -0.87*** (n=32)
- `repe_task_vectors.layer_task_vector_cosine_sim.mean    ` -0.86*** (n=32)
- `repe_task_vectors.layer_task_vector_cosine_sim.q75     ` -0.86*** (n=32)
- `causality_ablation.baseline_loss                       ` -0.85*** (n=32)
- `interpretability_prediction_entropy.median_entropy     ` -0.85*** (n=32)
- `interpretability_prediction_entropy.mean_top5_prob     ` +0.85*** (n=32)
- `causality_attention_knockout.baseline_loss             ` -0.84*** (n=32)
- `repe_task_vectors.layer_task_vector_cosine_sim.q50     ` -0.84*** (n=32)
- `dynamics_sharpness.baseline_loss                       ` -0.84*** (n=32)
- `repe_refusal_direction.direction_norm                  ` +0.83*** (n=30)
- `repe_refusal_direction.mean_projection_gap             ` +0.83*** (n=30)
- `geometry_perplexity.ppl_overall                        ` -0.83*** (n=31)
- `geometry_perplexity.mean_nll_nats                      ` -0.83*** (n=31)
- `geometry_schatten.schatten_4_per_layer.q75             ` -0.83*** (n=32)
- `geometry_intrinsic_dim.sample_size                     ` +0.83*** (n=32)
- `interpretability_prediction_entropy.mean_entropy       ` -0.83*** (n=32)
- `topology_homology.mean_persistence_h0.slope            ` +0.82*** (n=32)

### Partial correlates with composite, controlling for log(N_params) (n≥20)
- 731 features survive the n≥20 power filter.
- **285 / 731** remain FDR-significant at q<0.05 after partialling out log N_params — features carrying signal BEYOND model scale.

**Top 25 features ranked by |partial ρ| (n≥20):**
- `repe_task_vectors.layer_task_vector_cosine_sim.min     ` partial ρ=-0.82*** (n=32) [repe-tv]
- `repe_task_vectors.layer_task_vector_cosine_sim.std     ` partial ρ=+0.82*** (n=32) [repe-tv]
- `geometry_contextualization.per_layer.n_words_tracked.q25` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_contextualization.per_layer.n_words_tracked.max` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_contextualization.per_layer.n_words_tracked.q75` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_contextualization.per_layer.n_words_tracked.q50` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_contextualization.per_layer.n_words_tracked.mean` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_contextualization.per_layer.n_words_tracked.min` partial ρ=-0.82*** (n=32) [geom-context]
- `geometry_lid.lid_min                                   ` partial ρ=+0.80*** (n=32) [geom-lid]
- `interpretability_waa.layer_waa_alignments.mean         ` partial ρ=-0.78*** (n=32) [interp-waa]
- `repe_task_vectors.layer_task_vector_cosine_sim.mean    ` partial ρ=-0.77*** (n=32) [repe-tv]
- `causality_ablation.loss_ablate_5pct                    ` partial ρ=-0.77*** (n=32) [other]
- `geometry_tokenizer_efficiency.vocab_size               ` partial ρ=+0.77*** (n=32) [geom-token]
- `repe_task_vectors.layer_task_vector_cosine_sim.q50     ` partial ρ=-0.77*** (n=32) [repe-tv]
- `geometry_hubness.hubness_k10_gini                      ` partial ρ=+0.74*** (n=32) [geom-hubness]
- `geometry_cka.min_offdiag_cka                           ` partial ρ=+0.74*** (n=32) [geom-cka]
- `geometry_schatten.matrix_nuclear_norm_per_layer.q50    ` partial ρ=+0.74*** (n=32) [other]
- `geometry_cka.std_offdiag_cka                           ` partial ρ=-0.74*** (n=32) [geom-cka]
- `repe_task_vectors.layer_task_vector_cosine_sim.q75     ` partial ρ=-0.73*** (n=32) [repe-tv]
- `causality_ablation.loss_ablate_1pct                    ` partial ρ=-0.73*** (n=32) [other]
- `interpretability_waa.mean_waa_alignment                ` partial ρ=-0.71*** (n=31) [interp-waa]
- `geometry_tokenizer_efficiency.compression_ratio        ` partial ρ=+0.71*** (n=32) [geom-token]
- `geometry_tokenizer_efficiency.fertility                ` partial ρ=+0.71*** (n=32) [geom-token]
- `geometry_tokenizer_efficiency.total_tokens             ` partial ρ=+0.71*** (n=32) [geom-token]
- `geometry_intrinsic_dim.sample_size                     ` partial ρ=+0.71*** (n=32) [geom-idim]

**Low-power filter removed** 0 features at n<20, of which 0 had spurious |ρ|>0.9 (likely chance inflation).

### LASSO multivariate selection
- Selected **26** features out of ~920 candidates.
- Training R² is overfit (n=32 × p≈900); see console output for held-out LOO/LOFO R².

**Top 20 LASSO coefficients (signed):**
- ↑ `repe_refusal_direction.direction_norm                  ` β=+0.0935  [repe-rd]
- ↑ `repe_task_vectors.layer_task_vector_norms.q50          ` β=+0.0659  [repe-tv]
- ↓ `geometry_cka.std_offdiag_cka                           ` β=-0.0559  [geom-cka]
- ↓ `repe_task_vectors.layer_task_vector_cosine_sim.q25     ` β=-0.0370  [repe-tv]
- ↓ `repe_task_vectors.layer_task_vector_cosine_sim.min     ` β=-0.0261  [repe-tv]
- ↓ `interpretability_logit_lens.entropy.min                ` β=-0.0227  [interp-ll]
- ↑ `interpretability_sparsity.layer_kurtosis.q25           ` β=+0.0203  [interp-sparse]
- ↑ `geometry_weight_norms.stable_rank_per_layer.q25        ` β=+0.0203  [geom-weight]
- ↑ `geometry_hsic.adjacent_hsic.min                        ` β=+0.0194  [geom-hsic]
- ↓ `causality_ablation.degradation_1pct                    ` β=-0.0174  [other]
- ↓ `geometry_hsic.input_to_layer_hsic.std                  ` β=-0.0127  [geom-hsic]
- ↓ `interpretability_attention_entropy.max_normalized_entropy_head` β=-0.0124  [interp-attH]
- ↓ `topology_betti_curve.betti_0_curve.min                 ` β=-0.0124  [other]
- ↓ `topology_homology.num_loops_h1.slope                   ` β=-0.0096  [other]
- ↑ `topology_betti_curve.betti_1_curve.q75                 ` β=+0.0088  [other]
- ↓ `interpretability_waa.layer_waa_alignments.q25          ` β=-0.0084  [interp-waa]
- ↓ `repe_task_vectors.layer_task_vector_cosine_sim.max     ` β=-0.0081  [repe-tv]
- ↓ `geometry_hsic.adjacent_hsic.std                        ` β=-0.0057  [geom-hsic]
- ↓ `interpretability_attention_entropy.avg_entropy_per_layer.max` β=-0.0051  [interp-attH]
- ↑ `causality_knowledge_neurons.localization_layer_mean    ` β=+0.0043  [caus-kneu]


## Q4. Category-level signal

### FDR-significant features per BLME major category
(partial Spearman with composite, controlling for log N_params, n≥20)

| Category | n features | FDR-sig | sig rate | max \|partial ρ\| | best feature |
|---|---:|---:|---:|---:|---|
| dynamics | 64 | 38 | 59.4% | 0.64 | `dynamics_coe.per_sample_coe_c.q75` |
| geometry | 306 | 137 | 44.8% | 0.82 | `geometry_contextualization.per_layer.n_words_tracked.mean` |
| interpretability | 173 | 66 | 38.2% | 0.78 | `interpretability_waa.layer_waa_alignments.mean` |
| repe | 30 | 11 | 36.7% | 0.82 | `repe_task_vectors.layer_task_vector_cosine_sim.min` |
| consistency | 26 | 7 | 26.9% | 0.62 | `consistency_icl_slope.mean_nll_0shot` |
| other | 59 | 15 | 25.4% | 0.69 | `topology_homology.max_persistence_h0.slope` |
| causality | 73 | 11 | 15.1% | 0.77 | `causality_ablation.loss_ablate_5pct` |


## Q5. EDG validation

### Effective Dimensionality Gradient (EDG) — novel metric
- **causality_knowledge_neurons.mean_attribution_gini** (n=32): ρ(.,Y)=+0.70, ρ(.,log N)=+0.63, partial ρ=+0.41
- **causality_knowledge_neurons.mean_top1_share** (n=32): ρ(.,Y)=+0.14, ρ(.,log N)=-0.14, partial ρ=+0.42*
- **causality_knowledge_neurons.mean_top1pct_share** (n=32): ρ(.,Y)=+0.68, ρ(.,log N)=+0.64, partial ρ=+0.37
- **causality_knowledge_neurons.localization_layer_mean** (n=32): ρ(.,Y)=+0.53, ρ(.,log N)=+0.30, partial ρ=+0.49*
- **causality_knowledge_neurons.attribution_layer_entropy** (n=32): ρ(.,Y)=+0.63, ρ(.,log N)=+0.75, partial ρ=+0.09
- **causality_knowledge_neurons.n_layers** (n=32): ρ(.,Y)=+0.65, ρ(.,log N)=+0.75, partial ρ=+0.15
- **interpretability_attention_graph.mean_edge_gini** (n=32): ρ(.,Y)=-0.35, ρ(.,log N)=-0.15, partial ρ=-0.39
- **causality_edge_attribution.attribution_gini** (n=32): ρ(.,Y)=+0.31, ρ(.,log N)=+0.32, partial ρ=+0.10
- **causality_edge_attribution.top1_layer_share** (n=32): ρ(.,Y)=-0.59, ρ(.,log N)=-0.56, partial ρ=-0.29
- **causality_edge_attribution.peak_attribution_layer** (n=32): ρ(.,Y)=-0.11, ρ(.,log N)=+0.07, partial ρ=-0.27
- **causality_edge_attribution.attribution_entropy** (n=32): ρ(.,Y)=+0.70, ρ(.,log N)=+0.78, partial ρ=+0.22
- **causality_edge_attribution.mean_layer_attribution_profile.mean** (n=32): ρ(.,Y)=-0.64, ρ(.,log N)=-0.76, partial ρ=-0.12
- **causality_edge_attribution.mean_layer_attribution_profile.std** (n=32): ρ(.,Y)=-0.52, ρ(.,log N)=-0.40, partial ρ=-0.36
- **causality_edge_attribution.mean_layer_attribution_profile.min** (n=32): ρ(.,Y)=-0.28, ρ(.,log N)=-0.53, partial ρ=+0.25
- **causality_edge_attribution.mean_layer_attribution_profile.max** (n=32): ρ(.,Y)=-0.60, ρ(.,log N)=-0.56, partial ρ=-0.31
- **causality_edge_attribution.mean_layer_attribution_profile.slope** (n=32): ρ(.,Y)=-0.46, ρ(.,log N)=-0.33, partial ρ=-0.34
- **causality_edge_attribution.mean_layer_attribution_profile.q25** (n=32): ρ(.,Y)=-0.47, ρ(.,log N)=-0.47, partial ρ=-0.18
- **causality_edge_attribution.mean_layer_attribution_profile.q50** (n=32): ρ(.,Y)=-0.66, ρ(.,log N)=-0.65, partial ρ=-0.31
- **causality_edge_attribution.mean_layer_attribution_profile.q75** (n=32): ρ(.,Y)=-0.71, ρ(.,log N)=-0.79, partial ρ=-0.23
- **edg** (n=32): ρ(.,Y)=+0.11, ρ(.,log N)=+0.15, partial ρ=-0.01
- **edg_early** (n=31): ρ(.,Y)=+0.64, ρ(.,log N)=+0.67, partial ρ=+0.27
- **edg_late** (n=30): ρ(.,Y)=-0.48, ρ(.,log N)=-0.29, partial ρ=-0.45*
- **benchmark_mmlu_clinical_knowledge_acc** (n=32): ρ(.,Y)=+0.96, ρ(.,log N)=+0.69, partial ρ=--

**Within Pythia (n=8):**
- edg: ρ(log N, edg)=-0.33  ρ(edg, composite)=-0.05
- edg_early: ρ(log N, edg_early)=+nan  ρ(edg_early, composite)=+nan
- edg_late: ρ(log N, edg_late)=+nan  ρ(edg_late, composite)=+nan


## Q6. Within-family analysis

### Within-family analysis

**pythia** (n=8): ρ(log N, composite) = +0.881 (p = 3.9e-03)
| model | N_params | composite |
|---|---:|---:|
| pythia-70m | 70M | 0.085 |
| pythia-160m | 162M | 0.086 |
| pythia-410m | 405M | 0.107 |
| pythia-1b | 1011M | 0.092 |
| pythia-1.4b | 1415M | 0.114 |
| pythia-2.8b | 2775M | 0.139 |
| pythia-6.9b | 6857M | 0.138 |
| pythia-12b | 11847M | 0.130 |

**gpt2** (n=4): ρ(log N, composite) = +0.800 (p = 2.0e-01)
| model | N_params | composite |
|---|---:|---:|
| gpt2-small | 124M | 0.092 |
| gpt2-medium | 355M | 0.116 |
| gpt2-large | 774M | 0.112 |
| gpt2-xl | 1500M | 0.126 |

**qwen3.5** (n=9): ρ(log N, composite) = +0.881 (p = 1.7e-03)
| model | N_params | composite |
|---|---:|---:|
| qwen3.5-0.8b | 800M | 0.439 |
| qwen3.5-0.8b-it | 800M | 0.467 |
| qwen3.5-2b | 2000M | 0.537 |
| qwen3.5-2b-it | 2000M | 0.572 |
| qwen3.5-4b | 4000M | 0.797 |
| qwen3.5-4b-it | 4000M | 0.752 |
| qwen3.5-9b | 9000M | 0.773 |
| qwen3.5-9b-it | 9000M | 0.742 |
| qwen3.5-27b-it | 27000M | 0.990 |

**llama3** (n=4): ρ(log N, composite) = +0.949 (p = 5.1e-02)
| model | N_params | composite |
|---|---:|---:|
| llama3-1b | 1236M | 0.206 |
| llama3-1b-it | 1236M | 0.405 |
| llama3-3b | 3213M | 0.565 |
| llama3-8b | 8030M | 0.689 |


## Q7. Base vs Instruct paired shifts

### Base → Instruct paired shifts (n=6 pairs)
- n_pairs max: 6 (llama3-1b, qwen3.5-{0.8, 2, 4, 9}b, gemma4-e4b).
- **102** of 731 evaluated features moved unanimously across all available pairs — 55 up, 47 down.

**Top 15 unanimous shifts by |cross-model-standardised Δ|:**
- ↑ `consistency_calibration.ece                            ` std_Δ=+2.00  d=2.35  [cons-cal]
- ↓ `consistency_calibration.calibration_intercept          ` std_Δ=-1.39  d=-1.94  [cons-cal]
- ↑ `consistency_format_robustness.mean_nll_overall         ` std_Δ=+1.11  d=0.52  [cons-fmt]
- ↑ `causality_ablation.loss_ablate_5pct                    ` std_Δ=+1.10  d=0.64  [other]
- ↑ `causality_ablation.loss_ablate_10pct                   ` std_Δ=+1.03  d=0.58  [other]
- ↑ `dynamics_sharpness.sam_perturbed_loss                  ` std_Δ=+1.02  d=0.49  [dyn-sharp]
- ↑ `dynamics_sharpness.baseline_loss                       ` std_Δ=+1.01  d=0.45  [dyn-sharp]
- ↑ `causality_attention_knockout.baseline_loss             ` std_Δ=+1.00  d=0.46  [caus-ko]
- ↑ `causality_ablation.loss_ablate_1pct                    ` std_Δ=+0.99  d=0.45  [other]
- ↑ `causality_ablation.loss_ablate_25pct                   ` std_Δ=+0.99  d=0.68  [other]
- ↑ `causality_ablation.baseline_loss                       ` std_Δ=+0.98  d=0.44  [other]
- ↓ `interpretability_attention_graph.max_sink_pagerank     ` std_Δ=-0.94  d=-0.55  [interp-attG]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.mean    ` std_Δ=+0.92  d=0.44  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_mean              ` std_Δ=+0.92  d=0.44  [dyn-grad]
- ↑ `consistency_calibration.brier_score                    ` std_Δ=+0.91  d=1.97  [cons-cal]

**Unanimous shifts by category:**
- geometry: 41
- interpretability: 23
- dynamics: 20
- causality: 11
- consistency: 5
- repe: 2

**Directional themes (from unanimous set):**
- **Calibration degradation** — 4 features unanimous, top:
  - ↑ `consistency_calibration.ece` std_Δ=+2.00
  - ↓ `consistency_calibration.calibration_intercept` std_Δ=-1.39
  - ↑ `consistency_calibration.brier_score` std_Δ=+0.91
- **Sharper minima (SAM, gradient norms, Hessian trace)** — 11 features unanimous, top:
  - ↑ `dynamics_sharpness.sam_perturbed_loss` std_Δ=+1.02
  - ↑ `dynamics_sharpness.baseline_loss` std_Δ=+1.01
  - ↑ `dynamics_gradient_flow.gradient_norm_per_layer.mean` std_Δ=+0.92
- **Higher surface-form NLL** — 6 features unanimous, top:
  - ↑ `consistency_format_robustness.mean_nll_overall` std_Δ=+1.11
  - ↑ `geometry_perplexity.bits_per_char` std_Δ=+0.37
  - ↑ `geometry_perplexity.mean_nll_nats` std_Δ=+0.31
- **Lower attention entropy** — 3 features unanimous, top:
  - ↓ `interpretability_attention_entropy.max_entropy_head` std_Δ=-0.20
  - ↓ `interpretability_attention_entropy.max_normalized_entropy_head` std_Δ=-0.20
  - ↓ `interpretability_attention_entropy.avg_entropy_per_layer.std` std_Δ=-0.08
- Refusal direction emergence: none significant
- **Lower activation sparsity / higher kurtosis** — 4 features unanimous, top:
  - ↓ `interpretability_sparsity.global_mean_l0` std_Δ=-0.12
  - ↓ `interpretability_sparsity.layer_l0_rates.mean` std_Δ=-0.12
  - ↓ `interpretability_sparsity.layer_l0_rates.q50` std_Δ=-0.10


## Q8. PCA / cross-family structure

### PCA of the feature matrix
Explained variance:
- PC1: 21.30%
- PC2: 16.38%
- PC3: 12.42%

- ρ(PC1, log N_params) = +0.63  ρ(PC1, composite) = +0.87
- ρ(PC2, log N_params) = -0.53  ρ(PC2, composite) = -0.16

**Family centroids in PCA space:**
| Family | n | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|---:|
| gemma4 | 4 | +8.96 | -2.67 | -8.12 |
| gpt2 | 4 | -11.80 | -2.17 | -5.74 |
| llama3 | 4 | -0.37 | -3.38 | +19.77 |
| olmo | 1 | -6.81 | +5.09 | +9.80 |
| phi | 1 | -5.98 | -8.79 | +3.03 |
| pythia | 8 | -13.41 | -1.53 | -4.55 |
| qwen3.5 | 9 | +14.95 | +5.72 | -1.74 |
| tinyllama | 1 | -1.64 | -2.68 | +15.62 |


## Misc. notable observations

### Notable observations
- **qwen3.5-27b-it** (27B) composite = 0.990  vs  **gemma4-31b** (31B) composite = 0.954
  Smaller-better-than-larger: True
  - mmlu_acc: qwen 0.865  gemma 0.845  (qwen − gemma = +0.021)
  - arc_challenge_acc: qwen 0.597  gemma 0.643  (qwen − gemma = -0.046)

**Worst ECE (top 5) — poorly calibrated models:**
- llama3-1b-it (llama3, 1236M): ECE = 0.030
- qwen3.5-4b-it (qwen3.5, 4000M): ECE = 0.028
- tinyllama-1.1b (tinyllama, 1100M): ECE = 0.023
- gpt2-xl (gpt2, 1500M): ECE = 0.022
- qwen3.5-0.8b-it (qwen3.5, 800M): ECE = 0.020

**Best ECE (top 5):**
- pythia-1.4b (pythia, 1415M): ECE = 0.007
- gemma4-e4b (gemma4, 4500M): ECE = 0.007
- pythia-410m (pythia, 405M): ECE = 0.007
- pythia-1b (pythia, 1011M): ECE = 0.006
- qwen3.5-9b (qwen3.5, 9000M): ECE = 0.005

**Highest refusal-direction separability (repe_refusal_direction.best_layer_separability_auc):**
- gpt2-small (gpt2): 1.000
- gpt2-medium (gpt2): 1.000
- gpt2-large (gpt2): 1.000
- gpt2-xl (gpt2): 1.000
- pythia-160m (pythia): 1.000
