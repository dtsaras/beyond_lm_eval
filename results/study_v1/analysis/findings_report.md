# BLME Study — Findings Report

*Generated from results/study_v1/aggregated.csv (32 models, 1252 columns) and analysis/*.csv outputs.*

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

### Univariate Spearman with composite (n=32)
- 169 / 1176 features FDR-significant at q<0.05.

**Top 20 by |ρ| (univariate, naturally dominated by scale):**
- `interpretability_waa.layer_waa_alignments.39           ` -1.00*** (n=5)
- `geometry_contextualization.per_layer.layer31.n_words_tracked` -0.95** (n=7)
- `geometry_contextualization.per_layer.layer31.self_similarity_corrected` -0.94* (n=6)
- `repe_task_vectors.layer_task_vector_cosine_sim.min     ` -0.92*** (n=29)
- `repe_task_vectors.layer_task_vector_cosine_sim.std     ` +0.91*** (n=29)
- `geometry_contextualization.per_layer.layer8.mev        ` -0.90 (n=5)
- `repe_refusal_direction.per_layer.layer41.direction_norm` -0.90 (n=5)
- `geometry_contextualization.per_layer.layer3.mev        ` -0.90 (n=5)
- `repe_refusal_direction.per_layer.layer41.mean_projection_gap` -0.90 (n=5)
- `causality_attention_knockout.baseline_loss             ` -0.89*** (n=26)
- `interpretability_sparsity.layer_l0_rates.slope         ` -0.87*** (n=17)
- `geometry_contextualization.per_layer.layer15.n_words_tracked` -0.87** (n=12)
- `repe_task_vectors.layer_task_vector_cosine_sim.mean    ` -0.87*** (n=29)
- `interpretability_prediction_entropy.mean_top5_prob     ` +0.86*** (n=30)
- `repe_task_vectors.layer_task_vector_cosine_sim.q75     ` -0.85*** (n=29)
- `interpretability_prediction_entropy.median_entropy     ` -0.85*** (n=30)
- `geometry_contextualization.per_layer.layer7.n_words_tracked` -0.85* (n=10)
- `dynamics_sharpness.baseline_loss                       ` -0.84*** (n=29)
- `repe_task_vectors.layer_task_vector_cosine_sim.q50     ` -0.84*** (n=29)
- `interpretability_prediction_entropy.mean_entropy       ` -0.84*** (n=30)

### Partial correlates with composite, controlling for log(N_params) (n=32)
- **113 / 1176** features remain FDR-significant at q<0.05 after partialling out log N_params — i.e., they carry predictive signal **beyond model scale**.

**Top 25 features ranked by |partial ρ|:**
- `interpretability_waa.layer_waa_alignments.39           ` partial ρ=-1.00***  [interp-waa]
- `causality_tracing.layer_4_aie                          ` partial ρ=-0.98  [caus-trace]
- `geometry_contextualization.per_layer.layer8.n_words_tracked` partial ρ=-0.95  [geom-context]
- `interpretability_logit_lens.layer33_entropy            ` partial ρ=-0.94  [interp-ll]
- `interpretability_logit_lens.layer34_entropy            ` partial ρ=-0.94  [interp-ll]
- `geometry_contextualization.per_layer.layer8.anisotropy_baseline` partial ρ=-0.94  [geom-context]
- `geometry_positional_decay.layer_positional_decay.layer_35` partial ρ=+0.94  [geom-posdec]
- `interpretability_logit_lens.layer35_acc                ` partial ρ=-0.94  [interp-ll]
- `geometry_contextualization.per_layer.layer5.n_words_tracked` partial ρ=-0.92*  [geom-context]
- `interpretability_logit_lens.layer32_entropy            ` partial ρ=-0.92  [interp-ll]
- `interpretability_waa.layer_waa_alignments.37           ` partial ρ=-0.92  [interp-waa]
- `geometry_positional_decay.layer_positional_decay.layer_30` partial ρ=+0.91  [geom-posdec]
- `geometry_positional_decay.layer_positional_decay.layer_34` partial ρ=+0.90  [geom-posdec]
- `interpretability_waa.layer_waa_alignments.29           ` partial ρ=+0.90  [interp-waa]
- `geometry_contextualization.per_layer.layer17.n_words_tracked` partial ρ=-0.89  [geom-context]
- `geometry_contextualization.per_layer.layer23.n_words_tracked` partial ρ=-0.88**  [geom-context]
- `geometry_contextualization.per_layer.layer8.mev        ` partial ρ=-0.87  [geom-context]
- `geometry_contextualization.per_layer.layer3.mev        ` partial ρ=-0.87  [geom-context]
- `interpretability_waa.layer_waa_alignments.12           ` partial ρ=-0.87*  [interp-waa]
- `geometry_positional_decay.layer_positional_decay.layer_25` partial ρ=+0.86  [geom-posdec]
- `interpretability_waa.layer_waa_alignments.6            ` partial ρ=-0.86*  [interp-waa]
- `geometry_contextualization.per_layer.layer17.mev       ` partial ρ=-0.84  [geom-context]
- `geometry_contextualization.per_layer.layer31.n_words_tracked` partial ρ=-0.84  [geom-context]
- `interpretability_waa.mean_waa_alignment                ` partial ρ=-0.83*  [interp-waa]
- `geometry_contextualization.per_layer.layer0.n_words_tracked` partial ρ=-0.83***  [geom-context]

### LASSO multivariate selection (LassoCV on standardized features → composite)
- Selected **34** features out of ~920 candidates.

**Top 20 LASSO coefficients (signed):**
- ↓ `geometry_tokenizer_efficiency.vocab_utilization        ` β=-0.0773  [geom-token]
- ↑ `causality_knowledge_neurons.localization_layer_mean    ` β=+0.0677  [caus-kneu]
- ↑ `repe_refusal_direction.best_layer                      ` β=+0.0579  [repe-rd]
- ↑ `geometry_spectral.avg_stable_rank                      ` β=+0.0398  [geom-spectral]
- ↑ `causality_tracing.layer_16_aie                         ` β=+0.0290  [caus-trace]
- ↓ `interpretability_sparsity.layer_kurtosis.mean          ` β=-0.0266  [interp-sparse]
- ↑ `geometry_cka.layers.q25                                ` β=+0.0221  [geom-cka]
- ↑ `geometry_cka.layers.mean                               ` β=+0.0167  [geom-cka]
- ↓ `interpretability_sparsity.layer_kurtosis.max           ` β=-0.0158  [interp-sparse]
- ↑ `repe_refusal_direction.per_layer.layer2.direction_norm ` β=+0.0156  [repe-rd]
- ↑ `geometry_weight_norms.n_layers                         ` β=+0.0153  [geom-weight]
- ↓ `repe_task_vectors.layer_task_vector_cosine_sim.q25     ` β=-0.0139  [repe-tv]
- ↑ `repe_refusal_direction.direction_norm                  ` β=+0.0133  [repe-rd]
- ↓ `geometry_contextualization.per_layer.layer23.self_similarity_raw` β=-0.0128  [geom-context]
- ↑ `interpretability_attention_rank.layer_mean_normalised_rank.max` β=+0.0114  [interp-attR]
- ↓ `causality_tracing.layer_27_aie                         ` β=-0.0108  [caus-trace]
- ↑ `geometry_hsic.min_adjacent_hsic                        ` β=+0.0101  [geom-hsic]
- ↓ `geometry_contextualization.per_layer.layer15.n_words_tracked` β=-0.0097  [geom-context]
- ↓ `causality_edge_attribution.mean_layer_attribution_profile.std` β=-0.0090  [caus-edge]
- ↑ `geometry_intrinsic_dim.intrinsic_dimension             ` β=+0.0085  [geom-idim]


## Q4. Category-level signal

### FDR-significant features per BLME major category
(partial Spearman with composite benchmark, controlling for log N_params)

| Category | n features | FDR-sig | sig rate | max \|partial ρ\| | best feature |
|---|---:|---:|---:|---:|---|
| other | 5 | 1 | 20.0% | 0.54 | `erank_utilization_last` |
| repe | 236 | 45 | 19.1% | 0.78 | `repe_refusal_direction.per_layer.layer32.separability_auc` |
| consistency | 38 | 4 | 10.5% | 0.70 | `consistency_format_robustness.mean_nll_cv_across_formats` |
| geometry | 499 | 41 | 8.2% | 0.95 | `geometry_contextualization.per_layer.layer8.n_words_tracked` |
| dynamics | 25 | 2 | 8.0% | 0.51 | `dynamics_sharpness.baseline_loss` |
| interpretability | 302 | 17 | 5.6% | 1.00 | `interpretability_waa.layer_waa_alignments.39` |
| causality | 71 | 3 | 4.2% | 0.98 | `causality_tracing.layer_4_aie` |


## Q5. EDG validation

### Effective Dimensionality Gradient (EDG) — novel metric
- **interpretability_attention_graph.mean_edge_gini** (n=15): ρ(.,Y)=-0.29, ρ(.,log N)=-0.18, partial ρ=-0.26
- **causality_knowledge_neurons.n_facts** (n=32): ρ(.,Y)=+nan, ρ(.,log N)=+nan, partial ρ=--
- **causality_knowledge_neurons.mean_attribution_gini** (n=30): ρ(.,Y)=+0.68, ρ(.,log N)=+0.74, partial ρ=+0.08
- **causality_knowledge_neurons.mean_top1_share** (n=30): ρ(.,Y)=+0.09, ρ(.,log N)=-0.02, partial ρ=+0.22
- **causality_knowledge_neurons.mean_top1pct_share** (n=30): ρ(.,Y)=+0.67, ρ(.,log N)=+0.73, partial ρ=+0.07
- **causality_knowledge_neurons.localization_layer_mean** (n=32): ρ(.,Y)=+0.52, ρ(.,log N)=+0.27, partial ρ=+0.52*
- **causality_knowledge_neurons.attribution_layer_entropy** (n=30): ρ(.,Y)=+0.72, ρ(.,log N)=+0.74, partial ρ=+0.21
- **causality_knowledge_neurons.n_layers** (n=32): ρ(.,Y)=+0.65, ρ(.,log N)=+0.75, partial ρ=+0.15
- **causality_edge_attribution.n_prompts** (n=30): ρ(.,Y)=+nan, ρ(.,log N)=+nan, partial ρ=--
- **causality_edge_attribution.attribution_gini** (n=30): ρ(.,Y)=+0.50, ρ(.,log N)=+0.43, partial ρ=+0.29
- **causality_edge_attribution.top1_layer_share** (n=30): ρ(.,Y)=-0.57, ρ(.,log N)=-0.56, partial ρ=-0.18
- **causality_edge_attribution.peak_attribution_layer** (n=30): ρ(.,Y)=+0.02, ρ(.,log N)=+0.09, partial ρ=-0.13
- **causality_edge_attribution.attribution_entropy** (n=30): ρ(.,Y)=+0.75, ρ(.,log N)=+0.78, partial ρ=+0.24
- **causality_edge_attribution.mean_layer_attribution_profile.mean** (n=30): ρ(.,Y)=-0.72, ρ(.,log N)=-0.74, partial ρ=-0.22
- **causality_edge_attribution.mean_layer_attribution_profile.std** (n=30): ρ(.,Y)=-0.63, ρ(.,log N)=-0.49, partial ρ=-0.49*
- **causality_edge_attribution.mean_layer_attribution_profile.min** (n=30): ρ(.,Y)=-0.37, ρ(.,log N)=-0.56, partial ρ=+0.32
- **causality_edge_attribution.mean_layer_attribution_profile.max** (n=30): ρ(.,Y)=-0.62, ρ(.,log N)=-0.54, partial ρ=-0.36
- **causality_edge_attribution.mean_layer_attribution_profile.slope** (n=30): ρ(.,Y)=-0.59, ρ(.,log N)=-0.50, partial ρ=-0.38
- **causality_edge_attribution.mean_layer_attribution_profile.q25** (n=30): ρ(.,Y)=-0.64, ρ(.,log N)=-0.61, partial ρ=-0.28
- **causality_edge_attribution.mean_layer_attribution_profile.q50** (n=30): ρ(.,Y)=-0.70, ρ(.,log N)=-0.70, partial ρ=-0.24
- **causality_edge_attribution.mean_layer_attribution_profile.q75** (n=30): ρ(.,Y)=-0.75, ρ(.,log N)=-0.72, partial ρ=-0.37
- **edg** (n=25): ρ(.,Y)=+0.26, ρ(.,log N)=+0.51, partial ρ=-0.35
- **edg_early** (n=24): ρ(.,Y)=+0.70, ρ(.,log N)=+0.75, partial ρ=+0.22
- **edg_late** (n=24): ρ(.,Y)=-0.45, ρ(.,log N)=-0.19, partial ρ=-0.53
- **benchmark_mmlu_clinical_knowledge_acc** (n=32): ρ(.,Y)=+0.96, ρ(.,log N)=+0.69, partial ρ=--

**Within Pythia (n=6):**
- edg: ρ(log N, edg)=+0.77  ρ(edg, composite)=+0.60
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
- **96** of 543 evaluated features moved unanimously across all available pairs — 43 up, 53 down.

**Top 15 unanimous shifts by |cross-model-standardised Δ|:**
- ↓ `consistency_calibration.calibration_intercept          ` std_Δ=-1.38  d=-2.45  [cons-cal]
- ↑ `consistency_calibration.ece                            ` std_Δ=+1.15  d=0.49  [cons-cal]
- ↑ `consistency_calibration.brier_score                    ` std_Δ=+1.06  d=0.47  [cons-cal]
- ↑ `consistency_format_robustness.mean_nll_overall         ` std_Δ=+1.05  d=0.52  [cons-fmt]
- ↓ `consistency_calibration.calibration_slope              ` std_Δ=-1.04  d=-0.46  [cons-cal]
- ↑ `dynamics_sharpness.sam_perturbed_loss                  ` std_Δ=+0.98  d=0.49  [dyn-sharp]
- ↑ `dynamics_sharpness.baseline_loss                       ` std_Δ=+0.97  d=0.45  [dyn-sharp]
- ↑ `dynamics_gradient_flow.gradient_norm_max               ` std_Δ=+0.97  d=0.45  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.max     ` std_Δ=+0.97  d=0.45  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.std     ` std_Δ=+0.93  d=0.45  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.mean    ` std_Δ=+0.93  d=0.45  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_mean              ` std_Δ=+0.93  d=0.45  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.q25     ` std_Δ=+0.90  d=0.42  [dyn-grad]
- ↑ `dynamics_gradient_flow.gradient_norm_per_layer.q50     ` std_Δ=+0.87  d=0.44  [dyn-grad]
- ↑ `interpretability_logit_lens.layer28_entropy            ` std_Δ=+0.69  d=0.72  [interp-ll]

**Unanimous shifts by category:**
- repe: 34
- geometry: 19
- interpretability: 15
- dynamics: 14
- causality: 7
- consistency: 5
- other: 2

**Directional themes (from unanimous set):**
- **Calibration degradation** — 4 features unanimous, top:
  - ↓ `consistency_calibration.calibration_intercept` std_Δ=-1.38
  - ↑ `consistency_calibration.ece` std_Δ=+1.15
  - ↑ `consistency_calibration.brier_score` std_Δ=+1.06
- **Sharper minima (SAM, gradient norms, Hessian trace)** — 14 features unanimous, top:
  - ↑ `dynamics_sharpness.sam_perturbed_loss` std_Δ=+0.98
  - ↑ `dynamics_sharpness.baseline_loss` std_Δ=+0.97
  - ↑ `dynamics_gradient_flow.gradient_norm_max` std_Δ=+0.97
- **Higher surface-form NLL** — 1 features unanimous, top:
  - ↑ `consistency_format_robustness.mean_nll_overall` std_Δ=+1.05
- Lower attention entropy: none significant
- **Refusal direction emergence** — 32 features unanimous, top:
  - ↓ `repe_refusal_direction.per_layer.layer26.separability_auc` std_Δ=-0.62
  - ↓ `repe_refusal_direction.per_layer.layer29.separability_auc` std_Δ=-0.45
  - ↓ `repe_refusal_direction.per_layer.layer25.separability_auc` std_Δ=-0.38
- Lower activation sparsity / higher kurtosis: none significant


## Q8. PCA / cross-family structure

### PCA of the feature matrix
Explained variance:
- PC1: 12.37%
- PC2: 12.11%
- PC3: 10.58%

- ρ(PC1, log N_params) = -0.53  ρ(PC1, composite) = -0.77
- ρ(PC2, log N_params) = -0.14  ρ(PC2, composite) = -0.35

**Family centroids in PCA space:**
| Family | n | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|---:|
| gemma4 | 4 | -18.46 | +10.99 | +8.41 |
| gpt2 | 4 | +11.19 | -5.81 | +15.86 |
| llama3 | 4 | -1.54 | -1.30 | -2.64 |
| olmo | 1 | +1.75 | +2.50 | -0.70 |
| phi | 1 | +9.51 | +10.36 | +2.77 |
| pythia | 8 | +8.95 | +6.19 | -1.17 |
| qwen3.5 | 9 | -5.50 | -8.84 | -8.32 |
| tinyllama | 1 | +1.84 | +1.65 | -4.32 |


## Misc. notable observations

### Notable observations
- **qwen3.5-27b-it** (27B) composite = 0.990  vs  **gemma4-31b** (31B) composite = 0.954
  Smaller-better-than-larger: True
  - mmlu_acc: qwen 0.865  gemma 0.845  (qwen − gemma = +0.021)
  - arc_challenge_acc: qwen 0.597  gemma 0.643  (qwen − gemma = -0.046)

**Worst ECE (top 5) — poorly calibrated models:**
- gemma4-e4b-it (gemma4, 4500M): ECE = 0.568
- qwen3.5-4b-it (qwen3.5, 4000M): ECE = 0.028
- tinyllama-1.1b (tinyllama, 1100M): ECE = 0.023
- gpt2-xl (gpt2, 1500M): ECE = 0.022
- qwen3.5-0.8b-it (qwen3.5, 800M): ECE = 0.020

**Best ECE (top 5):**
- pythia-410m (pythia, 405M): ECE = 0.007
- pythia-1b (pythia, 1011M): ECE = 0.006
- qwen3.5-9b (qwen3.5, 9000M): ECE = 0.005
- pythia-12b (pythia, 11847M): ECE = 0.000
- pythia-6.9b (pythia, 6857M): ECE = 0.000

**Highest refusal-direction separability (repe_refusal_direction.best_layer_separability_auc):**
- tinyllama-1.1b (tinyllama): 0.799
- gemma4-31b (gemma4): 0.781
- gemma4-e4b (gemma4): 0.764
- gpt2-xl (gpt2): 0.764
- gemma4-e2b (gemma4): 0.761
