# Beyond Benchmarks: Correlating Intrinsic LLM Properties with Downstream Performance

## Research Objective

Benchmark scores tell us *what* LLMs can do but not *why*. This study uses BLME to measure intrinsic geometric, topological, and mechanistic properties of 30+ language models and systematically correlates these with downstream benchmark performance. We aim to: (1) identify which intrinsic properties predict performance beyond model size alone, (2) characterize how instruction tuning alters representational geometry, and (3) propose a novel metric — the Effective Dimensionality Gradient (EDG) — that captures the information bottleneck structure of a model's representations.

---

## 1. Task Taxonomy

Of BLME's 51 diagnostic tasks, we classify each by how intrinsic it is to the model (vs. dependent on input data or dataset design). Only tasks in Tiers 1-3 are used as independent variables (predictors). Behavioral tasks are used as dependent variables or excluded.

### Tier 1 — Fully Intrinsic (weight-only, no data dependency)

| Task | Key Metrics | What It Captures |
|------|------------|------------------|
| `geometry_spectral` | avg_alpha, avg_stable_rank | Power-law exponent and stable rank of weight matrices |
| `geometry_hubness` | hubness_skew, gini | k-NN hub structure of the embedding matrix |
| `dynamics_stability` | stability_mean, stability_std | Jaccard overlap of k-NN neighborhoods in embedding space |
| `geometry_unembedding` | eff_rank, is_tied | Effective rank and structure of the LM head |

### Tier 2 — Mostly Intrinsic (data-dependent but measuring stable model tendencies)

| Task | Key Metrics | What It Captures |
|------|------------|------------------|
| `geometry_svd` | effective_rank, participation_ratio, avg_cosine_similarity | Isotropy and capacity utilization of representation space |
| `geometry_lid` | lid_mean, lid_std, lid_median | Local intrinsic dimensionality (MLE) |
| `geometry_collapse` | collapse_ratio, erank_per_layer, max_drop | Effective rank trajectory across layers |
| `geometry_lipschitz` | lipschitz_mean, lipschitz_max | Layer change ratios ||h_{l+1}-h_l||/||h_l|| |
| `geometry_intrinsic_dim` | intrinsic_dimension | Global Two-NN intrinsic dimension |
| `geometry_matrix_entropy` | mean_matrix_entropy, layer_matrix_entropies | Von Neumann entropy of covariance per layer |
| `geometry_mutual_info` | avg_adjacent_mi, information_compression_ratio | HSIC-based mutual information between layers |
| `geometry_rsa` | rsa_adjacent_mean, rsa_early_late | Spearman correlation between layer RDMs |
| `geometry_cka` | avg_adjacent_cka, cka_matrix | Centered kernel alignment between layers |
| `geometry_correlation_dimension` | correlation_dimension | Grassberger-Procaccia fractal dimension |
| `geometry_positional_decay` | mean_positional_decay_correlation | Spearman(distance, attention) for RoPE integrity |
| `geometry_consistency` | cosine_consistency_mean | Hidden-state / predicted-embedding alignment |
| `interpretability_induction_heads` | max_induction_score, avg_induction_score | Mechanistic induction head detection (synthetic data) |
| `interpretability_attention_entropy` | avg_entropy_total, per-layer entropy | Shannon entropy of attention distributions |
| `interpretability_sparsity` | global_mean_l0, global_mean_kurtosis | MLP activation sparsity and heavy-tailedness |
| `interpretability_superposition` | mean_polysemanticity_index, neuron_utilization_rate | Bimodality coefficient of neuron activations |
| `interpretability_waa` | mean_waa_alignment | Weight-activation alignment (top singular vectors) |
| `interpretability_logit_lens` | per-layer accuracy and entropy | Logit lens convergence profile |
| `interpretability_attention_graph` | mean_sink_pagerank, mean_edge_gini | Attention sink structure (PageRank centrality) |
| `causality_tracing` | max_aie, causal_entropy | ROME-style causal tracing (AIE per layer) |
| `causality_attention_knockout` | head_impact_gini_coefficient | Head importance distribution |
| `topology_homology` | per-layer persistence H0/H1 | Persistent homology features |
| `topology_persistence_entropy` | per-layer PE_H0, PE_H1 | Entropy of persistence lifespans |
| `topology_betti_curve` | simplification_ratio, betti_0_decay_rate | Betti number evolution across layers |
| `repe_task_vectors` | layer_task_vector_norms, cosine_sim | Task vector geometry |
| `repe_concept_separability` | layer_separability_auc, max_auc | Linear separability per layer |

### Tier 3 — Data-Dependent (valid with controlled corpus, see TASK_FIXES.md)

| Task | Status | What It Captures |
|------|--------|------------------|
| `geometry_mahalanobis` | Needs fixed OOD strategy | Mahalanobis distance for OOD detection |
| `geometry_information_fisher` | Needs more samples (>=50) | Empirical Fisher trace (curvature) |
| `geometry_categories` | Needs standardized categories | Category separation in embedding space |
| `interpretability_attribution` | Exploratory only | Semantic coherence of layer updates |
| `interpretability_attention_polysemanticity` | Exploratory only | SVD entropy of attention projections |
| `dynamics_coe` | Needs standardized prompts | Chain-of-embedding drift during generation |
| `repe_steering_effectiveness` | Needs fixed alpha/prompts | KL divergence under steering |
| `consistency_calibration` | Use as Y-variable | Expected Calibration Error |
| `consistency_paraphrase` | Needs better paraphrase set | Representation invariance to paraphrasing |
| `consistency_contrastive` | Needs length-controlled pairs | Factual vs. counterfactual probability |
| `consistency_knowledge_capacity` | Needs better tokenization | Memorization vs. generalization ratio |
| `consistency_contamination` | Needs reference baseline | Min-k% memorization detection |
| `interpretability_prediction_entropy` | Use as Y-variable | Output distribution entropy |
| `interpretability_probing` | Needs control task | Linear probe accuracy per layer |
| `consistency_logical` | Needs real logical pairs | Logical consistency via probability |
| `causality_ablation` | Needs dataset-mean ablation | Ablation robustness curve |
| `causality_circuit_quality` | Needs adaptive thresholds | Circuit faithfulness/minimality |
| `dynamics_interpolation` | Needs more pairs + steps | Latent space convexity |

### Excluded from Study

| Task | Reason |
|------|--------|
| `interpretability_sae_features` | Only works with GPT-2 small SAE dictionaries; cannot compare across models |
| `geometry_layer_change_ratio` | Removed — exact algorithmic duplicate of `geometry_lipschitz` |
| `geometry_perplexity` | Reclassified as dependent variable (Y), not predictor |

---

## 2. Evaluation Corpus

### Problem

The default cache (`cache.py:211-217`) uses 3 hardcoded sentences repeated cyclically. This provides only ~30-50 unique tokens — insufficient for any publishable result.

### BLME-Bench Standard Corpus

**Source:** WikiText-103 validation set (deterministic, publicly available, not in standard contamination lists).

**Construction:**
1. Select first 500 passages with >= 64 tokens
2. Truncate each to 128 tokens (per model's own tokenizer)
3. Discard passages < 32 tokens after tokenization
4. Final corpus: ~400-500 passages, ~50K-60K tokens per model

**Sample counts per task tier:**
- Tier 1 (weight-only): no corpus needed
- Tier 2 (hidden states): `num_samples=200`
- Tier 3 (expensive): `num_samples=10-50`
- Topology (O(n^3)): `num_samples=50`

**Implementation:** The cache already accepts arbitrary `{"text": ...}` lists via `_resolve_dataset()`. Pass the WikiText corpus through a driver script.

---

## 3. Model Zoo (~30 checkpoints)

### Within-family scaling series (control architecture, isolate size)

| Family | Checkpoints | Count |
|--------|-------------|-------|
| GPT-2 | 124M, 355M, 774M, 1.5B | 4 |
| Pythia (deduped) | 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B | 8 |
| Llama-3.x | 1B, 3B, 8B | 3 |
| Qwen-2.5 | 0.5B, 1.5B, 3B, 7B | 4 |

### Cross-family at ~1-3B (control size, compare architecture)

GPT-2 XL (1.5B), Pythia-1.4B, OLMo-1B, Llama-3.2-1B, Qwen-2.5-1.5B, Gemma-2B, TinyLlama-1.1B, Phi-2 (2.7B)

### Base vs. instruction-tuned pairs

Llama-3.2-1B / Instruct, Qwen-2.5-1.5B / Instruct, Qwen-2.5-7B / Instruct (3-5 pairs)

**Total: ~28-35 unique checkpoints**

---

## 4. Normalization for Cross-Model Comparability

### Dimension-dependent metrics

| Metric | Normalization |
|--------|---------------|
| effective_rank, participation_ratio, LID, intrinsic_dim | / d_model -> ratio in [0,1] |
| condition_number | log10 |
| matrix_entropy | / log(d_model) |
| fisher_trace | log or / d_model^2 |
| lipschitz_mean | Already a ratio |
| collapse_ratio | Already [0,1] |

### Layer-dependent metrics

Models have 12 to 80+ layers. For layer-wise profiles:
1. **Interpolation:** Resample to normalized depth axis [0.0, 1.0] with 20 evenly spaced points
2. **Summary statistics:** Value at depth 0.25/0.5/0.75, slope of linear fit, curvature (quadratic coefficient), min/max and their normalized depth positions

### Tokenizer differences

Same text produces different token counts across tokenizers. Normalize per-token statistics by each model's own token count. Cross-layer metrics are internally consistent within a model.

---

## 5. Benchmark Performance (Dependent Variables)

### Primary Y-variable

**Composite benchmark score** = mean of min-max normalized accuracies across:
- HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, MMLU (5-shot)

All benchmarks run via `lm_eval` with fixed seeds and standard task configurations.

### Secondary Y-variables

- Individual benchmark scores
- Expected Calibration Error (from `consistency_calibration`)
- Perplexity on evaluation corpus (from `geometry_perplexity`)

---

## 6. Statistical Analysis Plan

### Step 1: Univariate correlations
Spearman rho between each intrinsic metric and each benchmark. Benjamini-Hochberg FDR correction for multiple comparisons (~400 tests). Visualize as heatmap.

### Step 2: Partial correlations controlling for size
Partial Spearman rho(metric, benchmark | log_params). Key question: which metrics predict performance beyond what model size alone predicts?

### Step 3: Multivariate prediction
- Ridge regression (CV-tuned lambda) predicting composite score from all features
- LASSO for feature selection: minimal set of intrinsic metrics that best predicts performance
- Compare R^2 against naive baseline: log(params) only

### Step 4: Within-family analysis
Repeat steps 1-3 within Pythia (n=8) and GPT-2 (n=4) families separately. Controls for architecture and training data. Answers: do the same metrics correlate within a family as across families?

### Step 5: Base vs. instruction-tuned
Paired Wilcoxon signed-rank test per metric. Which intrinsic properties shift with instruction tuning? Which are invariant?

### Step 6: Clustering
PCA on (models x features) matrix. Hierarchical clustering (Ward linkage) on intrinsic features alone. Do models cluster by family, size, or training paradigm?

### Step 7: Novel metric evaluation
Compare EDG (Section 7) against all individual metrics for predictive power. Report incremental R^2.

### Statistical power
With n~30 models, minimum detectable correlation at alpha=0.05, power=0.8 is r~0.45 (medium-to-large effects). Within-family Pythia (n=8) can detect r>0.7 only — report as exploratory.

---

## 7. Novel Metric: Effective Dimensionality Gradient (EDG)

### Motivation

The information bottleneck principle (Tishby & Zaslavsky 2015; Shwartz-Ziv & Tishby 2017) suggests that good representations compress input information into task-relevant features. In a transformer, this should manifest as a progressive reduction in effective dimensionality from early to late layers. Models that compress smoothly should generalize better than those with erratic or non-monotonic dimensionality profiles.

### Definition

Given a model with L layers, let erank(l) = exp(H(sigma)) where sigma are the normalized singular values of the hidden-state covariance at layer l. This is already computed by `geometry_collapse` -> `erank_per_layer`.

```
erank_ratio(l) = erank(l) / d_model        # normalize by model width
EDG = Spearman(layer_index, erank_ratio)    # rank correlation over layers
```

### Interpretation

| EDG Value | Meaning |
|-----------|---------|
| ~ -1.0 | Smooth monotonic compression. Strong information bottleneck. |
| ~ 0.0 | No compression trend. Random dimensionality across layers. |
| > 0.0 | Dimensionality expansion with depth (unusual). |

**Hypothesis:** EDG closer to -1.0 correlates with better benchmark performance.

### Extended compression profile features

| Feature | Definition |
|---------|-----------|
| EDG_early | Spearman over layers 0 to L//3 |
| EDG_late | Spearman over layers 2L//3 to L-1 |
| erank_utilization_first | erank_ratio at layer 0 (initial capacity usage) |
| erank_utilization_last | erank_ratio at final layer (surviving capacity) |
| compression_smoothness | 1 - std(delta_erank) / mean(|delta_erank|) |

### Why EDG

- **Threshold-free:** Unlike composite scores (RCE), EDG uses Spearman rank correlation with no arbitrary cutoffs.
- **Scale-independent:** Rank-based, so naturally comparable across models with different d_model.
- **Zero new code:** Computed from existing `geometry_collapse` output.
- **Theoretically grounded:** Information bottleneck theory predicts that optimal representations compress input entropy into task-relevant features.

### Implementation

```python
from scipy.stats import spearmanr

erank_per_layer = results["geometry_collapse"]["erank_per_layer"]
d_model = model.config.hidden_size
ratios = [e / d_model for e in erank_per_layer]
edg, p_value = spearmanr(range(len(ratios)), ratios)
```

### Validation plan

1. Compute EDG for all models in the zoo
2. Correlate with composite benchmark score (expect significant negative correlation)
3. Partial correlation controlling for log(params) — does EDG predict beyond size?
4. Compare R^2 against every other individual metric
5. Within-family analysis: does EDG predict within the Pythia scaling series?
6. Base vs. instruct pairs: does instruction tuning systematically change EDG?

---

## 8. Paper Structure

1. **Introduction** — Benchmarks measure "what" not "why"; gap in systematic correlation studies
2. **Background** — Representation geometry (Ethayarajh 2019), information bottleneck (Tishby), spectral properties (Martin & Mahoney 2021), scaling laws (Kaplan 2020)
3. **Methodology** — Task taxonomy (Table 1), model zoo (Table 2), corpus design, normalization
4. **Results** — Univariate correlations (heatmap), partial correlations, LASSO feature selection, within-family analysis, base vs. instruct
5. **Effective Dimensionality Gradient** — Definition, results, predictive power, compression profile visualization
6. **Discussion** — Which metrics matter, limitations (correlation != causation, sample size), implications for model design
7. **Appendix** — Full metric definitions, per-model results, compute cost, sensitivity analysis

---

## 9. Code Changes Required

### New files needed
- `scripts/run_benchmark_study.py` — Driver script: iterates over model list, loads WikiText corpus, runs BLME per model, collects results matrix
- `scripts/analyze_correlations.py` — Analysis: normalizations, Spearman correlations, partial correlations, LASSO, PCA, figures, EDG computation

### Infrastructure changes
- WikiText corpus loader utility in `src/blme/` for reproducibility

### No changes needed
- All existing task implementations stay as-is
- Cache infrastructure already supports custom datasets via `_resolve_dataset()`
- Core evaluation dispatcher works unchanged

---

## 10. Verification Plan

1. Run BLME on GPT-2 small with WikiText corpus — validate all 30 Tier 1+2 tasks produce outputs
2. Run BLME on GPT-2 family (4 sizes) — check normalized metrics show expected scaling trends
3. Compute EDG for GPT-2 family — verify it is negative and magnitude increases with size
4. Run lm_eval benchmarks on GPT-2 family — verify correlation pipeline end-to-end
5. Spot-check: within Pythia, does `geometry_spectral.avg_alpha` correlate with benchmark score? (Expected: yes, per Martin & Mahoney 2021)
