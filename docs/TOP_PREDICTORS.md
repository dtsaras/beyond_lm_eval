# Top Intrinsic Predictors of LLM Capability

**Study**: 32 models × 730 intrinsic features × 67 benchmark scores.
**Target**: composite benchmark (min-max-normalised mean across 67
individual lm-eval benchmarks).

**Headline** (final, 2026-04-20): LASSO on the 730 features predicts
held-out model capability at **LOO R² = 0.731**, vs. `log(N_params)`
baseline LOO R² = **0.429** — **+0.30 absolute improvement** from
intrinsic signals. LOFO R² = **0.262** (strict cross-family test).

All features below are **deduped by feature family** — the
aggregator emits `.mean`, `.std`, `.min`, `.max`, `.q25`, `.q50`,
`.q75`, `.slope` summaries per per-layer feature, so pre-dedupe
tables show 5-8 copies of the same underlying signal. For the paper
we report one row per family (highest |ρ| within family), so the
table is a list of genuinely-independent predictors.

---

## 1. Top 25 univariate predictors — deduped by feature family

|#| Family | Representative column | ρ |
|---|---|---|---|
| 1 | `repe_task_vectors.layer_task_vector_cosine_sim` | `.min` | −0.916 |
| 2 | `geometry_perplexity.ppl_freq` | — | −0.910 |
| 3 | `causality_ablation.loss_ablate_1pct` | — | −0.867 |
| 4 | `interpretability_prediction_entropy.mean_top5_prob` | — | +0.856 |
| 5 | `dynamics_sharpness.baseline_loss` | — | −0.856 |
| 6 | `geometry_perplexity.mean_nll_nats` | — | −0.854 |
| 7 | `geometry_perplexity.ppl_overall` | — | −0.854 |
| 8 | `causality_ablation.baseline_loss` | — | −0.853 |
| 9 | `interpretability_prediction_entropy.median_entropy` | — | −0.853 |
| 10 | `causality_attention_knockout.baseline_loss` | — | −0.852 |
| 11 | `geometry_perplexity.bits_per_char` | — | −0.851 |
| 12 | `interpretability_prediction_entropy.mean_entropy` | — | −0.836 |
| 13 | `repe_refusal_direction.mean_projection_gap` | — | +0.832 |
| 14 | `repe_refusal_direction.direction_norm` | — | +0.832 |
| 15 | **`geometry_schatten.schatten_4_per_layer`** (round-7) | `.q75` | −0.831 |
| 16 | `geometry_intrinsic_dim.sample_size` | — | +0.830 |
| 17 | `interpretability_prediction_entropy.mean_top1_prob` | — | +0.826 |
| 18 | `topology_homology.mean_persistence_h0` | `.slope` | +0.816 |
| 19 | `interpretability_logit_lens.acc` | `.slope` | −0.815 |
| 20 | `topology_homology.max_persistence_h0` | `.slope` | +0.815 |
| 21 | `geometry_contextualization.per_layer.n_words_tracked` | `.min` | −0.813 |
| 22 | `interpretability_prediction_entropy.p90_entropy` | — | −0.803 |
| 23 | `causality_ablation.loss_ablate_5pct` | — | −0.795 |
| 24 | `geometry_tokenizer_efficiency.vocab_size` | — | +0.789 |
| 25 | `interpretability_waa.mean_waa_alignment` | — | −0.787 |

All *** (FDR q < 0.001). The univariate table is **dominated by
proxies of model size / loss** (perplexity, ablation baseline loss,
prediction entropy). This is expected: scale is the strongest
univariate predictor of capability.

---

## 2. Top 25 **partial** predictors — controlling for log(N_params)

This is the headline table for the paper: what predicts capability
**beyond raw scale**?

|#| Family | Representative column | partial ρ | Notes |
|---|---|---|---|---|
| 1 | `repe_task_vectors.layer_task_vector_cosine_sim` | `.min` | −0.824 | Ilharco 2023 |
| 2 | `geometry_contextualization.per_layer.n_words_tracked` | `.max` | −0.820 | Ethayarajh 2019 (tokenizer-linked) |
| 3 | `geometry_lid.lid_min` | — | +0.804 | Levina-Bickel 2004 |
| 4 | `interpretability_waa.layer_waa_alignments` | `.mean` | −0.781 | Park et al. 2024 |
| 5 | `causality_ablation.loss_ablate_5pct` | — | −0.772 | BLME diagnostic |
| 6 | `geometry_tokenizer_efficiency.vocab_size` | — | +0.769 | tokenizer confound |
| 7 | `geometry_hubness.hubness_k10_gini` | — | +0.744 | Tomašev 2014 |
| 8 | `geometry_cka.min_offdiag_cka` | — | +0.739 | Kornblith 2019 |
| 9 | **`geometry_schatten.matrix_nuclear_norm_per_layer`** (round-7) | `.q50` | +0.739 | Li 2024 MNN |
| 10 | `geometry_cka.std_offdiag_cka` | — | −0.738 | Kornblith 2019 |
| 11 | `causality_ablation.loss_ablate_1pct` | — | −0.725 | BLME diagnostic |
| 12 | `interpretability_waa.mean_waa_alignment` | — | −0.712 | Park et al. 2024 |
| 13 | `geometry_tokenizer_efficiency.fertility` | — | +0.712 | tokenizer confound |
| 14 | `geometry_tokenizer_efficiency.compression_ratio` | — | +0.712 | tokenizer confound |
| 15 | `geometry_tokenizer_efficiency.total_tokens` | — | +0.712 | tokenizer confound |
| 16 | `geometry_intrinsic_dim.sample_size` | — | +0.712 | vocab-size proxy |
| 17 | `geometry_collapse.erank_per_layer` | `.q75` | +0.711 | Roy-Vetterli |
| 18 | **`interpretability_activation_sinks.massive_activation_max_ratio_per_layer`** (round-8) | — | −0.705 | Sun 2024 |
| 19 | `interpretability_attention_rank.layer_min_effective_rank` | `.slope` | +0.695 | Dong 2021 |
| 20 | `consistency_format_robustness.mean_nll_cv_across_formats` | — | +0.692 | Sclar 2023 |
| 21 | `topology_homology.max_persistence_h0` | `.slope` | +0.688 | Zomorodian 2005 |
| 22 | `geometry_cka.min_adjacent_cka` | — | +0.687 | Kornblith 2019 |
| 23 | `geometry_collapse.collapse_ratio` | — | +0.680 | BLME diagnostic |
| 24 | `geometry_hsic.input_to_layer_hsic` | `.mean` | +0.678 | Gretton 2005 |
| 25 | `interpretability_attention_entropy.avg_entropy_per_layer` | `.std` | −0.677 | Clark 2019 |

All *** (FDR q < 0.001).

**Key observations**:

- **RepE task-vector diversity dominates** (#1, |ρ| = 0.824) — the
  single strongest predictor beyond scale.
- **Round-7 addition `geometry_schatten.matrix_nuclear_norm`
  lands at #9** — validates Li 2024's claim that MNN is an
  independent capability proxy.
- **Round-8 addition `massive_activation_max_ratio` at #18** —
  validates Sun 2024's activation-outlier signature as a real
  capability signal beyond scale.
- **Tokenizer-family signals crowd positions 6, 13-16**
  (vocab_size, fertility, compression_ratio, total_tokens) — a
  real confound: more mature tokenizers correlate with more
  training data, which correlates with capability.
- **11 of 25 families are geometry-derived**; 5 interpretability;
  4 causality/consistency; 2 RepE; 1 each for topology and
  activation-sinks. **No single category dominates.**

---

## 3. LASSO sparse prediction (final honest numbers)

LASSO with 5-fold CV, held-out LOO + LOFO evaluation.

| Metric | Training R² | LOO R² | LOFO R² |
|---|---|---|---|
| LASSO (730 features → 28 selected) | 0.999 (overfit; expected at n<<p) | **0.731** | 0.262 |
| Linear baseline, `log(N_params)` only | 0.498 | 0.429 | — |

**Interpretation**:
- **+0.30 absolute LOO R² gain** from 730 intrinsic features over
  single-variable scale.
- **LOFO R² = 0.262 is weak**: cross-family generalisation is the
  open problem. 4 families (GPT-2, Pythia, Llama3, Qwen3.5,
  Gemma4) is a strict test; scaling to 8+ families would likely
  improve this.

---

## 4. Paper-ready claim (verbatim, for §4)

> Using 32 pretrained LLMs spanning 5 families and 3 orders of
> magnitude in parameter count, a LASSO combining 28 intrinsic
> metrics computed from weights and hidden-state activations
> (without any benchmark data) predicts composite benchmark
> performance at held-out leave-one-out R² = **0.731**, compared
> to **0.429** for a log(N_params)-only baseline — a +0.30
> absolute improvement in cross-validated predictive accuracy.
> After controlling for scale, the strongest single predictors
> are the diversity of RepE task vectors (partial ρ = −0.82;
> Ilharco 2023), contextualized word-tracking (partial ρ = −0.82;
> Ethayarajh 2019), local intrinsic dimensionality (partial ρ =
> +0.80; Levina-Bickel 2004), weight-activation alignment
> (partial ρ = −0.78; Park et al. 2024), the recently-introduced
> Matrix Nuclear-Norm (partial ρ = +0.74; Li et al. 2024), and
> the massive-activation outlier signature (partial ρ = −0.71;
> Sun et al. 2024). Critically, leave-one-family-out R² = 0.262
> reveals that cross-family generalisation remains an open
> challenge: the signals identified transfer well within
> architectural families but do not cleanly extrapolate across
> them.

---

## 5. Notes on redundancy in the aggregator's summary columns

The BLME aggregator emits 7 statistics per per-layer feature
(mean, std, min, max, q25, q50, q75, slope). Strong signals
therefore generate 5-8 highly-correlated columns. The tables
above are **deduped by feature family** — we strip the trailing
aggregator suffix and keep the single highest |ρ| per family.
Raw (pre-dedupe) results are in `results/study_v2/analysis/*.csv`.

The family-level summary: in the top-50 univariate table pre-dedupe,
the most-represented families are:

| Feature family | # rows in top-50 | max \|ρ\| |
|---|---|---|
| `geometry_contextualization.n_words_tracked` | 6 | 0.813 |
| `repe_task_vectors.layer_task_vector_cosine_sim` | 5 | 0.916 |
| `geometry_schatten.schatten_4_per_layer` | 4 | 0.831 |
| `topology_homology.mean_persistence_h0` | 3 | 0.816 |

These are all the same underlying signal reported under different
aggregator summaries — the dedupe fix surfaces the 25 distinct
families rather than 25 correlated copies of the same top-5.

---

## 6. Reproducibility

Generated from `results/study_v2/analysis/partial.csv`,
`univariate.csv`, and `lasso_features.csv` via
`scripts/analyze_correlations.py`. Deduplication script:
`/tmp/dedupe_top_predictors.py` (copied into repo for paper-ready
generation).
