# Changelog

## 0.2.0

### New tasks
- `dynamics_entropy_calibration`: entropy calibration error EntCE (Cao et al. 2025).
- `geometry_dime`: DiME augmentation invariance across depth (Skean et al. 2025). It needs the new `augment` extra (`nlpaug==1.1.11`).
- `geometry_zipfian_symmetry`: Zipfian symmetry of the embedding tables (Yokoi et al. 2024).
- `geometry_delta_hyperbolicity`: relative delta-hyperbolicity of prompt token clouds (HypLoRA).
- `interpretability_inactive_heads`: inactive-head fraction (Sandoval-Segura et al. 2026).
- `geometry_prompt_id`: prompt-level intrinsic dimension (Viswanathan et al. 2026).
- `geometry_ffn_spectral_utilization`: FFN activation spectral utilisation (Jha & Reagen 2025).
- Also new since 0.1.0: `geometry_alpha_req`, `geometry_spectral_signature`, `geometry_gride`, `geometry_information_imbalance`, `geometry_farms_alpha`, `causality_layer_skip`, `consistency_icl_score`, `geometry_representation_dispersion`, `geometry_predictive_information`, `geometry_neighborhood_overlap`, `geometry_ww_randomized`, `interpretability_logit_lens_kl`, `geometry_rogue_dimensions` and `geometry_lm_head_free_energy`. The README lists all 21 with their parity tests.

### Fixes that change numbers
- **One-point-per-sequence clouds read 20-word windows.** Each passage contributes the last token of a 20-word window, chosen from its text alone, as Cheng et al. do. Before, the cloud took the last token of each whole passage. This changes every cloud key of 13 tasks: `geometry_intrinsic_dim`, `geometry_lid`, `geometry_vendi_score`, `geometry_magnitude`, `geometry_correlation_dimension`, `geometry_gride`, `geometry_information_imbalance`, `geometry_neighborhood_overlap`, `geometry_alpha_req`, `topology_betti_curve`, `topology_homology`, `topology_persistence_entropy` and `topology_persistence_landscape`. `points_readout: passage` restores whole passages; that route is now also capped at the model's context.
- **An appended EOS is stripped wherever a task tokenises its own input**, as the shared cache already did. The change affects tokenizers that append EOS (OLMo). Keys:
  - `convexity_gap` (interpolation);
  - `growth_log_ratio_relative` (perturbation_growth);
  - `gradient_flow_log_ratio_in_out`;
  - `representation_sensitivity` and `S_unit`;
  - `positional_decay_nosink_write_weighted`;
  - `effective_depth_90`, `total_block_influence` and `max_block_influence`;
  - `prediction_alignment_mean` and `prediction_alignment_std` (fallback route).
- `geometry_categories`: both words of a relation pair use one token form, so `{relation}_consistency` changes.
- `geometry_collapse`: `qdl_entropy_min` reads the first 64 tokens of every passage. The whole-passage value is `_meta_qdl_entropy_min_whole_passage`.
- `geometry_positional_decay`: `positional_decay_nosink_write_weighted` scores the first 256 content tokens of long WikiText documents (`documents: dataset` keeps the run's passages).
  - Qwen3.5's linear-attention blocks are skipped, so the key is no longer NaN on those models.
  - An output projection called with a keyword input is captured.
- `dynamics_perturbation_growth`: the headline is now the relative growth `growth_log_ratio_relative` = log(r_L / r_0); see the renames below.
- `geometry_perplexity`: a special token the tokenizer adds owns no bytes, so `bits_per_byte` is finite on Llama-3 tokenizers (it was NaN).
- `dynamics_generation_diversity`: completions are cut at the first of every stop id, not only the tokenizer's EOS. This changes `distinct_*`, `mean_distinct_*`, `mean_self_bleu`, `mean_phrase_repetition_rate` and the sampling-entropy keys.
- `dynamics_interpolation`: context pairs are distinct and drawn without replacement (`convexity_gap`).
- `geometry_rsa`: a constant row is dropped from every correlation RDM. `rsa_adjacent_min_corrdist` and `rsa_early_late_corrdist` are no longer NaN when such a row exists.
- `geometry_phd_dimension`: when no text yields a per-text estimate, `phd_dimension` is NaN with a reason. It used to fall back silently to the pooled estimate.
- Topology:
  - an empty barcode gives NaN persistence entropy (`*_pe_h1`, `*_pe_h1_norm`); it used to give 0;
  - `topology_homology`'s mean H1 lifetime is NaN without loops;
  - the `*_norm` twins divide by the mean pairwise distance of the de-duplicated cloud.
- `dynamics_gradient_flow`: blocks called with `hidden_states=` are captured. `gradient_flow_log_ratio_in_out` was NaN or wrong on such models.
- `geometry_representation_sensitivity`:
  - `S_unit` is finite on a final norm without affine parameters (OLMo);
  - the head-linearity guard fails closed, so `representation_sensitivity` and `S_unit` are NaN, with a reason, when the guard cannot run.
- `geometry_schatten`: blocks with routed experts get NaN with a reason, and `schatten_1` / `schatten_4` average the remaining blocks. Dense models are unchanged.
- `geometry_predictive_information`: special tokens are never targets. This changes `predictive_info_*` on EOS-appending tokenizers.
- `geometry_alpha_req`: non-finite rows are masked and counted, and huge finite rows no longer overflow, which affects `alpha_req` and `rankme`. Finite clouds are unchanged.
- `dynamics_stability`: the noise is drawn on the CPU, so `stability_mean` from a GPU run equals the CPU value.
- Phi-1/1.5/2: the final norm at `model.final_layernorm` is found. Phi results of logit_lens, logit_lens_kl, unembedding, representation_sensitivity, gradient_flow, zigzag_persistence and bias_weat change.
- Gemma 3n / 4: the final logit softcap is read from the text config. This changes the tasks that decode through `common.apply_lm_head`: `interpretability_logit_lens`, `interpretability_logit_lens_kl` and `convexity_gap`.
- `_meta_` only: `_meta_gride_peak_layer_frac` uses the hidden-state index / L, as the other peak depths do.

### Key renames and demotions to `_meta_`
- `dynamics_perturbation_growth`: `growth_log_ratio` → `_meta_growth_log_ratio_absolute`; the new headline is `growth_log_ratio_relative`.
- `geometry_unembedding`: `embedding_alignment_mean`, `embedding_alignment_std` and `embedding_high_alignment_frac` move to `_meta_`.
- `dynamics_stability`: `stability_std` moves to `_meta_`.
- `geometry_mahalanobis`: `mean_mahalanobis_id`, `mean_mahalanobis_ood` and `ood_separation_gap` move to `_meta_`. Their per-dimension twins stay headline.
- `topology_betti_curve`: the headline is `betti_1_curve_mean`; `max_betti_1` moves to `_meta_`.
- `geometry_rsa`: `_meta_n_degenerate_rows_jittered` → `_meta_n_constant_rows_dropped_corrdist`.

Apart from the perturbation_growth headline, these moves leave values unchanged.

### Docs and packaging
- `docs/VALIDATION.md` is generated from the study's verification trackers. It replaces `SIGNOFF_MATRIX.md`, `TASK_STATUS.md`, `TOP_PREDICTORS.md` and `VERIFICATION_STATUS.md`.
- The README groups the tasks into three tiers:
  - the 51 paper-core tasks;
  - the 21 methods added in this release;
  - the experimental interpretability, causality and consistency tasks.
- Task metadata notes say what each task computes and name the test that pins it. Provenance claims that no test backs were removed.
- Identity-block invariance is declared per placement (between blocks, before block 0, after the last block).
- `transformers>=5.2.0,<5.11` (5.2.0 and 5.10.2 are tested). The new `augment` extra installs `nlpaug==1.1.11`.

### Earlier changes in 0.2.0

#### Added
- `blme.tasks._tokens`: `select_rows` (token policies `all`, `exclude_pos0`, `last`, `mean`, `mean_excl_pos0` with per-row document and position), `massive_token_share` / `position0_share`, and `hidden_states_for_task`, one cache-or-collect gate with explicit parameters. `collect_hidden_states` gains `per_sample=` (per-document lists that record each row's token position, in memory and under disk offload) and `max_length=`. Defaults are unchanged.
- `blme.tasks._precision.fp32_compute`: computes in fp32 on an fp16/bf16 model and restores every parameter AND buffer to its original dtype, also on error, with no size gate.

#### Changed
- **Python ≥ 3.10 is now required.** numpy 2.x and transformers 5.x need it, and the package evaluates PEP 604 annotations (`bool | None`) at import time. The CI matrix is now 3.10–3.12.
- **scikit-learn is now a core dependency**, because five paper-core tasks need it.
- **The default corpus changed** (`load_default_corpus`, used when no dataset is passed). Numbers move for runs that relied on it; runs that pass their own dataset are unaffected.
  - It loads `Salesforce/wikitext` (`wikitext-103-raw-v1`, validation) and keeps passages of at least 64 characters after stripping, returned stripped. The trailing newline of every WikiText row is a massive-activation sink on pythia and TinyLlama.
  - `diverse=True` is now the default: draws follow one fixed van der Corput order, so every draw is spread across the split and `load_default_corpus(n)` is a prefix of `load_default_corpus(m)` for `n < m`. `diverse=False` gives the split-order prefix.
  - Requesting more passages than exist (1691) raises `ValueError`, and a dataset that cannot be loaded raises `RuntimeError`. The three hardcoded fallback sentences are gone. `evaluate()` caps its corpus request at the corpus size, because some `num_samples` count vocabulary rows.
  - New `load_long_documents(n, min_tokens, tokenizer)` returns whole articles from `EleutherAI/wikitext_document_level` in the same nested order.

#### Fixed
- **bfloat16 storage no longer crashes.** numpy has no bfloat16 dtype, so `.numpy()` raised `TypeError` on bf16 hidden states in three places:
  - disk offload in `collect_hidden_states` (`BLME_DISK_OFFLOAD=1`), which now stores the bf16 bit pattern and reads it back as bf16, bit for bit;
  - the self-collect routes of `geometry_lipschitz` and `geometry_rsa`, which now widen bf16 exactly to float32 (new helper `geometry.utils.as_numpy`), so they match the cache route bit for bit.

  Shipped and study numbers are unchanged: those paths raised before.
- **`geometry_phd_dimension` with `pooling: mean`** no longer raises `AttributeError` (`.double()` was called on numpy arrays). Document means are now taken in float64 on all three routes. Shipped and study configs use `pooling: all_tokens`, so no shipped number moves.
- **`geometry_hubness`**: a repeated entry in `k_values` doubled every occurrence count for that k. Duplicates are now dropped in order and counted in `_meta_k_values_duplicates_dropped`. Non-positive k is rejected. The default `[10, 50, 100]` is unaffected.
- **`repe_refusal_direction`** returns an error when scikit-learn is missing instead of raising `NameError`, as `repe_concept_separability` already did.
