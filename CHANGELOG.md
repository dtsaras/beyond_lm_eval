# Changelog

## Unreleased

- **Two exploratory tasks** (BLME constructs from the 2026-11-21 research memo, section 4; certification `proxy-only`, tier `experimental`, not in the paper core). Every other task is unchanged: the edits outside the two new modules only register them.
  - `geometry_readout_dual_spectrum`: the Jeffreys dispersion `jeffreys_dispersion` = 1/2 E J(p_x, p_x') = tr Cov(z, p) of the next-token distributions, and the dual spectrum kappa = sqrt(eig(Cov(h) Cov(W^T p))) of the readout, exactly invariant to any invertible affine re-basing of the state the head reads: `dual_coupling` (sum kappa), `dual_coherence`, `dual_effective_rank` (and `_frac`), `dual_tail_exponent`. Word-final positions (from character offsets), sinks excluded and removed exactly from the streamed float64 moments, N pinned at 32 d rows; every position is `_meta_`. On the memo prototype's own protocol it reproduces the prototype's D_J, sum kappa, coherence, effective rank and decay exponent to under 1e-6 relative on gpt2 and pythia-160m-deduped.
  - `dynamics_context_influence`: I(k), the Jensen-Shannon change of the next-token distribution when the token k positions back is resampled from the corpus unigram, scored only at the last 64 positions of 1024-token windows so every term has nearly full context and only the lag varies (base lags log-uniform on [1, 945], 16 replacements per window, lags to 1008; every lag weighs the same within its bin); `influence_decay_exponent`, `influence_far_mass` (bins from lag 9) and `influence_horizon_bytes`; the binned and per-lag profiles (lags in tokens and bytes), term counts and bootstrap SEs are `_meta_`. Forwards in fp32 arithmetic at every size (`compute_dtype`): bf16 forwards raised gpt2's far mass 20x. Every random stream derives from `seed`. `_meta_causality_ok` flags a divergence before the replaced position above 1e-6 (recorded, never an error). Scoring every position after the replaced token, as the memo's prototype did, mixes lag with position: on a model whose response depends on position only it reports a decay exponent of 1.14, against 0.04 here.
  - **Decisions:** `docs/validation/decisions.yaml` (six for each task). **Tests:** `tests/tasks/test_readout_dual_spectrum.py`, `tests/tasks/test_context_influence.py`, `tests/tasks/test_exploratory_task_guards.py` (the declaration, unusable-cache and offload guards applied to the two tasks, and `blme.evaluate` on tiny-random-gpt2 and gpt2). Receipts pending.

- **Docs:** receipt `2026-11-20-gpu-linalg` (source e5dc33e, 967 linked cases on transformers 5.2.0, 5.10.2, 5.22.2 and 5.23.0, 0 failures) now backs every row of `docs/validation/metrics.json`. New decision entry: `dynamics_sharpness.top_eigenvalue_estimate` is PyHessian's signed eigenvalue of largest magnitude, which is negative on gpt2-355m, phi-2 and phi-4 (an isolated negative eigenvalue of the last-block Hessian outweighs lambda_max); a cross-model ranking of sharpness should read its magnitude.

## 0.2.5 — 2026-11-21

Two defects found by an independent review of the study sweep: Gemma windows without BOS in strided perplexity, and Gemma-4-12B sampling its banned modality tokens in the generation tasks.

- **Strided windows without BOS.** `strided_perplexity` (the HuggingFace recipe: one joined stream, 1024-token windows, stride 512) gave the tokenizer's BOS only to the stream's first window.
  - **Why it matters:** a model that always sees its BOS first is off distribution in every later window.
  - **Measured on the sweep:** corpus perplexity exceeded per-document perplexity by 2.7-37x on gemma2-2b/9b/27b, gemma3-12b/27b and gemma4-e2b/e4b. On gemma-2-9b, `ppl_overall` was 493 against 13.2 per document, and `bits_per_byte` 1.93. On the other 59 models the ratio was 0.67-1.03.
  - **Fix:** with `ppl_window_bos: auto` (the new default), when the tokenizer prepends a BOS, every window is that BOS followed by `max_length - 1` content tokens, sliding over the content by `min(stride, max_length - 1)`. The BOS is context only. The first window is unchanged, every content token is scored exactly once, and byte attribution is unchanged. `never` restores the old behaviour.
  - **Affected:** `ppl_overall`, `mean_nll_nats`, `bits_per_byte` and `bits_per_char` on every model whose tokenizer prepends a BOS (Gemma, Llama, Mistral, Yi, TinyLlama, ...). Tokenizers without a BOS (GPT-2, Pythia, Qwen, ...) give bit-identical results. The per-document keys are unchanged.
  - **Recorded:** `_meta_ppl_window_bos`.
- **Generation ignored the model's `suppress_tokens`.** `_no_model_generation_defaults` unsets every decoding default a model ships (temperature, top_k, repetition_penalty, ...), so all models sample under the task's own settings. It also unset `suppress_tokens` and `begin_suppress_tokens`, which say which tokens are valid output at all.
  - **Effect:** Gemma-4-12B ships `suppress_tokens` for `<image|>` and `<audio|>`. At the study's settings (temperature 0.8, top_k 0, 32 tokens) it sampled `<image|>` in 39 of 64 completions, 27% of all sampled tokens. The completion filter strips them, leaving fragmented, repetitive text: `mean_phrase_repetition_rate` 0.158 against a population median of 0.0055. With the suppression kept, it sampled none.
  - **Fix:** `generation_diversity` and `coe` pass `keep_validity=True`, so the validity fields stay applied, and each run records `_meta_generation_validity_config`. `entropy_calibration` and `self_consistency` measure the model's full sampling distribution, so they keep the default (`keep_validity=False`, everything unset) and record the fields as ignored, as before.
  - **Scope:** of the 66 study checkpoints only gemma-4-12B sets either field, so every other model is unchanged.
- **Stale guard:** `test_round5_determinism.py::test_repe_steering_hook_casts_vector_to_hidden_dtype` looked for the literal `(alpha * vec).to(out_t.dtype)`. Since 0.2.2 the cast is `.to(out_t.device, out_t.dtype)`, so the test had failed outside every receipt's linked selectors. The code was correct; the guard now matches any `.to(...)` that casts to `out_t.dtype`.
- **Tests:** `tests/tasks/test_perplexity_window_bos.py`. Every window starts with BOS; values match an independent window loop on raw logits; no token is skipped when the stride equals the window; tokenizers without a BOS are unchanged. All 8 fail on 0.2.4. `tests/tasks/test_generation_diversity_stop_ids.py::test_model_suppress_tokens_still_apply`: suppressed tokens are never sampled, and other defaults are still ignored. It fails on 0.2.4.
- **Receipts:** `2026-11-21-v025` covers all 109 tasks, CPU only, on four Transformers stacks: 1001 cases per stack, 0 failures, 8 skips (the GPU tests). Ten files did not pass in the parallel run and passed on a serial rerun. `2026-11-21-cka-gpu-v025` repeats `geometry_cka`'s GPU-path cases (20 per stack, all passed). Both pin the study at 1d94455, with fractal at 1000 documents.

## 0.2.4 — 2026-11-21

Parallel item sets, for measuring how much a result depends on the texts it reads. Item set 0 (the default) is unchanged: every value of a 0.2.3 run is reproduced.

- **`BLME_ITEM_SET`** (`blme.cache.item_set`, 0 or 1; anything else raises). Item set 1 draws every corpus by the same process from a disjoint pool of the same size:
  - the default corpus (`load_default_corpus`) reads WikiText-103 test passages where set 0 reads validation passages: same filter, stripping and nested order. `default_corpus_size()` is set 0's size (1,691) in either set, so tasks that read the whole corpus read the same count;
  - `load_long_documents` (fractal, PHD) returns the next `n` qualifying articles of the same order;
  - `geometry_mahalanobis`' extended corpus walks test, then train, skipping every passage of set 0's draw;
  - `geometry_representation_dispersion`'s release-protocol rows come from the test split.
- **Provenance:** results record `environment.item_set`; fractal's `_meta_document_source` and PHD's `_meta_phd_documents` end in `:item1` in set 1. Tasks with bundled items (categories, RepE prompts, CounterFact) read the same items in both sets.
- **Tests:** `tests/test_item_sets.py`, on synthetic splits.
- **Receipts:** `2026-11-21-v024` covers all 109 tasks, CPU only, on four Transformers stacks: 979 cases per stack, 0 failures, 8 skips (the GPU tests). Eleven files hit FUSE I/O errors at start-up in the parallel run and passed on a serial rerun. `2026-11-21-cka-gpu` repeats `geometry_cka`'s GPU-path cases on 0.2.4, because `cache.py`, `results.py` and `_version.py` are in its import closure: 20 cases per stack, all passed.

## 0.2.3 — 2026-11-20

One precision defect in `geometry_cka`, found by the study sweep: `_meta_max_adjacent_cka` exceeded 1 (up to 1.0095) on 9 of 61 models, all 12–34B.

- **Biased linear CKA on the GPU** (0.2.1–0.2.2). `evaluate()` caches each layer's ||X^T X||_F and passes it to `_linear_cka`. Since 0.2.1 the GPU path computes the numerator ||Y^T X||_F^2 in float64, but the cached norms were still computed in float32 on the CPU. Their error is almost all in the float32 Frobenius reduction, about 3e-5 relative on a 7168 × 7168 Gram of random data and up to about 5e-3 on real activations. Every biased CKA value carried the product of its two layers' norm errors, so CKA(X, X) came out up to 1e-2 above 1, growing with width.
  - **Fix:** the norms are computed on the linear-algebra device with the numerator's arithmetic.
  - **Affected keys:** `early_late_cka`, `min_adjacent_cka`, `min_offdiag_cka` and their `_meta_` summaries. The debiased twins were float64 throughout and are unchanged; the CPU path is unchanged.
  - **New `_meta_max_abs_self_cka_minus_one`:** the matrix diagonal, which passes through the cached norms; about 1e-16 on the GPU and 0 on the CPU.
  - **Why the tests missed it:** `tests/test_gpu_linalg.py` called `_linear_cka` without cached norms, and the receipts hid CUDA. The new GPU tests drive `evaluate()`.
- **Receipts:** `2026-11-20-v023` covers all 109 tasks, CPU only, on four Transformers stacks. It ran 979 cases per stack: 0 failures, and 8 skips, the GPU tests. `2026-11-20-cka-gpu` covers `geometry_cka` with a GPU visible, so its `evaluate()` parity cases take the study's path: 20 cases per stack, all passed.

## 0.2.2 — 2026-11-20

Failure paths only: every value a 0.2.1 run produced without an error or a fallback is unchanged. Found by the study sweep's first models split across two GPUs and by MPS-packed GPUs running out of memory.

- **Multi-GPU device maps.** `geometry_hubness` compared the input and output embeddings with `torch.equal` across two cards, and `repe_steering_effectiveness` added the steering vector on the first card to a block output on another. Both failed outright on every model split over two GPUs; both now work on the block's or the table's own device.
- **`safe_svdvals` on a matrix already on the GPU** (a weight) now recomputes on the CPU after any CUDA failure, as CPU inputs already did. Before, an out-of-memory on a shared card raised, and `geometry_spectral` left that module (the lm_head, on 4 sweep models) out of its averages, with only a `_meta_` count and a warning to show it.
- **`eigvalsh_f64`** (new, `geometry/utils.py`): float64 eigenvalues with the same fallbacks. cuSOLVER 11 (CUDA 12.x) rejects syevd from n = 27,392 (`CUSOLVER_STATUS_INVALID_VALUE`; 25,600 works), the FFN width of six 24–34B models, so `geometry_ffn_spectral_utilization` failed on them. A CUDA failure now goes to the command in `BLME_EIGVALSH_HELPER` (a newer cuSOLVER in another environment: matrix and eigenvalues exchanged as `.npy`), and otherwise to LAPACK on the CPU. `_meta_ffn_eigensolver_counts` records which solver ran. cuSOLVER 11 and 12 agree to 1e-16 relative at n = 25,600, where both run.

## 0.2.1 — 2026-11-20

Performance and provenance; the study sweep's GPUs sat idle while single-threaded CPU code ran.

- **Large linear algebra runs on the GPU when one is available.** `safe_svdvals`, and the new `gpu_svdvals` / `gpu_svd` / `gpu_eigh`, move float64 SVDs and eigendecompositions of large CPU matrices to CUDA. This covers the SVD, collapse, effective-rank, Schatten, matrix-entropy, Procrustes, unembedding and spectral paths. CKA's layer-pair GEMMs, and the kNN loops of `dynamics_stability` and the unembedding purity diagnostic, run on the GPU too, batched through `knn_sorted_batched` (float64 direct chord distances, stable index tie-break).
  - **Numerics:** the same quantities in the same precision, with only the reduction order different. A CUDA out-of-memory falls back to the CPU, and `BLME_CPU_LINALG=1` forces the CPU path, which is the one every test, parity fixture and blind review ran.
  - **Stability:** the GPU path computes distances in float64 where the CPU loop used float32, so neighbour sets can differ where two candidates tie within float32 rounding at rank k. Qwen3-1.7B: 295 of 300 queries identical, `stability_mean` 8e-5 apart; gpt2 and gemma-3: identical.
  - **Speed:** 280–360× on stability; `_meta_knn_device` and `_meta_linalg_device` record the path taken.
- **Provenance:** `blme_version` is now the imported source tree's version (`blme/_version.py`), not installed package metadata, which was stale when the source came from an exported tree. `git_hash` is taken in the package's own directory, not the caller's, and is None for an export; the envelope also records `blme_source`.
- **Docs:** the conventions blind review E had to infer (predictive information's split rule, RankMe's eigenvalue convention, delta-hyperbolicity's prompts and rows, BOS rows in the FFN and delta keys, and others).

## 0.2.0 — 2026-11-19

The library version the study's structural sweep runs (source 998d8aa). Every task has a numerical receipt (`docs/validation/receipt-2026-11-19-sweep-config.json`); the 53 study tasks have also had a blind end-to-end review (`docs/validation/independent_reviews.yaml`); `docs/METRICS.md` documents each task.

### Verification campaign and protocol decisions (2026-11-19)

- **Fix:** `safe_norm` sums squares in float64. A float32 `norm()` of a parameter-sized vector is 4e-4 off at 7M coordinates and 3e-3 at 50M, so `sam_sharpness` and `asam_sharpness` were 0.1–0.15% off on gpt2's last block and the error grew with model size (blind review D). **Values change** for `dynamics_sharpness` (SAM, ASAM, the power iteration's normalisation, the critical step's start) and slightly for `dynamics_gradient_flow`.
- `dynamics_sharpness`: `asam_variant` chooses the ASAM normaliser, `invariant` (default, unchanged) or `release` (the released asam.py rule); the other is recorded as `_meta_asam_sharpness_<other>`.
- Docs from blind review D: truncation lengths (Mahalanobis and contextualization 128 tokens, sharpness 64), what `effective_depth_90` accumulates, contextualization's layer selection, Procrustes' chain end and slope axis, the position-0 rule of trajectory curvature, sharpness's batch and objective, generation diversity's units, fractal's lag grids, concept separability's folds and seed noise, refusal's prompt sets, task vectors' layers, landscape seed noise, zigzag's subsets, topology's layer indices, calibration's line fit.

- Design choices that the evidence leaves open are parameters, with the reference protocol as the default. Where the alternative is cheap, every run also records it as `_meta_`. The choices are in `docs/validation/decisions.yaml`, and `docs/METRICS.md` documents every task (generated by `scripts/metrics_doc.py`).
  - Interpolation scores every distinct pair by default (`num_pairs: null`), removing its ~7% pair-sampling noise. **Values change.**
  - The Betti curve reads β₁ at `hop_radius` (2, the released read); 1 hop is recorded.
  - Collapse reads the valley over `qdl_layers` (all blocks); the interior variant is recorded.
  - CoE ends its chain at `chain_end` (`final_norm`, the release); the block-output variant is recorded.
  - Sharpness stops Hutchinson by `hutchinson_stop` (`pyhessian`); `rse` stops at a relative-SE target.
  - Intrinsic dimension defaults to layerwise mode, whose `lid_*` keys are its headline. The static mode had none.
  - LID defaults to layerwise mode too. Before, `lid_mean` read the post-norm output by default but the deepest raw block output in layerwise mode (gpt2: 7.62 vs 21.75), so the key changed meaning with the flag. **Values change** for runs that relied on the old default.
  - Intrinsic dimension's `_meta_` token cloud runs only when CUDA is available (`token_cloud: auto`). On the CPU ball tree it exceeded `task_timeout`, which discarded the headlines.
  - Hubness can drop near-duplicate rows of its draw (`drop_near_duplicates`, default off), i.e. untrained in-vocabulary rows such as Llama-3's reserved tokens or Qwen3's never-emitted byte tokens; their count and the other choice's values are recorded. Superposition has the same switch, opt-in.
  - One-point-per-sequence clouds count windows whose last token carries a massive activation (`_meta_window_n_massive_points`); `massive_points: drop` removes them at every depth.
- `repe_steering_effectiveness`: `directional_consistency`, `directional_success_rate` and `steering_sign_reversal` become `_meta_` (same values, `_meta_` names). They summarise projections onto the steering direction and sit near 1 by construction, so they do not show whether steering works.
- Docs: magnitude's headline uses magnipy's 30-scale grid and cosine distance, not `n_scales`; CKNNA's outlier clamp, what the HSIC headlines are, and spectral's module set are now stated.

- Mahalanobis OOD keeps whole articles on one side of each fit/held-out split. The passage-level split held out WikiText passages whose sibling passages from the same article were in the Gaussian fit. On eight checkpoints, keeping articles whole raised the in-distribution distance per dimension by 40–93% and lowered AUROC by 3–19%. Phi-2 fell from 0.924 to 0.746, below GPT-2, so the leak changed model rankings. The extended corpus rows carry `article_id`. `_meta_split_unit`, `_meta_n_split_groups` and `_meta_split_groups_shared_max` record the split. Rows without source IDs keep the earlier seeded permutation exactly.
- Mahalanobis OOD no longer drops a split when LAPACK's MRRR eigensolver fails on the shrunk covariance. At n_fit < d that covariance has d − n_fit equal eigenvalues, and SmolLM3-3B raised `LinAlgError("Internal Error.")`. The same covariance is inverted with the divide-and-conquer solver. `_meta_n_precision_fallbacks` and `_meta_n_failed_splits` are derived. Fits that succeed are unchanged.
- Weight spectral analysis counts modules whose analysis raised in `_meta_n_modules_failed`. Before, they were left out of the averages with only an INFO log.
- The results envelope records the attention kernel the tasks ran with in `config.attn_implementation`. On Gemma-4-E2B, bf16 eager and sdpa differ, with mean NLL 2.2434 against 2.2407. Transformers 5.10.2 and 5.22+ choose different defaults, and `evaluate()` switches the model to eager when a co-run task needs attention weights.
- Test fixes for a fresh install with the newest packages allowed by the pins:
  - pytest 9.2 rejects `approx` on nested values;
  - the GUDHI-dependent topology lifts are skipped without GUDHI;
  - the magnitude cache test's documents no longer share a 20-word window. The duplicate point made magnipy's convergence target unreachable on every stack.

- Strided bits per byte and character can recover the exact denominator when every token after an inserted BOS is scored and its decoded suffix exactly reproduces the source text.

- Representation sensitivity differentiates a declared final tanh softcap and checks its reconstructed logits against the model. Gemma-4 perturbation growth uses the original token IDs for per-layer embeddings during both forwards.
- The held-out Procrustes readout can run separately and can use a float64 Gram eigensolver on CPU or CUDA. The archive backfill records its cutoff, rank, input articles and solver. The ordinary task still returns both the in-sample and document-held-out scores.
- Positional attention decay ranks causal rows in batches, with average ranks for ties. The exhaustive example recipe includes a labeled panel for domain bits per byte and paraphrase LiDAR.

- Document-held-out Procrustes R² now selects a truncated-SVD cutoff on an inner document split, with a numerical floor of `sqrt(eps_float32)`, then refits on all outer training documents. Results include both splits and each layer pair’s selected cutoff and rank. Previously, the float64 solver could amplify directions below the resolution of the cached activations.
- Generation-diversity sampling entropy now normalizes and reduces one step of processed logits at a time in float64. The first-quarter, last-quarter and change-in-entropy readouts use the original score positions for content tokens when a completion contains padding.

### Earlier 0.2.0 work

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
- `geometry_procrustes_linearity`: all 17 keys are fitted without position 0 and the massive-activation rows (token policy `exclude_sinks`, one mask per token shared by every layer, as `geometry_cka`). Position-0 rows hold 94% of the centred sum of squares on gpt2 and set every key: `procrustes_linearity_mean` is 0.978 with them and 0.919 without. The every-position values of the released notebook are `_meta_incl_pos0_procrustes_linearity_*`.
- `topology_zigzag_persistence`: each point is the last token of a 20-word window of a passage, as in the other topology tasks. It used to be the last token of the whole passage. `zz_true_adjacent_mean` and `zz_true_mean_bar_len` change; `points_readout: passage` restores whole passages. Sites, tokenisation without special tokens and the barcode are unchanged.
- `geometry_unembedding`: `unembedding_eff_rank` and `unembedding_eff_rank_centered` read only the rows the tokenizer can emit (ids below `len(tokenizer)`). Models whose unembedding has padded rows change; the all-rows values are `_meta_unembedding_eff_rank_all_rows` and `_meta_unembedding_eff_rank_centered_all_rows`, beside `_meta_n_unused_rows`.
- `geometry_schatten`: every model scores the first 256 tokens of the long WikiText-103 train articles `geometry_positional_decay` reads (`seq_tokens`; `documents: dataset` scores the run's passages in the same window). It used to score each full passage, so the token count followed the passage and the tokenizer. Every key changes. `_meta_schatten_seq_tokens` and `_meta_schatten_documents` replace `_meta_schatten_max_length`.
- Long articles are selected by whitespace-word count (`load_long_documents(min_words=)`), so every tokenizer gets the same articles:
  - `geometry_phd_dimension` keeps articles of ≥ 512 words (it kept ≥ 300 tokens);
  - `dynamics_fractal` keeps articles of ≥ 4096 words (it kept ≥ 4096 tokens);
  - `geometry_positional_decay` keeps articles of ≥ `seq_tokens` + 1 words.
- **An appended EOS is stripped wherever a task tokenises its own input**, as the shared cache already did. The change affects tokenizers that append EOS (OLMo). Keys:
  - `convexity_gap` (interpolation);
  - `growth_log_ratio_relative` (perturbation_growth);
  - `gradient_flow_log_ratio_in_out`;
  - `S_unit` and `_meta_representation_sensitivity_raw`;
  - `positional_decay_nosink_write_weighted`;
  - `effective_depth_90`, `total_block_influence` and `max_block_influence`;
  - `prediction_alignment_mean` and `prediction_alignment_std` (fallback route).
  - `dynamics_generation_diversity`, `dynamics_coe`, `dynamics_sharpness`, `geometry_neural_collapse` (both routes) and `geometry_cknna` (fallback route) also strip it. Each records `_meta_n_trailing_special_stripped`.
- `geometry_categories`: both words of a relation pair use one token form, so `{relation}_consistency` changes.
- `geometry_collapse`: `qdl_entropy_min` reads the first 64 tokens of every passage. The whole-passage value is `_meta_qdl_entropy_min_whole_passage`.
- `geometry_positional_decay`: `positional_decay_nosink_write_weighted` scores the first 256 content tokens of long WikiText-103 train articles (`documents: dataset` keeps the run's passages).
- Long articles for `geometry_schatten` and `geometry_positional_decay` come from the train split (`doc_split: train`, 28,831 articles of at least 257 words), as for `geometry_phd_dimension` and `dynamics_fractal`. The document-level validation split holds 60, so any `num_samples` above 60 used to return an error; the study's `geometry_schatten` (200 articles) failed on every model. Selection is otherwise unchanged:
  - the articles are in nested order, so the k-article draw is a prefix of the n-article draw;
  - every tokenizer gets the same list;
  - a draw comes from one split only, and a split that is too short is an error that names it.
  
  The split used is `_meta_schatten_doc_split` / `_meta_doc_split`. `doc_split: validation` reproduces the earlier articles for N ≤ 60.
  - Qwen3.5's linear-attention blocks are skipped, so the key is no longer NaN on those models.
  - An output projection called with a keyword input is captured.
- `dynamics_perturbation_growth`: the headline is now the relative growth `growth_log_ratio_relative` = log(r_L / r_0); see the renames below.
- `geometry_perplexity`: a special token the tokenizer adds owns no bytes, so `bits_per_byte` is finite on Llama-3 tokenizers (it was NaN).
- `dynamics_generation_diversity`: completions are cut at the first of every stop id, not only the tokenizer's EOS. This changes `distinct_*`, `mean_distinct_*`, `mean_self_bleu`, `mean_phrase_repetition_rate` and the sampling-entropy keys.
- `dynamics_generation_diversity` and `dynamics_coe` decode under the task's own settings only. On transformers 5.x, `generate()` fills unset fields from `model.generation_config`, so a model that ships decoding defaults (`repetition_penalty`, `no_repeat_ngram_size`, `suppress_tokens`, ...) used to change both tasks' samples. Those fields are now unset for the call and restored afterwards; `_meta_ignored_generation_config` lists them.
- `dynamics_entropy_calibration` sampled under the same leaked decoding defaults (tiny gpt2 with a shipped `suppress_tokens` / `repetition_penalty` / `no_repeat_ngram_size`: EntCE −0.0025290 instead of −0.0025465). It now isolates the call in the same way and reports `_meta_ignored_generation_config`. Values change only for models that ship such fields.
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
  - the head-linearity guard fails closed, so `S_unit` and `_meta_representation_sensitivity_raw` are NaN, with a reason, when the guard cannot run.
- `geometry_schatten`: blocks with routed experts get NaN with a reason, and `_meta_schatten_1_mlp` / `_meta_schatten_4_mlp` average the remaining blocks. Dense models are unchanged.
- `geometry_predictive_information`: special tokens are never targets. This changes `predictive_info_*` on EOS-appending tokenizers.
- `geometry_alpha_req`: non-finite rows are masked and counted, and huge finite rows no longer overflow, which affects `alpha_req` and `rankme`. Finite clouds are unchanged.
- `dynamics_stability`: the noise is drawn on the CPU, so `stability_mean` from a GPU run equals the CPU value.
- Phi-1/1.5/2: the final norm at `model.final_layernorm` is found. Phi results of logit_lens, logit_lens_kl, unembedding, representation_sensitivity, gradient_flow, zigzag_persistence and bias_weat change.
- Gemma 3n / 4: the final logit softcap is read from the text config. This changes the tasks that decode through `common.apply_lm_head`: `interpretability_logit_lens`, `interpretability_logit_lens_kl` and `convexity_gap`.
- `_meta_` only: `_meta_gride_peak_layer_frac` uses the hidden-state index / L, as the other peak depths do.
- `_meta_` only: `_meta_correlation_dimension_under_resolved` compares the reported fixed-window `correlation_dimension` with the Eckmann-Ruelle bound 2 log10 N. It used to compare the demoted plateau fit.
- `geometry_lm_head_free_energy`: the float32 Gram and `eigvalsh` run on the CPU wherever the weights are, so a model on a GPU gets the CPU value. gpt2 in bfloat16 on an RTX 3090 returned `lm_head_bfe_w = NaN`: the float32 rounding of WᵀW (0.026) exceeds its smallest eigenvalue (0.020), and cuSOLVER returned that eigenvalue non-positive.
  - When the float32 spectrum has a non-positive eigenvalue but the float64 spectrum is positive beyond its rounding, the headline is the float64 value and `_meta_lm_head_bfe_w_from_float64` is True.
  - Every NaN headline carries `_meta_lm_head_bfe_w_nan_reason`. `_meta_lm_head_min_eigenvalue_float32` is new.
  - Values that were finite on the CPU are unchanged, bit for bit.
- **A task that fails while parameters are promoted no longer corrupts the rest of the run.** On 24 GB GPUs, gemma-4-E2B and Yi-1.5-34B in bf16 ran out of memory inside `fp32_compute` (gradient_flow, interpolation, perturbation_growth, sharpness). A cast back to bf16 in its `finally` then ran out of memory too, because the traceback still held the block's tensors. The loop stopped with parameters left in float32, and every later task failed with `expected mat1 and mat2 to have the same dtype` (29 tasks on gemma-4-E2B, 13 on Yi).
  - `fp32_compute` now keeps restoring after a failed cast and retries it through host memory.
  - `evaluate()` restores any parameter dtype a task changed before the next task runs, and appends the count to that task's error (or `_meta_param_dtypes_restored`).
- Crash only: `topology_persistence_landscape` returns an error, as `topology_betti_curve` and `topology_persistence_entropy` do, when no passage holds a 20-word window. It used to raise `ValueError`.

### Key renames and demotions to `_meta_`
- `dynamics_perturbation_growth`: `growth_log_ratio` → `_meta_growth_log_ratio_absolute`; the new headline is `growth_log_ratio_relative`.
- `geometry_unembedding`: `embedding_alignment_mean`, `embedding_alignment_std` and `embedding_high_alignment_frac` move to `_meta_`.
- `dynamics_stability`: `stability_std` moves to `_meta_`.
- `geometry_mahalanobis`: `mean_mahalanobis_id`, `mean_mahalanobis_ood` and `ood_separation_gap` move to `_meta_`. Their per-dimension twins stay headline.
- `topology_betti_curve`: the headline is `betti_1_curve_mean`; `max_betti_1` moves to `_meta_`.
- `geometry_rsa`: `_meta_n_degenerate_rows_jittered` → `_meta_n_constant_rows_dropped_corrdist`.
- `geometry_procrustes_linearity`: `_meta_excl_pos0_procrustes_linearity_mean` and `_meta_excl_pos0_procrustes_linearity_adjusted_mean` are removed (the headline now excludes position 0), and the `meta_excl_pos0` option becomes `meta_incl_pos0`.
- `geometry_schatten`: `schatten_1` → `_meta_schatten_1_mlp` and `schatten_4` → `_meta_schatten_4_mlp`. They are in the MLP activation's units and grow with d_ff. The headline is `rankme_last`, `row_normalized_rankme_last`, `row_normalized_matrix_nuclear_norm_last` and `row_normalized_schatten_inf_last`.
- `topology_homology`: the raw lifetimes `{first,mid,last}_layer_mean_persistence_h0`, `*_max_persistence_h0` and `*_mean_persistence_h1` move to `_meta_{first,mid,last}_layer_*`. The headline is their `_norm` twins and `*_num_loops_h1`.
- `geometry_representation_sensitivity`: `representation_sensitivity` → `_meta_representation_sensitivity_raw` and `sensitivity_std` → `_meta_representation_sensitivity_raw_std`. The headline is `S_unit`.
- `geometry_intrinsic_dim`: `intrinsic_dimension` leaves HEADLINE_KEYS; the headline is `lid_first_layer`, `lid_mid_layer` and `lid_last_layer` (`layerwise: true`). The library default is still static mode (`layerwise: false`), which emits only `intrinsic_dimension`, so a default run has no headline key for this task.
- `topology_persistence_landscape`: the raw-unit H1 summaries `{first,mid,last}_layer_h1_mean_landscape_integral`, `*_landscape_max_1`, `*_landscape_norm_1` and `*_landscape_integrals` move to `_meta_{first,mid,last}_layer_h1_*`. The headline is the nine `_mpd` twins.
- `dynamics_coe`: `mean_magnitude_change` → `_meta_mean_magnitude_change` and `std_magnitude_change` → `_meta_std_magnitude_change` (the representation's units).
- `geometry_neural_collapse`: `nc1_cdnv` → `_meta_nc1_cdnv` (it scales as 1/c² with the feature scale). The headline is `nc1_cdnv_galanti`.

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
- `dynamics_coe` and `dynamics_generation_diversity` set `max_new_tokens` on the `GenerationConfig` (per call in coe) instead of passing it next to it, which transformers deprecates. No value changes. The generation-config isolation helpers live in `blme.tasks.common`.
- `blme.experts`: on transformers 5.10, routed-expert (MoE) models call `torch._grouped_mm`, which torch 2.8 lacks on CPU. A model loaded or evaluated on CPU without that kernel is switched to the eager experts implementation; the switch is recorded as `config.experts_implementation_fallback` in the result envelope. GPU runs and dense models are unchanged.
- `dynamics_perturbation_growth`: the docs state that `growth_log_ratio_relative` is a BLME formulation, not Li et al.'s quasi-Lyapunov exponent (their Eq. 9 has no state-norm normalisation and a 1/(n - m) factor); `_meta_growth_rate_per_block` is the closest analogue.

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
