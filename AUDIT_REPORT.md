# BLME Library Audit — Correctness Review

**Date:** 2026-04-17
**Scope:** `src/blme/**` correctness vs. paper claims; root causes of NaN/constant/failed measurements in `results/study_v1/aggregated.csv` (32 models × ~1250 columns).

**Update 2026-04-17 (round 1):** bugs #1–#5 fixed with regression tests.

**Update 2026-04-17 (round 2):** HIGH items #6, #7, #8, #9, #10, #12, #15, #16, #17, #18 fixed.

**Update 2026-04-19 (round 6, final independent sweep):** an independent final-pass agent found four additional bugs that the round-5 pass missed. All fixed with regression tests and the two study-affecting tasks re-patched across all 32 models:

* **#57 double-norm on `hidden_states[-1]`** (`consistency/bias.py:113`, `geometry/neural_collapse.py:216`) — both files applied ``final_norm`` to ``outputs.hidden_states[-1]``. But the round-4 logit_lens fix (#48) established empirically that ``hidden_states[-1]`` is *already* post-final-norm in transformers 5.x (``lm_head(hidden_states[-1]) == outputs.logits`` to within FP noise). The extra norm double-normalised the features, corrupting WEAT d-values and NC1 on every RMSNorm architecture (Llama, Qwen, Gemma). Post-fix: NC1 ``nc1_within_class_collapse`` range dropped from [0, 8.23] (spuriously large on tied-head GPT-2s) to [0, 1.09] — consistent with Papyan et al. 2020 expectations for un-collapsed representations.
* **#58 hook leak in `interpretability/attention_polysemanticity.py:80-114`** — forward hooks registered at line 80 and removed via an unguarded ``for h in handles: h.remove()`` at line 113. A tokenizer or forward failure in the sample loop would leak hooks into subsequent tasks using the same model instance. Wrapped the body in try/finally so hook removal is guaranteed, and switched from deprecated ``torch.svd`` to ``torch.linalg.svdvals`` while we were there.
* **#59 `topology/betti_curve.py:150` un-normalised slope** — ``betti_0_decay_rate`` was ``np.polyfit(np.arange(num_layers), betti_0_curve, 1)[0]``. Same cross-model-comparability bug that round-4 fixed in ``dynamics/gradient_flow`` (#45); regressing on normalised depth ``x / (n_layers - 1)`` makes the slope per-unit-depth and comparable across 12-layer and 80-layer models. Post-fix: slopes are in [-2.22, 0.73], mostly negative (topological simplification with depth, per Naitzat et al. 2020).
* **#60 `dynamics/generation_diversity.py:191` log-underflow** — replicated the ``softmax(x) + log(probs.clamp(min=1e-12))`` pattern that round-4 prediction_entropy (#48) fixed to ``F.log_softmax(x)``. ``scores_stack.float()`` upstream puts things in fp32 so the bias is smaller than on bf16 logits, but the log-sum-exp form is strictly more accurate and consistent with the rest of the library.

**Update 2026-04-19 (round 5, previously-un-audited tasks):** a final targeted re-audit of the ~20 tasks that prior rounds hadn't touched surfaced a handful of determinism and cross-model-comparability bugs. All fixed with regression tests:

* **#49 WEAT word-position tokenisation bug** (`consistency/bias.py:67-77`) — the helper tokenised each target word standalone (``"John"``) then searched the templated sentence's ids for the standalone id. On BPE tokenisers (GPT-2, Llama, Qwen, Gemma) in-context word ids differ (the leading space changes the token), so the search almost always failed and silently fell back to ``pos = len(input_ids) - 2``. Every target word ended up attributed to the **same end-of-sentence position**, giving near-identical hidden states for all names and WEAT d-statistics ≈ 0 regardless of actual bias. Fixed via a new ``_find_word_token_position`` helper that uses ``return_offsets_mapping=True`` on the templated text.
* **#50 `causality_ablation.py:98`** — ``torch.randperm(dim)[:num_ablate]`` used the global RNG, so the ablation mask re-rolled every call. The degradation curve at each ``k_pct`` was therefore noisy across reruns. Fixed to seed with a deterministic ``torch.Generator`` keyed on ``(l_idx, k_pct)`` so the same feature coordinates are ablated every rerun — and the same across models at matched ``(layer, k_pct)``.
* **#51 `causality_edge_attribution.py:98`** — the corruption shuffle used an unseeded ``torch.randperm``. Same reproducibility bug as ``ablation``. Fixed to seed per-prompt. (The underlying off-manifold-shuffle design flaw is still there — a curated counterfactual pair dataset would be the proper fix — but at least the numbers are reproducible now.)
* **#52 `dynamics/trajectories.py:55-56`** — ``random.sample(samples, 2)`` called the global Python RNG module. Each run picked different pairs. Fixed to use a seeded ``random.Random(0)``.
* **#53 `interpretability/attention_polysemanticity.py:64-65`** — ``random.sample(target_modules, 4)`` was also unseeded, so each run sampled a different 4 layers out of the model. Fixed with a seeded ``random.Random(0)``.
* **#54 `interpretability/sae_features.py:81`** — ``target_layer = num_layers // 2`` ignored the SAE's trained hook point. On GPT-2 small the default SAE is trained on ``blocks.8.hook_resid_pre`` but the code applied it to the middle layer (index 6). The reported L0 counts therefore measured an un-trained-for hidden state. Fixed to parse the layer index out of ``sae_id`` (``blocks.8.*`` → 8) with the middle-layer behaviour as the fallback.
* **#55 `repe_steering_effectiveness` dtype mix** (`representation_engineering.py:338,342`) — the steering hook added a fp32 task vector to a bf16/fp16 hidden state via ``out_t[:, -1, :] += alpha * vec``. PyTorch silently up-casts the LHS to fp32, so the residual stream after the hook has a *different dtype* than it did in the unablated forward — every subsequent layer sees a wider tensor and the KL divergence is no longer comparing apples to apples. Fixed by casting ``(alpha * vec)`` to ``out_t.dtype`` before the in-place add.
* **#56 `topology/persistence_landscape.py:98,100`** — used the deprecated ``np.trapz``; now falls back through ``getattr(np, "trapezoid", np.trapz)`` so NumPy 2.x doesn't emit deprecation warnings and older NumPy still works.

**Update 2026-04-18 (round 4, aggregator + parallel audit):** post-aggregation bias fixes and an independent re-audit of the less-scrutinised tasks turned up nine more issues. All fixed with regression tests:

* **#36 aggregator depth bias** — five tasks (`interpretability_logit_lens`, `causality_tracing`, `geometry_contextualization.per_layer`, `geometry_matrix_entropy.layer_matrix_entropies`, `geometry_positional_decay.layer_positional_decay`, `interpretability_waa.layer_waa_alignments`) emit per-layer absolute-index columns. The old `_flatten_dict` flattened these to one column per absolute layer, so `layer_31` was only filled for models with ≥32 layers — introducing a strong size-correlated missingness pattern into PCA/Lasso. Rewrote `_flatten_dict` to regroup layer-indexed keys (including bare-integer `{"0": ..., "1": ...}` dicts and flat `layer_N_metric` top-level keys) into sorted lists and emit architecture-agnostic summaries (mean / std / slope / q25 / q50 / q75). Feature count dropped from 1251 to 642 on v2.
* **#37 `__deprecated_inverted` rename removed** — the old cache shift-by-one bug was fixed in round 1; the aggregator's kludge renaming `geometry_perplexity.*` columns is now obsolete and was misleading users into avoiding correct data. Removed.
* **#38 gemma4-e4b-it perplexity nulled** — same chat-template / tokenisation bug as calibration. Base `gemma4-e4b` reports `ppl_overall = 9.56`; the instruction-tuned variant reports `ppl_overall = 7311` (770× higher). One-model nullification keeps the metric from being dominated by this single outlier in Pearson aggregates.
* **#39 `cond_number` numerical-rank cap** — `geometry_svd.cond_number` was reporting 1.4×10⁸ for GPT-2 models because `S[0] / S[-1]` on a near-rank-deficient point cloud ends up dominated by the floating-point floor at S[-1]. Cap the denominator at `S[0] * max(shape) * eps` so the ratio reports the effective conditioning of the numerical-rank subspace. `numerical_rank` is now exposed so reviewers can see the cutoff. After patching all 32 models the max dropped from 1.4×10⁸ → 7.3×10³.
* **#40 `ppl_rare` threshold** — the old code thresholded the **full vocabulary** by argsort of token counts. With a ~12 k-token eval corpus in a 50–200 k-vocab model, ~90 % of ids have count 0, so the "bottom 20 %" set was dominated by never-seen ids; `cnt_rare` was 0 and `ppl_rare` returned `+inf` for every model. Fixed to threshold only **observed** tokens. Patched all 32 models.
* **#41 Mahalanobis in-sample bias** — `_compute_mahalanobis_distances(X_id, X_id)` was fitting the Gaussian on the same samples it was scoring. In-sample Mahalanobis is biased downward by leverage (≈ p / n), inflating `ood_separation_gap`. Fixed to use a held-out 50/50 split on the ID samples; exposes `n_id_fit` / `n_id_score` for reviewer transparency.
* **#42 LID non-determinism** — `_compute_lid_for_matrix` used `np.random.choice` without a fixed seed, so reruns produced different LID numbers. Switched to a seeded `default_rng(0)`.
* **#43 homology spelling** — output keys were `"persistance"` (misspelt). Downstream aggregation silently dropped the columns or produced double columns. Fixed to `"persistence"`.
* **#44 `topology_betti_curve` threshold** — the per-layer threshold was the median of H0 death values, which moves with each model's point-cloud geometry. β₀ across models was computed at different threshold values and therefore not comparable. Switched to the median of the cloud's pairwise-distance distribution — a geometry-invariant scale that makes β₀ cross-model comparable.
* **#45 `dynamics_gradient_flow.slope`** — regressing `log(‖grad_l‖)` on raw layer index `l` gives a slope that scales as 1 / n_layers; a 32-layer model and a 64-layer model with identical profiles report different slopes. Fixed to regress on normalised depth `l / (n_layers − 1)`.
* **#46 `interpretability_induction.causal_validation`** — was normalising `(top_drop − rand_drop)` by `baseline_acc`. Small models with baseline ≈ 0.5 got artificially inflated scores relative to large models with baseline ≈ 0.98. Fixed to a raw accuracy difference (naturally bounded in [−1, 1]).
* **#47 `attention_graph.gini` zero-guard** — the Gini formula divided by `cum_edges[-1]` without a zero-guard. On a fully-masked attention matrix the denominator is 0 and the result is ±inf, which poisoned the mean across heads / layers. Now guards the zero case explicitly.
* **#48 `prediction_entropy` numerical stability** — was computing entropy via `softmax` + `log(clamp(..., 1e-12))`. On bf16 / fp16 logits over a 150 k-vocab, low-probability tokens underflow to 0 before `log` and get silently replaced by the clamp floor, biasing the entropy sum. Fixed to compute `log_softmax(logits)` directly (log-sum-exp stable) and derive probs via `exp`.

**Update 2026-04-17 (round 3, paper-faithfulness audit):** re-audited the library against the actual papers via WebSearch + official reference code. Discovered the round-1/2 fixes were *functionally* correct but had several material deviations from the published definitions. All addressed:

* **#26 matrix_entropy** — was pooling tokens across every sample into one giant covariance; missing the per-row L2 normalisation from Def. 4.1 and the `/ log d` normalisation from Def. 4.3. Now computes entropy per sentence and averages; row-L2-normalises after centering; divides by `log d`. Emits the last-layer value as the headline metric (matching Wei et al.'s convention).
* **#27 tracing (ROME)** — was only sweeping 5 evenly-spaced layers and using one noise draw per sample. Now sweeps every layer by default and averages over `n_noise_samples=10` noise draws. Also emits `traced_layers` metadata.
* **#28 attention caching** — was caching attentions for 5 tasks when only `attention_entropy` actually reads the cached tensors. Now only `attention_entropy` triggers attention caching; the other 4 attention-consuming tasks still get the `attn_implementation="eager"` force-switch but don't pay the cache memory cost.
* **#29 orphaned test block** — a prior Edit had removed the `def` line of `test_unembedding_alignment_is_meaningful_when_tied`, so its assertions were executing *inside* `test_intrinsic_dim_never_below_one` under the wrong test name. Extracted to a proper test.
* **#30 CoE (Wang et al. 2025)** — now emits the paper's headline scores **CoE-R (Eq. 5)** and **CoE-C (Eq. 7)** plus the Eq. 3 normalised mean-mag / mean-ang. Hand-verified on an orthogonal 3-point toy chain (CoE-R = 0 exactly, CoE-C = 1 exactly). Documented our prompt-only last-token position as a known departure from Wang et al.'s "mean-pool over generated output tokens" choice.
* **#31 cache list-length sync** — `get_prediction_stats` could silently drop hidden entries while keeping logits/labels, breaking downstream `zip`. Fixed so all three lists are emitted in lock-step.
* **#32 Two-NN (Facco et al. 2017)** — removed the `max(1.0, d_est)` floor (paper does not floor; Fig. 3 explicitly shows `d < 1` as a diagnostic signal); removed bottom-1% trim (paper only trims top 10%); switched to `F = i/N` matching the paper and `scikit-dimension`.
* **#33 neural_collapse** — NC1 was numerically exploding (3×10⁷ for pythia-70m) because `np.linalg.pinv(Σ_B, rcond=1e-10)` inverted near-zero off-subspace eigenvalues in D ≫ K·n_per_class space. Fixed by projecting Σ_W onto Σ_B's top-(K−1) eigenvector subspace before inversion (Papyan et al. 2020 §2.3 formulation). Exposes `nc1_subspace_rank` so reviewers can verify the projection.
* **#34 tokenisation-boundary bugs** (format_robustness, icl_slope, logical, contrastive, paraphrase) — added a shared `score_continuation(model, tokenizer, prompt, answer)` helper in `tasks/common.py` that locates the prompt/answer split via `return_offsets_mapping=True` (fast tokenisers) or a prefix-length fallback (slow tokenisers). Replaces five ad-hoc, BPE-merge-sensitive slicing paths. Also rewrote `contrastive` to score only the target tokens given the shared prompt (paper's definition) and `paraphrase` to use the last-token state instead of mean-pool (BOS-attractor bias on Llama/Gemma).
* **#35 effective_rank convention** — three tasks (collapse, isotropy, unembedding) were computing Roy-Vetterli effective rank with `p = σ/Σσ`; the paper convention is `p = σ²/Σσ²`. Added `geometry/utils.py::effective_rank(S)` helper and routed all three callers through it.

Plus two hygiene items from the code-review agent: WAA's `torch.randperm` now uses a seeded `torch.Generator` so reruns are deterministic; `refusal_direction`'s depth-quantile interpolation now normalises against the model's actual layer count and emits NaN when the requested depth is outside the surviving range.

**Remaining HIGH items not yet addressed**: #11 edge_attribution (token-shuffle corruption is off-manifold; positional mismatch), #19 position_sensitivity (measures paraphrase NLL with 60-80-word contexts, not "lost-in-the-middle" retrieval). Both are larger conceptual changes.

## TL;DR

The library works end-to-end, but a non-trivial fraction of the published numbers are either measuring something different from the named metric, silently computed over a degenerate input, or architecture-dependent in a way that invalidates cross-model comparisons. Five bugs by themselves explain most of the suspicious values and most of the “failed to run” cells. The other findings are smaller but will bias correlation/PCA analyses.

**Confidence legend:** **[V]** verified against the code and/or results CSV; **[H]** high confidence from code inspection, not end-to-end reproduced; **[M]** plausible, needs a unit test to confirm.

---

## CRITICAL bugs (most of the “suspicious” numbers come from these)

### 1. Hidden-state cache flattens across samples → per-sample metrics get a cross-sample token cloud  **[V]**
**`src/blme/cache.py:272-296`**

The cache concatenates every sample’s `(T_i, D)` hidden states into a single `(ΣT_i, D)` tensor per layer. Many consumers treat the resulting rows as *samples*, e.g. `geometry_cka`, `geometry_rsa`, `geometry_lid`, `geometry_matrix_entropy`, `geometry_mahalanobis`. The per-sample axis is lost. `_sample_lengths` is stored but not applied in `get_hidden_states`.

**Consequence:**
- CKA is computed over a pile of tokens drawn from many documents rather than over matched per-sample representations (Kornblith 2019 uses per-example rows).
- RSA loads the first 200 rows of the concatenated tensor, i.e. tokens from the first sentence or two — local residual structure dominates, inflating layer-to-layer Spearman to ~1.
- Matrix entropy (see #2) degenerates to a single row per layer.

**Fix direction:** expose `get_hidden_states(..., per_sample=True)` backed by `_split_by_samples`; audit each consumer and pick the correct accessor explicitly.

### 2. `geometry_matrix_entropy` — 100% NaN for every model in the study  **[V]**
**`src/blme/tasks/geometry/matrix_entropy.py:46-69`**

On the cache branch the code reduces `(N_tokens, D)` to a single mean row per layer, then wraps it in a length-1 outer list: `all_hidden_states = [[ all_layers[li].mean(dim=0, keepdim=True) for li in range(n_layers) ]]`. Downstream `H_l` becomes a 1×D matrix, the centering step zeros it, `svdvals` returns zeros, and `rho = eigen / sum(eigen) = 0/0 = NaN`. The non-cache branch also mean-pools each sample down to a single row, so even without cache you get at most `num_samples=10` rows in D≈2–14k dimensions — too rank-deficient to yield a meaningful von Neumann entropy.

**Verified:** `geometry_matrix_entropy.*` is all-NaN for all 32 models in the aggregated CSV.

**Fix direction:** use per-sample per-layer mean pooling (giving `(num_samples, D)` per layer), and prefer a regularized covariance (e.g., Ledoit–Wolf) before SVD. Or, follow Wei et al. 2024 and compute entropy on token-level features rather than pooling.

### 3. `geometry_spectral` misses every weight matrix on GPT-2  **[V]**
**`src/blme/tasks/geometry/spectral.py:27`**

```python
TARGET_MODULES = (torch.nn.Linear, torch.nn.Conv1d)
```
GPT-2’s QKV/MLP projections are `transformers.pytorch_utils.Conv1D` (capital D), which matches neither. Only `lm_head` (`nn.Linear`) is analysed, so the power-law exponent α and stable rank are computed from a single matrix. **Verified:** all four GPT-2 models have `std_alpha = 0` and `min_alpha = max_alpha = median_alpha = avg_alpha` (in aggregated CSV).

**Fix direction:** include `transformers.pytorch_utils.Conv1D` in `TARGET_MODULES`. Several other tasks already do this correctly (`sparsity.py`, `weight_norms.py`) — copy the guard from there.

### 4. Attention knockout zeros the wrong tensor; breaks on architectures with `hidden_size ≠ num_heads × head_dim`  **[V]**
**`src/blme/tasks/causality/attention_knockout.py:84-109`**

- Hook is attached to the whole `self_attn` module and zeroes a slice of its output. But `self_attn`’s output is the post-`o_proj` residual contribution of shape `(B, T, hidden_size)`. Zeroing a contiguous slice there zeros a chunk of the mixed representation, not a single head — `o_proj` has already combined all heads. Correct head knockout zeroes the *pre-`o_proj`* concatenation `(B, T, num_heads·head_dim)`.
- `head_size = getattr(config, "head_dim", None) or hidden_size // num_heads`. For models where these differ (Gemma 2/3 hidden_size ≠ num_heads×head_dim), the slicing either overflows the last dim or indexes the residual stream in a way that doesn’t correspond to a head. **Verified:** the task fails on all four Gemma 4 models (+ pythia-6.9b/12b, likely OOM).

**Fix direction:** hook `self_attn.o_proj` with a `register_forward_pre_hook` and zero the `[h·d:(h+1)·d]` slice of the *input*. Read `head_dim` from the attention module when the config value is ambiguous.

### 5. `consistency_calibration` / `geometry_perplexity` / `geometry_prediction_alignment` are off-by-one on the cache path  **[H]**
**`src/blme/cache.py:156-192`, consumed by:**
- `src/blme/tasks/consistency/calibration.py:28-36`
- `src/blme/tasks/geometry/perplexity.py:43-55`
- `src/blme/tasks/geometry/consistency.py:35-57`

The non-cache helper `collect_prediction_stats` (`geometry/utils.py:185-196`) explicitly shifts logits and labels by 1 for next-token prediction. `ModelOutputCache.get_prediction_stats` returns unshifted logits and the raw `input_ids` as labels. So when `use_cache=True`, ECE and cross-entropy are computed comparing the distribution at position `t` to the token at position `t` (not `t+1`). That is a much easier task (identity-ish) and artificially deflates the loss for verbose tokens.

Note that `consistency_calibration` has `use_cache: false` in `defaults.yaml`, which neutralises this in the current config. But `geometry_perplexity` and `geometry_prediction_alignment` both rely on cache when it is enabled (see `core.py:153-156`). The `aggregate_results.py` sidebar comment on line 471 (“__deprecated_inverted”) indicates the perplexity column was already noticed to be wrong — this is the root cause.

**Fix direction:** in `ModelOutputCache.get_prediction_stats`, shift logits, labels, and per-sample hidden states by one position before returning, matching `collect_prediction_stats`.

---

## HIGH — wrong or misleading for some architectures / explains the failure pattern

### 6. `geometry_unembedding` purity is always 0 and alignment is tautological  **[V]**
**`src/blme/tasks/geometry/unembedding.py:79-91, 106-116`**

- *Purity (line 79):* random 2000-id sample from a 50–130 k vocab almost never lands on the ~200 categorised tokens; `scores` is typically empty so `purity_mean = 0.0`. **Verified:** 0.0 for all 32 models.
- *Alignment (lines 106-116):* when the LM head is tied, `W_out == E_in` exactly, so per-row cosine = 1. **Verified:** `embedding_alignment_mean = 1.0, embedding_high_alignment_frac = 1.0` whenever `unembedding_is_tied = True`; ~0 otherwise.

Both metrics therefore carry no signal beyond `unembedding_is_tied`. Additionally `assets/categories.json` contains list-of-list entries (`singular_plural`, `present_past`) that `tokenizer.encode(w)` does not handle.

**Fix direction:** iterate over `cat_labels.keys()` (no random vocab sampling). Measure alignment via an activation-based comparison (e.g., `h @ E_in^T` vs `h @ W_out^T`), not raw weight rows, so tied models still give meaningful numbers.

### 7. Attention-consuming tasks don’t use the cache, re-run a forward pass with `output_attentions=True`, and SDPA silently drops weights  **[H]**
**`interpretability/attention.py:47`, `attention_graph.py`, `attention_rank.py`, `induction.py`, `head_roles.py` (all do their own `model(..., output_attentions=True)`)**

The failure pattern in the CSV — every Llama 3 and every Qwen 3.5 missing `interpretability_attention_*` and `interpretability_induction_heads` — is consistent with the attention weights coming back as `None` under SDPA. `model_zoo.py` sets `attn="eager"` for every model, but `wrapper.py:159-160` only plumbs `attn_implementation` into `from_pretrained` if the recipe passed it. On modern `transformers`, some forward paths override the eager request (notably when KV cache / Flash is preferred); individual tasks then bail with `{"error": "…"}`.

Even when eager is honoured, `ModelOutputCache` is never populated with attentions (`core.py:180: need_attn = False`), so each attention task runs another forward pass, tripling peak memory for 8B-class models.

**Fix direction:**
1. Wire `need_attn = True` when any requested task consumes attention, and teach the cache to store attentions.
2. In `wrapper.py`, when attention tasks are scheduled, call `model.config._attn_implementation = "eager"` and (for newer `transformers`) `model.set_attn_implementation("eager")` before the first forward pass.

### 8. `causality_tracing.max_causal_layer` is stored as a **string**, so aggregation drops it  **[V]**
**`src/blme/tasks/causality/tracing.py:304-306`**

```python
max_layer = max((k for k in results if "_aie" in k), key=results.get)
results["max_causal_layer"] = max_layer  # e.g. "layer_21_aie"
```
`aggregated.csv` reports 29/32 any-filled but 0/32 all-filled for `causality_tracing`; the flattener in `aggregate_results.py` either coerces the string to NaN or filters it out. The causal-localization summary is therefore absent.

**Fix direction:** also emit `"max_causal_layer_idx": int(max_layer.split("_")[1])`.

### 9. `causality_tracing` uses a fixed noise scale instead of 3 σ_embedding  **[H]**
**`src/blme/tasks/causality/tracing.py:84-88, 226-232`**

Meng et al. (ROME) use `noise_std = 3 × σ(E)` where σ is computed over the model’s input embedding matrix. `noise_std = 0.1` means the corruption is ~5× too strong on Llama-class models (embedding σ ≈ 0.02) and too weak on others. `max_restoration <= 0` skips are then uneven across models; AIE values are not cross-model comparable.

**Fix direction:** `noise_std = 3.0 * model.get_input_embeddings().weight.float().std().item()`.

### 10. `causality_tracing` also normalises AIE by clean–corrupted gap  **[V]**
**`src/blme/tasks/causality/tracing.py:277-281`**

ROME reports `p(target | restored) − p(target | corrupted)`. The code divides by `max_restoration = p_clean − p_corrupted`, which can be near-zero or negative; the `max(mean_aie, 1e-10)` clamp in the entropy computation silently hides the problem. Use the raw AIE instead, or expose both.

### 11. `causality_edge_attribution` corrupts by shuffling tokens; gradients are compared at mismatched positions  **[H]**
**`src/blme/tasks/causality/edge_attribution.py:97-160`**

Two issues:
1. **Off-manifold corruption.** EAP/EAP-IG (Syed 2024, Hanna 2024) linearises around a *semantically parallel* counterfactual; a random permutation of tokens moves activations far from the clean manifold so the 1st-order Taylor approximation is invalid.
2. **Position mismatch.** `c_h[i]` is pulled from the permuted run at index `i` — which is a different token than `clean_h[i]`. The subtraction is apples-to-oranges and the signal is dominated by positional embeddings. If you keep the shuffle fallback, invert it with `torch.argsort(perm)` before subtracting.

### 12. `interpretability_weight_activation_alignment` re-runs one forward pass per layer and holds a D×D activation matrix  **[H]**
**`src/blme/tasks/interpretability/weight_activation_alignment.py:98-128`**

The hook is registered inside the layer loop, so the whole corpus is pushed through the entire model `num_layers` times. Then the branch at line 123 computes `cov = all_acts.T @ all_acts` — a D×D matrix (up to ~14 336² for Llama 3 8B) — and calls `torch.linalg.eigh`. This easily trips the 600 s per-task timeout on Llama 3-3B/8B and on all Qwen 3.5 base models; CSV confirms 22/32 missing. For the models where it does complete (gpt2-small, gpt2-xl, pythia-2.8b, gemma4-e4b in the CSV), alignment is `1.0`, which strongly suggests the activation matrix collapsed to a single direction.

**Fix direction:** hook all `down_proj` modules before the forward pass and collect inputs in one pass; subsample to ≤5 000 tokens before SVD; use `torch.linalg.svd(..., full_matrices=False)` instead of `torch.svd`. Also, “1.0” outputs should return NaN rather than be reported as a valid measurement.

### 13. `geometry_collapse`, `geometry_isotropy`, `geometry_unembedding` all use `p = S / ΣS` for effective rank  **[V]**
**`geometry/collapse.py:64-68`, `geometry/isotropy.py:32-35`, `geometry/unembedding.py:45-50`**

The Roy–Vetterli effective rank is defined on `p = σ² / Σσ²` (eigenvalues of the Gram matrix) or equivalently `exp(H(σ²/Σσ²))`. Using raw singular values instead of their squares gives a larger number and changes the scaling — the reported “effective rank” is 1-norm-normalised, not Schatten. Pick one convention and use it across the codebase so the feature is comparable cross-task.

### 14. `geometry_neural_collapse` is numerically unstable for small N and pinv(Σ_B)  **[H]**
**`src/blme/tasks/geometry/neural_collapse.py:105-119`**

With K = 5 classes and 8 samples each, Σ_B has rank ≤ 4 in D ≈ 2048-space. `np.linalg.pinv` with `rcond=1e-10` on a rank-deficient matrix amplifies noise; the aggregated CSV shows pythia-70m NC1 ≈ 3.2×10⁷ and several models in the 10²–10³ range, which are numerically meaningless rather than real NC1 values. Also, the final representation is mean-pooled over the full sequence (line 181-190) rather than taking the last-token hidden state as in Papyan-Han-Donoho 2020.

**Fix direction:** PCA-project to `d < n_per_class` before computing NC; or add Tikhonov regularisation to Σ_B; return NaN when `rank(Σ_B) < K-1`.

### 15. `dynamics_coe` measures a generation-step trajectory at the last layer, not Wang et al.’s layer chain  **[V]**
**`src/blme/tasks/dynamics/coe.py:45-70`**

Wang et al. (ICLR 2025) define Chain-of-Embedding as the *across-layer* sequence `h_t^{(0)}, h_t^{(1)}, …, h_t^{(L)}` for a fixed token t, and derive magnitude/angle change from that. The code instead takes `out.hidden_states[-1][0, -1]` at successive greedy-decoded token steps. This is a token-to-token trajectory on the final layer — a different object whose relationship to correctness detection is not the one the paper documents.

**Fix direction:** per sample, run one forward pass and iterate over `out.hidden_states[i][0, -1]` for `i in range(1, len(out.hidden_states))`.

### 16. `dynamics_gradient_flow` backpropagates the argmax logit, not a loss, with all parameters frozen  **[H]**
**`src/blme/tasks/dynamics/gradient_flow.py:65-97`**

- All parameters are `requires_grad_(False)`. The captured hook tensors are manually set to require grad, so `.backward()` produces gradients only w.r.t. those activations. That is effectively a Jacobian-of-output-logit measurement, not what Pascanu 2013 calls gradient flow.
- Backpropagating `logits[target_id]` where `target_id = argmax` ties the measurement to the model’s own greedy prediction; cross-model comparisons measure very different directions.
- On SDPA/flash kernels the graph may not support gradients through attention activations and quietly returns zero.

**Fix direction:** follow `dynamics/sharpness.py`’s pattern — use `F.cross_entropy` on shifted logits/labels, keep the parameters trainable for this task, and gate on `attn_implementation == "eager"`.

### 17. `repe_refusal_direction` and related tasks emit nested `per_layer` dicts with model-dependent keys  **[V]**
**`src/blme/tasks/representation_engineering.py:488-512`**

The aggregator flattens `per_layer.layerN.*` to columns. Because models have different layer counts, only the first layer’s summaries are present for all 32 rows (hence CSV reports 32/32 any-filled but 1/32 all-filled). Downstream PCA/Lasso then treats deep-model-only columns as random missingness correlated with size.

**Fix direction:** emit only architecture-agnostic summaries at the top level (e.g., AUC at normalised depths 0.0, 0.25, 0.5, 0.75, 1.0, plus the scalar best-layer AUC).

### 18. `geometry_intrinsic_dim` (Two-NN) reports values < 1 for several GPT-2 models  **[V]**
**`src/blme/tasks/geometry/intrinsic_dim.py:85-106`**

Two-NN (Facco et al. 2017) has a lower bound of 1 by construction. The CSV shows `intrinsic_dimension = 0.11` for gpt2-small and `0.18` for gpt2-xl — impossible. The cause is near-duplicate neighbours in vocab space (`r1 ≈ r2`): `log(r2/r1) → 0`, so `1 / log(μ)` blows up and the mean swings negative/subunitary. Facco et al. recommend the linear-regression form on `−log(1−F(μ))` vs `log(μ)` precisely to avoid this. Current code already clips `d > 1e-6`, but this doesn’t help when two **non-identical** neighbours are still nearly equidistant.

**Fix direction:** either (a) switch to the linear-regression form, or (b) trim the top 10 % of μ, per Facco.

### 19. Position-sensitivity (“needle-in-haystack”) evaluates paraphrase NLL instead of retrieval accuracy, and measures a range of 60–80 words  **[H]**
**`src/blme/tasks/consistency/position_sensitivity.py:185-225`**

- The distractors are ~60–80-word passages, so the needle is at most tens of tokens from the edges. “Lost in the middle” (Liu 2023) requires contexts of thousands of tokens to show position sensitivity; current setup has no headroom for the effect.
- At `rel_pos = 1.0`, the fact ends up directly adjacent to its recall paraphrase, so NLL collapses from local copying rather than retrieval ability.
- The metric averages NLL of a paraphrase that reuses words from the fact — this measures surface overlap, not retrieval.

**Fix direction:** generate distractors of configurable length (e.g., 1k/2k/4k tokens) and measure whether the argmax of the recall query’s answer tokens equals the fact’s key tokens.

### 20. Tokenisation boundary bugs in `format_robustness`, `icl_slope`, `logical`, `contrastive`, `paraphrase` (H1–H3, H8–H9 from consistency audit)  **[H]**
**various — cited in the consistency audit**

Several tasks tokenise `prompt` and `prompt + answer` independently and then slice logits at `prompt_len`. For BPE/SentencePiece tokenisers this is only correct if the joined tokenisation equals the concatenation, which fails whenever a leading space or merge crosses the boundary. Cross-model comparisons are biased because the failure mode is tokeniser-dependent.

**Fix direction:** tokenise the prompt, then tokenise the answer with `add_special_tokens=False`, and concatenate ids. Make `add_special_tokens` consistent across both branches.

---

## MEDIUM — methodology smells, cross-model bias, or wasted compute

- **`geometry_perplexity.ppl_rare` uses a vocab-index threshold on an argsort of counts, not a frequency quantile.** `src/blme/tasks/geometry/perplexity.py:40-44`. If 80 % of the vocab has count 0, `rare_ids` ends up being an arbitrary chunk of unseen tokens and `ppl_rare` is NaN. The CSV shows `ppl_rare` all-NaN.
- **`geometry_positional_decay` correlates raw softmax attention with |i−j|,** but because each row sums to 1 and rows have different lengths under the causal mask, Spearman is biased toward negative regardless of architecture. `src/blme/tasks/geometry/positional_decay.py:75-81`. Do the correlation per row and average.
- **`geometry_correlation_dimension` fits GP dimension on 100 mean-pooled points and never reports R².** `src/blme/tasks/geometry/correlation_dimension.py:92-93`. Use `pooling="all_tokens"`, bump `max_points` to ≥5 000, and enforce R² ≥ 0.99 on the scaling regime.
- **`interpretability_superposition` hooks the whole MLP output** — the residual-stream write — rather than the down-proj input (the sparse intermediate). The 2026-04-15 fix in `sparsity.py` should be mirrored here. `src/blme/tasks/interpretability/superposition.py:85-103`.
- **`interpretability_head_roles` issues `H×T²` `.item()` calls per sample**, syncing CPU/GPU inside the inner loop. Combined with bf16 attention tensors this blows the per-task timeout on 8B-class models. Vectorise the previous-token and duplicate-token metrics.
- **`geometry_contextualization` treats multiple positions of the same token in the same sentence as separate occurrences**, inflating self-similarity. Ethayarajh 2019 specifies across-context comparisons; pick one occurrence per token per sample. `src/blme/tasks/geometry/contextualization.py:150-160`.
- **`interpretability_logit_lens` is correctly cited as nostalgebraist 2020 but applies final_norm only once.** For multi-modal Gemma-4 runs, `out.hidden_states` may be wrapped differently; add a length check vs `get_num_layers(model)` and fall back gracefully.
- **`ModelOutputCache.get_attentions` is dead code** — no task is ever wired through it, and `core.py:180` always sets `need_attn = False`. Wiring it removes duplicated forward passes and the OOMs they cause for the 8B–27B models.
- **`wrapper.py` passes `torch_dtype="auto"` as a string**; newer `transformers` expects `torch.dtype` or the new `dtype=` kwarg. Today it works, tomorrow it may not. Drop the kwarg when the user says "auto".
- **`scripts/aggregate_results.py` `_flatten_dict` produces per-layer columns keyed by absolute layer index** (`task.layer_0`, `task.layer_1`, …). Models with fewer layers then look “missing at layer 31” — a systematic depth bias in the downstream PCA/Lasso. Convert layer-indexed dicts to lists first so the existing list-summary path (mean/std/slope/q25/q50/q75) is used.
- **Signal-based timeout in `core.py:197-217` references `old_handler` in `finally`** without initialising it, so a failure before `signal.signal(...)` binds the name will shadow the true exception with a `NameError`. Initialise `old_handler = None` and test in `finally`.
- **`consistency_calibration` echoes config metadata (`n_bins`, `n_samples_analyzed`, etc.)** into the result dict, which the aggregator flattens as scalar features. `consistency_icl_slope.shot_counts.*` and `consistency_format_robustness.n_formats` similarly leak — they appear as constants across all 32 models (verified). Strip config echoes before writing the envelope.

---

## LOW / style

- `utils.py:17-28` seeds Python/numpy/torch but doesn’t set `torch.backends.cudnn.deterministic` or `torch.use_deterministic_algorithms(True)`.
- Several files use `torch.svd` (deprecated) instead of `torch.linalg.svd`.
- `cache.py:380-384` fallback corpus is three sentences repeated; silent and non-comparable with runs that succeed in downloading WikiText. Consider raising instead of warning when the dataset load fails.
- `causality/attention_knockout.py:22-23`, `causality/tracing.py:78-82` cite fabricated-looking 2026 references; the real ones are Meng 2022, Geva 2023, Michel 2019, Voita 2019.
- `topology/*` tasks build a point cloud from 20 mean-pooled sentence vectors in D ≈ 2–8 k; at that sample size `H_1` is numerically noise. Either push to per-token clouds or label these metrics exploratory.
- `topology/homology.py:102` metric name typo `persistance` — affects CSV column names.
- `tests/` coverage is sparse on the metric math itself; bugs above slip through because tests check shapes and keys, not values. A dedicated per-task numerical regression test (ground-truth input → expected output) would catch most of these.

---

## What the user should do next

In priority order:

1. Fix bugs **#1, #2, #3, #4, #5** — they invalidate most of the headline tables in the paper.
2. Fix **#6, #7, #8, #9, #10, #11** — they bias cross-model comparisons or hide data you already collected.
3. Re-run only the affected tasks: `geometry_spectral` (GPT-2 only), `geometry_matrix_entropy` (all), `geometry_unembedding` (all), `causality_*` (all), attention-family interpretability (Llama/Qwen). Everything else can be re-aggregated from existing per-model JSON.
4. Add numerical regression tests: for each task, feed a fixed small-model + fixed seed and pin the expected output. This will catch future drift and give you confidence that the re-run numbers are real.
5. After the re-run, revisit the PCA/Lasso analyses — the depth-correlated missingness from **#17** and the aggregation issues from the MEDIUM list will still bias any scaling-law claims until fixed.

The library is close to right — the metrics are mostly the intended ones — but the cache layer and a handful of hooks are doing subtly different arithmetic than the papers assume. Fixing those is a few days of targeted work; the study does not need to be rerun end-to-end, just on the affected tasks.
