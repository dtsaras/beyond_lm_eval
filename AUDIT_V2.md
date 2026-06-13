# BLME Audit V2 — Publication-Blocking Review

**Date:** 2026-06-13 · **Branch:** `audit-v2` · **Auditor:** Claude (multi-agent + manual verification)
**Scope:** correctness of all 74 diagnostic tasks, the analysis pipeline (`scripts/`), and the paper's quantitative claims (`paper/main.tex`), ahead of an open-source + arXiv release.

This document is **complementary** to `AUDIT_REPORT.md` (rounds 1–6, immutable history). It records findings that prior rounds did not reach — principally in the **analysis pipeline and the paper's headline numbers**, which earlier rounds did not audit.

> **Status:** This audit was **partially completed**. A multi-agent run audited 37 of 74 tasks plus the full analysis/infra layer before exhausting an API quota; the per-task adversarial-verification stage did not run. The pipeline/infra findings below were **re-verified by hand** against the code and the committed study CSVs and are high-confidence. The per-task findings in §5 are **candidate findings pending adversarial verification** (see §7 roadmap). The single most important finding (§1) is fully verified, quantified, and fixed.

---

## 1. CRITICAL — the v3 headline is an artifact of a parameter-count bug

**Finding.** `scripts/aggregate_results.py` mapped each model to `MANUAL_PARAM_COUNTS_M`, a dict that contained **only the 32 v2 models**, then computed `log_n_params = np.log(n_params_M.fillna(1) * 1e6)`. Every v3-only model — **26 of the 58**, including *all* four 70B/72B anchors and every Qwen-2/2.5/3 and Gemma-1/2/3 checkpoint — was silently assigned **1 million parameters**, i.e. `log_n_params = log(1e6) = 13.8155`. Of those, **14 carry benchmark scores and enter the regression**, and they are disproportionately the *highest-capability* models in the zoo (qwen2.5-72b y=0.97, llama3.3-70b-it y=0.91, qwen2.5-32b y=0.93, qwen3-32b y=0.92, …).

A baseline that predicts capability from `log(params)` is therefore handed the five strongest models all labelled "1 M params." This **manufactures** the paper's entire v3 narrative.

**Verification.** Confirmed directly in `results/study_v3/aggregated.csv` (26 rows with the sentinel value) and quantified by reproducing the exact published pipeline (`scripts/audit_v2_param_bug_impact.py`, result in `results/audit_v2/param_bug_impact.json`):

| Quantity | Paper (v3) | Reproduced (buggy) | **Corrected** |
|---|---|---|---|
| LASSO LOO R² | 0.78 | 0.777 ✓ | 0.78 (robust) |
| Baseline log(N) LOO R² | **0.06** ("collapsed") | 0.056 ✓ | **0.63** |
| LOO gain over baseline | **+0.72** | +0.721 ✓ | **+0.16** |
| LASSO LOFO R² | 0.50 | 0.503 ✓ | 0.50 |
| Baseline log(N) LOFO R² | ~0 / neg | −0.26 ✓ | **0.58** |
| LOFO: who wins? | "LASSO ≫ baseline" | — | **baseline ≥ LASSO** |

My pipeline reproduces every published number on the buggy data (validating the comparison), then shows that with **correct parameter counts**:

- The log(N) baseline **does not collapse**. It is **0.63 LOO / 0.58 LOFO** — *better* than v2's 0.43, exactly as one expects when 70B models are added.
- The honest within-distribution gain of the intrinsic LASSO over scale is **+0.16 LOO**, not +0.72.
- **Cross-family, the corrected baseline (0.58) beats the intrinsic LASSO (0.50).** The paper's flagship generalization claim ("LOFO jumps to 0.50, beating the collapsed baseline on 11 of 13 families") **inverts** once the bug is fixed.

**Fix (committed `70b4d5a`).** Added correct counts for all 26 v3 models; fall back to the existing `12·L·d²` size estimate when a manual count is missing; and **hard-assert** that no analyzed model has ≤1 params (never silently `fillna(1)` again). The v3 analysis outputs and the paper's §3.2/§4/abstract numbers must be regenerated from the corrected `log_n_params`.

---

## 2. The "intrinsic beats scale" claim, re-examined honestly

Two further issues compound §1; both are verified against the code and CSVs.

**2a. Architecture/size features leak into the "intrinsic" feature matrix (HIGH).**
`scripts/bootstrap_lasso_r2.py` / `analyze_correlations.py` exclude only `benchmark_*`, `composite_benchmark`, and a few name columns from `X`. As a result the LASSO is given a **working size proxy and raw architecture counts as "features"**: `n_params_est` (= 12·L·d², a clean size estimate), `d_model`, `n_layers`, `n_heads`, `vocab_size`, and task-emitted depth echoes such as `causality_tracing.traced_layers.{mean,std,…}` (literally `list(range(n_layers))`), `geometry_cka.n_layers`, `geometry_schatten.n_layers`, `causality_knowledge_neurons.n_layers`, etc. So the headline comparison pits "intrinsic features **plus a correct size proxy**" against "a **corrupted** size proxy" — rigged on both sides. `geometry_cka.n_layers` is LASSO-selected feature #21 in v3.

**2b. Circularity — the feature matrix contains the training objective (HIGH, conceptual).**
WikiText NLL *is* the pretraining objective, and that it predicts benchmarks is already established (observational scaling, Ruan et al., which the paper cites). Yet the feature matrix includes `geometry_perplexity.{ppl_*, mean_nll_nats, bits_per_char}`, `interpretability_prediction_entropy.{…}`, intermediate-layer logit-lens confidence, and **three independent copies of corpus cross-entropy loss** emitted as side-products (`causality_ablation.baseline_loss`, `causality_attention_knockout.baseline_loss`, `dynamics_sharpness.baseline_loss`). "Intrinsic predicts capability" therefore partly reduces to "loss predicts capability."

**The decisive ablation (verified; `results/audit_v2/intrinsic_ablation.json`).** I re-fit the LASSO on a **strictly-structural** feature set (671 features: size *and* all likelihood/loss/confidence columns removed) vs. the **corrected** baseline:

| Feature set | LOO R² | LOFO R² |
|---|---|---|
| Full (as published, 737 feats) | 0.777 | 0.503 |
| Size/arch removed (715) | 0.784 | 0.497 |
| **Structural-only (671; no size, no likelihood)** | **0.788** | **0.552** |
| Corrected log(N) baseline | 0.629 | 0.576 |

**Good news for the thesis:** the within-distribution signal is **real and robust** — stripping out *both* size and every behavioral/likelihood feature leaves LOO at 0.79 (intrinsic structure genuinely predicts capability, +0.16 over scale). **Bad news for the headline:** cross-family, the cleanest intrinsic model (0.552) **does not beat** a correct size baseline (0.576).

**Defensible reframing for the paper:** intrinsic representational structure carries a modest but robust capability signal beyond scale *within distribution* (LOO +0.16, surviving the harshest behavioral-feature ablation); *across families* it performs at the level of scale. Drop the "baseline collapses / +0.72 / beats 11-of-13-families" framing entirely.

**2c. The dependent variable is ~79% MMLU, not "six benchmarks" (HIGH).**
`paper/main.tex:351` says the composite is "the mean of min-max normalized accuracies across all six benchmarks." The code (`aggregate_results.py:559-570`) takes `mean(axis=1)` over **all 78 `benchmark_*` columns, 62 of which are MMLU** (overall + 4 category aggregates + 57 subtasks). MMLU is thus counted ~62× against 16 single counts; the target is a ~79%-MMLU-weighted average. Worse, per-model benchmark coverage differs across v3 (67 vs 76–78 columns), so different models' composites average **different benchmark sets**, and min–max normalization over the zoo makes the target depend on zoo membership (v2 and v3 `composite_benchmark` are different variables). Fix: compute the composite from a fixed, declared list of group-level scores with identical coverage.

---

## 3. Verified pipeline/infra findings (from the infra auditor, hand-checked)

| # | Sev | File | Issue | Status |
|---|---|---|---|---|
| infra:1 | CRIT | aggregate_results.py | param `fillna(1)` → §1 | **FIXED `70b4d5a`** |
| infra:2 | HIGH | analyze_correlations.py, *_lasso | size/arch features leak into X → §2a | documented; deny-pattern proposed |
| infra:3 | HIGH | aggregate_results.py:126 | `_summarise_list` `.slope` regresses on raw `np.arange` index, not normalized depth → every `*.slope` feature scaled by 1/(L−1) and confounded with depth (same class as task fixes #45/#59, still present in the aggregator) | pending regen |
| infra:4 | HIGH | paper §3.4 vs aggregate_results.py | Paper describes a 20-point depth-interpolation + 7-vector `[v(.25),v(.5),v(.75),β1,β2,z_min,z_max]` with curvature β2 and **extremum depths**; code emits mean/std/min/max/slope/q25/q50/q75 with no interpolation, no β2, and min/max as **values** not depths | reconcile text↔code |
| infra:5 | HIGH | aggregate_results.py:559 | composite Y ≈ 79% MMLU → §2c | pending regen |
| infra:6 | MED | core.py:194-224 | force-eager attention switch nested inside `if cache_tasks` → a selective run of only the non-cache attention tasks (`attention_graph/rank/head_roles/induction_heads`) gets `attentions=None` and errors | proposed: hoist block |
| infra:7 | MED | core.py:240-263 | signal handler `old_handler` UnboundLocalError off main thread + float-timeout TypeError → kills the whole eval loop | **FIXED `119eb75`** + tests |
| infra:8 | MED | mutual_info.py:62, intrinsic_dim.py:60, isotropy.py:41 | residual unseeded sampling (global RNG) — same class as fixes #42/#50-53; one also makes the no-cache fallback (10 random tokens/sample) differ from the cache path | proposed: seed |
| infra:9 | LOW | utils.py | `set_global_seed` omits cuDNN determinism flags (safe to add) and `use_deterministic_algorithms` (opt-in, `warn_only`) | proposed |
| infra:10 | LOW | models/wrapper.py:146 | deprecated `torch_dtype=` kwarg, `_resolve_dtype` returns `'auto'` under a `torch.dtype` annotation, legacy `load_in_8bit/4bit` instead of `BitsAndBytesConfig` | proposed |
| infra:11 | LOW | cache.py:100 | stale docstring claims CKA/RSA/LID "need" `per_sample=True`; shipped consumers correctly use the flat cloud — invites a future "fix" that would silently change locked numbers. Also RSA builds its RDM from only the first 200 tokens (≈1–2 passages) | doc + RSA sampling |
| infra:12 | LOW | tests/conftest.py | `DummyTokenizer` ignores text, returns unseeded random ids, no `return_offsets_mapping` → the offset paths from fixes #34/#49 are never exercised; py3.9 CI leg can't install transformers 5.x | test hardening |
| infra:13 | LOW | results.py | fixed `results.json` filename (silent overwrite); non-RFC8259 NaN/Inf in JSON; `print_results_table` assumes dict results | proposed |

Also reproduced this session: the OpenBLAS `corrupted size vs. prev_size` crash on `geometry_intrinsic_dim` (TwoNN) when `OPENBLAS_NUM_THREADS` exceeds 64 on many-core hosts — cap threads in the runner or document.

---

## 4. New metrics added (committed `7e6b98d`)

Both were **independently flagged as the top-two portfolio gaps** by the premise critic (gap #3 RMT activation spectra; gap #4 across-token trajectory geometry), which corroborates their value.

- **`geometry_trajectory_curvature`** — discrete curvature of per-token hidden-state trajectories across layers (Hosseini & Fedorenko, NeurIPS 2023). The **first** BLME metric that uses sequence order. Per-sample (`per_sample=True`), fp32, BOS-skipped; reports early/mid/late curvature, straightening ratio, and slope over *normalized* depth. Golden tests: collinear→0, right-angle zigzag→π/2, hexagon→π/3, degenerate→no-NaN. gpt2 smoke: curvature ≈2.0 rad with a negative depth slope (straightening), as the hypothesis predicts.
- **`geometry_mp_bulk_deviation`** — Marchenko–Pastur bulk-deviation analysis of the activation correlation spectrum (MP 1967; BBP spiked-covariance 2005). Complements the existing HT-SR α on *weights* with an RMT diagnostic on *activations*. Reports outlier fraction, spike energy, KS distance to the γ-matched MP law, plus fixed-γ variants for cross-model comparability. Golden tests: iid-null→~0 outliers, planted spike→detected, constant-dim→dropped, edge/CDF sanity.

Architecture/sampling metadata in both is namespaced under `_meta_` so it cannot repeat the §2a leakage. Task count 72 → 74; `tests/test_completeness.py` updated; 30 new golden tests pass.

---

## 5. Per-task candidate findings (37/74 audited; **pending adversarial verification**)

Treat these as leads, not verdicts — the skeptic/adjudicator stage did not run. Citation-ID corrections must be web-verified before applying (the auditor's *replacement* IDs may themselves be wrong).

**Wrong / missing citations (HIGH-value, low-risk once verified):**
- `geometry_schatten` — cites "Wei et al. 2025, arXiv:2509.25359"; that arXiv ID is Yusupov et al. (no author "Wei"), and its actual finding is that Schatten/MOM mostly reflect **output length** once controlled. Also `schatten_2` is degenerate after the code's row-L2 normalization (‖Z_norm‖_F = √N_kept).
- `geometry_isoscore` — cited arXiv:2207.10341 is "UFO: Unified Feature Optimization" (ECCV 2022); correct IsoScore is **2108.07344**. **Plus a CRITICAL math claim:** the code L1-normalizes the eigenvalue vector (sum = d) but IsoScore Algorithm 1 specifies **L2** normalization — would change every value. *Verify before fixing.*
- `geometry_neural_collapse` — cited arXiv:2008.03465 is a brain-MRI paper; correct Papyan–Han–Donoho NC is **2008.08186**.
- `interpretability_waa` — misattributes arXiv:2311.03658 ("The Linear Representation Hypothesis", Park–Choe–**Veitch**) to "Park, Choe, Wattenberg, Jegelka 2024"; also unseeded `svd_lowrank` range-finder and a √(2/πD) chance-baseline confound.
- `interpretability_attention_rank`, `interpretability_superposition`, `geometry_representation_sensitivity`, `geometry_tokenizer_efficiency`, `consistency_logical`, `consistency_position_sensitivity` — flagged WRONG_ATTRIBUTION (details in `results/audit_v2/` JSON dumps).

**Implementation bugs (HIGH) producing degenerate/NaN shipped columns:**
- `interpretability_attention_entropy` — non-finite per-head entropies not filtered before aggregation → headline scalars NaN.
- `interpretability_attention_graph` — PageRank on a causal graph is structurally confounded (token-0 quasi-absorbing); unguarded NaN poisons mean/max sink-PageRank (NaN for pythia-6.9b/12b in v3).
- `interpretability_sae_features` — off-by-one hook point (applies blocks.8 SAE to `hidden_states[9]`).
- `interpretability_superposition` — bimodality coefficient may be **direction-inverted** vs polysemanticity; empty-hook layers zero-filled instead of NaN.
- `consistency_self_consistency` — `model.generate()` inherits each checkpoint's shipped `generation_config` for unspecified knobs → not comparable across models.
- `consistency_bias_weat`, `geometry_isoscore` — see above.

**Confirmed design flaws (the known open items + more):**
- `consistency_position_sensitivity` (#19) — confirmed: measures ~60-token paraphrase NLL, not lost-in-the-middle retrieval. → redesign as `consistency_position_retrieval`.
- `consistency_logical` — the LAMBADA-midpoint split is gone at HEAD but the replacement (5 bundled premise/conclusion items) is still not genuine entailment. → redesign as `consistency_entailment`.
- `geometry_positional_decay`, `geometry_categories` — DESIGN_FLAW (tokenizer-dependent membership breaks cross-model comparability).
- `causality_edge_attribution` (#11) — not yet reached this run (C2 unit failed); still open.

**Verdict distribution over the 37 audited:** ~12 CORRECT / CORRECT_WITH_CAVEATS that are genuinely fine; the remainder split across citation fixes, value-changing bugs, and design flaws. Full structured dumps: `results/audit_v2/cat_{geometry,interpretability,consistency}.json`.

---

## 6. Organization & premise (senior-reviewer view)

- **Premise is sound but the headline is over-claimed.** After the §1/§2 corrections the honest contribution is: a careful, audited catalogue of 74 intrinsic diagnostics + the finding that representational structure adds a modest, robust *within-distribution* signal beyond scale, with cross-family generalization at the level of scale. That is publishable; the current framing is not.
- **Taxonomy is a provenance scheme, not a measurement one, and it leaks.** `geometry_perplexity` is used as both a predictor (X) and a secondary target (Y); `geometry_prediction_alignment` lives in `geometry/consistency.py` and measures a behavioral quantity; `geometry_representation_sensitivity` is a confidence functional. Publish a single 74-row taxonomy table with one consistent Tier-1/2/3 definition and an explicit X/Y/confound role per task.
- **Redundancy: "731 features" are ~30–60 effective dimensions.** Nine near-duplicate clusters (effective-rank/isotropy ×8; dimension estimators ×4; the 4 topology tasks re-summarize one diagram; CKA == normalized HSIC exactly; attention-concentration ×3; the likelihood/confidence cluster; massive-activation/outlier; RepE directions; the size/tokenizer echo cluster). Report the effective dimensionality of the feature set.
- **Top portfolio gaps** (the critic's ranking): (1) Pythia intermediate-checkpoint longitudinal validation — the cheapest credible answer to correlation-vs-causation; (2) tokenizer-free/byte-level or multilingual corpus — the acknowledged tokenizer confound; (3) RMT activation spectra **[now added]**; (4) across-token trajectory geometry **[now added]**; (5) prequential/MDL compression; (6) Fisher-information geometry; (7) weight-graph modularity; (8) representation perturbation-robustness.

---

## 7. What remains (post-quota-reset roadmap)

The multi-agent audit should be **resumed** (the workflow supports `resumeFromRunId`, which replays the cached completed agents) to finish:

1. **Audit the 35 untouched tasks:** all of causality (×6, incl. open #11), dynamics (×6), topology (×4), repe (×4), the dimension-estimator geometry unit (×4), interpretability I2/I4 (×7), consistency S2 (×4).
2. **Run the adversarial-verification stage** (code-lens + literature-lens skeptics + adjudicator) over the §5 candidate findings — *especially* web-verify every citation-ID correction and the IsoScore L1/L2 math claim before applying.
3. **Implement the two design redesigns** under new names (`consistency_position_retrieval`, `consistency_entailment`), deprecating the old names so the locked v2 columns are never silently redefined.
4. **Regenerate the v3 analysis** from corrected `log_n_params`, with size/arch and likelihood features quarantined from the headline model, and update `paper/main.tex` (abstract, §3.2, §4) to the honest numbers; fix §3.3 corpus size (paper says 500; study used 200), §3.4 normalization text↔code, and the composite-Y definition.
5. **Apply the remaining safe infra fixes** (infra:6, :8, :9, :10, :13) with regression tests.

## Commits on `audit-v2` so far
- `7e6b98d` feat: `geometry_trajectory_curvature` + `geometry_mp_bulk_deviation` (+30 golden tests)
- `70b4d5a` fix: correct v3 parameter counts (CRITICAL §1) + impact script
- `119eb75` fix(core): per-task timeout no longer crashes the eval loop (infra:7) + regression tests

## Reproducibility
- `OPENBLAS_NUM_THREADS=8 python scripts/audit_v2_param_bug_impact.py` → `results/audit_v2/param_bug_impact.json` (§1)
- intrinsic ablation → `results/audit_v2/intrinsic_ablation.json` (§2)
- raw per-task findings → `results/audit_v2/cat_*.json`
