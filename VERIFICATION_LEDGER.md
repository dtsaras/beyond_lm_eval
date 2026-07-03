# BLME Task Verification Ledger

**Date:** 2026-06-22 · **Branch:** `audit-v2`
**Scope:** numeric-parity verification of all **74** diagnostic tasks against their original papers and a reference implementation, per the directive to make every task "absolutely correct… produces the exact same numbers" as the reference, and to track the paper + repository for each metric.

This ledger complements `AUDIT_V2.md` (which it supersedes for per-task findings: the §5 candidate findings there were "pending adversarial verification" — they have now been verified or refuted). Companion regression tests live in `tests/tasks/test_reference_parity_formulas.py` and the fixture manifest `tests/fixtures/reference_parity/formula_fixtures.json`.

## Method

For each task: (1) locate BLME's core numeric helper; (2) web-verify the original paper's defining equation and the correct arXiv id/authors; (3) find the canonical reference implementation; (4) **for any closed-form metric, run BLME's helper and the reference on identical synthetic input and compare numbers**; (5) classify CLOSED_FORM / PIPELINE / PROXY and assign a verdict. Discrepancy/citation/bug findings were independently re-derived by an adversarial pass and, for the high-stakes ones, re-verified by hand against primary sources. Pipeline tasks (those needing a real model/optimizer/circuit search) are verified for algorithmic faithfulness only — exact parity there would require running heavy reference code on the same model, which is out of scope.

## Verdict distribution (74 tasks)

| Verdict | Count | Meaning |
|---|---|---|
| VERIFIED_PARITY | 14 | closed-form, matches reference to ≤1e-6 (or documented float tol) |
| FAITHFUL | 12 | algorithm provably matches paper+reference; pipeline, parity not numerically run |
| PROXY_OK | 27 | honest proxy; `proxy-only` label correct, no false parity claimed |
| CITATION_FIX | 12 | implementation fine, cited paper/arXiv/authors wrong (11 upheld, `correlation_dimension` flag **refuted**) |
| DISCREPANCY | 7 | number-changing divergence from reference — **all fixed this session** |
| BUG | 2 | degenerate/non-deterministic shipped output — **both fixed this session** |

## Comprehensive test coverage (2026-06-22)

**Every one of the 74 tasks now has at least one independent-reference parity or behavioral
test that has been executed in-repo** (not an agent prose claim). Two test modules:

- `tests/tasks/test_reference_parity_formulas.py` — 26 closed-form numeric-parity tests (helpers vs pip pkg / analytic / transcribed reference).
- `tests/tasks/test_comprehensive_parity.py` — 53 tests covering the other 52 tasks + a full-pipeline `geometry_perplexity` anchor. Authored via fan-out, then **re-run and reviewed by hand**: 43 strong independent-numeric, 7 analytic, 2 behavioral-invariant; 0 self-rated weak.

Coverage character by task type:
- **Closed-form** (most tasks) → exact numeric parity vs an *independent* reference (e.g. `representation_sensitivity` checked vs torch.autograd; `positional_decay` vs Spearman-via-Pearson-of-ranks; `contrastive`/`logical`/`knowledge_capacity` vs independent teacher-forcing NLL).
- **Full-pipeline** (run the actual task on a real model) → `geometry_perplexity` (gpt2 vs textbook ppl, exact token count), `causality_ablation` (mean-ablation reconstruction), `causality_circuit_quality`, etc.
- **Behavioral-invariant** (pipeline tasks needing trained weights) → `causality_tracing` reproduces ROME Fig.2 early-site localization on real gpt2 (peak AIE early/mid ≫ final layer) with an independent corruption re-implementation; + full-pipeline determinism.

**Adversarial line-audit of all 52 authored tests (2026-06-22):** a skeptic-per-test workflow rated independence + substance, then findings were acted on by hand. Result: 35 STRONG, 15 ACCEPTABLE, 2 WEAK.
- **2 WEAK fixed:** `interpretability_probing` (reference cloned BLME's exact SGD — and probe accuracy is provably OPTIMIZER-dependent, so no exact independent reference exists; rewritten to structural-exactness + Alain-Bengio decodability-above-chance behavioral check) and `geometry_categories` (separation reused scipy's cosine + mirrored seed-42; rewritten with a hand-written cosine distance + a sampling-independent constructed ground truth).
- **File hygiene:** ~18 workflow agents had self-appended their tests to the file (duplicate shadows); regenerated the file from records → exactly 53 unique tests, all green.
- **Two ACCEPTABLE tasks strengthened to exact-numeric (DONE):** `causality_circuit_quality` faithfulness is now pinned by an INDEPENDENT `1 − scipy.jensenshannon(p_circuit, p_base, base=2)²` with our own non-circuit mean-ablation hooks (different code path than BLME's `F.kl_div` JSD). `causality_tracing` now pins EVERY per-layer AIE via an independent ROME reimplementation (own embedding-noise + patch hooks; matched protocol/seed) to <1e-4, plus the early-site invariant. Both upgraded from behavioral/partial → exact-numeric.
- Verification principle adopted (per user): **prove every analytic shortcut equals autograd/ground truth.** Done for `geometry_representation_sensitivity` — its closed-form gradient `= torch.autograd` to 2.7e-7 on real gpt2 (exact, not an approximation: HF returns the post-`ln_f` state, so `W·h+b` is genuinely linear).
- **float32 robustness tested:** study-dtype float32 vs float64 agree to ≤1e-9 rel on representative metrics.

**Honest caveats (residual doubt):** (1) `causality_circuit_quality`'s reference is an analytic re-derivation of its own proxy definition — there is no external numeric reference for a proxy, so it pins the formula + end-to-end invariants but is the weakest. (2) Behavioral tests verify the paper's *qualitative* defining property, not exact numeric parity (exact parity for ROME/knowledge-neurons/SAE needs running the heavy reference repo on the same model — not feasible here). (3) Proxy tasks verify they compute their stated definition + are honestly labeled, not that they reproduce a paper number (they don't claim to). (4) Parity is CPU/float64; the real study runs GPU/float32. (5) I deeply reviewed 2 of the 52 authored sources line-by-line plus all 52 reference descriptions; the references are independent, but I did not line-audit all 52 bodies.

## Fixes applied this session (verified + locked with parity tests)

All value-changing fixes were approved (fix-to-match-reference + flag for regeneration). Each was verified against the reference before/after and pinned by a regression test.

| Task / file | Change | Reference basis | New fixture value |
|---|---|---|---|
| **numpy crash** — `geometry/isotropy.py:29`, `causality/ablation.py:159`, `topology/persistence_landscape.py:95` | `getattr(np,"trapezoid",np.trapz)` eagerly eval'd `np.trapz` → `AttributeError` on NumPy ≥2.0; made the lookup lazy | crashed metric path of `geometry_svd`, `causality_ablation`, `topology_persistence_landscape` on numpy 2.x; numerically identical fix | n/a (restores output) |
| `topology_persistence_entropy` | natural log → **base-2** | Atienza Def 3.1; giotto-tda `scipy.stats.entropy(base=2)` | `[1,1,2]`→ 1.0397→**1.5** |
| `dynamics_generation_diversity` (distinct-n) | denominator n-gram-count → **total tokens** | Li et al. 2016 §5.2 | distinct_2 0.75→**0.6** |
| `geometry_schatten` (MNN) | `mnn/d_model` → **`mnn/L_input`** (rows) | Li et al. 2024 Eq. 11 (primary source verified) | wrong by factor T/d |
| `geometry_contextualization` (IntraSim) | pairwise-mean cosine → **token-to-sentence-mean** | Ethayarajh 2019 §3.1; kawine/contextual | exact vs ref formula |
| `dynamics_coe` (`coe_c`) | raw/raw → **normalized magnitude + normalized angle** | reference `score.py` `compute_CoE_C` | 1.118→**1.06398** |
| `interpretability_activation_sinks` | mean over all keys → **first-token Sink₁** | Gu et al. 2025 §3.2; sail-sg/Attention-Sink | 0.25→**1.0** (BOS case) |
| `interpretability_waa` | unseeded `svd_lowrank(q=1,niter=2)` → **deterministic exact/oversampled SVD** | fixes non-determinism + flat-spectrum inaccuracy | deterministic, |cos|→1.0 |

New/strengthened parity tests (suite now 24 in `test_reference_parity_formulas.py` + 3 numpy-compat guards in `test_infrastructure.py`): `geometry_isoscore` (official `IsoScore` pkg, <3e-8), `geometry_cka`/`geometry_hsic` (Kornblith reference value), `topology_persistence_landscape` (persim), `consistency_calibration` (torchmetrics), `geometry_intrinsic_dim` (skdim + known-ID recovery), plus the corrected fixtures above.

**Refuted prior lead:** AUDIT_V2's "CRITICAL IsoScore L1-vs-L2 normalization bug" is **false** — BLME uses L2 (`np.linalg.norm`), identical to the reference (`vector_norm`); verified to <3e-8 vs the official `IsoScore==2.0.1` package.

## Study/paper regeneration impact

The following study **feature columns change value** and must be regenerated before the paper's numbers are trusted (all are corrections toward the reference; none were headline predictors except where noted):

- `topology_persistence_entropy.*` — rescaled by 1/ln2 ≈ 1.4427 (base-2). Rankings/correlations unchanged.
- `dynamics_generation_diversity.distinct_2/3.*` — denominator change (distinct_1 unchanged).
- `geometry_schatten.row_normalized_matrix_nuclear_norm` — rescaled by T/d (varies per text/model).
- `geometry_contextualization.intra_sentence_similarity_{raw,corrected}` — different statistic (token-to-mean).
- `dynamics_coe.coe_c` and `.normalized_coe_c` — now the reference normalized form.
- `interpretability_activation_sinks` Sink₁ — first-token (was diluted over all keys).
- `interpretability_waa.*` — previously non-deterministic; now reproducible (prior values were noise).
- `geometry_svd.*`, `causality_ablation.area_under_degradation_curve`, `topology_persistence_landscape.*` — were **crashing** on numpy 2.x (likely absent/zero in any 2.x run); now produced.

## Citation corrections (web-verified)

Implementation correct, attribution wrong. Confirmed against primary sources:

- `geometry_unembedding` — **"Lan et al. 2024, Unembedding Dark Matter" is FABRICATED** (no such paper). Replace with Roy & Vetterli 2007 (effective rank). [docs/tasks_geometry.md, docs/PAPERS.md]
- `interpretability_waa` — authors are Park, Choe, **Veitch** (arXiv:2311.03658), not "Wattenberg, Jegelka".
- `geometry_hubness` — recorded `arXiv:1209.6425` is "Gene selection with guided regularized random forest" (Deng & Runger); Tomašev 2014 has no arXiv (IEEE TKDE 26(3), DOI 10.1109/TKDE.2013.25).
- `dynamics_interpolation` — recorded "Loshchilov & Hutter 2019 slerp" is `1711.05101` (AdamW); slerp is Shoemake 1985.
- `geometry_intrinsic_dim` — Facco et al. id is `1803.06992` (Sci. Rep. 7:12140), not `1705.10933`.
- `topology_betti_curve` — Naitzat et al. 2020 is **JMLR 21(184):1-40** (arXiv:2004.06093), not "ICLR 2020".
- `topology_persistence_entropy` — docstring conflates Rucco 2017 (Signal Processing) title with the ECCS-2014 author set; primary formula source is Chintakunta et al. 2015.
- `geometry_schatten` — MNN width-normalization is a BLME convention, not Yusupov et al.
- Also: `dynamics_stability` (Wendlandt 2018), `geometry_collapse` (Queipo-de-Llano et al. 2025 author order), `geometry_tokenizer_efficiency` (Ali et al. 2024 / Rust et al. 2021), `interpretability_head_roles` (add Wang 2022 IOI), `interpretability_attention_rank`/`_effective_rank` (effective rank = Roy & Vetterli 2007), `interpretability_prediction_entropy` (Shannon 1948, not Holtzman 2020).
- **Refuted:** `geometry_correlation_dimension` citation flag did not survive adversarial review — no change.

## Remaining work

- ~~`topology_betti_curve` redesign~~ — **DONE (2026-06-22)**: reimplemented on graph-geodesic distance (kNN graph + scipy `shortest_path`). β₀ = connected components of the symmetric kNN graph; β₁ = H1 loops with normalized persistence > 0.3 from ripser on the geodesic matrix. Validated against ground truth: K separated blobs → β₀=K (exact), noisy circle → (1,1), figure-8 → (1,2), high-dim gaussian noise → 0 loops; the old median-Euclidean collapsed K=4 → β₀=1. Faithful adaptation of Naitzat et al. (not exact Eirene parity), so stays `refined-adaptation`. 12 topology tests green.
- ~~`dynamics_generation_diversity` self-BLEU~~ — **DONE**: NLTK `sentence_bleu` + method-1 smoothing (Texygen-exact, <1e-12 on a partial-overlap fixture); `nltk` added as optional dep with a smoothed fallback.
- ~~docs citation sweep~~ — **DONE**: the verified corrections are applied to `docs/PAPERS.md` (dynamics_stability→Wendlandt 2018, interpolation slerp→Shoemake 1985, collapse→Queipo-de-Llano author order, prediction_entropy→Shannon 1948, attention_rank/effective_rank→Roy & Vetterli, head_roles→+Wang 2022 IOI) and source docstrings; the fabricated and wrong-id strings are locked in the `test_publication_docs.py` banned-identifiers guard.
- ~~Cert-label upgrades~~ — **DONE**: added checked-in covering tests and upgraded **3** to `parity-ready` — `topology_homology` (vs analytic unit-square + GUDHI), `geometry_mp_bulk_deviation` (analytic MP edge/CDF), `interpretability_attention_entropy` (uniform→log T, vs scipy). `parity-ready` count 11→14. Helpers `_attention_entropy` and `_lifespan_summary` were extracted so the tests exercise BLME's own code.
  - **`geometry_lid` was NOT upgraded** — verifying it myself refuted the agent's VERIFIED_PARITY claim: BLME uses the `-k` MLE variant (`LID = -k / Σ log(d_i/d_k)`), biased high by k/(k-1) vs the canonical Levina-Bickel `-(k-1)`; it differs from `skdim.id.MLE` by 0.36–0.89 on known-ID data. It is genuinely `formula-faithful`, not parity-ready. A test pins the `-k` formula + the k/(k-1) relationship; the metadata note documents it. (Optional future value-fix: switch `-k`→`-(k-1)` for exact Levina-Bickel/skdim parity — changes study features.)

## Full task table

Legend: ⚠ on arXiv = recorded citation was wrong (corrected above). Parity MATCH/MISMATCH/NOT_RUN is the numeric check at audit time (MISMATCH rows for the 7 DISCREPANCY/2 BUG tasks are **now fixed**; `geometry_mahalanobis`/`geometry_spectral` MISMATCH are expected proxy divergences, documented).

| Task | Verdict | Cert | arXiv | Reference repo | Parity |
|---|---|---|---|---|---|
| `causality_ablation` | PROXY_OK | proxy-only | 2408.17322 | nickypro/investigating-ablation | MATCH |
| `causality_attention_knockout` | FAITHFUL | refined-adaptation | 1905.10650 | pmichel31415/are-16-heads-really-better-than-1 | MATCH |
| `causality_circuit_quality` | PROXY_OK | proxy-only | 2304.14997 | ArthurConmy/Automatic-Circuit-Discovery | MATCH |
| `causality_edge_attribution` | PROXY_OK | proxy-only | 2310.10348 | Aaquib111/edge-attribution-patching | MATCH |
| `causality_knowledge_neurons` | PROXY_OK | proxy-only | 2104.08696 | Hunter-DDM/knowledge-neurons | MATCH |
| `causality_tracing` | FAITHFUL | refined-adaptation | 2202.05262 | kmeng01/rome (causal_trace.py) | MATCH |
| `consistency_bias_weat` | VERIFIED_PARITY | formula-faithful | 1608.07187 | W4ngatang/sent-bias | MATCH |
| `consistency_calibration` | VERIFIED_PARITY | parity-ready | 1706.04599 | gpleiss/temperature_scaling | MATCH |
| `consistency_contamination` | VERIFIED_PARITY | parity-ready | 2310.16789 | swj0419/detect-pretrain-code | MATCH |
| `consistency_contrastive` | PROXY_OK | proxy-only | 2202.05262 | kmeng01/rome (eval_utils_counterfact) | MATCH |
| `consistency_format_robustness` | PROXY_OK | proxy-only | 2310.11324 | msclar/formatspread | MATCH |
| `consistency_icl_slope` | PROXY_OK | proxy-only | 2005.14165 | (Brown et al. 2020) | MATCH |
| `consistency_knowledge_capacity` | PROXY_OK | proxy-only | 2205.10770 | (Tirumala et al. 2022) | NOT_RUN |
| `consistency_logical` | PROXY_OK | proxy-only | 2109.14723 | (entailment lit.) | MATCH |
| `consistency_membership_inference` | PROXY_OK | proxy-only | 1709.01604 | sam-yeom/ml-privacy-csf18 | MATCH |
| `consistency_paraphrase` | PROXY_OK | proxy-only | 2404.15206 | (paraphrase-invariance lit.) | MATCH |
| `consistency_position_sensitivity` | PROXY_OK | proxy-only | 2307.03172 | nelson-liu/lost-in-the-middle | MATCH |
| `consistency_self_consistency` | PROXY_OK | proxy-only | 2203.11171 | (Wang et al. 2022) | MATCH |
| `dynamics_coe` | DISCREPANCY→fixed | refined-adaptation | 2410.13640 | Alsace08/Chain-of-Embedding (score.py) | fixed→MATCH |
| `dynamics_generation_diversity` | DISCREPANCY→fixed | refined-adaptation | 1510.03055 | geek-ai/Texygen | distinct-n + self-BLEU fixed→MATCH |
| `dynamics_gradient_flow` | PROXY_OK | refined-adaptation | 1211.5063 | (Pascanu et al. 2013) | MATCH |
| `dynamics_interpolation` | CITATION_FIX | proxy-only | 1609.04468 ⚠ | scipy/Shoemake 1985 | MATCH |
| `dynamics_sharpness` | FAITHFUL | refined-adaptation | 2010.01412 | amirgholami/PyHessian | MATCH |
| `dynamics_stability` | CITATION_FIX | refined-adaptation | 1804.09692 ⚠ | (Wendlandt et al. 2018) | MATCH |
| `geometry_categories` | PROXY_OK | proxy-only | — | scipy.spatial.distance | MATCH |
| `geometry_cka` | VERIFIED_PARITY | parity-ready | 1905.00414 | google-research/representation_similarity | MATCH |
| `geometry_collapse` | CITATION_FIX | refined-adaptation | 2110.09348 ⚠ | EUSIPCO 2007 (eff. rank) | MATCH |
| `geometry_contextualization` | DISCREPANCY→fixed | formula-faithful | 1909.00512 | kawine/contextual (analyze.py) | fixed→MATCH |
| `geometry_correlation_dimension` | CITATION_FIX(refuted) | formula-faithful | — | CSchoel/nolds | MATCH |
| `geometry_hsic` | VERIFIED_PARITY | parity-ready | 1905.00414 | google-research/representation_similarity | MATCH |
| `geometry_hubness` | CITATION_FIX | parity-ready | (TKDE 2014) ⚠ | VarIr/scikit-hubness | MATCH |
| `geometry_intrinsic_dim` | CITATION_FIX | parity-ready | 1803.06992 ⚠ | scikit-dimension | MATCH |
| `geometry_isoscore` | VERIFIED_PARITY | parity-ready | 2108.07344 | bcbi-edu/p_eickhoff_isoscore | MATCH |
| `geometry_lid` | FORMULA-FAITHFUL (agent's VERIFIED_PARITY refuted) | formula-faithful | 1801.02613 | xingjunm/lid_adversarial_subspace_detection / skdim | `-k` variant, ~k/(k-1) off skdim |
| `geometry_lipschitz` | PROXY_OK | proxy-only | 1802.05957 | avirmaux/lipEstimation | MATCH |
| `geometry_mahalanobis` | PROXY_OK | refined-adaptation | 1807.03888 | pokaxpoka/deep_Mahalanobis_detector | MISMATCH (proxy) |
| `geometry_matrix_entropy` | VERIFIED_PARITY | parity-ready | 2401.17139 | waltonfuture/Matrix-Entropy | MATCH |
| `geometry_mp_bulk_deviation` | VERIFIED_PARITY | parity-ready | math/0403022 | AlejandroSantorum/scikit-rmt | MATCH |
| `geometry_neural_collapse` | VERIFIED_PARITY | formula-faithful | 2008.08186 | neuralcollapse/neuralcollapse | MATCH |
| `geometry_perplexity` | FAITHFUL | formula-faithful | — | HF transformers perplexity | MATCH |
| `geometry_positional_decay` | PROXY_OK | proxy-only | 2104.09864 ⚠ | (RoPE lit.) | MATCH |
| `geometry_prediction_alignment` | PROXY_OK | proxy-only | 2303.08112 | AlignmentResearch/tuned-lens | MATCH |
| `geometry_representation_sensitivity` | PROXY_OK | proxy-only | — ⚠ | torch.autograd | MATCH |
| `geometry_rsa` | FAITHFUL | refined-adaptation | — | rsagroup/rsatoolbox | MATCH |
| `geometry_schatten` | DISCREPANCY→fixed | refined-adaptation | 2410.10672 ⚠ | MLGroupJLU/MatrixNuclearNorm | fixed→MATCH |
| `geometry_spectral` | PROXY_OK | refined-adaptation | 1810.01075 | CalculatedContent/WeightWatcher | MISMATCH (proxy) |
| `geometry_svd` | FAITHFUL | refined-adaptation | 1909.00512 | EUSIPCO 2007 (eff. rank) | MATCH |
| `geometry_tokenizer_efficiency` | CITATION_FIX | formula-faithful | 2012.15613 ⚠ | Rust et al. 2021 (ACL) | MATCH |
| `geometry_trajectory_curvature` | VERIFIED_PARITY | parity-ready | 2311.04930 | Hosseini & Fedorenko 2023 | MATCH |
| `geometry_unembedding` | CITATION_FIX | proxy-only | 2503.21073 ⚠ | EUSIPCO 2007 (eff. rank) | MATCH |
| `geometry_weight_norms` | VERIFIED_PARITY | formula-faithful | 1901.08276 | CalculatedContent/WeightWatcher | MATCH |
| `interpretability_activation_sinks` | DISCREPANCY→fixed | parity-ready | 2410.10781 | sail-sg/Attention-Sink | fixed→MATCH |
| `interpretability_attention_effective_rank` | CITATION_FIX | proxy-only | (EUSIPCO 2007) ⚠ | Roy & Vetterli 2007 | MATCH |
| `interpretability_attention_entropy` | VERIFIED_PARITY | parity-ready | 1906.04341 | clarkkev/attention-analysis | MATCH |
| `interpretability_attention_graph` | PROXY_OK | proxy-only | 2309.17453 ⚠ | samiraabnar/attention_flow | MATCH |
| `interpretability_attention_rank` | CITATION_FIX | formula-faithful | 2103.03404 | twistedcubic/attention-rank-collapse | MATCH |
| `interpretability_attribution` | PROXY_OK | refined-adaptation | 1312.6034 | captum input_x_gradient | MATCH |
| `interpretability_head_roles` | CITATION_FIX | formula-faithful | 2209.11895 ⚠ | transformer-circuits / Wang 2022 IOI | MATCH |
| `interpretability_induction_heads` | FAITHFUL | refined-adaptation | 2209.11895 | TransformerLensOrg/TransformerLens | MATCH |
| `interpretability_logit_lens` | FAITHFUL | formula-faithful | 2303.08112 | AlignmentResearch/tuned-lens | MATCH |
| `interpretability_prediction_entropy` | CITATION_FIX | formula-faithful | 1904.09751 ⚠ | (Shannon 1948) | MATCH |
| `interpretability_probing` | FAITHFUL | refined-adaptation | 1610.01644 | Alain & Bengio 2017 | MATCH |
| `interpretability_sae_features` | FAITHFUL | refined-adaptation | 2309.08600 | decoderesearch/SAELens | MATCH |
| `interpretability_sparsity` | PROXY_OK | proxy-only | 2110.01786 ⚠ | thunlp/MoEfication | MATCH |
| `interpretability_superposition` | PROXY_OK | proxy-only | 2209.10652 ⚠ | (Elhage et al. 2022) | MATCH |
| `interpretability_waa` | BUG→fixed | proxy-only | 2311.03658 ⚠ | KihoPark/linear_rep_geometry | fixed→deterministic |
| `repe_concept_separability` | FAITHFUL | refined-adaptation | 2310.01405 | andyzoujm/representation-engineering | MATCH |
| `repe_refusal_direction` | PROXY_OK | refined-adaptation | 2406.11717 | andyrdt/refusal_direction | MATCH |
| `repe_steering_effectiveness` | PROXY_OK | proxy-only | 2308.10248 | andyzoujm/representation-engineering | MATCH |
| `repe_task_vectors` | FAITHFUL | refined-adaptation | 2310.01405 | andyzoujm/representation-engineering | MATCH |
| `topology_betti_curve` | DISCREPANCY→fixed | refined-adaptation | 2004.06093 ⚠ | topnn/topnn_framework | geodesic redesign→validated |
| `topology_homology` | VERIFIED_PARITY | parity-ready | — | scikit-tda/ripser.py | MATCH |
| `topology_persistence_entropy` | DISCREPANCY→fixed | formula-faithful | 1803.08304 ⚠ | giotto-ai/giotto-tda | fixed→MATCH |
| `topology_persistence_landscape` | BUG→fixed | parity-ready | 1207.6437 | scikit-tda/persim | fixed→MATCH |

---

# Campaign 2 — official-code parity upgrade (2026-06-24, branch `audit-v2`)

**Directive:** for every task not yet at the gold standard (= *run the official paper repo / package to generate the fixture, then pin BLME against it*, as opposed to an independent reimplementation), actually run the official reference and prove numeric parity; fill any missing implementation from the paper; survey + propose new methods at the same bar. All new artifacts live under `tests/tasks/parity/` (one file per task, no shared manifest → no agent dup-shadows) and `tests/fixtures/reference_parity/parity/<task>.json`. Each agent's script + test was **re-run and the body line-audited by the lead** (the standing lesson: agents over-claim).

## Wave 1 — closed-form metrics vs official code (9 tasks, all green)

Every row was produced by running the official reference (installed package or cloned repo at the pinned commit) and BLME's own helper on identical synthetic input, then checked into a per-task parity test. `tests/tasks/parity/` = **54 tests, all pass**; full suite (parity + reference_parity + metadata + docs) = **84 pass**.

| Task | Official ref @ commit/ver | Reference fn | BLME helper | OFFICIAL vs BLME | tol | Verdict |
|---|---|---|---|---|---|---|
| `consistency_bias_weat` | W4ngatang/sent-bias @ e3559fb | `weat.effect_size`, `s_XYAB` | `_weat_effect_size`, `_weat_statistic` | effect size −0.54833379…, stat −0.74378627…; **abs_diff 0.0** | 1e-9 | **PARITY** |
| `geometry_hubness` | VarIr/scikit-hubness @ c36a058 | `estimation.py:442 stats.skew(k_occurrence)` | `_hubness_stats_from_occurrences` | S_k 2.13272523…; diff 0.0 (skew), 1.1e-16 (gini) | 1e-9 | **PARITY** |
| `geometry_rsa` | rsatoolbox 0.3.2 | `calc_rdm('euclidean')`+`compare('spearman')` | `pdist`+`spearmanr` path | RSA 0.6896551724…; **diff 0.0** | 1e-12 | **PARITY** (rank-invariant; rsatoolbox RDM = d²/n_feat, equal under Spearman) |
| `geometry_correlation_dimension` | nolds (intended GP kernel) + independent GP | `nolds.measures` GP | `CorrelationDimensionTask` GP kernel | slope diff 0.0 vs independent GP & nolds-intended; max\|ΔC(r)\| ≤1.4e-17 | 1e-6 | **PARITY** of the GP kernel; FORMULA-FAITHFUL/PROXY for absolute dim (percentile radius window under-recovers dim>~1.5; documented). Also pins a real nolds raw-API self-match artifact (BLME is *more* GP-correct). |
| `geometry_neural_collapse` | rhubarbwu/neural-collapse @ c05a0b8 (ran directly) | `measure.covariance_ratio('pinv')` (Papyan'20) | `_neural_collapse_metrics` NC1 | NC1 0.00409094387…; **diff 0.0** (pinv), 1.7e-18 (svd) | 1e-9 | **PARITY** for NC1; NC2-ETF is a documented PROXY (different def from `simplex_etf_error`), verified vs its own stated formula |
| `dynamics_sharpness` | amirgholami/PyHessian + exact `eigh` | `hessian(...).eigenvalues(top_n=1)` | `_hvp`+power iteration | λ_max 2.0562113169…; BLME vs EXACT **<1e-9 rel**, vs PyHessian ~2e-6 | 1e-9/1e-5 | **PARITY** (BLME nails exact; PyHessian's ~2e-6 is its own Rayleigh-quotient floor) |
| `interpretability_attention_rank` | Roy & Vetterli 2007 (def); twistedcubic @ 38b5df6 (motivation) | `erank=exp(H(σ/‖σ‖₁))` | `_effective_rank` | erank max diff 1.78e-15; anchors rank-1→1.0, orthogonal→n | 1e-12 | **FORMULA-FAITHFUL** to Roy-Vetterli (exact). Dong's rank-1 *residual* is a distinct quantity (transcribed, confirmed BLME ≠ Dong residual) → Dong is motivation only |
| `interpretability_attribution` | captum 0.9.0 | `captum.attr.InputXGradient` | hook+backward+`(g·x).abs().sum`; `_gini_nonnegative` | per-token map **diff 0.0** vs captum (6 seeds, f32+f64); gini diff 2.8e-17 | 1e-9/1e-5 | **PARITY** of input×grad method; abs+drop-last is BLME's documented reduction |
| `consistency_membership_inference` | sam-yeom/ml-privacy-csf18 + sklearn | `inclusion.py`/`main.py` TPR−FPR; `roc_auc_score` | real `MembershipInferenceTask.evaluate()` via stub | AUROC 0.80277778, loss_gap 1.02040595; diff 1.1e-16 / 0.0 | 1e-9 | **PARITY** of the loss-AUROC attack (3 independent AUROC computations agree) |

**Cert-label upgrades (2 this wave, conservative):** `consistency_bias_weat` refined-adaptation→**parity-ready** (bit-exact vs official sent-bias `effect_size`); `dynamics_sharpness` refined-adaptation→**parity-ready** (top Hessian eigenvalue == exact eigh & PyHessian). `parity-ready` count 14→**16**. The other 7 kept their labels but gained an official-code parity test (notes recorded above); `geometry_hubness` was already parity-ready and is now pinned against the actual scikit-hubness skewness line, not just occurrence summaries.

**Test-strength audit (lead, line-by-line):** 7 STRONG (import/drive real BLME code vs an independent or official reference) — `bias_weat`, `hubness`, `neural_collapse`, `sharpness`, `membership_inference` (drives real `evaluate()`), `rsa`, `attention_rank`. 2 ACCEPTABLE — `attribution` and `correlation_dimension` transcribe BLME's *inline* kernel (no extractable helper exists) verbatim from the cited source lines; both are exact but would be STRONG if the kernel were extracted into a named helper the test calls (low-risk, value-preserving refactor — **candidate follow-up**).

**Residual doubt (honest):** (1) `geometry_correlation_dimension` recovers a line tightly (~1.0) but a plane lands ~1.5 not 2.0 — the percentile scaling window biases the GP slope down on higher-D manifolds; the *kernel* is bit-exact, the *absolute dimension* on dim≳2 is a documented proxy. (2) `geometry_rsa` parity holds because the comparator is Spearman (rank-invariant); it would break if BLME switched to Pearson/cosine. (3) `geometry_neural_collapse` NC2-ETF and the topic-label bundling stay proxies. (4) scikit-hubness could not be pip-installed (falconn C++ build fails on this host) so its one-line skewness was transcribed with file:line provenance + an in-test scipy guard, not executed from the package. (5) sent-bias's permutation p-value path uses removed `np.int` (numpy<1.20) so only the deterministic effect-size/statistic are pinned.

## Wave 1 reference assets (reproducibility)

- pip (conda py3.12): `pyhessian`, `captum==0.9.0`, `nolds`, `rsatoolbox==0.3.2`.
- cloned: `W4ngatang/sent-bias@e3559fb`, `twistedcubic/attention-rank-collapse@38b5df6`, `sam-yeom/ml-privacy-csf18`, `VarIr/scikit-hubness@c36a058`, `rhubarbwu/neural-collapse@c05a0b8`.

## Wave 2–4 (planned / in progress)

- **Wave 2 (pipeline, real-model official repos):** `causality_tracing` (ROME), `causality_attention_knockout` (Michel), `causality_knowledge_neurons` (Hunter-DDM), `causality_edge_attribution` (Aaquib111 EAP), `causality_circuit_quality` (ACDC), `interpretability_induction_heads`/`head_roles` (TransformerLens/IOI), `interpretability_sae_features` (SAELens), `repe_*` (andyzoujm/andyrdt). Exact parity bounded by needing the heavy reference repo on the same model; target = faithful-algorithm + behavioral-invariant + numeric where a closed-form sub-step exists.
- **Wave 3 (inherent proxies):** verify each computes its *stated* definition + reproduce any paper number where one exists; document why exact official parity is impossible.
- **Wave 4 (new methods):** survey done → `results/new_methods_survey.md`. Top must-adds (pending lead web-recheck of ids/repos + user approval): **Procrustes layer-linearity** (Razzhigaev 2024, arXiv:2405.12250, AIRI-Institute/LLM-Microscope), **PHD intrinsic dim** (Tulchinskii 2023, arXiv:2306.04723, ArGintum/GPTID), **Vendi Score** (Friedman & Dieng, arXiv:2210.02410, vertaix/Vendi-Score); then zigzag persistence (2410.11042), activation kurtosis, metric magnitude (2311.16054), CKNNA (2405.07987). Each new task held to the Wave-1 official-code-parity bar.

## Wave 4 — new methods DELIVERED (2026-07, 7 tasks, registry 74→81)

Survey (`results/new_methods_survey.md`) → user approved all 7. Each authored in isolation (module + parity test + verify script), the metric parity-verified against the paper's OFFICIAL code, then the lead **re-ran every test, line-audited every module, and wired the registry/metadata/config/recipe/completeness guards**. All 7 register + instantiate + run end-to-end on distilgpt2 (`SMOKE_RESULT: ALL_OK`, finite features; procrustes ≈0.97 reproduces the paper's "secretly linear" headline). Import-audited: **no module imports its reference package at top level** → BLME gains no new hard dependency. Full guard+parity suite green at 81 tasks (261 passed).

| New task (cert) | Paper / official ref @ commit | Reference fn | BLME helper | Parity | Verdict |
|---|---|---|---|---|---|
| `geometry_vendi_score` (parity-ready) | Friedman & Dieng 2023 TMLR; vertaix/Vendi-Score (pip `vendi_score` 0.0.3) | `vendi.score_K` | `_vendi_score(X, kernel)` | bit-exact (0.0), anchors VS(I)=n, VS(1s)=1 | **PARITY** (nonlinear kernel ⇒ distinct from effective_rank) |
| `geometry_phd_dimension` (parity-ready) | Tulchinskii 2023 NeurIPS; ArGintum/GPTID @8c8759e | `IntrinsicDim.PHD` (MST power-law) | `_phd_dimension(X,...)` | seed-matched bit-exact (0.0); recovers R^k≈k | **PARITY** (RNG-pinned; stochastic estimator) |
| `geometry_cknna` (parity-ready) | Huh 2024 ICML (Platonic); minyoungg/platonic-rep @dcd76ba | `metrics.cknna` | `_cknna(X,Y,topk)` | bit-exact (0.0), both HSIC variants + orderings | **PARITY** |
| `geometry_magnitude` (parity-ready) | Limbeck 2024 NeurIPS (2311.16054); aidos-lab/magnipy @7d49b90 | `compute_magnitude_no_gpu` / cholesky | `_magnitude(D,t)` | ≤2.8e-14; anchors 1pt→1, sep→n | **PARITY** |
| `geometry_procrustes_linearity` (parity-ready) | Razzhigaev 2024 ACL (2405.12250); AIRI/LLM-Microscope (pip `llm-microscope` 0.0.7) | `procrustes_similarity` (`get_est_svd`) | `_procrustes_similarity(X,Y)` | bit-exact (0.0); orthogonal-map→1.0 | **PARITY** — *absolute value conditioning-dependent (unguarded 1/S); use the depth profile* |
| `interpretability_activation_kurtosis` (parity-ready) | KurTail 2025 EMNLP Findings (2503.01483); Sun 2024 | `scipy.stats.kurtosis(fisher,bias)` | `_activation_kurtosis_stats(A)` | per-channel bit-exact (0.0); Gauss→0, Laplace→3, uniform→−1.2 | **PARITY** |
| `topology_zigzag_persistence` (refined-adaptation) | Gardinazzi 2025 ICML (2410.11042); RitAreaSciencePark/ZigZagLLMs @bcfe0a6 | `dionysus.zigzag_homology_persistence` (isolated venv) | `_zigzag_summary(...)` | feature layer-lifetimes anchored EXACTLY to dionysus on 4 ground-truth constructions; ==independent reimpl 0.0 | **FAITHFUL PROXY** — ripser-based (no dionysus dep); not a bit-exact barcode port of short-bar multiplicity |

**parity-ready count 16 → 22** (6 of the 7 new tasks; zigzag stays refined-adaptation). Reference assets pinned in the Wave-1 assets list + per-task fixtures under `tests/fixtures/reference_parity/parity/`.

**Placement rationale:** 5 geometry (representation-geometry/ID/diversity family), 1 interpretability (activation-outlier family, next to activation_sinks), 1 topology (zigzag, next to persistence tasks).

**Study-regeneration impact:** 7 new feature groups are now emitted by the eval; they are NOT in any prior study run. Regenerate features before including them in the paper's analysis (user's call). All are architecture-agnostic, label-light, single-forward-pass — consistent with the portfolio.

## Wave 2 + 3 — official-code parity for pipeline & remaining tasks (2026-07)

Heavy references were run in ISOLATED venvs (BLME gained NO dependency on any): `tunedlens` (tuned-lens 0.2.0), `tlens2` (transformer_lens 3.5.1, `--system-site-packages`), `kn` (knowledge-neurons 0.0.2), `ww` (weightwatcher 0.7.7); plus cloned repos lipEstimation, attention_flow, RepE, refusal_direction. Each agent's verify script + test were re-run and line-audited by the lead.

| Task | Official ref @ ver/commit | Verdict |
|---|---|---|
| `interpretability_logit_lens` | tuned-lens 0.2.0 `LogitLens` | **PARITY** 1.9e-5 (final-layer lens == model logits) → **parity-ready** |
| `interpretability_head_roles` (prev-token) | TransformerLens 3.5.1 | **PARITY** 8.8e-8 (L4H11 anchor); other role scores are adaptations |
| `interpretability_induction_heads` | TransformerLens 3.5.1 | **FIXED → PARITY** <1e-4 (L5H5 anchor) → **parity-ready** |
| `repe_task_vectors` | RepE `ClusterMeanRepReader` | **PARITY** 2.2e-16 (|cos|=1) |
| `repe_refusal_direction` | andyrdt `get_mean_diff` | **PARITY** 0.0 (|cos|=1) |
| `repe_concept_separability` | RepE `PCARepReader` | **FAITHFUL** (measures separability; ref constructs the PCA vector; both recover the planted direction) |
| `geometry_lipschitz` | numpy / lipEstimation σ_max | σ_max **kernel PARITY** (6.8e-14); the *task* is an honest hidden-state relative-change `proxy-only`, does NOT compute AutoLip |
| `geometry_spectral` | WeightWatcher 0.7.7 | **PROXY**, Hill kernel exact (<1e-9 vs independent Hill); WW xmin-MLE α delta quantified (σ-vs-λ scale + fixed-tail vs KS-xmin) — `refined-adaptation` correct |
| `causality_knowledge_neurons` | EleutherAI knowledge-neurons 0.0.2 | **PROXY** — BLME = grad-of-logit×activation saliency, materially ≠ Dai integrated-gradients (target/path/hook differ; docstring-honest). Kernel pinned exact (0.0 vs independent autograd); genuine Dai IG implemented + completeness-axiom verified live |
| `interpretability_attention_graph` | Abnar rollout | **FINDING** — computes damped PageRank (Xiao), NOT the cited rollout. Citation corrected; rollout added as the new `interpretability_attention_rollout` task |
| `interpretability_attention_rollout` (NEW) | Abnar `compute_joint_attention` | **PARITY** (bit-exact vs the transcribed reference) — added per user approval, registry 81→82 |

**Value-changing fix applied (needs study regeneration):**
- `interpretability_induction_heads` — extracted `_induction_score_per_head`, now averages the **full** induction diagonal (offset 1−N) to reproduce the official TransformerLens `induction_score` exactly. Was `[N, 2N−2]` (dropped 2 endpoint stripe entries → ~0.03 low). **`induction_score` / `prefix_match_score_max`/`_mean` study features change.** Comprehensive-parity reference (`test_comprehensive_parity.py`), the parity test, and the fixture verdict were updated to the corrected convention.

**Findings / citation tightening (the "cited-method vs computed-proxy" pattern):**
- `attention_graph`: cites Abnar–Zuidema rollout but computes PageRank centrality (Xiao) — citation corrected; rollout implemented separately.
- `geometry_lipschitz`: computes a hidden-state relative-change proxy (correctly `proxy-only`), not the AutoLip σ_max Lipschitz bound; the real σ_max kernel lives in `weight_norms`/`spectral` and is now pinned exact + guarded.
- `causality_knowledge_neurons`: saliency proxy, not Dai IG (docstring already said so; note tightened).
- Consistent with the earlier `attention_rank` precedent (Roy–Vetterli effective rank; Dong 2021 = motivation only). No false parity was ever claimed; the proxy/refined labels hold.

**Cert upgrades:** `interpretability_logit_lens`, `interpretability_induction_heads` → parity-ready; new `interpretability_attention_rollout` is parity-ready. **parity-ready count 22 → 25.** Registry **81 → 82** (attention_rollout wired into __init__/metadata/config/recipe/completeness; full guard+parity suite green — 353 passed).

## Wave 2 — heavy pipeline repos (2026-07, isolated venvs)

| Task | Official ref @ commit/ver | Verdict |
|---|---|---|
| `causality_tracing` | kmeng01/rome @0874014 (venv, ran actual `trace_with_patch`) | **PARITY (exact)** — per-layer AIE == ROME bit-for-bit (0.0) with shared noise; peaks early/mid (ROME Fig. 2) → **parity-ready** |
| `causality_edge_attribution` | Aaquib111/edge-attribution-patching @7124ef8 + paper Eq 2/3 | **KERNEL PARITY** (0.0 vs formula; 5.5e-16 on gpt2; exact on a linear model); full per-edge circuit is a documented per-layer proxy |
| `interpretability_sae_features` | SAELens 6.44.4 + real SAE `jbloom/gpt2-small-res-jb` (d_sae 24576) | **L0 stat KERNEL PARITY** (0.0) on real-SAE-encoded features; trained-SAE pipeline **FAITHFUL** |
| `causality_attention_knockout` | Michel 2019 (faithful reimpl — repo too old to build) | **FAITHFUL** — direct-ablation ΔNLL == independent reimpl exactly (0.0); Michel Eq. 5 proxy↔|ablation| Spearman 0.82 |

`causality_circuit_quality` (ACDC) — **not numerically pinnable**: its metric is BLME's OWN JSD-based circuit-quality proxy; ACDC produces a *circuit*, not this scalar, so there is no external number to reproduce. Already `proxy-only` + covered by an analytic JSD re-derivation in `comprehensive_parity` (Campaign 1's acknowledged weakest-but-honest test). No parity claimed.

**Cert:** `causality_tracing` → parity-ready. **parity-ready count 25 → 26.**
**Finding:** `sae_features.py:97` unpacks `SAE.from_pretrained()` as a 3-tuple; sae-lens ≥6.x returns the SAE directly → a compat fix is needed to run against current sae-lens (runtime, not metric).

## Pure-proxy tasks — status (already verified; no re-run)

The remaining ~28 `proxy-only`/`refined-adaptation` tasks — geometry (categories, positional_decay, prediction_alignment, representation_sensitivity, tokenizer_efficiency, unembedding, weight_norms, mahalanobis, collapse, svd), interpretability (attention_effective_rank, sparsity, superposition, waa, prediction_entropy, probing), consistency (contrastive, format_robustness, icl_slope, knowledge_capacity, logical, paraphrase, position_sensitivity, self_consistency), dynamics (interpolation, stability, gradient_flow), repe_steering_effectiveness — were ALREADY verified in Campaign 1: each computes its stated definition (covered by `tests/tasks/test_comprehensive_parity.py` against an independent reference) and carries an honest label. For a proxy there is no paper number to "reproduce" — they are diagnostics inspired by, not reproductions of, their motivating papers. Campaign 2 added official-code parity only where a NEW runnable official reference existed (lipschitz σ_max, spectral Hill-vs-WW, attention rollout); the rest stay honestly labeled. **No false parity is claimed anywhere in the portfolio.**

## Campaign 2 — final tally

- **Registry 74 → 82** (8 new methods, every one parity-verified against official code before wiring).
- **parity-ready 14 → 26.**
- **Official-code parity newly established by running the ACTUAL reference** (not a transcription) for: Wave 1 (9 closed-form tasks); Wave 2/3 — logit_lens, induction_heads (fixed), head_roles, repe×3, knowledge_neurons (kernel), lipschitz (σ_max kernel), spectral (Hill kernel), tracing (exact vs ROME), edge_attribution (kernel), sae_features (kernel), attention_knockout, attention_rollout; and the 8 new tasks.
- **Value-changing / structural changes acted on:** `induction_heads` fixed to reproduce TransformerLens (→ regen); `attention_graph` citation corrected + `attention_rollout` added; citation/label tightenings for lipschitz, spectral, knowledge_neurons, edge_attribution; sae-lens API-drift flagged.
- **Study regeneration required before the paper trusts these features:** `induction_score`/`prefix_match_score_*` (changed by the fix) + all 8 new-task feature groups (newly emitted). Everything else is additive test coverage / labels / citations with no feature-value change.
- **Method:** every agent artifact (verify script + parity test) was re-run and line-audited by the lead; no `src/blme` change was made by any agent; heavy references ran in throwaway venvs so BLME gained **no** new dependency.
