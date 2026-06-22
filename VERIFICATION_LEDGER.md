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
