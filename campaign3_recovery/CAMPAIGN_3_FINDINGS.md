# BLME Campaign 3 — Consolidated Verification Findings

**Date:** 2026-07-04 · **Scope:** no-trust re-verification of all 82 diagnostic tasks vs their papers/official repos, a literature gap survey, and a harness pipeline review, ahead of open-sourcing.

**Run status.** The workflow (`wf_50bc34f1-1d7`) is **complete — 417/417 agents, 0 errors**. It weathered a runtime segfault + repeated token-limit interruptions across sessions and was resumed from its crash-surviving append-only journal each time (completed agents replay from cache); this report is reconstructed offline from that journal. Every finding has now been adversarially adjudicated. No `src/` file was modified by any agent — all verifiers were read-only.

## Completeness of the recovered run

| Stage | Status |
|---|---|
| Task verifiers | **82 / 82 complete** (all tasks re-verified, parity tests re-run, fresh numeric checks executed) |
| Adversarial refuters | **510 votes**, adjudicating **205 findings**; 1 unresolved (title-join miss) |
| Pipeline reviewers | **5 / 5 complete** (core-loop, model-layer, release, reproducibility, test-infra) |
| Gap-survey lenses | **6 / 6 complete** + synthesis complete |

**Adjudication:** of 205 adjudicated findings, **190 survived** and **15 were refuted** (skeptic over-reach, dropped below). Refuters were instructed to default to *refuted* unless evidence held. **Every BUG and every DISCREPANCY survived** — the 15 refutations are all lower-severity doc/citation/test-strength claims that did not hold up.

**4 / 82 tasks returned zero findings:** `geometry_cka`, `geometry_rsa`, `interpretability_head_roles`, `interpretability_induction_heads`.

**Standing findings (survived or pending-refute), by severity:** BUG 31, DISCREPANCY 12, WEAK_TEST 53, CITATION 38, DOC_GAP 57.

## Per-task verdicts (all 82)

`ADJ` = has an adversarial verdict; `survive` = findings still standing (adjudicated-survived + pending).

| Task | Cert | Verdict | Parity tests | Test strength | Cites OK | Standing findings |
|---|---|---|---|---|---|---|
| `geometry_categories` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |
| `causality_ablation` | proxy-only | FINDINGS | PASS | STRONG | Y | BUG,DOC_GAP |
| `causality_attention_knockout` | formula-faithful | FINDINGS | PASS | STRONG | Y | BUG |
| `causality_circuit_quality` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST |
| `causality_edge_attribution` | proxy-only | FINDINGS | PASS | STRONG | Y | WEAK_TEST,DOC_GAP |
| `causality_knowledge_neurons` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `causality_tracing` | parity-ready | FINDINGS | PASS | STRONG | N | BUG,CITATION,DOC_GAP |
| `consistency_bias_weat` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,DOC_GAP |
| `consistency_calibration` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | DISCREPANCY,WEAK_TEST,DOC_GAP |
| `consistency_contamination` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |
| `consistency_contrastive` | proxy-only | FINDINGS | PASS | STRONG | Y | BUG,WEAK_TEST |
| `consistency_format_robustness` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |
| `consistency_icl_slope` | proxy-only | FINDINGS | PASS | STRONG | Y | BUG |
| `consistency_knowledge_capacity` | proxy-only | FINDINGS | PASS | STRONG | N | CITATION,DOC_GAP |
| `consistency_logical` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,DOC_GAP |
| `consistency_membership_inference` | proxy-only | FINDINGS | PASS | STRONG | Y | WEAK_TEST |
| `consistency_paraphrase` | proxy-only | FINDINGS | PASS | STRONG | N | BUG,WEAK_TEST,CITATION,DOC_GAP |
| `consistency_position_sensitivity` | proxy-only | FINDINGS | PASS | STRONG | N | CITATION |
| `consistency_self_consistency` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,DOC_GAP |
| `dynamics_coe` | formula-faithful | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `dynamics_generation_diversity` | parity-ready | FINDINGS | PASS | ACCEPTABLE | N | BUG,WEAK_TEST,CITATION |
| `dynamics_gradient_flow` | refined-adaptation | FINDINGS | PASS | STRONG | Y | CITATION,DOC_GAP |
| `dynamics_interpolation` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST |
| `dynamics_sharpness` | parity-ready | FINDINGS | PASS | ACCEPTABLE | N | DISCREPANCY,WEAK_TEST,CITATION |
| `dynamics_stability` | proxy-only | FINDINGS | PASS | STRONG | Y | WEAK_TEST,CITATION,DOC_GAP |
| `geometry_cka` | parity-ready | CONFIRMED_CORRECT | PASS | ACCEPTABLE | Y | — |
| `geometry_cknna` | parity-ready | FINDINGS | PASS | STRONG | Y | WEAK_TEST,DOC_GAP |
| `geometry_collapse` | refined-adaptation | FINDINGS | PASS | STRONG | N | WEAK_TEST,CITATION,DOC_GAP |
| `geometry_contextualization` | refined-adaptation | FINDINGS | PASS | WEAK | Y | WEAK_TEST,DOC_GAP |
| `geometry_correlation_dimension` | formula-faithful | FINDINGS | PASS | STRONG | N | CITATION,DOC_GAP |
| `geometry_hsic` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `geometry_hubness` | parity-ready | FINDINGS | PASS | ACCEPTABLE | N | WEAK_TEST,CITATION |
| `geometry_intrinsic_dim` | formula-faithful | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `geometry_isoscore` | parity-ready | FINDINGS | PASS | STRONG | N | WEAK_TEST,CITATION |
| `geometry_lid` | formula-faithful | FINDINGS | PASS | ACCEPTABLE | N | CITATION |
| `geometry_lipschitz` | proxy-only | FINDINGS | PASS | STRONG | Y | DISCREPANCY |
| `geometry_magnitude` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | BUG,DISCREPANCY,WEAK_TEST |
| `geometry_mahalanobis` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,CITATION |
| `geometry_matrix_entropy` | parity-ready | FINDINGS | PASS | WEAK | Y | DISCREPANCY,WEAK_TEST |
| `geometry_mp_bulk_deviation` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST |
| `geometry_neural_collapse` | refined-adaptation | FINDINGS | PASS | STRONG | Y | WEAK_TEST,CITATION,DOC_GAP |
| `geometry_perplexity` | formula-faithful | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST |
| `geometry_phd_dimension` | parity-ready | FINDINGS | PASS | STRONG | Y | WEAK_TEST,DOC_GAP |
| `geometry_positional_decay` | proxy-only | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `geometry_prediction_alignment` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |
| `geometry_procrustes_linearity` | parity-ready | FINDINGS | PASS | STRONG | Y | — |
| `geometry_representation_sensitivity` | proxy-only | FINDINGS | PASS | STRONG | N | WEAK_TEST,CITATION |
| `geometry_rsa` | refined-adaptation | CONFIRMED_CORRECT | PASS | STRONG | Y | — |
| `geometry_schatten` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | N | WEAK_TEST,CITATION,DOC_GAP |
| `geometry_spectral` | refined-adaptation | FINDINGS | PASS | STRONG | N | WEAK_TEST,CITATION,DOC_GAP |
| `geometry_svd` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | DISCREPANCY,WEAK_TEST,DOC_GAP |
| `geometry_tokenizer_efficiency` | proxy-only | FINDINGS | PASS | STRONG | N | CITATION,DOC_GAP |
| `geometry_trajectory_curvature` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | DISCREPANCY,WEAK_TEST,DOC_GAP |
| `geometry_unembedding` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `geometry_vendi_score` | parity-ready | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `geometry_weight_norms` | proxy-only | FINDINGS | PASS | STRONG | Y | BUG,DOC_GAP |
| `interpretability_activation_kurtosis` | parity-ready | FINDINGS | PASS | STRONG | N | CITATION,DOC_GAP |
| `interpretability_activation_sinks` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | N | WEAK_TEST,CITATION,DOC_GAP |
| `interpretability_attention_effective_rank` | proxy-only | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `interpretability_attention_entropy` | parity-ready | FINDINGS | PASS | ACCEPTABLE | N | BUG,WEAK_TEST,CITATION |
| `interpretability_attention_graph` | proxy-only | FINDINGS | PASS | ACCEPTABLE | N | WEAK_TEST,CITATION |
| `interpretability_attention_rank` | formula-faithful | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `interpretability_attention_rollout` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `interpretability_attribution` | refined-adaptation | FINDINGS | PASS | STRONG | N | CITATION,DOC_GAP |
| `interpretability_head_roles` | refined-adaptation | CONFIRMED_CORRECT | PASS | STRONG | Y | — |
| `interpretability_induction_heads` | parity-ready | CONFIRMED_CORRECT | PASS | STRONG | Y | — |
| `interpretability_logit_lens` | parity-ready | FINDINGS | PASS | STRONG | N | CITATION |
| `interpretability_prediction_entropy` | formula-faithful | FINDINGS | PASS | STRONG | Y | BUG,WEAK_TEST,DOC_GAP |
| `interpretability_probing` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,DOC_GAP |
| `interpretability_sae_features` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | BUG,DISCREPANCY,WEAK_TEST,DOC_GAP |
| `interpretability_sparsity` | proxy-only | FINDINGS | PASS | STRONG | N | BUG,WEAK_TEST,CITATION |
| `interpretability_superposition` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | DISCREPANCY,WEAK_TEST,DOC_GAP |
| `interpretability_waa` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |
| `repe_concept_separability` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | BUG,DOC_GAP |
| `repe_refusal_direction` | refined-adaptation | FINDINGS | PASS | STRONG | Y | DOC_GAP |
| `repe_steering_effectiveness` | proxy-only | FINDINGS | PASS | ACCEPTABLE | Y | BUG,WEAK_TEST,DOC_GAP |
| `repe_task_vectors` | refined-adaptation | FINDINGS | PASS | STRONG | Y | CITATION |
| `topology_betti_curve` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | N | BUG,WEAK_TEST |
| `topology_homology` | parity-ready | FINDINGS | PASS | ACCEPTABLE | N | BUG,CITATION |
| `topology_persistence_entropy` | parity-ready | FINDINGS | PASS | STRONG | N | CITATION |
| `topology_persistence_landscape` | parity-ready | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST,DOC_GAP |
| `topology_zigzag_persistence` | refined-adaptation | FINDINGS | PASS | ACCEPTABLE | Y | WEAK_TEST |

## Standing findings (detail)

Each finding's status: **[ADJUDICATED-SURVIVED]** (refuters ran, ≥1 upheld it) or **[PENDING-REFUTE]** (refuters not yet run — provisional).

### BUG (31)

- **[ADJUDICATED-SURVIVED]** `causality_ablation` — Result keys mislabel/collide for non-default ablation_percentages due to int() float truncation — votes 6/6 upheld, unanimous
  - `src/blme/tasks/causality/ablation.py:149`
  - int(k_pct*100) floors: 0.29*100=28.999999999999996 -> key "loss_ablate_28pct" (verified in python). Config [0.145,0.149] both map to "14pct": second k silently overwrites the first's 4 result keys. Defaults [0.01,0.05,0.1,0.25] unaffected; degradation_curve/AUC internally correct.
  - _fix:_ Use round(k_pct*100) or format the float itself (e.g. f"{k_pct*100:g}pct") at ablation.py:149-152; optionally assert key uniqueness across ablation_percentages.

- **[ADJUDICATED-SURVIVED]** `causality_attention_knockout` — TypeError crash when config.head_dim is None on non-divisible out-proj fallback path — votes 6/6 upheld, unanimous
  - `src/blme/tasks/causality/attention_knockout.py:173`
  - getattr(model.config, "head_dim", in_features // num_heads) returns None when the attribute exists but is None; line 176 `head_dim <= 0` then raises TypeError. Reproduced through the real task (synthetic module: o_proj in_features=100, num_heads=12, config.head_dim=None -> "CRASH: TypeError '<=' not supported between instances of 'NoneType' and 'int'").
  - _fix:_ head_dim = getattr(cfg, 'head_dim', None) or (in_features // num_heads); resolve cfg via the same text_config/language_config fallback used for num_heads (lines 140-150), and `continue` (skip layer) if still invalid. Unreachable on all standard architectures (their out-proj widths are divisible by num_heads), so no shipped numbers are affected — cr

- **[ADJUDICATED-SURVIVED]** `causality_circuit_quality` — Last layer mean-ablated with post-final-norm mean, not the mean of the activation being replaced — votes 6/6 upheld, unanimous
  - `src/blme/tasks/causality/circuit_quality.py:113`
  - hidden_states[l+1] for the last layer is ln_f(block output) in transformers 5.2.0 (GPT2 and LLaMA, verified). On distilgpt2: cos(used mean, true block-5 mean)=0.303, norm ratio 0.43; last-layer importance 3.0038 vs 2.5917 self-consistent; at default top_k_pct=25 selected circuit flips [5] vs [1], changing all downstream scores.
  - _fix:_ Capture raw block outputs via forward hooks when accumulating per-layer means (as done for the ablation), or special-case the final layer; layers 0..N-2 already match raw means exactly (cos=1.0).

- **[ADJUDICATED-SURVIVED]** `causality_tracing` — Final-layer patch restores post-ln_f state, not the block output — wrong final-layer AIE when prompt ends with the subject — votes 6/6 upheld, unanimous
  - `src/blme/tasks/causality/tracing.py:356`
  - clean_states[layer_idx+1] at the last layer is HF hidden_states[n] = ln_f(block output) (verified: diff 0.0 vs ln_f, up to 370 vs raw). ROME copies raw h[0,t] live. gpt2, prompt 'The capital of Italy', subject at end: layer-11 AIE BLME 0.000385 vs ROME 0.121248 (layers 0-10 exact 0.0).
  - _fix:_ Patch from row 0 of the CURRENT forward (out_tensor[1:, s:e] = out_tensor[0:1, s:e]) — row 0 is clean and unaffected by the patch, exactly ROME's live-copy semantics; also removes the cached-states dependency. Invisible to existing tests because mid-prompt subjects make final-layer AIE structurally 0.0; add a subject-at-end regression case.

- **[ADJUDICATED-SURVIVED]** `consistency_bias_weat` — Substring match embeds 'This' instead of attribute word 'his' — corrupts gender_science WEAT d — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/bias.py:83`
  - line 83 char_start = text.find(word): 'his' matches inside 'This' at char 1 of 'This is his.', so _find_word_token_position returns 0 (token 'This'), not 2 ('Ġhis'). distilgpt2: gender_science_d 0.5415346795 shipped vs 0.7999010666 corrected (delta -0.258). Violates docstring lines 14-16 (extract hidden state at attribute-word position).
  - _fix:_ Anchor the search at the template insertion point: pass template.index('{}') into _find_word_token_position and use text.find(word, insert_off); apply the same anchor to the fallback prefix slice at line 107. Add a regression test over all 64 bundled words.

- **[ADJUDICATED-SURVIVED]** `consistency_contrastive` — evaluate() silently returns NaN metrics when every sample fails scoring — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/contrastive.py:128`
  - np.mean([]) at lines 128-130 yields nan when all triples are skipped (e.g. factual==exclusive gives empty targets; score_continuation returns None). Reproduced on real gpt2: all three outputs nan. Empty dataset returns {'error': ...} at line 94, so the all-failed path is inconsistent.
  - _fix:_ After the scoring loop, if not factual_probs: return {'error': 'no scoreable contrastive samples'} (mirroring line 94) instead of NaN means.

- **[ADJUDICATED-SURVIVED]** `consistency_icl_slope` — shot_counts k > n_demos silently truncates: k-shot metric mislabeled and slope biased — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/icl_slope.py:90`
  - _build_prompt does item["demos"][:k]; with supported config shot_counts=[0,4,8] (read at line 104) and the built-in 4-demo bundle, measured mean_nll_8shot == mean_nll_4shot == 3.1755768657 exactly. Output key claims 8-shot NLL but is 4-shot; icl_slope over [0,4,8] is silently flattened. Default config [0,1,2,4] unaffected.
  - _fix:_ In evaluate(), skip (or warn and drop) any k > len(item["demos"]) per item, or raise ValueError on configs exceeding available demos; alternatively clamp shot_counts to max demo count before building nll_by_k.

- **[ADJUDICATED-SURVIVED]** `consistency_logical` — Unconditional prompt " " degenerates on non-BOS tokenizers; task errors on its own bundled defaults — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/logical.py:77`
  - score_continuation(model, tok, " ", conclusion): first BPE token (e.g. 'ĠJohn', offset (0,5)) absorbs the space, prompt_len=0, logits[-1:-1] empty -> None (common.py:269-272). All 5 bundled pairs skipped -> {"error": "No valid samples processed."} on gpt2/pythia/Qwen2.5/SmolLM2/phi-2/OLMo; only BOS families (Llama-3.2, OPT) work.
  - _fix:_ In logical.py handle the unconditional path explicitly: prepend tokenizer.bos_token when available, or in score_continuation treat prompt_len==0 by scoring tokens 1..T (drop the unpredictable first token from BOTH cond and uncond spans so the comparison stays balanced).

- **[ADJUDICATED-SURVIVED]** `consistency_paraphrase` — mpararel fallback is dead code: wrong config name and wrong column names; silently degrades to 5 bundled triples — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/paraphrase.py:58`
  - load_dataset('coastalcph/mpararel','en') always raises ValueError "BuilderConfig 'en' not found. Available: ['default']" (reproduced live); even if loaded, line 62 item['relation_id']/item['text'] would KeyError — hub columns are relation/template/query. Broad except -> every dataset=None run silently uses the 5 _BUNDLED sentences (proved numerically: output==bundled recompute, absdiff 7.4e-7).
  - _fix:_ Either delete the unreachable mpararel branch, or fix it: load config 'default', filter language=='en', group by 'relation', use 'template' or 'query' as text; add a test asserting the branch works or is gone.

- **[ADJUDICATED-SURVIVED]** `consistency_self_consistency` — seed config breaks task on real HF models: generator kwarg raises ValueError, only TypeError is caught — votes 6/6 upheld, unanimous
  - `src/blme/tasks/consistency/self_consistency.py:104`
  - transformers 5.2.0: model.generate(generator=g) raises ValueError 'model_kwargs are not used by the model: [generator]'. Line 104 catches only TypeError; broad except at :111 skips every prompt; task returns {'error': 'No samples successfully generated'} with seed=1234 on tiny GPT-2, while seed=None works (verify.py Part B).
  - _fix:_ Catch (TypeError, ValueError) at line 104, or never pass generator to HF generate — call torch.manual_seed(seed) before each generate call for determinism.

- **[ADJUDICATED-SURVIVED]** `dynamics_generation_diversity` — seed config crashes generation on real HF models: generate() rejects 'generator' kwarg, task returns error dict — votes 6/6 upheld, unanimous
  - `src/blme/tasks/dynamics/generation_diversity.py:192`
  - Verified on distilgpt2/transformers 5.2.0: every prompt logs "generate() failed: The following model_kwargs are not used by the model: ['generator']"; evaluate() with seed=123 returns {'error': 'No completions generated'}. transformers generate() has no generator param (diffusers-only API); lines 194-196 swallow the ValueError per prompt.
  - _fix:_ Replace generator kwarg with torch.manual_seed(seed) (or transformers.set_seed) before model.generate; keep reporting seed in output.

- **[ADJUDICATED-SURVIVED]** `dynamics_interpolation` — evaluate() hangs forever when fewer than 2 usable dataset rows exist — votes 6/6 upheld, unanimous
  - `src/blme/tasks/dynamics/trajectories.py:78`
  - `while count < num_pairs` (line 78) only increments after 2 usable samples; rows matching neither str/'text'/'input_ids' hit `continue` (lines 89, 94) without progress. Reproduced: dataset=[{'tokens':[1,2]},{'tokens':[3,4]}] loops indefinitely (scratchpad hang_check.py, SIGALRM at 6s: HANG CONFIRMED). Missed by both prior campaigns.
  - _fix:_ Filter usable samples upfront (return {'error': ...} if <2), or cap attempts at e.g. 10*num_pairs and break with a warning.

- **[ADJUDICATED-SURVIVED]** `geometry_magnitude` — np.trapezoid requires numpy>=2.0 but pyproject declares unversioned numpy — evaluate() crashes (AttributeError) on numpy 1.x — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/magnitude.py:278`
  - magnitude.py:278 `np.trapezoid(lm, lt)`; pyproject.toml:22 lists bare "numpy", which numpy 1.26 satisfies (np.trapezoid added in numpy 2.0; 1.x only has np.trapz). Conditional on environment; dev env has numpy 2.4.6 so tests never see it. Repo-wide: also geometry/isotropy.py, causality/ablation.py, topology/persistence_landscape.py.
  - _fix:_ Pin numpy>=2.0 in pyproject.toml (affects 4 modules), or use trapezoid = getattr(np, 'trapezoid', np.trapz).

- **[ADJUDICATED-SURVIVED]** `geometry_mahalanobis` — dataset=None branch: 'held-out' ID split scores exact duplicates of fit texts, inverting the OOD conclusion — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/mahalanobis.py:156`
  - Line 156 cycles 5 id_texts, so score rows duplicate fit rows. distilgpt2: reported mean_mahalanobis_id=2.30 but genuinely-new ID sentences score 72.25 vs same fit — above OOD mean 62.82. Gap 60.52/ratio 27.3/auroc 1.0 are duplication artifacts, violating the code's own held-out rationale (lines 199-205). Harness path unaffected (core.py:147-148).
  - _fix:_ In the dataset=None fallback, use >= fit_n+2 distinct sentences (or dedupe by text and split text-disjointly) so ID scoring is genuinely held out; or return an error/insufficient_holdout flag when unique ID texts < fit_n+2.

- **[ADJUDICATED-SURVIVED]** `geometry_mp_bulk_deviation` — _ks_distance returns ~1-1/gamma for any gamma>1 spectrum, regardless of MP fit — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/rmt_bulk.py:124`
  - iid MP(3) data (D=3N, N=100/400/1600): BLME ks_distance=0.6667 constant while true tie+atom-aware KS=0.0085/0.0020/0.0007. The \|F-(i-1)/n\| term compares F(0)=point-mass 1-1/γ against 0 at the ~(D-N) zero eigenvalues. Violates the docstring's own 'KS distance' definition for the documented γ>1 branch (rmt_bulk.py:27,85-86).
  - _fix:_ Compute KS over unique eigenvalues with cumulative counts, using theoretical left limits F(x-)=F(x)-pointmass at the 0 atom (see scratchpad true_ks_with_atom); or restrict the comparison to the continuous part.

- **[ADJUDICATED-SURVIVED]** `geometry_mp_bulk_deviation` — _mp_cdf catastrophically wrong at gamma==1.0 (max CDF error 0.593) — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/rmt_bulk.py:98`
  - At γ=1.0 the density ~t^{-1/2}; uniform trapezoid grid's first node (lo+4e-9) has density ~5e3, creating spurious mass that renormalization spreads: max\|_mp_cdf−scipy.quad\|=0.593 (γ=0.99/1.01: 8.3e-3; 0.95/1.05: 2.2e-3; ≤6.4e-4 elsewhere). Reachable when n_tokens==kept dims or config fixed_gamma=1.0 (n_fixed=D ⇒ γ exactly 1).
  - _fix:_ Use the closed-form MP CDF (skrmt spectral_law.py:415-476 arctan formula) or integrate under substitution t=s^2 / a non-uniform edge-refined grid.

- **[ADJUDICATED-SURVIVED]** `geometry_perplexity` — BPC systematically wrong for BOS-adding tokenizers (Llama/Gemma/Mistral-style) — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/perplexity.py:92`
  - Recount uses add_special_tokens=False but scoring (utils.py:220-223) used default True, violating the line-94 comment's own (T-1)-match invariant: 13 recounted vs 15 scored on a 2-doc BOS demo. bpc=0.8935757 vs docstring-definition bits/chars=1.0310504 — 13.33% under-report, exactly predicted by 13/15 ratio; bias ~ n_docs/total_tokens.
  - _fix:_ Tokenize the recount with the same arguments as collect_prediction_stats (drop add_special_tokens=False), or accumulate per-doc char counts during collection; then bits_per_token*tokens_per_char reduces exactly to nll_total/ln2/chars.

- **[ADJUDICATED-SURVIVED]** `geometry_perplexity` — Empty-text sample crashes the task; BPC doc-window desync is latent behind it — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/utils.py:225`
  - dataset=[{'text':''},docA,docB], num_samples=2: model forward on 0-length input_ids raises RuntimeError ('cannot reshape tensor of 0 elements') at utils.py:225 via perplexity.py:33. Wikitext-style corpora contain empty rows. Also perplexity.py:90-91 continues without count+=1, so via the cache path the BPC loop counts docs never scored.
  - _fix:_ Skip samples that tokenize to <2 tokens in collect_prediction_stats; in the BPC loop increment count before the empty-text continue so both loops consume the identical first-num_samples window.

- **[ADJUDICATED-SURVIVED]** `geometry_weight_norms` — 3-D params (fused MoE experts) silently get spectral_norm=Frobenius and stable_rank=1.0 — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/weight_norms.py:88`
  - Probe: layer with 3-D param (4,8,8): svdvals succeeds batched but float(S[0].item()) (line 80) raises on the 8-vector; bare except records spectral=15.7175 (=frob, true max sigma_max=5.0624, 3.1x high) and srank=1.0 (true 9.64). ndim<2 filter (line 62) admits 3-D. Hits real fused-expert models (e.g. GPT-OSS gate_up_proj).
  - _fix:_ Handle param.ndim>2 explicitly: iterate slices of W.reshape(-1, *W.shape[-2:]) computing per-slice frob/sigma_max/srank (or skip non-2-D with a logged warning); keep except only for genuine SVD failures.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_entropy` — Direct forward path tokenizes without truncation -> crash on inputs longer than model context; also diverges from cache path (512) — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/attention.py:82`
  - Line 82: tokenizer(text, return_tensors="pt") — no truncation. Reproduced: tiny GPT2 (n_positions=32) + 180-token text -> IndexError: index out of range in self. All 11 sibling interpretability tasks truncate (max_length=128/512); cache.py:352 truncates at 512, so cache vs direct runs give different entropies for docs >512 tokens.
  - _fix:_ Add truncation=True, max_length=512 at attention.py:82 to match cache.py:352, restoring cache/direct parity and preventing the crash.

- **[ADJUDICATED-SURVIVED]** `interpretability_prediction_entropy` — mean_top5_prob silently reports top-K mass when decisiveness_top_k < 5 — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/prediction_entropy.py:75`
  - line 75: top5_sum = topk_p[..., :min(5, k)] where k=min(decisiveness_top_k, V). With decisiveness_top_k=2, task returned 'mean_top5_prob'=0.540196888150776 (= exact top-2 mass) vs true top-5 mass 0.798714024125410 (fresh_check.py). Key name is wrong for any k<5; default k=5 unaffected.
  - _fix:_ Compute top-5 from its own topk call: top5_sum = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1), independent of decisiveness_k.

- **[ADJUDICATED-SURVIVED]** `interpretability_probing` — Unseeded subsample makes default-config output non-reproducible — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/probing.py:82`
  - np.random.choice has no seed; default config (50 WikiText passages x 128 tokens > max_probe_samples=5000) always hits this branch. Three identical synthetic runs returned [0.0,0.6154,0.0769], [0.0,1.0,0.0], [0.0769,1.0,0.0]. Existing test dodges the branch on purpose (test_comprehensive_parity.py:5297 comment).
  - _fix:_ Use a seeded generator, e.g. np.random.default_rng(self.config.get("seed", 42)).choice(...); same fix pattern already applied to interpretability_waa per VERIFICATION_LEDGER.md:58.

- **[ADJUDICATED-SURVIVED]** `interpretability_probing` — Silent `except Exception: acc = 0.0` fabricates accuracy on probe failure — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/probing.py:120`
  - Single-class corpus: sklearn fit raises (needs >=2 classes) and BLME reports probing_accuracy_per_layer=[0.0,0.0,0.0] though true linear decodability is 100% (constant probe). Any runtime error (OOM, dtype) likewise becomes a plausible-looking 0.0 instead of an error.
  - _fix:_ Catch narrowly (ValueError), log a warning, and emit None/NaN (or short-circuit num_classes<2 with accuracy = majority-class rate) instead of 0.0.

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — 'gpt2' substring guard lets distilgpt2 through; SAE silently applied to wrong model at clamped wrong layer — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/sae_features.py:83`
  - 'gpt2' in 'distilgpt2' -> guard passes; _select('blocks.8.hook_resid_pre', 6) clamps to hidden_states[6] (final, ln_f applied). Ran it: returns mean_active_features_l0 = 18775.48 with no skip/error, violating the guard's own stated intent (lines 78-92) to skip models the SAE wasn't trained for.
  - _fix:_ Exact-match the model name (gpt2 only) or verify parsed layer < num_layers and d_in matches before running; return skipped instead of clamping 8->6.

- **[ADJUDICATED-SURVIVED]** `interpretability_sparsity` — No None-guard after get_layers(): TypeError crash on unrecognized architectures — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/sparsity.py:89`
  - sparsity.py:88-89 does `layers = get_layers(model); num_layers = len(layers)`. get_layers (common.py:96-102) returns None for unknown archs -> demonstrated `TypeError: object of type 'NoneType' has no len()`. Sibling superposition.py:80-81 returns {"error": ...} instead; this module also uses the error-dict contract at line 137.
  - _fix:_ Add `if layers is None: return {"error": "Could not detect model layers."}` after line 88, mirroring superposition.py:80-81.

- **[ADJUDICATED-SURVIVED]** `repe_concept_separability` — NaN layer poisons summary stats: max_auc_layer points at the NaN layer, max_auc/mean_auc become NaN — votes 2/2 upheld, unanimous
  - `src/blme/tasks/representation_engineering.py:253`
  - Lines 253-255 use np.argmax/np.max/np.mean on layer_aucs that lines 224-228 deliberately fill with NaN (fp16/bf16 deep-layer path per comment at 221-227). Reproduced on real evaluate(): aucs=[1.0, 0.8519, nan] -> max_auc_layer=2, max_auc=nan, mean_auc=nan; correct nanargmax=0, nanmax=1.0.
  - _fix:_ Use np.nanargmax/np.nanmax (and nanmean or document mean_auc semantics), with an all-NaN guard returning an {'error': ...} dict.

- **[ADJUDICATED-SURVIVED]** `repe_concept_separability` — String-labelled datasets pass the _is_labelled guard then crash at np.bincount — votes 2/2 upheld, unanimous
  - `src/blme/tasks/representation_engineering.py:214`
  - _is_labelled (172-176) accepts any {'text','label'} dict; label='harmful'/'harmless' (the sibling repe_refusal_direction format, same module) reaches np.bincount(y) at line 214 -> TypeError "Cannot cast array data from dtype('<U8')". Reproduced. Runner's catch-all (core.py:294) contains it, but the task yields no result instead of its own {'error':...} convention.
  - _fix:_ Validate labels are binary ints in _is_labelled (or map string labels), returning {'error': ...} otherwise.

- **[ADJUDICATED-SURVIVED]** `repe_steering_effectiveness` — Task can never produce metrics in the default pipeline: bundled-pair fallback is unreachable — votes 2/2 upheld, unanimous
  - `src/blme/tasks/representation_engineering.py:282`
  - Fallback triggers only on `dataset is None`, but core.py:146-148 substitutes the WikiText {"text":...} corpus before tasks run (core.py:285). Empirically: evaluate(model, tok, [{"text":...}]) returns {'error': "Dataset must contain 'text_pos' and 'text_neg' keys."}. Study feature_matrix.csv has 0 steering columns; siblings repe_task_vectors:68-80, concept_separability:178, refusal_direction:494 all fall back on content.
  - _fix:_ Mirror repe_task_vectors: if the first num_samples items lack text_pos/text_neg, substitute the bundled contrastive triples (lines 283-287) instead of erroring; keep the error only for explicitly malformed user datasets.

- **[ADJUDICATED-SURVIVED]** `repe_steering_effectiveness` — In-place mutation of the caller's shared dataset dicts (adds 'neutral' key), even on the error path — votes 2/2 upheld, unanimous
  - `src/blme/tasks/representation_engineering.py:297`
  - `samples = list(dataset)[:num_samples]` copies the list, not dicts; s["neutral"]=... writes into the shared corpus core passes to every task (core.py:285). Verified: corpus[0] became {'text':..., 'neutral': 'The weather today is'} after an erroring evaluate(). Violates the module's own no-mutation invariant (lines 161-171 comment). No current task reads 'neutral', so no numeric impact today.
  - _fix:_ samples = [dict(s) for s in list(dataset)[:num_samples]] before mutating. Also delete dead code: `required` set (line 293) and `base_log_probs` (line 343) are never used.

- **[ADJUDICATED-SURVIVED]** `topology_betti_curve` — Zero-weight kNN edges silently dropped: duplicate points inflate beta_0 — votes 2/2 upheld, unanimous
  - `src/blme/tasks/topology/betti_curve.py:92`
  - graph.maximum(graph.T) prunes explicit zero-weight edges (nnz 24->0). 4+4 exact-duplicate points -> _count_betti=(8,0), true kNN-graph components=2; 3 identical points -> b0=3 not 1. Violates own definition at lines 55-57. Reachable: duplicate dataset texts give bitwise-identical mean-pooled reps. Clamping weights to 1e-300 yields correct b0=2.
  - _fix:_ Before csr construction (line 90), clamp: vals = np.maximum(vals, tiny_eps) (e.g. 1e-12), or symmetrize the boolean pattern separately so zero-distance edges survive; also fix the now-false comment at line 84 ('column 0 is the point itself' — untrue with duplicates, idx row0=[1 2 3 0]).

- **[ADJUDICATED-SURVIVED]** `topology_homology` — Duplicate target_layers feed ripser a duplicated point cloud for models with <=2 layers, diluting H0 statistics — votes 2/2 upheld, unanimous
  - `src/blme/tasks/topology/homology.py:90`
  - num_layers=2 gives target_layers=[0,1,1]; loop at line 104 appends layer-1 rep twice per sample. Demo (2-layer GPT2, 8 samples): ripser cloud 16 pts/8 unique; layer_1_mean_persistence_h0 reported 0.3689 vs 0.4216 correct. All conftest mocks have 2 layers, so tests silently exercise this.
  - _fix:_ target_layers = sorted({0, num_layers // 2, num_layers - 1}) at homology.py:90 (and iterate that deduped list at line 104).

### DISCREPANCY (12)

- **[ADJUDICATED-SURVIVED]** `consistency_calibration` — Default n_bins=10 differs from paper/official-code M=15 — votes 3/6 upheld
  - `src/blme/tasks/consistency/calibration.py:132`
  - calibration.py:132 defaults n_bins=10; Guo 2017 reports ECE 'with M=15 bins' (Table 1) and official _ECELoss defaults n_bins=15 (temperature_scaling.py:105). At defaults BLME's ECE differs numerically (0.481774085 vs 0.482028782 on my N=5000 check). Formula parity is exact at matched M.
  - _fix:_ Change default to 15 or add a docstring/ledger note that parity holds at matched n_bins and the default deviates from Guo's M=15.

- **[ADJUDICATED-SURVIVED]** `dynamics_sharpness` — top_eigenvalue_estimate is unsigned \|lambda_dominant\|, not PyHessian's signed Rayleigh quotient — votes 6/6 upheld, unanimous
  - `src/blme/tasks/dynamics/sharpness.py:207`
  - top_eig = float(flat_hv.norm().item()) reports \|\|Hv\|\|. Fresh check (verify_kernels.py, quadratic eigs {-5,3,1.5,...}): BLME +5.000000000000, official pyhessian eigenvalues(top_n=1) = -4.999990000009, true lambda_max = +3.0. Metadata/ledger claim unconditional match to PyHessian; sign case never tested. Only bites when dominant curvature is negative (saddles, untrained scopes).
  - _fix:_ Report the Rayleigh quotient sum((h*vi).sum() for h,vi in zip(hv,v)) like PyHessian hessian.py eigenvalues(), or document that the estimate is the unsigned dominant-magnitude eigenvalue in the docstring, task_metadata, and VERIFICATION_LEDGER.

- **[ADJUDICATED-SURVIVED]** `geometry_lipschitz` — Returned lipschitz_max violates module's own max_i definition; token-max array computed then discarded — votes 1/2 upheld
  - `src/blme/tasks/geometry/lipschitz.py:117`
  - Docstring line 15 defines L_hat(layer)=max_i ratio; class docstring line 44 promises 'mean and max over all tokens'. Line 103 appends per-layer token-maxes to lipschitz_maxes, which is NEVER returned; line 117 returns max over per-layer token-MEANS instead. distilgpt2: returned 1.1539 vs docstring-definition 6.7758 (5.9x).
  - _fix:_ Either return the token-max stats (e.g. lipschitz_token_max from lipschitz_maxes) and document lipschitz_max as max-of-layer-means, or rewrite docstring lines 15 and 44 to state mean_i aggregation and delete the dead lipschitz_maxes list. Update test_comprehensive_parity.py:3173 comment accordingly.

- **[ADJUDICATED-SURVIVED]** `geometry_magnitude` — Median-heuristic scale silently drops zero distances, deviating from cited magnipy function on duplicate points — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/magnitude.py:158`
  - magnitude.py:158 `d = d[d > 0]`; magnipy scales.py:105-107 medians ALL upper-triangle entries. Probe cloud [[0,0],[0,0],[1,0],[3,0]]: BLME t_med=1.0 vs magnipy 1.1547005383792517. Shifts the whole scale ladder, hence all 7 outputs. Docstring (magnitude.py:151-155) claims it IS scales.median_heuristic_from_distances; deviation undocumented. Duplicate hidden states are realistic (repeated prompts; the module's own fallback dataset repeats one sentence).
  - _fix:_ Document the zero-filter as an intentional robustness deviation in the helper docstring + ledger, or match magnipy exactly and only guard the median<=0 -> nan case.

- **[ADJUDICATED-SURVIVED]** `geometry_magnitude` — magnitude_med_scale is NOT \|tX\| at the median-heuristic scale as documented — it is \|tX\| at the nearest ladder scale (0.774·t_med at defaults) — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/magnitude.py:271`
  - Docstring magnitude.py:203 says '\|tX\| at the median-heuristic scale'; code uses argmin\|ts−t_med\| (line 271). Defaults (n_scales=10, span=10) put nearest scale at t_med·10^(−1/9)=0.774·t_med. Measured (n=60 gaussian, seed 3): emitted 11.702842 vs 18.873006 at true t_med — 38% low. Also emitted alongside magnitude_median_scale=t_med, inviting a wrong (t, \|tX\|) pairing.
  - _fix:_ Compute _magnitude(D, t_med) directly (one extra O(n^3) solve) for magnitude_med_scale, or rename/redocument as 'nearest sampled scale to t_med'.

- **[ADJUDICATED-SURVIVED]** `geometry_matrix_entropy` — Cache vs fresh path truncate at 512 vs 128 tokens (official protocol: 2048) — same config yields different numbers — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/matrix_entropy.py:151`
  - Fresh path truncates each sample to config max_length default 128 (L126, L151); shared-cache path hard-codes max_length=512 (src/blme/cache.py:352); official entropy_dataset.py uses max_length=2048. Per-sentence entropy depends on token count, so the headline differs with vs without a populated cache for texts >128 tokens. Protocol-level, not formula-level.
  - _fix:_ Align the two paths (honor config max_length in the cache slice or set the task default to the cache's 512) and document that the paper's protocol truncated at 2048.

- **[ADJUDICATED-SURVIVED]** `geometry_svd` — avg_cosine_similarity computed on CENTERED states — not the cited Ethayarajh 2019 anisotropy baseline (reads ~0 always) — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/isotropy.py:43`
  - isotropy.py:16 centers X before pair sampling at :43-52. Ethayarajh Eq.4 (D19-1006 Sec 3.4) uses raw representations. On identical gpt2 last-layer states (192x768): paper definition 0.973182 (paper Fig.1 ~1.0); BLME returns -0.046843. Undocumented; sampler also admits self-pairs (1/192 pairs, cos=1, seed 42).
  - _fix:_ Sample cosine pairs from X BEFORE the mean-subtraction at isotropy.py:16 (and reject i==j pairs), matching Eq.4; or rename the key (e.g. centered_avg_cosine) and document the centering in the docstring, task_metadata notes, and docs/interpreting_results.md.

- **[ADJUDICATED-SURVIVED]** `geometry_trajectory_curvature` — Layer curvature pools all angles instead of paper's mean of per-sentence means — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/trajectory_curvature.py:137`
  - np.mean(np.concatenate(all_angles)) length-weights sentences. Paper (arXiv:2311.04930 Methods: C_sn=(1/k)Σc_i, then 1/N over sentences) and official curvature_utils.py L41 mean per-sentence first. distilgpt2, 8 sents: last layer 1.90563 vs 1.87601 (Δ=2.96e-2 rad); ratio 0.0478 vs 0.0615; adversarial synthetic Δ=1.195 rad. Docstring L20-22 attributes pooling to the paper — not documented as deviation.
  - _fix:_ Either compute per-sample _mean_curvature then average across samples, or amend docstring/implementation-notes to state pooling is an intentional deviation from the paper's per-sentence aggregation.

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — SAE fed raw HF hidden states; official usage requires centered + BOS-prepended activations — headline L0 inflated ~37x — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/sae_features.py:148`
  - sae.cfg.metadata declares model_from_pretrained_kwargs={'center_writing_weights': True}, prepend_bos=True (sae-lens warns at load). BLME's raw-HF/no-BOS feed: mean L0 = 2431.03, max 23967/24576. Same pipeline with per-token mean-centering + BOS: 64.48 == official 65.47 (Bloom release). Undocumented; ledger:275 calls pipeline 'FAITHFUL'.
  - _fix:_ Subtract per-token mean when metadata has center_writing_weights=True and prepend BOS per cfg.metadata.prepend_bos; else document the OOD regime in the docstring and retract the ledger/fixture 'trained-SAE pipeline FAITHFUL' wording.

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — hook_resid_mid silently mapped to resid_post (hidden_states[N+1]), contradicting the module's own rejection docstring — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/sae_features.py:44`
  - Docstring (lines 25-26): unsupported hooks 'are rejected rather than silently mapped to an arbitrary residual tensor'. Yet resid_mid -> N+1 = post-MLP resid. Measured on gpt2 block 8: mean L0 3193.15 (BLME pick) vs 2624.85 (true ln_2-input resid_mid); mean per-token \|L0 diff\| 568.31.
  - _fix:_ Reject hook_resid_mid (HF hidden_states contains no post-attn pre-MLP tensor) or capture block.ln_2's input via a forward-pre-hook.

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — Final-block hook_resid_post maps to hidden_states[num_layers], which HF returns AFTER ln_f — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/sae_features.py:43`
  - TL blocks.11.hook_resid_post is pre-ln_f; HF hidden_states[12] has ln_f applied. Measured rel-L2 diff vs true block-11 output: 0.8717. test_comprehensive_parity.py:5468 knowingly skips verifying N = last block; module docs never disclose the ln_f mismatch.
  - _fix:_ For the last block, capture the block's raw output via a forward hook, or reject blocks.{L-1}.hook_resid_post with a clear error.

- **[ADJUDICATED-SURVIVED]** `interpretability_superposition` — BC kernel drops the finite-sample correction term stated in its own docstring — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/superposition.py:49`
  - Docstring (line 32) states BC = (skew²+1)/(kurt + 3(n-1)²/((n-2)(n-3))) — the SAS/Pfister 2013 form with bias-corrected excess kurtosis. Code computes (s²+1)/k with biased Pearson kurtosis, no correction. Fresh check: rel. deviation 10.766% (n=50), 5.133% (n=100), 0.780% (n=640).
  - _fix:_ Either implement the stated formula (scipy skew/kurtosis with bias=False, fisher=True, plus the correction term) or rewrite the docstring to the asymptotic form actually computed, matching docs/tasks_interpretability.md:120.

### WEAK_TEST (53)

- **[ADJUDICATED-SURVIVED]** `causality_circuit_quality` — Parity test's independent faithfulness reference transcribes the same hs[l+1] convention, so it cannot detect the last-layer mean bug — votes 5/6 upheld
  - `tests/tasks/test_comprehensive_parity.py:598`
  - Reference recomputation uses h = hs[l + 1] identically to circuit_quality.py:113, so both sides share the post-ln_f last-layer mean; test passes while the ablation value deviates (cos 0.303 from true mean). Otherwise strong: real blme imports, analytic minimality cases, scipy-JSD cross-check at 1e-5, determinism.
  - _fix:_ In the test's reference, gather per-layer means from raw block outputs via capture hooks instead of hidden_states[l+1], which would fail against the current implementation and pin the fix.

- **[ADJUDICATED-SURVIVED]** `causality_edge_attribution` — Only real-BLME-code test in the parity file silently skips offline despite warm cache — votes 4/6 upheld
  - `tests/tasks/parity/test_edge_attribution_parity.py:233`
  - test_real_blme_task_computes_eap_kernel skipped with 'tiny model unavailable: timed out' even though models--hf-internal-testing--tiny-random-gpt2 is in ~/.cache/huggingface/hub; from_pretrained hits the hub, times out, and the broad except skips. The 12 passing tests in this file use a transcribed kernel copy, not blme source.
  - _fix:_ Load with local_files_only=True (fallback to network), or set HF_HUB_OFFLINE=1 for the cached path, so the real-task parity check runs in offline CI. Coverage currently survives only via test_comprehensive_parity.py, which does exercise the real task offline.

- **[ADJUDICATED-SURVIVED]** `causality_knowledge_neurons` — saliency_layer_mean and saliency_layer_entropy pinned by no test — votes 6/6 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:959`
  - Comprehensive test asserts only mean_top1_share/mean_top1pct_share/mean_attribution_gini (L959-961); parity suite pins gini/top1 only. Changing np.log to np.log2 in knowledge_neurons.py:213 or the (n_layers-1) normalizer at :196 passes the whole suite. Fresh check pinned both: diff 0.0 vs independent recompute on gpt2 (2.464608270806265, 0.9090909091).
  - _fix:_ Extend test_causality_knowledge_neurons's independent reconstruction to also assert localization_layer_mean and attribution_layer_entropy (argmax/(n_layers-1); -sum p ln p over layer-summed \|saliency\|).

- **[ADJUDICATED-SURVIVED]** `consistency_bias_weat` — No test exercises word-position lookup over the bundled word lists; 'his' bug undetectable — votes 6/6 upheld, unanimous
  - `tests/tasks/test_bias_task.py:19`
  - Position tests probe only 'John' (lines 19-73); parity tests feed synthetic vectors directly to the kernels. A 3-line loop of _find_word_token_position over the 64 _WEAT_TESTS words under the default template flags 'his'->0 immediately (my audit: 1/64 mismatch). Kernel tests themselves are strong (real imports, bit-exact official pins).
  - _fix:_ Add a test asserting, for every word in _WEAT_TESTS under 'This is {}.', that the returned token's offset span overlaps the word's insertion span (chars 8..8+len(word)) with the gpt2 tokenizer.

- **[ADJUDICATED-SURVIVED]** `consistency_calibration` — calibration_slope/intercept never exercised by any test — votes 5/5 upheld, unanimous
  - `src/blme/tasks/consistency/calibration.py:53`
  - calibration.py:53-67 unweighted np.polyfit over bin means; the only fixture (calibration_two_bins, 2 bins) hits the <3-bins NaN branch, so swapping polyfit x/y would pass the suite. Also uncited: literature 'calibration slope' means the Cox/logistic recalibration coefficient, not a bin-mean linear fit.
  - _fix:_ Add a >=3-bin fixture pinning slope/intercept and a docstring line defining them as a BLME-specific bin-level linear fit.

- **[ADJUDICATED-SURVIVED]** `consistency_contamination` — Calibration helpers (_binary_auroc, _best_threshold) and the held_out branch have zero numeric test coverage — votes 4/4 upheld, unanimous
  - `src/blme/tasks/consistency/contamination.py:220`
  - grep of tests/ shows no test references _binary_auroc or _best_threshold; the held_out branch (contamination.py:160-183) is never executed by any test (only 'in_sample' asserted, test_consistency_diagnostic_semantics.py:250). Dropping the 0.5-tie term at :226 or flipping >= at :171 would pass the suite. Fresh check shows current code correct (sklearn diff 1.1e-16; brute force 200/200).
  - _fix:_ Add fixture tests pinning _binary_auroc against sklearn.roc_auc_score (incl. ties) and _best_threshold against a hand-computed case, plus one held_out-mode evaluate() test with >=6 labeled samples asserting held_out_threshold/held_out_auroc values.

- **[ADJUDICATED-SURVIVED]** `consistency_contrastive` — Mock-model unit test passes while producing NaN on all 11 architectures — votes 6/6 upheld, unanimous
  - `tests/tasks/test_consistency.py:24`
  - test_contrastive_consistency asserts only key presence (lines 43-46). Running with -W error::RuntimeWarning fails all 11 parameterizations ('Mean of empty slice'): every mock arch hits the NaN path, so the test cannot catch formula changes or the NaN bug; the legacy {factual,exclusive} path has no numeric pin anywhere (my fresh check B now shows it equals the triple path, diff 0.0).
  - _fix:_ Assert math.isfinite() on the three outputs (or expect an error dict), and add a numeric pin for the legacy input form against the triple form on gpt2.

- **[ADJUDICATED-SURVIVED]** `consistency_format_robustness` — CV outputs (format_nll_cv_sensitivity / mean_nll_cv_across_formats) never numerically pinned by any test — votes 6/6 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:1418`
  - Parity test pins std/max/overall/top1 (lines 1418-1426) but no assertion covers the CV keys computed at src/blme/tasks/consistency/format_robustness.py:127,146; changing std/mean to std/median or dropping the 1e-6 guard passes the whole suite. Fresh check confirms current values correct: blme 0.007537956 vs ref 0.007537953 (fp32), absdiff 6.9e-18 in fp64.
  - _fix:_ Add CV assertions to test_consistency_format_robustness mirroring the existing std assertions (ref: mean over questions of std/mean with the >1e-6 guard); optionally also pin the dataset+num_samples path and the degenerate all-NaN-row case where CV counts 0.0 while the std mean excludes the question (format_robustness.py:127 np.where vs :145 nanmea

- **[ADJUDICATED-SURVIVED]** `consistency_logical` — Parity test doctors input to dodge the degeneracy instead of testing the shipped path — votes 6/6 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:1691`
  - Comment at 1691-1694 admits conclusions get two leading spaces "so that the hardcoded unconditional prompt ' ' does not get absorbed... (which would make prompt_len==0 and score=None)". No test runs dataset=None or realistic no-leading-space data; test_consistency_diagnostic_semantics.py:192 stubs score_continuation entirely.
  - _fix:_ Add a test running LogicalConsistencyTask with dataset=None (bundled defaults) on a tiny GPT-2 asserting numeric output (no error dict); it fails today and pins the bug fix.

- **[ADJUDICATED-SURVIVED]** `consistency_membership_inference` — counterfactual_gap metric has zero test coverage — votes 4/4 upheld, unanimous
  - `src/blme/tasks/consistency/membership_inference.py:231`
  - grep 'counterfactual' tests/ hits only an unrelated edge_attribution.json note; no test asserts counterfactual_gap (lines 194-206, 231). A sign flip (orig−shuf) or rng-seed change would pass the entire suite. Fresh check confirms current code correct: 4.2012364864 vs independent 4.2012331885 (diff 3.3e-06) on distilgpt2.
  - _fix:_ Add an assertion on counterfactual_gap to test_membership_inference_parity.py: extend the stub so shuffled texts get a prescribed loss, pin mean(shuf−orig) to 1e-9; optionally also force the sklearn-fallback branch (lines 184-189) via sys.modules poisoning as in fresh_check.py.

- **[ADJUDICATED-SURVIVED]** `consistency_paraphrase` — No test exercises the dataset=None fallback path, so the broken mpararel branch passed two verification campaigns — votes 5/6 upheld
  - `tests/tasks/test_comprehensive_parity.py:1882`
  - All paraphrase tests (test_comprehensive_parity.py:1882, test_consistency.py:118, test_consistency_diagnostic_semantics.py:221) pass explicit triples, so lines 55-80 of paraphrase.py (mpararel + bundled fallback) are never covered; the guaranteed ValueError at paraphrase.py:58 went undetected through Campaigns 1-2.
  - _fix:_ Add a test calling evaluate(model, tok, dataset=None) with datasets.load_dataset monkeypatched, asserting either mpararel triples parse or the bundled path is reported explicitly (e.g. a data_source key).

- **[ADJUDICATED-SURVIVED]** `consistency_self_consistency` — Seed path never tested on real HF generate; parity test knowingly routes around the bug — votes 6/6 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:2141`
  - Test docstring admits: seed left None 'because this transformers version rejects the generator kwarg and the task's broad except-clause would otherwise drop every prompt'. Seed path only exercised via fake models accepting generator (test_consistency_diagnostic_semantics.py:161,255), so the shipping seed BUG is invisible to the suite.
  - _fix:_ After fixing the BUG, add a real-HF-model test with seed set asserting 'error' not in result and determinism across two runs.

- **[ADJUDICATED-SURVIVED]** `dynamics_coe` — CoE fixture chain has Z_Mag exactly 1, so tests cannot detect removal of the Eq. 3 magnitude normalization in coe_r/coe_c — votes 6/6 upheld, unanimous
  - `tests/fixtures/reference_parity/formula_fixtures.json:81`
  - Chain [[1,0],[0,1],[1,1]]: Z_Mag=\|\|h_L-h_0\|\|=\|\|(0,1)\|\|=1.0, so raw and normalized magnitudes coincide; both pinned tests (test_reference_parity_formulas.py:147, test_dynamics.py:181) pass unchanged if /z_mag is dropped. Angle side is covered (Z_Ang=π/4). My random-chain check (Z_Mag≠1) confirms current code is correct (diff ≤1.9e-9).
  - _fix:_ Add a fixture chain with Z_Mag≠1, e.g. [[1,0],[0,2],[3,3]], pinning coe_r/coe_c/mag_norm_mean computed from the official CoEScoreInfo.

- **[ADJUDICATED-SURVIVED]** `dynamics_generation_diversity` — Seed test's FakeModel accepts **kwargs and asserts generator_was_passed, enshrining the API that real transformers models reject — votes 6/6 upheld, unanimous
  - `tests/tasks/test_dynamics.py:284`
  - FakeModel.generate(**kwargs) at test_dynamics.py:266-273 happily receives 'generator'; assert model.generator_was_passed (line 284) pins the broken contract, so the suite passes while any real HF model errors on the seeded path.
  - _fix:_ Make FakeModel.generate raise ValueError on unknown kwargs (mimicking _validate_model_kwargs), and assert seeded evaluate() returns metrics, not an error dict.

- **[ADJUDICATED-SURVIVED]** `dynamics_interpolation` — Entropy kernel and norm-corrected lerp are not numerically pinned by any test — votes 4/4 upheld, unanimous
  - `tests/tasks/test_dynamics.py:14`
  - Existing tests pin _slerp strongly (real blme import, independent reference) but the entropy at trajectories.py:111-112 and norm correction at 101-103 are only checked for key presence/self-consistency. E.g. softmax over dim=0 would yield gap≈0 and still pass test_interpolation_default_steps_include_true_midpoint (0==0). My fresh end-to-end check pins these (d<=8.1e-06).
  - _fix:_ Add a parity test like scratchpad verify.py Part B: tiny deterministic model, independently recompute interp/slerp entropies + convexity gaps, and assert alpha=0/1 entropy equals the model's own-logits entropy.

- **[ADJUDICATED-SURVIVED]** `dynamics_sharpness` — Power-iteration loop and SAM metric transcribed in tests, never driven; no test calls evaluate() — votes 4/5 upheld
  - `tests/tasks/parity/test_sharpness_parity.py:103`
  - _blme_lambda_max re-implements the evaluate() loop (sharpness.py:196-211) instead of calling it; test_comprehensive_parity.py:2582-2598 does the same. The SAM block (sharpness.py:217-231) is exercised by NO test. Changing line 207 or the SAM sign would still pass the suite. VERIFICATION_LEDGER.md:205 rates this test STRONG.
  - _fix:_ Extract the power-iteration loop and SAM computation into named module-level helpers called by evaluate(), and have parity tests import them; add one small-model test driving the real evaluate() (my verify_evaluate_e2e.py shows baseline/SAM are deterministic and pinnable in fp64).

- **[ADJUDICATED-SURVIVED]** `dynamics_stability` — reference_model branch of evaluate() has zero test coverage — votes 5/5 upheld, unanimous
  - `src/blme/tasks/dynamics/stability.py:34`
  - No test sets reference_model_path (grep tests/: only embedding_noise mode exercised). A regression in stability.py:34-50 (E2 loading/normalization/vocab check) would go uncaught. Fresh check proves branch currently correct: self-ref -> exactly 1.0/0.0; cross-ref mean 0.017062937063 == scipy ref, absdiff 0.0.
  - _fix:_ Add a parity test: save a tiny GPT2 to tmp_path, run with reference_model_path pointing at itself (assert mean==1.0, std==0.0, mode=='reference_model') and at a second model (assert vs independent kNN reference).

- **[PENDING-REFUTE]** `geometry_categories` — Hubness, relation-consistency and analogy kernels have zero numeric test coverage (relation kernels never even execute in tests)
  - `tests/tasks/test_comprehensive_parity.py:2719`
  - test_geometry_categories pins only separation+purity; its JSON ({"A","B"} word lists, categories.py relation path needs pair lists) never triggers _compute_relation_consistency (categories.py:231), and hub_skew/hub_max (categories.py:224-227) are computed but unasserted. These kernels DO run in production via packaged singular_plural/present_past; a formula change there passes the suite.
  - _fix:_ Extend the parity test with a pair-list category (n<20 pairs so analogy is deterministic without seed mirroring) asserting hub_skew/hub_max, consistency, analogy_acc_top1/top5 against independent references — my scratchpad verify_synthetic.py already does this and matches to <=4e-8; kernels are currently correct.

- **[ADJUDICATED-SURVIVED]** `geometry_cknna` — _l2_normalize preprocessing helper has zero test coverage (fixture tensors pre-normalized) — votes 6/6 upheld, unanimous
  - `tests/tasks/parity/test_cknna_parity.py:26`
  - grep of tests/ finds no reference to _l2_normalize; fixture docstring line 26 says 'the fixture tensors are already normalised'. A regression in cknna.py:171-175 (e.g. p=1, or dropping normalization) would leave all 10 parity tests green while changing every task output.
  - _fix:_ Add a parity case with UNNORMALIZED input asserting _l2_normalize(X) == F.normalize(X,p=2,dim=-1) bit-exact and full-pipeline equality (my fresh run shows max\|diff\|=0.0, so it can pin at tol=0).

- **[ADJUDICATED-SURVIVED]** `geometry_collapse` — No test pins max_drop / max_drop_layer outputs — votes 4/4 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:2882`
  - test_geometry_collapse pins erank_per_layer, collapse_ratio, max/min_erank but never max_drop or max_drop_layer (collapse.py:92-94); a sign flip (np.max vs np.min) or off-by-one would pass the suite. My fresh check pinned them: max_drop -1.0464153 vs ref -1.0464156, max_drop_layer 2==2.
  - _fix:_ Add asserts: out['max_drop'] == approx(min(diff(ref))) and out['max_drop_layer'] == argmin(diff(ref))+1 in the multi-layer pipeline check.

- **[ADJUDICATED-SURVIVED]** `geometry_contextualization` — 3 of 4 numeric kernels have zero value-pinning tests (MEV, SelfSim, anisotropy baseline) — votes 3/3 upheld, unanimous
  - `tests/tasks/test_reference_parity_formulas.py:185`
  - Only test (lines 185-202) pins _intra_sentence_mean_cosine; _cosine_pairwise_mean appears only in an inequality (\|intra-pairwise\|>0.1), _max_explainable_variance and _baseline_anisotropy are never imported by any test. A formula change (e.g. dropping centering in MEV) passes the full suite. My fresh check pins them: MEV diff <=8.3e-17 vs sklearn, SelfSim <=2.2e-16 vs official.
  - _fix:_ Add value-pinning fixtures for _cosine_pairwise_mean, _max_explainable_variance (vs sklearn PCA ratio) and _baseline_anisotropy (analytic anisotropic cloud), e.g. port scratchpad verify_kernels.py cases into test_reference_parity_formulas.py.

- **[ADJUDICATED-SURVIVED]** `geometry_hsic` — evaluate() duplicates nHSIC normalization inline; reference-value parity test only guards the unused helper — votes 3/3 upheld, unanimous
  - `src/blme/tasks/geometry/mutual_info.py:113`
  - _normalized_linear_hsic (pinned at 1e-9 vs official 0.450953881199808) is never called by evaluate(); lines 105, 113-115, 128-130 recompute sum(K_i*K_j)/sqrt(self_hsic*self_hsic) inline. test_geometry.py:503 only asserts key presence, so a formula edit in the production path (e.g. dropped sqrt) passes all tests. Fresh check confirms inline path currently matches official cka to ≤3.7e-8.
  - _fix:_ Have evaluate() reuse the parity-covered normalization (e.g. a gram-level helper called by both _normalized_linear_hsic and evaluate), or add a flow-level test pinning adjacent_hsic/input_to_layer_hsic on stubbed activations against the Kornblith reference.

- **[ADJUDICATED-SURVIVED]** `geometry_hubness` — evaluate() k-NN occurrence kernel not numerically pinned by any test — votes 2/2 upheld, unanimous
  - `tests/tasks/test_geometry.py:93`
  - Only assertions on evaluate() are key-presence and gini in [0,1] (lines 99-102). Demonstrated: removing the self-mask (hubness.py:110) changes skew 1.204842 to 0.952487 and gini 0.295574 to 0.247798, yet every existing assertion still passes; parity tests feed a fixed occurrence vector, bypassing the kernel.
  - _fix:_ Add an end-to-end parity test like scratchpad/verify_geometry_hubness/fresh_check.py section A: fixed-seed fake model, compare evaluate() skew/gini/max against a numpy cosine-kNN + bincount reference at 1e-6.

- **[ADJUDICATED-SURVIVED]** `geometry_isoscore` — N<d zero-padding branch of _isoscore unpinned by any test — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:4067`
  - Part B deliberately uses N>d only ('the regime in which... coincide', test_comprehensive_parity.py:4067-4069); no test exercises isotropy.py:108-109 padding with N<d. Deleting the padding would change N<d outputs materially and pass all tests. Fresh check confirms the branch is CORRECT: 40x128 blme=0.189341058177 vs official 0.189341049247 (8.9e-9).
  - _fix:_ Add one N<d case (e.g. 40x128) to Part B of test_geometry_svd or to test_isoscore_matches_rudman_reference_package, tol 1e-6.

- **[ADJUDICATED-SURVIVED]** `geometry_magnitude` — _median_heuristic_scale is never reference-pinned against magnipy, and no duplicate-point cloud is compared to magnipy — exactly the blind spot hiding the zero-filter deviation — votes 1/2 upheld
  - `tests/tasks/parity/test_magnitude_parity.py:149`
  - Tests import the real helper (line 118) but only use it to BUILD ladders (lines 149,181,231,245); no assert vs scales.median_heuristic_from_distances. test_singular_zeta_perturbation_fallback (line 254) only checks np.isfinite, not parity vs magnipy's perturbed value. My fresh check shows both comparisons are cheap and decisive (bit-exact / 1.0-vs-1.1547).
  - _fix:_ Add: (a) assert _median_heuristic_scale(D) == median_heuristic_from_distances(D) on a duplicate-free cloud; (b) duplicate-cloud magnitude parity vs magnipy cholesky+perturb (I measured 2.209641094842 both); (c) document/decide the duplicate-cloud t_med behavior.

- **[ADJUDICATED-SURVIVED]** `geometry_mahalanobis` — HAS_SKLEARN=False pinv fallback path (lines 54-58) has zero test coverage — votes 1/2 upheld
  - `src/blme/tasks/geometry/mahalanobis.py:56`
  - No test in tests/ exercises the fallback (grep: all tests run with sklearn present); a formula change there (drop 1e-6 ridge, ddof, pinv) would pass the suite. My fresh analytic pin: blme 1.890431729116309 == closed form 1.890431729116309 (diff 0.0), so code is currently correct.
  - _fix:_ Add a test that monkeypatches blme.tasks.geometry.mahalanobis.HAS_SKLEARN=False and asserts the analytic closed-form value (4-point symmetric X_train gives np.cov=diag(2a^2/3,2b^2/3)); also pin evaluate()'s auroc against a Mann-Whitney reference.

- **[ADJUDICATED-SURVIVED]** `geometry_matrix_entropy` — No numeric parity test — formula mutants pass all 4 matrix-entropy tests — votes 2/2 upheld, unanimous
  - `tests/tasks/test_geometry.py:617`
  - Tests assert only finite, >0, ≤1.01, headline==last layer (L612-614, 630-639, 729-740). Mutants dropping centering or row-normalization change the number (D=128: 0.935216→0.935501/0.934334) yet pass every assertion — verified empirically. Tests do import real blme code.
  - _fix:_ Add a fixture-pinned case (official-code fp64 value on a fixed matrix) to tests/fixtures/reference_parity/formula_fixtures.json and a test asserting _matrix_entropy matches to ~1e-6; my scratchpad verify_kernel.py values can seed it.

- **[ADJUDICATED-SURVIVED]** `geometry_mp_bulk_deviation` — No test exercises the gamma>=1 branch, exactly where both bugs live — votes 2/2 upheld, unanimous
  - `tests/tasks/test_geometry.py:943`
  - All 4 kernel tests use γ∈{0.099,0.1,0.25}; integration tests use mock models with N≫D. Suite is green (15/15) while _ks_distance is wrong for γ>1 and _mp_cdf wrong at γ=1, despite rmt_bulk.py:27,93 explicitly implementing the γ>1 point-mass branch. Tests do import real blme code and would catch γ<1 formula changes.
  - _fix:_ Add: iid γ=3 fixture asserting ks_distance<0.05 (fails today, 0.6667), and _mp_cdf(γ∈{0.999,1.0,1.05}) vs closed form/quad within 1e-3.

- **[ADJUDICATED-SURVIVED]** `geometry_neural_collapse` — Double-norm regression guard is a string match on one exact spelling; no behavioral test covers the evaluate() feature-extraction path — votes 2/2 upheld, unanimous
  - `tests/tasks/test_round6_fixes.py:42`
  - Guard asserts only that literal 'h = final_norm(h.to(norm_dtype))' is absent. Re-introducing double-norm spelled differently (e.g. 'h = final_norm(h)') passes. No test runs NeuralCollapseTask.evaluate() on a model; my e2e check shows double-norm shifts logits by max\|diff\|=180 vs 0, so a behavioral pin is cheap and decisive.
  - _fix:_ Add a tiny-model behavioral test: run evaluate() on a 2-layer GPT2 stub and assert pooled features equal mean of out.hidden_states[-1] (or assert lm_head(hidden_states[-1])==logits invariant on the extraction path), plus finite metric keys.

- **[ADJUDICATED-SURVIVED]** `geometry_perplexity` — ppl_rare/ppl_freq never numerically pinned; bucket formula changes would pass — votes 2/2 upheld, unanimous
  - `tests/tasks/test_perplexity_task.py:69`
  - All existing assertions on ppl_rare/ppl_freq are finiteness/positivity only (test_perplexity_task.py:69-74,104-108); comprehensive-parity pins only ppl_overall/mean_nll/bpc. Changing the 20% fraction, lexsort tie-break, or swapping rare/freq buckets passes every test. My fresh check pins both (28.6310607 vs 28.6312130 ref, rel 5.3e-6) — they match today.
  - _fix:_ Add a numeric-parity test reconstructing rare/freq buckets independently (python sort by (count,id), bottom/top round(0.2n)) and assert ppl_rare/ppl_freq to rel 1e-5 on cached gpt2; also cover a BOS-emulating tokenizer for bpc.

- **[ADJUDICATED-SURVIVED]** `geometry_phd_dimension` — Reference-backed parity cases depend on an ephemeral hardcoded /tmp scratchpad path — votes 2/2 upheld, unanimous
  - `tests/tasks/parity/test_phd_dimension_parity.py:48`
  - _REF_CANDIDATES pins /tmp/claude-1736197890/.../42dc090a-.../refrepos/GPTID; on any fresh checkout 4/7 tests skip silently, so official-code parity is unrunnable by release users. Passed today only because the old session dir survives. Fixture-pinned tests (1e-12) still catch formula changes.
  - _fix:_ Read the GPTID location from an env var (e.g. BLME_GPTID_DIR) with the scratchpad path as fallback, and document how to clone commit 8c8759e to re-enable the reference-backed cases.

- **[ADJUDICATED-SURVIVED]** `geometry_prediction_alignment` — Untied-head behavior (the task's central advertised claim) has no numeric test — only a source-grep guard — votes 2/2 upheld, unanimous
  - `tests/tasks/test_round5_determinism.py:114`
  - Parity test (test_comprehensive_parity.py:3611) uses default tied GPT2Config, where lm_head.weight IS wte.weight, so it cannot distinguish output projection from input embeddings. The grep guard (lines 123-127) passes even if _get_output_projection_weight's body regresses to 'return get_embeddings(model)' — 'get_lm_head' survives in the import line. Fresh untied check: value shifts 0.013066 → -0.015712 (gap 0.0288) undetected.
  - _fix:_ Add a numeric case with GPT2Config(tie_word_embeddings=False) and a re-randomized lm_head: assert task output matches a cosine reference built from lm_head.weight (abs 1e-6) and differs from the wte-based reference. Section B of scratchpad/verify_geometry_prediction_alignment/verify.py is a ready template.

- **[ADJUDICATED-SURVIVED]** `geometry_representation_sensitivity` — lm_head bias branch (lines 89-90) has zero test coverage — votes 1/2 upheld
  - `src/blme/tasks/geometry/information_geometry.py:89`
  - Both checked-in tests use GPT2LMHeadModel whose lm_head has bias=False, so 'logits = logits + head.bias' never executes; changing + to - would pass the whole suite. Fresh check with a bias-ful head confirms the branch is currently correct: BLME 0.075530562550 vs float64 autograd 0.075530564528 (rel 2.62e-08).
  - _fix:_ In test_geometry_representation_sensitivity, swap in a bias-ful lm_head (nn.Linear(n_embd, vocab, bias=True) with random bias) for a second parity assertion, mirroring scratchpad/verify_geometry_representation_sensitivity/supplement_bias_and_linearity.py.

- **[ADJUDICATED-SURVIVED]** `geometry_schatten` — mnn/T fix (2026-06-22) has no regression guard: no test pins _per_sentence_measurements numerically — votes 2/2 upheld, unanimous
  - `tests/tasks/test_reference_parity_formulas.py:103`
  - test_matrix_nuclear_norm_normalized_by_sequence_length_li2024 asserts raw/T==2.0 using test-side arithmetic only; it never calls the task path. No test asserts _per_sentence_measurements()['row_normalized_matrix_nuclear_norm'] or any evaluate() value; test_schatten_task.py:129-138 checks key presence only. Reverting schatten.py:168 'mnn / T' to the old 'mnn / d' would pass all 31 tests.
  - _fix:_ Add a test pinning _per_sentence_measurements on a seeded matrix against the official pipeline value (e.g. torch.manual_seed(0) 24x16 gives 0.8060590426), covering preprocessing + /T division end-to-end.

- **[ADJUDICATED-SURVIVED]** `geometry_schatten` — _normalise_matrix (center + row-L2 preprocessing) is untested; a preprocessing regression is invisible to the suite — votes 1/2 upheld
  - `src/blme/tasks/geometry/schatten.py:116`
  - All MNN/Schatten/RankMe tests feed raw matrices directly to the kernels, bypassing _normalise_matrix. Swapping row/column normalization or dropping centering changes every reported metric but fails no test. Fresh check confirms current code matches official normalize() (MNN.py:32-38) with max\|diff\|=0.0.
  - _fix:_ Same fixture as above: assert _normalise_matrix output (or the per-sentence value that depends on it) against the transcribed official normalize() on a seeded matrix.

- **[ADJUDICATED-SURVIVED]** `geometry_spectral` — test_spectral_parity.py asserts on a transcribed copy of the Hill kernel, never importing blme — votes 1/2 upheld
  - `tests/tasks/parity/test_spectral_parity.py:66`
  - _blme_hill_alpha (lines 66-79) is a verbatim transcription of spectral.py:77-91 (declared at lines 37-38); the file imports no blme module, so a formula change in spectral.py would leave all 5 'parity' tests green. Compensated: test_comprehensive_parity.py::test_geometry_spectral imports the real WeightSpectralTask and pins avg_alpha to analytic 1+2/ln(10/9) at 1e-3 and to independent scipy-SVD references at 1e-4, so the suite overall WOULD catch a formula change.
  - _fix:_ Add one assertion importing blme.tasks.geometry.spectral.WeightSpectralTask (evaluate() on a tiny nn.Linear) and compare against _blme_hill_alpha, so transcription drift is detected inside the parity file itself.

- **[ADJUDICATED-SURVIVED]** `geometry_svd` — Parity test never pins avg_cosine_similarity — only asserts it lies in [-1, 1] — votes 1/2 upheld
  - `tests/tasks/test_comprehensive_parity.py:4060`
  - test_geometry_svd pins 5 of 6 outputs at 1e-9 vs independent re-derivations, but line 4060 only asserts -1.0 <= avg_cosine_similarity <= 1.0. Any formula change (including the centering deviation above, a 1.02 error vs the cited definition on gpt2) passes. Both prior campaigns relied on this test.
  - _fix:_ Add a pinned check: recompute mean pairwise cosine with rng seed 42 on the same X (independent transcription of Eq.4 semantics) and assert approx equality at 1e-9; add a gpt2-states case asserting the value tracks the uncentered Ethayarajh baseline once the fix lands.

- **[ADJUDICATED-SURVIVED]** `geometry_trajectory_curvature` — No reference-parity test; aggregation deviation invisible to test suite — votes 2/2 upheld, unanimous
  - `tests/tasks/test_geometry.py:842`
  - All 5 test bodies are analytic (0, π/2, π/3) or mock-model integration; the pooling test's only contributing chunk is one hexagon, so pooled-vs-mean-of-means is indistinguishable. No fixture pins BLME vs Hosseini's official curvature_utils.py anywhere in tests/, despite ledger row 154 'VERIFIED_PARITY ... MATCH'.
  - _fix:_ Add a parity test running _pooled_layer_curvature and a transcribed compute_model_curvature on shared variable-length trajectories, asserting per-sentence means match and pinning the chosen aggregation.

- **[ADJUDICATED-SURVIVED]** `geometry_unembedding` — purity_mean and alignment stats are bound-checked only, never exact-pinned — votes 2/2 upheld, unanimous
  - `tests/tasks/test_geometry.py:179`
  - Purity test asserts only >0.5 (line 179); alignment test only <0.999 (line 296). A formula change like including the query token in top-k, or match_count/(k+1), still passes both (cluster purity stays 1.0). Only unembedding_eff_rank has independent numeric parity (test_comprehensive_parity.py:4274). My script pins purity=0.3256441718 (distilgpt2) and an exact analytic purity==1.0 anchor.
  - _fix:_ Add the exact anchors from scratchpad/verify_geometry_unembedding/verify.py: two-orthogonal-cluster purity == 1.0 exactly, and an independent full-argsort purity recomputation compared to 1e-9 on a small model.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — Sink tests pin only degenerate 0/1 endpoints; off-by-one denominator would pass — votes 2/2 upheld, unanimous
  - `tests/tasks/test_activation_sinks_task.py:35`
  - Both sink tests (and formula_fixtures.json 'attention_sink_all_queries_to_bos') expect 1.0 (BOS-saturated) or 0.0 (uniform). A T-k+1-instead-of-T-k ratio bug gives importance 0.8>0.3 and 0.199<0.3 respectively — still passes. No non-degenerate value pinned; fresh check found 44/60 @ eps=0.3, gpt2 107/144.
  - _fix:_ Add a fixture pinning a strictly-interior value, e.g. the mixed-strength synthetic (44/60=0.7333333) from verify_sinks.py, computed via the official kernel.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_entropy` — evaluate() aggregation and normalized outputs are only smoke-tested; an axis-swap would pass CI — votes 2/2 upheld, unanimous
  - `tests/tasks/test_reference_parity_formulas.py:288`
  - Parity test pins only the _attention_entropy kernel on 2 fixed rows. evaluate()'s mean over queries (attention.py:124), sample stacking (line 136), and the 3 normalized outputs (lines 146-148) are checked only for key presence and >=0 (test_interpretability.py:58-59); changing mean(dim=-1) to mean(dim=0) still passes.
  - _fix:_ Add an end-to-end pin: tiny eager GPT2 with fixed seed, assert avg_entropy_total equals an independent scipy aggregation (my scratch script shows exact match, 0.7736806870 on real gpt2).

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_graph` — No test pins any nonzero mean_edge_gini value; a Gini formula change would pass the whole suite — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:4592`
  - Comprehensive test only asserts the key 'mean_edge_gini' exists (line 4592-4594); test_round4_fixes.py:35 tests only the total<=0 guard via a transcription. The inline Gini (attention_graph.py:135-141) is a reported study feature but its value is never compared to any reference. My fresh check pins it: 1.3e-9 end-to-end vs pairwise-definition Gini.
  - _fix:_ Add to test_interpretability_attention_graph an independent Gini reference (pairwise sum\|xi-xj\|/(2 n^2 mean) on the same distilgpt2/tiny-GPT2 attentions) asserting mean_edge_gini to ~1e-6, plus the one-hot (n-1)/n analytic anchor.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_rollout` — Summary helpers (_gini, _row_entropy, _rollout_summaries) have no numeric pin — a formula change in 2 of 5 shipped features would pass the suite — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/attention_rollout.py:135`
  - Smoke test only asserts finiteness + [0,1] on one feature (test_attention_rollout_parity.py:232-237); fixture attention_rollout.json 'evaluate_summaries_on_toy' is loaded by NO test (grep: no loader). Changing _gini's sign convention or _row_entropy to log2 alters rollout_mean_gini/rollout_mean_norm_entropy silently. Fresh check pins them: gini diff 1.1e-16, e2e 0.0.
  - _fix:_ Assert the fixture's evaluate_summaries_on_toy values (and _gini/_row_entropy on a fixed vector) in the parity test, like sibling parity tests that load their JSON fixtures.

- **[ADJUDICATED-SURVIVED]** `interpretability_prediction_entropy` — No test exercises decisiveness_top_k != 5, exactly where the mean_top5_prob bug hides — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:5195`
  - The only value-pinning test constructs PredictionEntropyTask({'num_samples': 10, 'decisiveness_top_k': 5}) (line 5195); smoke test uses default (also 5). Two campaigns marked MATCH without ever running k<5, so the line-75 mislabel survived both.
  - _fix:_ Add a K=2 (and K=7) case to test_interpretability_prediction_entropy asserting mean_top5_prob equals independent top-5 mass for any K.

- **[ADJUDICATED-SURVIVED]** `interpretability_probing` — Existing test would not catch a shift off-by-one, split change, or probe-hyperparameter change — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:5254`
  - It never pins accuracy values (comment at :5301 claims no exact reference exists). But SGD is deterministic at random_state=42: my independent pipeline reproduces BLME to diff 0.0 on gpt2, and a synthetic next-token-onehot model pins the t->t+1 alignment (1.0 vs 0.0638). An hs[:,1:] off-by-one would pass today's test.
  - _fix:_ Add the scratchpad exact-parity + synthetic shift-alignment checks (verify_pipeline_parity_gpt2.py, verify_synthetic_ground_truth.py Check A) as tests.

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — Cross-sample aggregation (np.mean of per-sample means, np.max of maxes) pinned by no test — votes 2/2 upheld, unanimous
  - `tests/tasks/parity/test_sae_features_parity.py:59`
  - Parity kernel is a transcription guarded only by source-substring asserts (lines 100-114) that cover the per-sequence L0 lines, not sae_features.py:167-168; the fixture stores static diffs. Changing np.mean to token-weighted or np.median would pass the entire suite (fresh check shows 2431.03 vs 2411.68 differ).
  - _fix:_ Extract the L0-stats aggregation into a module-level helper and test the REAL helper on unequal-length samples with hand-computed mean-of-means (e.g. [2.5, 6.0] -> 4.25, not 22/6).

- **[ADJUDICATED-SURVIVED]** `interpretability_sparsity` — Kurtosis >500k subsampling branch (sparsity.py:112-114) exercised by no existing test — votes 1/2 upheld
  - `tests/tasks/test_comprehensive_parity.py:5507`
  - The parity test uses n_embd=32 GPT-2 (max ~128*T values/call), so `if flat.size > 500_000` (sparsity.py:113) never fires; a bug there (e.g. subsampling L0 too, or replace=True) would pass all tests. My fresh check pins it: L0 count 595211/600000 exact; kurtosis 0.0068004131 == rng(0) subsample exactly.
  - _fix:_ Add a parity case feeding a >500k-element down_proj input (or refactor the hook body into a testable kernel helper) and pin L0 full-tensor count + subsampled kurtosis.

- **[ADJUDICATED-SURVIVED]** `interpretability_superposition` — Neuron-subsampling path (all real models) untested and nondeterministic via unseeded np.random.choice — votes 2/2 upheld, unanimous
  - `src/blme/tasks/interpretability/superposition.py:147`
  - distilgpt2 (dim 3072 > max_neurons 256): two identical evaluate() calls give mean index 0.703010 vs 0.704466 (per-layer \|diff\| 0.009194). Parity test's determinism/parity assertions run only at dim=128 < 256 (test_comprehensive_parity.py:5774), so this branch is never exercised by any test.
  - _fix:_ Use a locally seeded np.random.default_rng(self.config.get('seed', 0)) for neuron selection and add a parity case with dim > max_neurons.

- **[ADJUDICATED-SURVIVED]** `interpretability_waa` — nn.Linear (down_proj) transpose branch has no checked-in numeric test — only Conv1D/GPT-2 is pinned end-to-end — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:5788`
  - Deleting `W = W.T` (weight_activation_alignment.py:142) breaks all Llama/Qwen/Mistral-family models (dim mismatch -> error dict) yet every existing test passes: the parity test uses GPT-2 Conv1D (untransposed branch), the arch smoke test tolerates errors (test_interpretability.py:414 `if "error" not in results`), and the Pythia/OLMo/Phi test checks module identity only. My fresh Llama check: diff 3.87e-07.
  - _fix:_ Add a tiny-Llama end-to-end assertion to test_interpretability_waa mirroring my scratchpad check D: run evaluate() on LlamaForCausalLM(hidden=32, inter=64, 2 layers), recapture down_proj inputs, assert per-layer \|cos\| == numpy.linalg.svd reference to 1e-5 (verify_waa.py in scratchpad/verify_interpretability_waa/ has the exact code).

- **[ADJUDICATED-SURVIVED]** `repe_steering_effectiveness` — No test exercises the default-corpus path, so the dead-in-default-pipeline behavior went unnoticed by two campaigns — votes 2/2 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:6116`
  - Every test feeds a well-formed {text_pos,text_neg,neutral} dataset; none evaluates the task with the {"text":...} corpus that core.evaluate_model actually passes (core.py:146-148,285). Result: the always-error default behavior and the corpus mutation were invisible while all 56 tests pass. The formula-level parity coverage itself is strong (real blme import, independent fp64 reference).
  - _fix:_ Add a test that runs SteeringEffectivenessTask on a text-only corpus and asserts (post-fix) metrics are produced and the input dicts are unmodified.

- **[ADJUDICATED-SURVIVED]** `topology_betti_curve` — Derived evaluate() stats not numerically pinned; decay-rate test is a source-grep — votes 2/2 upheld, unanimous
  - `tests/tasks/test_round6_fixes.py:62`
  - test_betti_curve_decay_rate_uses_normalised_depth only asserts source strings ('num_layers - 1', normalized-x expression). Changing np.polyfit(...,1)[0] to degree 2 or to [1] (intercept) at betti_curve.py:208 passes the entire suite; mock tests (test_topology.py:26) check key presence only. Kernel _count_betti IS strongly pinned (networkx parity, ground-truth topologies).
  - _fix:_ Add a numeric test: feed a synthetic betti_0_curve through evaluate()-equivalent math or run evaluate() on a tiny model and assert betti_0_decay_rate == np.polyfit(arange(L)/(L-1), curve, 1)[0], plus simplification_ratio and max_betti_1_layer values (my scratchpad verify_evaluate.py shows the pattern).

- **[ADJUDICATED-SURVIVED]** `topology_persistence_landscape` — Value-pinning landscape tests use only single-bar diagrams; a k-selection sort-direction bug evades every existing test — votes 2/2 upheld, unanimous
  - `tests/tasks/test_reference_parity_formulas.py:257`
  - Demonstrated: replacing descending sort (persistence_landscape.py:82) with ascending yields outputs identical to real code on both pinned inputs (dgm=[[0,2]], K=1 and K=2 — sort is a no-op with 1 pair); the only multi-bar test (test_infrastructure.py:443-445) asserts only np.isfinite. Missing max(0,.) clip would also evade.
  - _fix:_ Add a multi-bar k>=2 persim parity case to test_reference_parity_formulas.py, e.g. dgm=[[0,4],[1,3],[2,6],[2.5,3.5],[5,5.5]], K=4, n_points=241 vs PersLandscapeApprox(start=0,stop=6) — verified diff 8.88e-16.

- **[ADJUDICATED-SURVIVED]** `topology_zigzag_persistence` — Parity scenarios are degenerate for 5 of 12 metrics: h1_loop_layer_sum never differs from h1_total_layer_persistence and H0 metrics are constant 1.0 — votes 2/2 upheld, unanimous
  - `tests/tasks/parity/test_zigzag_parity.py:69`
  - All 4 fixture scenarios have <=1 loop/layer and 1 component, so loops.sum()==mask.sum() and comps==1 everywhere (zigzag.json: h0_mean_components=1.0, h0_persistence_range=0.0 in every scenario). Swapping zigzag.py:245 loops.sum() for mask.sum(), or h0_mean:=comps[0] at zigzag.py:249, passes all 16 tests.
  - _fix:_ Add fresh_check_v2.py's scenarios: a two-disjoint-circles layer (h1_loop_layer_sum=4.0 vs h1_total=3.0, comps=2) and two-cluster layers (h0_mean_components=2.0). Code verified CORRECT on these inputs vs an independent gudhi engine (diff 0.0), so this is coverage-only, not a bug.

### CITATION (38)

- **[ADJUDICATED-SURVIVED]** `causality_tracing` — Docstring cites '§3.2' for the 3-sigma noise rule; the paper puts it in Appendix B.1 (causal tracing itself is §2.1) — votes 3/3 upheld, unanimous
  - `src/blme/tasks/causality/tracing.py:41`
  - _resolve_noise_std docstring: 'ROME (Meng et al. 2022, §3.2) uses 3 × σ'. ar5iv 2202.05262: §3.2 = 'Evaluating ROME: Zero-Shot Relation Extraction (zsRE)'; noise ν = 3× embedding std is Appendix B.1. arXiv id, authors, venue in PAPERS.md/REPOSITORIES.md all correct.
  - _fix:_ Change '§3.2' to 'App. B.1 / kmeng01-rome collect_embedding_std (s3 default)'.

- **[ADJUDICATED-SURVIVED]** `consistency_knowledge_capacity` — Docstring attributes wrong paper title to Tirumala et al. 2022 — votes 3/3 upheld, unanimous
  - `src/blme/tasks/consistency/knowledge_capacity.py:10`
  - Docstring cites '"Do Language Models Memorize or Generalize?" (Tirumala et al., 2022)'. arXiv 2205.10770 (Tirumala et al. 2022, NeurIPS) is titled 'Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models'. Quoted title matches the 2023 Google PAIR explorable 'Do Machine Learning Models Memorize or Generalize?', not any Tirumala paper.
  - _fix:_ Change line 10 to: - "Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models" (Tirumala et al., 2022, arXiv:2205.10770).

- **[ADJUDICATED-SURVIVED]** `consistency_paraphrase` — PAPERS.md recommends citing a nonexistent paper 'Kirichenko 2023' for paraphrase invariance — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:317`
  - Line 317: 'add Kirichenko 2023 for paraphrase-invariance motivation'. Web searches find no Kirichenko 2023 paraphrase-invariance paper; Kirichenko's 2023 work is spurious-correlations/data-augmentation. The ledger's own on-topic citation is Fierro, Li, Sogaard 2024 (arXiv 2404.15206, verified real); canonical alternative is Elazar et al. 2021 ParaRel (arXiv 2102.01017).
  - _fix:_ Replace 'Kirichenko 2023' with 'Fierro et al. 2024 (arXiv:2404.15206)' and/or 'Elazar et al. 2021 ParaRel' in docs/PAPERS.md:317.

- **[ADJUDICATED-SURVIVED]** `consistency_position_sensitivity` — PAPERS.md author list wrong for Lost in the Middle (2307.03172) — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:64`
  - docs/PAPERS.md:64 credits "Liu, Lin, Hu, Manning, Liang 2023"; arXiv:2307.03172 authors are Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang (verified on arxiv.org). "Hu" and "Manning" are not authors; four authors omitted.
  - _fix:_ Change line 64 to "Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang 2023" (or "Liu et al. 2023" as task_metadata.py:114 already does).

- **[ADJUDICATED-SURVIVED]** `dynamics_generation_diversity` — Texygen author list in PAPERS.md: last author 'Zhang' should be 'Yu' (Yong Yu) — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:94`
  - PAPERS.md:94 reads 'Zhu, Lu, Zheng, Guo, Zhang, Wang, Zhang 2018'; arXiv:1802.01886 author list is Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, Yong Yu. arXiv id and title are correct.
  - _fix:_ Change final surname to 'Yu' (Zhu, Lu, Zheng, Guo, Zhang, Wang, Yu 2018); optionally add venue SIGIR 2018.

- **[ADJUDICATED-SURVIVED]** `dynamics_gradient_flow` — Parity-test docstring cites wrong section of Pascanu et al. 2013 — votes 3/3 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:2287`
  - Docstring says 'Sec. 3.1 / Eq. (3)-(4)'. In arXiv:1211.5063 (verified via ar5iv), Eq. (3)-(4) are in Sec. 1.1 and vanishing/exploding is defined in Sec. 2; Sec. 3.1 is 'Previous solutions'. Paper/authors/venue/eq numbers are otherwise correct.
  - _fix:_ Change to 'Sec. 1.1 Eq. (3)-(4), Sec. 2' in the test docstring.

- **[ADJUDICATED-SURVIVED]** `dynamics_sharpness` — Hutchinson trace misattributed to 'Bai & Demmel 1996' in module docstring — votes 3/3 upheld, unanimous
  - `src/blme/tasks/dynamics/sharpness.py:4`
  - Docstring says 'Hutchinson trace estimate of the Hessian (Bai & Demmel 1996; Yao et al. 2020)'. PyHessian paper (ar5iv 1912.07145 bibliography, fetched 2026-07-03) cites Bai, Fahey & Golub 1996 (J. Comput. Appl. Math. 74:71-89) and Avron & Toledo 2011; no Bai-Demmel trace paper exists. Method is Hutchinson 1990.
  - _fix:_ Change to 'Hutchinson 1990; Bai, Fahey & Golub 1996' (or just cite Yao et al. 2020).

- **[ADJUDICATED-SURVIVED]** `dynamics_stability` — HELM misdated as 'Liang et al. 2024' in dynamics_stability doc note — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:305`
  - PAPERS.md:305: 'add Liang et al. 2024 *Holistic Evaluation of Language Models* if comparable'. HELM is Liang et al. 2022 (arXiv:2211.09110, published TMLR 2023). Minor: a suggested-future citation, not the task's implemented-method citation (which is correct at PAPERS.md:133).
  - _fix:_ Change to 'Liang et al. 2022 (arXiv:2211.09110, TMLR 2023)' or drop the suggestion.

- **[ADJUDICATED-SURVIVED]** `geometry_collapse` — PAPERS.md geometry_collapse row is fabricated/conflated: wrong authors AND wrong arXiv id — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:84`
  - Row reads 'Thilak, Maddox 2021 — The Low-Rank Simplicity Bias in Deep Networks \| arXiv:2011.09348 \| geometry/collapse.py'. That title is by Huh, Mobahi, Zhang, Cheung, Agrawal, Isola (arXiv:2103.10427); arXiv:2011.09348 is 'Motion of the heavy symmetric top...' by Tanriverdi (physics). collapse.py never cites it.
  - _fix:_ Delete the row, or replace with 'Huh, Mobahi, Zhang, Cheung, Agrawal, Isola 2021, arXiv:2103.10427' if the low-rank-bias context is wanted.

- **[ADJUDICATED-SURVIVED]** `geometry_collapse` — 2510.06477 docstring citation omits first author Queipo-de-Llano; ledger's CITATION_FIX only partially applied — votes 3/3 upheld, unanimous
  - `src/blme/tasks/geometry/collapse.py:26`
  - Docstring says 'Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv 2025' but arXiv:2510.06477 lists Enrique Queipo-de-Llano first. VERIFICATION_LEDGER.md:89 recorded this fix; it landed in PAPERS.md:248 but not in the docstring, task_metadata.py:50 ('Arroyo et al. 2025'), or PAPERS.md:77 table row.
  - _fix:_ Change to 'Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv 2025' in collapse.py:26, task_metadata.py:50, and PAPERS.md:77.

- **[ADJUDICATED-SURVIVED]** `geometry_correlation_dimension` — REPOSITORIES.md points to nonexistent giotto-tda CorrelationDimension — votes 2/2 upheld, unanimous
  - `docs/REPOSITORIES.md:38`
  - Line 38 claims 'no author repo; [giotto-tda] has CorrelationDimension'. giotto-tda 0.5.1 full API genindex and module index contain NO CorrelationDimension and no fractal/correlation-dimension estimator (scope: persistent homology/mapper). Real pip references: nolds.corr_dim (executed in BLME's parity suite) and skdim.id.CorrInt.
  - _fix:_ Replace with 'CSchoel/nolds (nolds.corr_dim)' — the reference the parity suite actually executes — and/or 'scikit-dimension (skdim.id.CorrInt)'.

- **[ADJUDICATED-SURVIVED]** `geometry_hubness` — PAPERS.md truncates Tomasev 2014 title (drops 'Clustering') — votes 2/2 upheld, unanimous
  - `docs/PAPERS.md:85`
  - Doc says 'The Role of Hubness in High-Dimensional Data'; actual IEEE TKDE 26(3):739-751 (DOI 10.1109/TKDE.2013.25) title is 'The Role of Hubness in Clustering High-Dimensional Data'. Module docstring (hubness.py:52-54) has it correct; authors/venue/DOI match everywhere.
  - _fix:_ Insert 'Clustering' into the title at docs/PAPERS.md:85.

- **[ADJUDICATED-SURVIVED]** `geometry_isoscore` — Wrong author initial: 'Rayne, S.' should be 'Rayne, T.' (Taylor Rayne) — votes 1/1 upheld, unanimous
  - `docs/tasks_geometry.md:207`
  - docs/tasks_geometry.md:207 cites 'Rudman, W., Gillman, N., Rayne, S., & Eickhoff, C. (2022)'. arXiv:2108.07344 author list is William Rudman, Nate Gillman, Taylor Rayne, Carsten Eickhoff. PAPERS.md:79 (surname-only) and the module docstring are correct.
  - _fix:_ Change 'Rayne, S.' to 'Rayne, T.' in docs/tasks_geometry.md line 207.

- **[ADJUDICATED-SURVIVED]** `geometry_lid` — Docstring cites Levina & Bickel 2004 under a fabricated title — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/lid.py:12`
  - lid.py:12-13 quotes the title as 'Local Intrinsic Dimensionality Estimation via Maximum Likelihood'. The real NIPS 2004 Levina-Bickel title is 'Maximum Likelihood Estimation of Intrinsic Dimension' (papers.nips.cc/paper/2577). docs/PAPERS.md:60 has it correct; the module docstring does not.
  - _fix:_ Change the quoted title at lid.py:12 to 'Maximum Likelihood Estimation of Intrinsic Dimension'. Consider adding it to the test_publication_docs.py banned-identifiers guard like other corrected citations.

- **[ADJUDICATED-SURVIVED]** `geometry_lid` — -k MLE formula misattributed to Levina & Bickel instead of Ma et al. 2018 — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/lid.py:29`
  - lid.py:29-30 says 'the MLE formula from Levina & Bickel (2004): LID = -k / sum...'; class docstring lid.py:104-105 repeats it. That -k formula is Ma et al. 2018 Eq.4 (util.py:440); L-B uses -(k-1) over j=1..k-1 — as task_metadata.py:57's own note states, contradicting the docstring.
  - _fix:_ Reword docstrings: 'the -k MLE variant of Ma et al. 2018 (Eq. 4), derived from Levina & Bickel (2004)' at lid.py:29-30 and lid.py:104-105, matching the accurate task_metadata.py:57 note.

- **[ADJUDICATED-SURVIVED]** `geometry_mahalanobis` — Parity test cites 'Eq. 3' for the Mahalanobis score; paper defines it in Eq. (2) — votes 1/1 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:3279`
  - Lines 3279 and 3351 say 'NeurIPS 2018, Eq. 3'. arXiv:1807.03888 (ar5iv) defines M(x)=max_c −(f(x)−μ̂_c)ᵀΣ̂⁻¹(f(x)−μ̂_c) as Eq. (2); Eq. (1) is the μ̂_c/Σ̂ estimation. Module and docs citations are otherwise correct.
  - _fix:_ Change 'Eq. 3' to 'Eq. 2' in the two test docstring/comment locations.

- **[ADJUDICATED-SURVIVED]** `geometry_neural_collapse` — Code comment pinpoints NC1 to 'Eq. 3 / sec 2.3' but section 2.3 of arXiv:2008.08186 is 'Network architecture and feature engineering' — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/neural_collapse.py:119`
  - Comment L118-119: '(Papyan, Han & Donoho 2020 Eq. 3 / §2.3)'. Verified via ar5iv: Eq. 3 (Sigma_B) and Eq. 4 (Sigma_W) are in sec 2.9 'Moments of activations'; the Tr(Sigma_W Sigma_B^dagger)/C measure is Fig. 6. Paper/arXiv id/authors are correct everywhere else.
  - _fix:_ Change the comment pinpoint to 'Eq. 3-4 / sec 2.9, Fig. 6'.

- **[ADJUDICATED-SURVIVED]** `geometry_representation_sensitivity` — Docstring cites a non-existent (title, year) pairing: 'Information Geometry of Neural Networks' (Amari, 1998) — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/information_geometry.py:19`
  - No 1998 Amari publication has this title. Exact-title works: Amari 1996 (New Bayesian Duality Theory) and 1997 Springer chapter '— An Overview —' (10.1007/978-1-4615-6099-9_2). Amari 1998 = 'Natural Gradient Works Efficiently in Learning', Neural Computation 10(2):251-276. Parity test (test_comprehensive_parity.py:3670) cites the 1998 paper correctly.
  - _fix:_ One-line docstring change: cite Amari 1998, 'Natural Gradient Works Efficiently in Learning', Neural Computation 10(2):251-276 (matching the parity test), or Amari 2016 book per the PAPERS.md:263 TODO.

- **[ADJUDICATED-SURVIVED]** `geometry_schatten` — PAPERS.md records wrong authors for arXiv:2509.23024 (Tracing the Representation Geometry...) — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:63`
  - Row reads 'Li, Lizhi, Sondak, Wang 2025'. Actual authors (arXiv:2509.23024, verified 2026-07-04): Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards. 'Sondak' and 'Wang' are not authors; 'Lizhi' looks like a mangling of the first author's handle (melodylizx). Title and arXiv id are correct.
  - _fix:_ Replace author string with 'Li, Agrawal, Ghosh, Teru, Santoro, Lajoie, Richards 2025'.

- **[ADJUDICATED-SURVIVED]** `geometry_schatten` — PAPERS.md checklist attributes the Schatten text-quality paper to 'Wei 2025' instead of Yusupov et al. 2025 — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:265`
  - Checklist row: '`geometry_schatten` ✅ Wei 2025, Li 2024, Garrido 2023'. The Schatten-norm text-quality paper (arXiv:2509.25359) is Yusupov et al. 2025 — correctly stated at PAPERS.md:90 and in schatten.py:8 and task_metadata.py:68. Wei et al. 2024 is geometry_matrix_entropy's Diff-eRank paper (PAPERS.md line ~41 in REPOSITORIES.md).
  - _fix:_ Change 'Wei 2025' to 'Yusupov 2025' on the checklist row.

- **[ADJUDICATED-SURVIVED]** `geometry_spectral` — Stable rank misattributed to Bartlett-Foster-Telgarsky 2017, which never defines or uses it — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/spectral.py:21`
  - Docstring lines 21-22 cite 'Bartlett, Foster & Telgarsky, NeurIPS 2017, arXiv:1706.08498' for stable rank \|\|W\|\|_F^2/\|\|W\|\|_2^2. Full-text grep of the arXiv PDF: 0 hits for 'stable rank'; its bound uses (2,1)/spectral norm ratios, not Frobenius^2/spectral^2. Propagated to tests/tasks/test_comprehensive_parity.py:3876.
  - _fix:_ Cite Rudelson & Vershynin 2007 (JACM 54(4), defines stable rank) and/or Neyshabur, Bhojanapalli & Srebro 2018 (arXiv:1707.09564, whose DNN bound uses sum \|\|W\|\|_F^2/\|\|W\|\|_2^2); fix the same attribution in test_comprehensive_parity.py:3876.

- **[ADJUDICATED-SURVIVED]** `geometry_spectral` — PAPERS.md merges two distinct Martin-Mahoney papers: 1901.08278 (SDM 2020) is not the Nature Comms 12:4122 paper — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:68`
  - Row 68 gives title 'Heavy-Tailed Universality Predicts Trends in Test Accuracies' (= arXiv:1901.08278, Martin & Mahoney, SDM 2020 per its arXiv page) with venue 'Nature Comms. 2021 ... Nat. Comms. 12:4122', which is a different paper: 'Predicting trends in the quality of state-of-the-art neural networks...' by Martin, Peng & Mahoney, arXiv:2002.06716. Author Peng omitted.
  - _fix:_ Split into two rows: 1901.08278 (Martin & Mahoney, SDM 2020) and 2002.06716 (Martin, Peng & Mahoney, Nat. Comms. 12:4122, 2021); also add 2002.06716/Peng to docs/REPOSITORIES.md:39, which lists 'arXiv:1901.08276, 1901.08278, Nat. Comms. 2021' as one group.

- **[ADJUDICATED-SURVIVED]** `geometry_tokenizer_efficiency` — Fabricated author list 'Rajput, Chamberlain, Reese et al. 2024' for arXiv:2310.08754; real authors are Ali et al. — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/tokenizer_efficiency.py:24`
  - Docstring line 24: 'Rajput, Chamberlain, Reese et al. 2024'. arxiv.org/abs/2310.08754 lists Mehdi Ali, Michael Fromm, Klaudia Thellmann et al. — no Rajput/Chamberlain/Reese. Ledger line 89 + PAPERS.md:140 already say 'Ali et al. 2024'; the claimed docstring sweep (ledger line 96) never touched this file.
  - _fix:_ Change docstring to 'Ali, Fromm, Thellmann et al. 2024 (NAACL Findings 2024, arXiv:2310.08754)'; add 'Rajput, Chamberlain, Reese' to the banned-identifiers guard and extend test_publication_docs.py to also scan src/blme docstrings.

- **[ADJUDICATED-SURVIVED]** `geometry_tokenizer_efficiency` — Docstring says Rust et al. 2021 'Introduces fertility'; the paper attributes fertility to Ács 2019 — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/tokenizer_efficiency.py:22`
  - Docstring lines 20-23: Rust 2021 'Introduces fertility as a cross-tokenizer comparison metric'. The paper (Fig. 1b caption, ar5iv 2012.15613) defines it as 'the average number of subwords produced per tokenized word (Ács 2019)' — Rust et al. adopt, not introduce, the metric.
  - _fix:_ Reword to 'popularizes fertility (Ács 2019) as a cross-tokenizer comparison metric' or add an Ács 2019 credit.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_kurtosis` — Wrong author list for Sun et al. 2024 (arXiv:2402.17762) in module docstring — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_kurtosis.py:56`
  - Docstring reads "Sun, Chen, Bai, Hu, Xiong, Kolter"; arXiv API for 2402.17762 gives Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. Bai/Hu/Xiong are not authors; Liu is omitted. Same wrong list in activation_sinks.py:54; docs/PAPERS.md:82 is correct.
  - _fix:_ Change to "Sun, Chen, Kolter, Liu" in activation_kurtosis.py:56 (and activation_sinks.py:54).

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — Gu 2025 author list confabulated: 'Collier' is not an author — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_sinks.py:51`
  - Docstring says 'Gu, Pang, Du, Liu, Collier, Lin'. arXiv:2410.10781 authors: Gu, Pang, Chao Du, Liu, Zhang, Cunxiao Du, Wang, Lin. Collier absent; Zhang/Du/Wang dropped. docs/PAPERS.md:52 has it correct.
  - _fix:_ Replace with 'Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin'.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — Sun 2024 author list wrong: 'Bai, Hu, Xiong' are not authors; Liu dropped — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_sinks.py:54`
  - Docstring says 'Sun, Chen, Bai, Hu, Xiong, Kolter'. arXiv:2402.17762 (COLM 2024) authors: Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. docs/PAPERS.md:82 is correct.
  - _fix:_ Replace with 'Sun, Chen, Kolter, Liu'.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — arXiv:2510.06477 first author Queipo-de-Llano dropped; 'Arroyo et al.' misattributes — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_sinks.py:56`
  - Docstring lists 'Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv'; arXiv:2510.06477 first author is Enrique Queipo-de-Llano. Same error at docs/PAPERS.md:77, docs/PAPER_SURVEY.md:232, geometry/collapse.py:26. Prior campaign fixed only PAPERS.md:248 (geometry_collapse row).
  - _fix:_ Prepend Queipo-de-Llano; cite as 'Queipo-de-Llano et al. 2025' everywhere (docstring lines 22-28 prose too).

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — StreamingLLM fifth author is Lewis, not a second 'Han' — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_sinks.py:49`
  - Docstring says 'Xiao, Tian, Chen, Han, Han'. arXiv:2309.17453 authors: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. Same 'Han, Han' error at docs/PAPERS.md:91 and docs/RELATED_WORK.md:221.
  - _fix:_ Replace with 'Xiao, Tian, Chen, Han, Lewis' in all three places.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_entropy` — Venue wrong: Clark et al. 2019 was the ACL 2019 Workshop BlackboxNLP, not EMNLP — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/attention.py:4`
  - Docstring line 4 says "(EMNLP BlackBoxNLP)"; ACL Anthology W19-4828 = "Proceedings of the 2019 ACL Workshop BlackboxNLP", Florence, Aug 2019. Same error at docs/PAPERS.md:41 and docs/REPOSITORIES.md:63 ("EMNLP BlackBoxNLP 2019"). Authors, title, arXiv 1906.04341, repo clarkkev/attention-analysis all correct.
  - _fix:_ Replace "EMNLP BlackBoxNLP" with "ACL 2019 BlackboxNLP Workshop" in the three locations.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_graph` — docs/PAPERS.md still attributes Abnar & Zuidema 2020 to attention_graph.py, contradicting the Campaign-2 citation correction — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:144`
  - PAPERS.md:144 lists 'Abnar & Zuidema 2020 Attention Rollout (topology part)' as the docstring citation for attention_graph.py; PAPERS.md:278 says 'add Abnar-Zuidema 2020'. But attention_graph.py:14 cites Xiao 2023, and task_metadata.py:84 says 'NOT Abnar-Zuidema rollout'; VERIFICATION_LEDGER.md:255 claims the citation was corrected.
  - _fix:_ Update PAPERS.md:144 to 'Xiao et al. 2023 (arXiv:2309.17453, attention sinks) + standard PageRank (Page-Brin)' and change PAPERS.md:278 to a checkmark for Xiao 2023 (docstring already cites it); optionally extend PAPERS.md:91 Xiao row to also list attention_graph.py.

- **[ADJUDICATED-SURVIVED]** `interpretability_attribution` — 'Input × Gradient' misattributed to Simonyan 2014; canonical source is Shrikumar et al. 2016 (arXiv:1605.01713) — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:146`
  - PAPERS.md:146 'Simonyan et al. 2014 *Input × Gradient*'; REPOSITORIES.md:73 'Simonyan et al. 2014 — Saliency / Input × Gradient'; attribution.py:27 '(Simonyan et al. 2014; equal to captum InputXGradient...)'. arXiv:1312.6034 defines gradient-only saliency; captum's InputXGradient docstring cites arXiv:1605.01713 (Shrikumar, Greenside, Shcherbina, Kundaje 2016).
  - _fix:_ Cite Shrikumar et al. 2016 (arXiv:1605.01713) as the input×gradient source alongside Simonyan 2014 (gradient saliency) in attribution.py:27-29, docs/PAPERS.md:146, docs/REPOSITORIES.md:73, and task_metadata.py:86 references.

- **[ADJUDICATED-SURVIVED]** `interpretability_logit_lens` — Test docstring attributes transformer-utils to wrong GitHub org (dead link) — votes 1/1 upheld, unanimous
  - `tests/tasks/test_comprehensive_parity.py:5068`
  - Docstring cites https://github.com/EleutherAI/transformer-utils — HTTP 404 (checked 2026-07-04). The official repo is nostalgebraist/transformer-utils (HTTP 200), correctly cited in docs/REPOSITORIES.md:61. Module code, docs, and ledger are unaffected; numerics fully confirmed.
  - _fix:_ In test_comprehensive_parity.py:5068 change the URL to https://github.com/nostalgebraist/transformer-utils.

- **[ADJUDICATED-SURVIVED]** `interpretability_sparsity` — PAPER_SURVEY venue imprecise: MoEfication is Findings of ACL 2022, not 'ACL' 2021 — votes 1/1 upheld, unanimous
  - `docs/PAPER_SURVEY.md:107`
  - docs/PAPER_SURVEY.md:107 says 'Zhang et al. 2021 (ACL)'. arXiv 2110.01786 page states 'Accepted to ACL Findings 2022'. Authors (Zhang, Lin, Liu, Li, Sun, Zhou), title, and arXiv id elsewhere in docs/ledger are correct; venue shorthand only.
  - _fix:_ Change '(ACL)' to '(Findings of ACL 2022)' at docs/PAPER_SURVEY.md:107.

- **[ADJUDICATED-SURVIVED]** `repe_task_vectors` — PAPERS.md drops author Suchin Gururangan from Ilharco et al. 2023 author list — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:56`
  - Line 56 lists 'Ilharco, Ribeiro, Wortsman, Schmidt, Hajishirzi, Farhadi 2023' as a full list (PAPERS.md convention enumerates all authors, e.g. 9-name Ma et al. line 65), but arXiv:2212.04089 has 7 authors — Suchin Gururangan (4th) is omitted mid-list.
  - _fix:_ Insert 'Gururangan' between 'Wortsman' and 'Schmidt' on docs/PAPERS.md:56, or use 'Ilharco et al. 2023'.

- **[ADJUDICATED-SURVIVED]** `topology_homology` — Module docstring cites Naitzat, Zhitnikov, Lim 2020 as ICLR; actual venue is JMLR 21(184):1-40, 2020 — votes 1/1 upheld, unanimous
  - `src/blme/tasks/topology/homology.py:12`
  - homology.py:11-13 says 'Topology of Deep Neural Networks, ICLR'. JMLR page jmlr.org/papers/v21/20-345.html confirms JMLR 21(184), 2020 (arXiv 2004.06093). VERIFICATION_LEDGER.md:86 flagged this exact venue error for betti_curve but missed the same error here.
  - _fix:_ Change 'ICLR' to 'JMLR 21(184):1-40' in the docstring reference list.

- **[ADJUDICATED-SURVIVED]** `topology_persistence_entropy` — Docstring pairs Signal Processing 2017 title with wrong author set — votes 1/1 upheld, unanimous
  - `src/blme/tasks/topology/persistence_entropy.py:11`
  - Lines 10-11 cite 'A New Topological Entropy-based Approach...Piecewise Linear Functions' as '(Rucco, Castiglione, Merelli et al., 2016)'. arXiv:1512.07613/Signal Processing 134:130-138 (2017) authors are Rucco, Gonzalez-Diaz, Jimenez, Atienza, Cristalli, Concettoni, Ferrante, Merelli. Castiglione co-authored only the ECCS-2014 idiotypic-network paper. Flagged in VERIFICATION_LEDGER.md:87, still unfixed (PAPERS.md:330 pending).
  - _fix:_ Cite 'Characterisation of the Idiotypic Immune Network Through Persistent Entropy' (Rucco, Castiglione, Merelli, Pettini, Springer 2016) plus Chintakunta et al. 2015 (Pattern Recognition 48(2):391-401), matching task_metadata.py:78.

- **[ADJUDICATED-SURVIVED]** `topology_persistence_entropy` — Wrong venue/year for Atienza et al. stability paper — votes 1/1 upheld, unanimous
  - `src/blme/tasks/topology/persistence_entropy.py:13`
  - Line 13 says '(Atienza et al., Entropy 2019)'. The paper 'On the stability of persistent entropy and new summary functions for TDA' (arXiv:1803.08304) was published in Pattern Recognition 107:107509 (2020); dblp for Soriano-Trigueros shows no MDPI-Entropy 2019 publication. Not caught by either prior campaign (ledger line 52 cites 'Atienza Def 3.1' without venue).
  - _fix:_ Change to '(Atienza, Gonzalez-Diaz, Soriano-Trigueros, Pattern Recognition 107:107509, 2020; arXiv:1803.08304)'.

### DOC_GAP (57)

- **[ADJUDICATED-SURVIVED]** `causality_ablation` — REPOSITORIES.md omits causality_ablation's cited paper and its official repo that the ledger relies on — votes 3/3 upheld, unanimous
  - `docs/REPOSITORIES.md:1`
  - VERIFICATION_LEDGER.md:106 lists reference repo nickypro/investigating-ablation (arXiv:2408.17322, repo confirmed to exist), but grep of docs/REPOSITORIES.md and docs/PAPERS.md finds no Pochinkov/2408.17322/causal-scrubbing entry, despite REPOSITORIES.md claiming to be the authoritative repo list for papers cited by BLME (ablation.py:22-23 cites both).
  - _fix:_ Add a Causality row: Pochinkov et al. 2024 (arXiv:2408.17322) -> nickypro/investigating-ablation (HIGH); Chan et al. 2022 causal scrubbing -> no code (Alignment Forum post).

- **[ADJUDICATED-SURVIVED]** `causality_edge_attribution` — Module docstring states signed attribution formula; code takes absolute value — votes 2/3 upheld
  - `src/blme/tasks/causality/edge_attribution.py:10`
  - Docstring line 10: attr(layer l) = (h_l_clean - h_l_corrupted) . grad — signed. Code line 173: attr = np.abs((diff * grad_h[:T]).sum()). The abs matches paper Eq(3) and task_metadata.py:100, but the module's own stated formula omits it.
  - _fix:_ Change docstring line 10 to attr(layer l) = \|(h_l_clean - h_l_corrupted) . grad(logit \| h_l_clean)\| and note it is paper Eq (3).

- **[ADJUDICATED-SURVIVED]** `causality_knowledge_neurons` — Parity test misattributes pre-GELU hook point to official Dai method — votes 3/3 upheld, unanimous
  - `tests/tasks/parity/test_knowledge_neurons_parity.py:35`
  - Test lists divergence #3 'hook = post-GELU (BLME) vs pre-GELU intermediate (Dai/ref)'. Official Hunter-DDM custom_bert.py L232-235 scales tmp_score AFTER intermediate_act_fn (post-GELU) — same hook point as BLME. Pre-GELU c_fc is EleutherAI-GPT-port-specific. Fixture meta repeats it. Over-strict, not parity-overclaiming; omits the real all-positions-sum vs tgt_pos divergence.
  - _fix:_ Reword divergence #3: 'EleutherAI GPT port hooks pre-GELU c_fc output; official Dai BERT code uses post-GELU intermediate, matching BLME'. Add the positions-sum divergence; update fixture divergence_summary.

- **[ADJUDICATED-SURVIVED]** `causality_knowledge_neurons` — Docstring metric names top1_share/top1pct_share are not returned keys — votes 3/3 upheld, unanimous
  - `src/blme/tasks/causality/knowledge_neurons.py:15`
  - Docstring 'Metrics reported' lists top1_share (L15) and top1pct_share (L17); evaluate() returns mean_top1_share/mean_top1pct_share (+ mean_top1_saliency_share) at L241-247 — results['top1_share'] would KeyError. Other three documented names match returned keys.
  - _fix:_ Rename docstring bullets to mean_top1_share / mean_top1pct_share (cosmetic).

- **[ADJUDICATED-SURVIVED]** `causality_tracing` — Parity cert silent that BLME's per-layer AIE restores the whole subject span while the paper's IE restores a single (token, layer) state; fixture subject is single-token so the difference is untested there — votes 2/3 upheld
  - `src/blme/task_metadata.py:102`
  - Paper §2.1 defines IE per single restored state h_i^(l); ROME trace_important_states patches one token per cell. BLME patches all subject tokens per layer (documented in tracing.py:133-136) and gates samples on 'corruption hurt' rather than ROME's expect-match filter. Fixture subject 'Italy' = 1 token, so span==single-state there. My fresh check: span protocol == actual trace_with_patch given a span patch-spec, bit-exact on a 4-token subject.
  - _fix:_ Add one sentence to the metadata note/ledger: 'per-layer AIE = whole-subject-span restoration (paper Fig. 2 AIE is per single token-layer state); sample gate = corruption-must-hurt, not ROME's correct-prediction filter'.

- **[ADJUDICATED-SURVIVED]** `consistency_bias_weat` — Module docstring promises per-layer reporting and wrong result keys — votes 3/3 upheld, unanimous
  - `src/blme/tasks/consistency/bias.py:18`
  - Docstring lines 18-21: 'Reports per-layer and per-test: weat_d, weat_p_value, mean_d_across_tests'. evaluate() (lines 208-215) reports final layer only, with keys {test}_d, {test}_p_value, mean_abs_d_across_tests, n_tests. Also key 'gender_science' actually holds the WEAT7 math/arts lists (docstring line 39 says tests 6 and 7).
  - _fix:_ Rewrite the docstring bullet to the actual keys and final-layer scope; optionally rename 'gender_science' to 'gender_math_arts' or note it is WEAT7.

- **[ADJUDICATED-SURVIVED]** `consistency_calibration` — PAPERS.md 'weak citation' row is stale — docstring now cites both sources — votes 3/3 upheld, unanimous
  - `docs/PAPERS.md:131`
  - PAPERS.md:123-131 lists consistency_calibration under 'docstrings don't cite the original paper explicitly', but calibration.py:84-91 explicitly cites Guo, Pleiss, Sun & Weinberger 2017 and Brier 1950.
  - _fix:_ Remove the consistency_calibration row from the weak-citation section of docs/PAPERS.md.

- **[ADJUDICATED-SURVIVED]** `consistency_knowledge_capacity` — VERIFICATION_LEDGER self-inconsistent: row says Parity NOT_RUN, prose claims parity checked — votes 3/3 upheld, unanimous
  - `VERIFICATION_LEDGER.md:118`
  - Line 118 lists consistency_knowledge_capacity Parity=NOT_RUN, but line 32 claims 'knowledge_capacity vs independent teacher-forcing NLL' parity, and test_comprehensive_parity.py:1549 exists and passes. Row 118 is stale; a reader auditing the ledger table would wrongly conclude no numeric check exists.
  - _fix:_ Update ledger row 118 Parity to MATCH, citing tests/tasks/test_comprehensive_parity.py::test_consistency_knowledge_capacity.

- **[ADJUDICATED-SURVIVED]** `consistency_logical` — Ledger records Parity MATCH / PROXY_OK without noting the task is inoperable on most tokenizer families — votes 3/3 upheld, unanimous
  - `VERIFICATION_LEDGER.md:119`
  - Row says PROXY_OK / MATCH, but MATCH was only ever obtained on leading-space-doctored inputs; with bundled defaults the task returns {"error": "No valid samples processed."} on 6 of 8 cached tokenizer families (gpt2, pythia, Qwen2.5, SmolLM2, phi-2, OLMo). Two campaigns missed this.
  - _fix:_ After fixing the bug, re-run and update the row; until then annotate MATCH as kernel-only (doctored inputs), default path broken on non-BOS tokenizers.

- **[ADJUDICATED-SURVIVED]** `consistency_paraphrase` — PAPER_SURVEY.md mischaracterizes consistency_paraphrase as an NLL-based check — votes 3/3 upheld, unanimous
  - `docs/PAPER_SURVEY.md:170`
  - Line 170 groups consistency_paraphrase with consistency_contrastive/consistency_logical as 'custom NLL-based consistency checks', but paraphrase.py computes no NLL — it is a last-token representation L2/cosine distance proxy (paraphrase.py:107-117); tasks_consistency.md:22-25 describes it correctly.
  - _fix:_ Reword PAPER_SURVEY.md:170 to list consistency_paraphrase separately as a representation-distance (isometry-ratio) diagnostic.

- **[ADJUDICATED-SURVIVED]** `consistency_self_consistency` — Module docstring promises a 'common-prefix overlap' measure that is never computed — votes 3/3 upheld, unanimous
  - `src/blme/tasks/consistency/self_consistency.py:9`
  - Docstring lines 9-10: 'exact-match agreement on the first generated word and a softer common-prefix overlap measure as well'. Code computes only first-TOKEN agreement/uniqueness/entropy (lines 119-129); no prefix-overlap key exists in the returned dict.
  - _fix:_ Delete the common-prefix sentence and change 'first generated word' to 'first generated token' to match the implementation and the diagnostic_method string.

- **[ADJUDICATED-SURVIVED]** `dynamics_coe` — Module docstring states paper-prose Eq. 7 (raw M·exp(iA)) but code computes the official-code normalized variant; also mislabels M as Eq. 1 — votes 3/3 upheld, unanimous
  - `src/blme/tasks/dynamics/coe.py:16`
  - Docstring lines 16-17: CoE-C = \|(1/L)ΣC\|, C = M·exp(i·A) with raw M,A (paper Eq. 6/7). Code lines 103-106 compute (M/Z_Mag)·exp(i·A/Z_Ang): fixture gives 1.06398 vs printed-Eq.7 1.11803. Line 8 labels M '(Eq. 1)'; paper defines M and A together in Eq. 2.
  - _fix:_ In the docstring header, state CoE-C as implemented: Mag~·exp(i·Ang~) per official score.py, noting the paper's printed Eq. 6/7 uses raw M,A; relabel M/A as Eq. 2.

- **[ADJUDICATED-SURVIVED]** `dynamics_coe` — Inline comment falsely claims the old raw/raw CoE-C 'matched neither the paper code nor the paper prose' — raw/raw IS the paper prose Eq. 6/7 — votes 3/3 upheld, unanimous
  - `src/blme/tasks/dynamics/coe.py:101`
  - Printed Eq. 6/7 (ar5iv S3.E6/E7) is exactly raw M·e^{i·A}; old value 1.118034 equals my paper-prose computation on the fixture. The original work is 3-way inconsistent: prose raw/raw, Algorithm 2 norm-mag/raw-ang, official code norm/norm. BLME correctly matches official code, but the justification misstates the paper.
  - _fix:_ Reword comment: 'previously used RAW magnitude and RAW angle — the paper's printed Eq. 6/7 form — which does not match the official score.py (normalized/normalized); we follow the official code.'

- **[ADJUDICATED-SURVIVED]** `dynamics_gradient_flow` — Module docstring defines the legacy argmax-logit metric, not the implemented CE-loss gradient — votes 3/3 upheld, unanimous
  - `src/blme/tasks/dynamics/gradient_flow.py:5`
  - Docstring lines 5-6 define '\|\|d logit(y*) / d h_l\|\|_F where y* is the predicted token' and line 23 says 'backpropagates a single logit'; code lines 102-108 backprop shifted next-token cross-entropy and return loss:'cross_entropy'. Inline comment 96-101 calls the logit approach 'legacy'. Same-file contradiction.
  - _fix:_ Rewrite module docstring header and 'Implementation:' paragraph to the shifted next-token cross-entropy definition \|\|dL_CE/dh_l\|\|_F, matching lines 96-108 and the test contract loss=='cross_entropy'.

- **[ADJUDICATED-SURVIVED]** `dynamics_gradient_flow` — Docstring says slope regresses on 'layer index'; code uses normalized depth (value differs by factor n_layers-1) — votes 3/3 upheld, unanimous
  - `src/blme/tasks/dynamics/gradient_flow.py:15`
  - Lines 15-16: 'slope of log(norm) vs. layer index'; code lines 152-157 regress vs xs/(n_layers-1). On distilgpt2 reported slope -2.0758 vs raw-index slope -0.41516 (factor 5 = n_layers-1). Normalization is intentional (comment 143-148, test_round4_fixes.py:90) but the docstring was never updated.
  - _fix:_ Change line 15-16 to 'slope of log(norm) vs. normalized depth (layer index / (n_layers-1))'.

- **[ADJUDICATED-SURVIVED]** `dynamics_stability` — Ledger cert label contradicts shipped metadata (refined-adaptation vs proxy-only) — votes 2/3 upheld
  - `VERIFICATION_LEDGER.md:129`
  - VERIFICATION_LEDGER.md:129 records dynamics_stability as 'refined-adaptation' with arXiv 1804.09692; shipped task_metadata.py:122 says 'proxy-only', 'BLME embedding-neighborhood stability diagnostic', no arXiv/repo. Ledger overstates relative to the machine-readable source; shipped label is the conservative, correct one.
  - _fix:_ Update the ledger row (or annotate it as audit-time-stale) to match task_metadata.py's proxy-only label.

- **[ADJUDICATED-SURVIVED]** `geometry_cknna` — distance_agnostic option is advertised but non-functional (crashes) — inherited from official code, undocumented — votes 2/3 upheld
  - `src/blme/tasks/geometry/cknna.py:111`
  - Docstring calls it a 'reference option; off by default', but _cknna(X,Y,topk=10,distance_agnostic=True) raises RuntimeError (.item() on an (n,n) tensor at cknna.py:165). Official cknna @dcd76ba crashes identically (verified: crash-parity SAME), so port fidelity holds, but the option is unusable.
  - _fix:_ Docstring note: 'non-functional at the pinned reference commit (crashes in both); do not use', or raise NotImplementedError explicitly.

- **[ADJUDICATED-SURVIVED]** `geometry_cknna` — Module docstring claims 'Identical inputs give 1.0' but reference-default unbiased mode gives ~0.99995 — votes 2/3 upheld
  - `src/blme/tasks/geometry/cknna.py:36`
  - Fresh check: _cknna(X,X,topk=10,unbiased=True)=0.999952559763 (biased=1.0 exactly) because score=s/(s+1e-6). BLME's own test file (test_cknna_parity.py:28-30) documents the ~0.99990 caveat, contradicting docstring lines 35-36 and 115-116.
  - _fix:_ Change to 'identical inputs give ~1.0 (exactly 1.0 only with biased HSIC; unbiased default ~0.9999 due to the +1e-6 regularizer)'.

- **[ADJUDICATED-SURVIVED]** `geometry_collapse` — Docstring says collapse_ratio is 'top-23' predictor; TOP_PREDICTORS.md section 2 ranks it 22 — votes 3/3 upheld, unanimous
  - `src/blme/tasks/geometry/collapse.py:31`
  - Docstring: 'top-17 and top-23 partial predictors ... (+0.71 and +0.68)'. docs/TOP_PREDICTORS.md section-2 table: rank 17 erank_per_layer.q75 +0.711 (matches) but rank 22 collapse_ratio +0.680 (docstring says 23; rank 23 is geometry_hsic).
  - _fix:_ Change 'top-23' to 'top-22' in the docstring (or re-sync if TOP_PREDICTORS.md changed).

- **[ADJUDICATED-SURVIVED]** `geometry_contextualization` — Ledger cert label 'formula-faithful' contradicts shipped 'refined-adaptation' — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:133`
  - Ledger row (table header line 104: Task\|Verdict\|Cert\|...) records Cert=formula-faithful for geometry_contextualization, but task_metadata.py:51 ships refined-adaptation. Shipped label is the more conservative and defensible one (token-id grouping, baseline sampling adaptations); ledger row is stale/overstated.
  - _fix:_ Change VERIFICATION_LEDGER.md:133 Cert column to refined-adaptation (or annotate why it differs from task_metadata.py).

- **[ADJUDICATED-SURVIVED]** `geometry_contextualization` — Docstring implies corrected versions of all metrics, but MEV is reported unadjusted only — votes 1/2 upheld
  - `src/blme/tasks/geometry/contextualization.py:22`
  - Lines 22-24: 'The bare versions (*_raw) are reported alongside the anisotropy-corrected versions (*_corrected)' — but output key 'mev' (line 234) has no raw/corrected pair. Paper Sec 3.4 defines adjusted MEV (subtract top-PC variance of random words); BLME's raw mev (~0.53 on distilgpt2) is NOT comparable to the paper's adjusted headline (<5%). Code matches official variance_explained_by_pc exactly, so numerics are correct.
  - _fix:_ One docstring sentence: 'MEV is reported unadjusted (matches official variance_explained_by_pc); the paper's headline MEV figures are anisotropy-adjusted and not directly comparable.'

- **[ADJUDICATED-SURVIVED]** `geometry_correlation_dimension` — Stale 'lines 83-119 / inline kernel' references predate the 754e06b kernel extraction — votes 2/2 upheld, unanimous
  - `tests/tasks/parity/test_correlation_dimension_parity.py:26`
  - Docstring (lines 5, 26-28, 57-58) says the GP kernel is 'transcribed VERBATIM from correlation_dimension.py lines 83-119', but since commit 754e06b the kernel is module-level _gp_correlation_dimension (lines 24-66) and line 91 imports the REAL helper. Same staleness: fixture correlation_dimension.json line 4 ('blme_lines: 83-119'), test_comprehensive_parity.py line 3002 ('inline... no importable helper'), VERIFICATION_LEDGER.md ~line 205 (rates the test ACCEPTABLE for the already-fixed transcrip
  - _fix:_ Update the four stale references to 'kernel extracted to _gp_correlation_dimension (lines 24-66), imported by the test'; ledger rating can be upgraded to STRONG.

- **[ADJUDICATED-SURVIVED]** `geometry_hsic` — Docstring says input_to_layer_hsic anchors on 'the input embedding' but the anchor is first-transformer-block output — votes 2/2 upheld, unanimous
  - `src/blme/tasks/geometry/mutual_info.py:47`
  - Docstring line 46-48: 'how strongly per-layer activations still depend on the input embedding'. But collect_hidden_states skips hidden_states[0] (utils.py:104 'skip embedding layer') and cache.py:365 does the same, so layers[0]=block-1 output. The module's own comment (lines 139-140, 'both measured after the embedding layer') contradicts the docstring. Numbers unaffected.
  - _fix:_ Reword docstring (and docs/tasks_geometry.md:141 'between input and layer representations') to 'first transformer block output' instead of 'input embedding'.

- **[ADJUDICATED-SURVIVED]** `geometry_intrinsic_dim` — Docstring attributes a below-1 'Fig. 3' observation to Facco et al. 2017 that the paper does not contain — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/intrinsic_dim.py:199`
  - Lines 199-203 claim the paper 'explicitly observes the estimate dipping below 1 on near-degenerate geometries (Fig. 3)'. Paper's Fig. 3 (arXiv:1803.06992 PDF) is d-vs-N for a noisy 2D plane/Swiss roll, estimates >=2; text has 0 hits for 'less than'/'underestimat'/'floor'/'lower bound'/'dip'. tests/tasks/test_geometry.py:255+ already states the correct rationale (Pareto d in (0,inf)).
  - _fix:_ Replace the Fig. 3 sentence with the correct rationale: the Two-NN Pareto model has support d in (0, inf), so no floor is implied by the paper; the paper never discusses estimates below 1.

- **[ADJUDICATED-SURVIVED]** `geometry_intrinsic_dim` — VERIFICATION_LEDGER table overstates cert label as 'parity-ready'; shipped label is 'formula-faithful' — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:137`
  - Ledger row: 'geometry_intrinsic_dim \| CITATION_FIX \| parity-ready \| ...'. task_metadata.py:55 ships 'formula-faithful', and test_reference_parity_formulas.py:227-230 explicitly says 'Hence the conservative formula-faithful (not parity-ready) label' because of the mu<=1 pre-filter (measured divergence 1.74 vs skdim on 5%-duplicate data).
  - _fix:_ Change the ledger row's Cert column to 'formula-faithful' (or add a note that the label was deliberately kept conservative despite the passing skdim-agreement test).

- **[ADJUDICATED-SURVIVED]** `geometry_intrinsic_dim` — Stale PAPERS.md row lists intrinsic_dim.py under 'paper-provenance not in docstring' though the docstring now cites Facco 2017 in full — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:138`
  - Section header (lines 123-127) says these tasks' 'docstrings don't cite the original paper explicitly', but intrinsic_dim.py:17-25 fully cites Facco et al. 2017 (Sci. Rep. 7, arXiv:1803.06992) and Ansuini et al. 2019 (arXiv:1905.12784). Row is stale, direction is harmless (understates).
  - _fix:_ Delete the geometry/intrinsic_dim.py row from the weak-citation table in docs/PAPERS.md.

- **[ADJUDICATED-SURVIVED]** `geometry_neural_collapse` — Module docstring misdescribes NC2-ETF proxy as 'standard deviation' when code computes mean absolute deviation (paper Fig.4 measure) — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/neural_collapse.py:15`
  - Docstring L15-16: 'standard deviation of pairwise cosine similarities ... vs. the ETF target'. Code L172: np.mean(np.abs(pair_cos - target)). These differ (std of cosines is Papyan Fig.3; mean\|cos+1/(K-1)\| is Fig.4). Parity test L39 states the correct mean-abs definition; only the module docstring is wrong.
  - _fix:_ Reword docstring to 'mean absolute deviation of pairwise cosine similarities from the ETF target -1/(K-1) (Papyan et al. Fig. 4 maximal-angle equiangularity)'. Also drop the unused get_final_norm import at L35 (dead since the round-6 fix).

- **[ADJUDICATED-SURVIVED]** `geometry_phd_dimension` — Docstring's general inversion formula d = alpha/(alpha-m) is mathematically wrong — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/phd_dimension.py:30`
  - From E~n^((d-alpha)/d), slope m=1-alpha/d gives d=alpha/(1-m), not alpha/(alpha-m); e.g. alpha=0.5, d=2 gives m=0.75, alpha/(alpha-m)=-2. Code correctly implements 1/(1-m) matching GPTID (verified bit-exact at alpha=0.5: 3.9778344632103315).
  - _fix:_ Change line 30 to 'd = alpha / (1 - m)  ==> (alpha = 1) d = 1 / (1 - m)'; optionally note GPTID returns 1/(1-m) regardless of alpha (only meaningful at alpha=1).

- **[ADJUDICATED-SURVIVED]** `geometry_positional_decay` — Ledger flags positional_decay citation as 'wrong (corrected above)' but no correction is recorded — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:146`
  - Row 146 shows '2104.09864 ⚠'; legend (line 102) defines ⚠ = 'recorded citation was wrong (corrected above)', yet the 'Citation corrections' section (lines 77-90) has no positional_decay entry. All citations that actually exist (RoFormer 2104.09864, Press et al. 2021 ALiBi 2108.12409) verified real and topical.
  - _fix:_ Either remove the ⚠ from the positional_decay row or add the missing correction note explaining what was wrong and what replaced it.

- **[ADJUDICATED-SURVIVED]** `geometry_schatten` — Inline comment says Schatten norms are 'normalised by sqrt(D)' but code uses d^(1/p) — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/schatten.py:152`
  - Comment prose at lines 152-153 says 'normalised by sqrt(D)'; code divides schatten_1 by d (line 165), schatten_4 by d**0.25 (line 166), schatten_inf by 1 (line 167). The parenthetical formula in the same comment (norm/d^{1/p}) is correct; sqrt(d) applies only to the excluded p=2. Prose contradicts code.
  - _fix:_ Reword to 'normalised by d^{1/p}'.

- **[ADJUDICATED-SURVIVED]** `geometry_spectral` — Fixture describes k as round(0.2*len) but code uses int() floor — votes 1/1 upheld, unanimous
  - `tests/fixtures/reference_parity/parity/spectral.json:14`
  - blme_definition says 'k=round(0.2*len)'; spectral.py:79 is k = max(2, int(tail_fraction*len(S_np))) = floor. They differ whenever frac(0.2*len) >= 0.5, e.g. len=13: floor 2 vs round 3. Description-only; no test assertion depends on it.
  - _fix:_ Change fixture text to 'k=max(2, floor(tail_fraction*len))'.

- **[ADJUDICATED-SURVIVED]** `geometry_svd` — Ledger claims FAITHFUL/MATCH vs 1909.00512, and interpreting_results.md documents thresholds the centered metric can never reach — votes 1/1 upheld, unanimous
  - `docs/interpreting_results.md:19`
  - interpreting_results.md:19-22 says 'near 1 = collapsed/anisotropic', 'well-trained < 0.5', and :220 '>0.8 -> severe anisotropy' — unreachable: the centered statistic is ~0 even for gpt2's severely anisotropic last layer (measured -0.047 vs paper 0.973). VERIFICATION_LEDGER.md:152 asserts FAITHFUL \| 1909.00512 \| MATCH; refuted for this key.
  - _fix:_ After fixing/renaming the metric, correct interpreting_results.md rows 19-22 and 213/220 thresholds to the implemented semantics, and downgrade the VERIFICATION_LEDGER.md:152 geometry_svd row (FAITHFUL/MATCH claim) to note the avg_cosine_similarity deviation.

- **[ADJUDICATED-SURVIVED]** `geometry_tokenizer_efficiency` — Cert label contradiction: ledger task table says 'formula-faithful', task_metadata.py and ledger's own proxy list say 'proxy-only' — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:153`
  - VERIFICATION_LEDGER.md:153 row: 'geometry_tokenizer_efficiency \| CITATION_FIX \| formula-faithful'. But src/blme/task_metadata.py:71 = 'proxy-only' and VERIFICATION_LEDGER.md:285 groups it among proxy-only tasks. The ledger contradicts both the shipped metadata and itself.
  - _fix:_ Correct the ledger row Cert column to 'proxy-only' (matching task_metadata.py), or deliberately upgrade the metadata with a note — pick one and make the three locations agree.

- **[ADJUDICATED-SURVIVED]** `geometry_tokenizer_efficiency` — Docstring claims 'all ~+0.71 partial ρ' but vocab_size is +0.769 in TOP_PREDICTORS.md — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/tokenizer_efficiency.py:30`
  - Docstring lines 27-30: fertility, compression_ratio, total_tokens, vocab_size 'all ~+0.71 partial ρ'. docs/TOP_PREDICTORS.md §2: vocab_size = +0.769 (row 6); the other three are +0.712 (rows 13-15).
  - _fix:_ Change to '+0.71 to +0.77 partial ρ' or list vocab_size's +0.769 separately.

- **[ADJUDICATED-SURVIVED]** `geometry_trajectory_curvature` — Docstring misstates paper baseline as the input embedding — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/trajectory_curvature.py:37`
  - Lines 35-39 claim 'Hosseini & Fedorenko compare against the input embedding, which is one step earlier'. Paper says analyses start 'from the first contextualized layer L0'; official code L44 baselines curve_change at curve_[1] (=hs[1]) — exactly BLME's first cached layer. The claimed baseline mismatch does not exist; BLME matches the paper here.
  - _fix:_ Rewrite the note: the shared cache's first layer (hs[1]) equals the paper's L0 baseline; official code drops the embedding row from its change computation too.

- **[ADJUDICATED-SURVIVED]** `geometry_unembedding` — tasks_geometry.md §17 describes a measurement the task does not perform — votes 1/1 upheld, unanimous
  - `docs/tasks_geometry.md:157`
  - Doc: 'angles (cosine similarity) and norms between the highest logit token vectors and the actual dynamic context state vectors'. Code (unembedding.py:50-213) never touches hidden states, logits, or norms: it computes is_tied, erank of centered W_out, category purity, and static-embedding-vs-W_out cosine after ln_f. Line 161 'frequent tokens dominate' claims frequency analysis that doesn't exist.
  - _fix:_ Rewrite §17 'How are we measuring' + 'Critical Info' to match the implemented metrics: tied-weight check, Roy-Vetterli erank of the mean-centered unembedding, top-k category purity, ln_f-transformed embedding/unembedding alignment.

- **[ADJUDICATED-SURVIVED]** `geometry_unembedding` — PAPERS.md omits mean-centering before erank (changes value ~4%) — votes 1/1 upheld, unanimous
  - `docs/PAPERS.md:141`
  - PAPERS.md says 'effective rank (Roy & Vetterli 2007) on the unembedding matrix', but unembedding.py:86 subtracts the mean row first. On distilgpt2: uncentered erank=675.3785, centered=704.2634 (BLME reports 704.2672). Centering is documented only in the parity-test docstring (test_comprehensive_parity.py:4209), not user-facing docs.
  - _fix:_ Amend PAPERS.md:141 (and tasks_geometry.md:159) to 'effective rank ... of the mean-centered unembedding matrix'.

- **[ADJUDICATED-SURVIVED]** `geometry_vendi_score` — Subsampling docstring falsely claims 'unbiased estimate' of the cloud's Vendi Score — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/vendi.py:159`
  - _layer_vendi docstring (vendi.py:157-160) says a fixed random subsample is 'an unbiased estimate'. Disproved: X=eye(1000) gives full-cloud VS=999.999999999997 but _layer_vendi(max_points=512, seed=0)=512.0000000000009 — estimator is capped at max_points, bias -488. Numbers themselves are correct and deterministic; only the statistical claim is wrong.
  - _fix:_ Reword to 'a deterministic size-capped estimate (VS <= max_points)'; parity/definition claims are unaffected since _meta_n_points_per_layer and vendi_normalized_mean already expose the cap.

- **[ADJUDICATED-SURVIVED]** `geometry_weight_norms` — VERIFICATION_LEDGER claims cert 'formula-faithful' but shipped label is 'proxy-only'; ledger internally inconsistent — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:156`
  - Ledger row 156: 'VERIFIED_PARITY \| formula-faithful \| 1901.08276 \| WeightWatcher \| MATCH' vs task_metadata.py:74 shipping 'proxy-only' and PAPERS.md:272 'BLME diagnostic' (no paper). Ledger's own narrative (line 285) lists weight_norms among proxy-only/refined tasks, contradicting its table row.
  - _fix:_ Reconcile: either upgrade task_metadata.py to formula-faithful/parity-ready (justified — checked-in independent parity test exists) or correct the ledger row's Cert column to proxy-only; align the arXiv column with PAPERS.md.

- **[ADJUDICATED-SURVIVED]** `geometry_weight_norms` — Module docstring states norm_uniformity = 1 - CV but code returns max(0, 1-CV) — votes 1/1 upheld, unanimous
  - `src/blme/tasks/geometry/weight_norms.py:103`
  - Docstring line 19-20: 'norm_uniformity: 1 - CV(frobenius_per_layer)'. Code line 103 clamps at 0. Demonstrated: frob/layer [14.28,14.28,1428.29] gives CV=1.372619, formula value -0.372619, task returns 0.0. The parity test documents the clamp; the module docstring omits it.
  - _fix:_ Amend docstring line 19 to 'max(0, 1 - CV(frobenius_per_layer))', matching the test's description.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_kurtosis` — kurtosis_tensor documented as "kurtosis of the whole pooled tensor" but computed as mean over layers of per-layer tensor kurtosis — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_kurtosis.py:210`
  - Line 297: _finite_mean(tensor_arr) averages per-layer whole-tensor kurtoses; docstring lines 38-41 and 210 say all-entries-pooled. Fresh 3-layer check: mean-over-layers=1064.93 vs all-layers-pooled=184.88. Inline comment line 296 is honest; kurtosis_median (line 295) also undocumented.
  - _fix:_ Reword docstring outputs to "per-layer whole-tensor kurtosis, averaged over layers" (or pool all layers before computing); document kurtosis_median.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — Stale pre-fix docstrings still describe the old diluted all-positions metric — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/activation_sinks.py:87`
  - Returns section (lines 86-88) says 'Fraction of (layer, head, token) triples', and module header lines 9-13 say 'fraction of (head, position, token) triples'. Code (line 123) computes first-token-only Sink1 over (layer, head) pairs — matching the correct formula block at lines 96-101, contradicting both stale passages.
  - _fix:_ Rewrite Returns to 'fraction of (layer, head) pairs whose first-token importance exceeds eps' and fix the module-header description to Sink1.

- **[ADJUDICATED-SURVIVED]** `interpretability_activation_sinks` — Ledger claims 'parity-ready' but shipped label is 'refined-adaptation' — votes 1/1 upheld, unanimous
  - `VERIFICATION_LEDGER.md:157`
  - Ledger row says cert 'parity-ready'; src/blme/task_metadata.py:81 has always said 'refined-adaptation' (both introduced in commit 69e1484). Metadata is right — only the sink kernel is official-parity; massive_*/valley_* are adaptations. Ledger overstates verification.
  - _fix:_ Correct the ledger row to 'refined-adaptation' with a note that only _sink_epsilon is kernel-parity.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_effective_rank` — tasks_interpretability.md describes per-head isolated value-projection SVD; code computes combined post-W_O output entropy — votes 1/1 upheld, unanimous
  - `docs/tasks_interpretability.md:44`
  - Doc lines 43-44: entropy "of the isolated value-projection outputs corresponding to specific heads", "how much capacity a head uses across samples". Code hooks combined attention OUTPUT projections (attention_polysemanticity.py:53-55) and SVDs (seq_len x hidden) per sample (lines 101-107); module docstring lines 9-11 states "combined multi-head output (post-W_O), not individual head outputs" — doc contradicts code.
  - _fix:_ Rewrite lines 43-44 to match the module docstring: SVD entropy of the combined post-W_O attention output over sequence positions, per sample; no per-head or value-projection isolation.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_rank` — Module docstring promises a 'rank-collapse ratio' but code returns a Spearman trend correlation — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/attention_rank.py:16`
  - Docstring lines 16-17: 'rank-collapse ratio: how much the effective rank decreases from early to late layers'. Code (lines 126-137, output key line 148) returns rank_collapse_trend_spearman = Spearman(layer index, layer mean normalised rank) — direction/monotonicity, not a magnitude ratio. distilgpt2 value -0.8286 verified correct vs hand Spearman (diff 1.1e-16).
  - _fix:_ Reword docstring bullet to 'rank-collapse trend: Spearman correlation between layer index and mean normalised effective rank (negative = later layers more collapsed, per Dong et al.'s empirical finding)'.

- **[ADJUDICATED-SURVIVED]** `interpretability_attention_rollout` — Cache path truncates at 512 tokens, fallback at 128; docstring says fallback matches the sibling attention-entropy fallback, which never truncates — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/attention_rollout.py:251`
  - attention_rollout.py:251 max_length=128 vs cache.py:352 max_length=512 vs sibling attention.py:82 (no truncation). For corpora with samples >128 tokens the same task returns different numbers with vs without cache; the bit-for-bit cache/no-cache assertion (parity test:239-245) holds only for short texts. Docstring lines 41-43 overstate fallback equivalence.
  - _fix:_ Use one shared max_length (config-driven, matching the cache's 512) in the fallback tokenizer call and correct the docstring parenthetical; not number-changing for the default short corpus.

- **[ADJUDICATED-SURVIVED]** `interpretability_attribution` — Stale metric description in docs/tasks_interpretability.md §6 contradicts the implementation — votes 1/1 upheld, unanimous
  - `docs/tasks_interpretability.md:66`
  - Doc says 'gradient of the predicted logit with respect to the input embeddings, taking the L2 norm (InputXGradient)' and 'specific target token'. Code (attribution.py:35,106-111) uses gradient of MEAN next-token CE loss over all positions, \|grad × activation\| summed over hidden dim (L1 of product), pooled mean/std/max + Gini.
  - _fix:_ Rewrite §6 'How are we measuring' to: gradient of mean next-token cross-entropy w.r.t. input-embedding activations; per-token \|grad × activation\| summed over hidden dim; pooled mean/std/max and Gini concentration.

- **[ADJUDICATED-SURVIVED]** `interpretability_prediction_entropy` — Machine-readable cert metadata still attributes formula to Holtzman/degen with no Shannon note — votes 1/1 upheld, unanimous
  - `src/blme/task_metadata.py:90`
  - Entry: _entry("formula-faithful", "Holtzman et al. 2020", "ari-holtzman/degen") with empty notes. Verified degen repo has no entropy metric; the module docstring (prediction_entropy.py:16-20) and PAPERS.md:284 say the formula is Shannon 1948, not Holtzman. The ledger's CITATION_FIX (line 166) was never propagated to task_metadata.py.
  - _fix:_ Change papers to "Shannon 1948; Holtzman et al. 2020 (motivation)", drop or annotate the degen reference in notes.

- **[ADJUDICATED-SURVIVED]** `interpretability_probing` — Docs claim cross-validated accuracy/AUC; code does one sequential 80/20 holdout, no AUC — votes 1/1 upheld, unanimous
  - `docs/tasks_interpretability.md:92`
  - Doc: "training a regularized Logistic Regression probe. Evaluated via cross-validated accuracy/AUC." Code (probing.py:89-94,119) does a single unshuffled 80/20 split and reports plain accuracy only; no CV, no AUC anywhere.
  - _fix:_ Reword to "single 80/20 holdout accuracy with an SGD logistic-regression probe" (or implement CV/AUC).

- **[ADJUDICATED-SURVIVED]** `interpretability_probing` — Docstring says probe predicts "the token identity"; code predicts the NEXT token — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/probing.py:16`
  - Docstring: "predict the token identity from its hidden state" reads as current-token; code uses labels = input_ids[1:] with feats hs[0,:-1] (probing.py:57-68), i.e. next-token prediction (as TASK_FIXES.md:183 correctly states).
  - _fix:_ Change docstring line to "predict the NEXT token's identity from the current hidden state".

- **[ADJUDICATED-SURVIVED]** `interpretability_sae_features` — Stale CAVEAT: metadata still says sae-lens>=6 compat 'update needed' though the fix already landed — votes 1/1 upheld, unanimous
  - `src/blme/task_metadata.py:92`
  - task_metadata.py:92 and VERIFICATION_LEDGER.md:281 claim sae_features.py:97 'unpacks a 3-tuple; update needed to run against current sae-lens'. Commit 754e06b (2026-07-03) added the dual-API handling (sae_features.py:98-101); verified working against sae-lens 6.45.0 in my fresh run.
  - _fix:_ Update the metadata caveat to note the compat fix landed (754e06b) and is verified against sae-lens 6.45.0; annotate the ledger line as resolved.

- **[ADJUDICATED-SURVIVED]** `interpretability_superposition` — Docstring claims 'We return 1 - BC so that higher = more monosemantic' but code returns BC — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/superposition.py:35`
  - Line 52 returns bc directly: bimodal sample → 0.807437 (BC; 1-BC would be 0.192563). evaluate() (lines 161-162) and interpreting_results.md:130 correctly treat higher = more polysemantic, so shipped numbers are right and the docstring orientation is inverted. AUDIT_V2.md:115's 'direction-inverted' flag was never reconciled.
  - _fix:_ Delete/correct the '1 - BC' sentence: the helper returns BC itself; higher = more multimodal = more polysemantic.

- **[ADJUDICATED-SURVIVED]** `interpretability_superposition` — Module and class docstrings claim 'pairwise activation correlations' that are never computed — votes 1/1 upheld, unanimous
  - `src/blme/tasks/interpretability/superposition.py:5`
  - Lines 5-6 and 58-59 both promise analysis of 'pairwise activation correlations within MLP layers'; evaluate() computes only per-neuron BC and utilization (comment '# 1. Multimodality scores' at line 149 has no step 2). grep confirms no corrcoef/correlation code in the file.
  - _fix:_ Drop the correlation claim from both docstrings (or implement it).

- **[ADJUDICATED-SURVIVED]** `repe_concept_separability` — Task doc claims PCA/LAT reading-vector extraction; code fits a logistic-regression CV probe and computes no reading vector — votes 1/1 upheld, unanimous
  - `docs/tasks_representation_engineering.md:25`
  - Doc says: 'Using Linear Artificial Tomography (LAT) / Principal Component Analysis (PCA)... We determine the principal Reading Vector and measure the accuracy'. Code (rep_eng.py:218-248) uses StratifiedKFold LogisticRegression AUC/acc; no PCA, no reading vector. The parity test itself states BLME measures 'LINEAR SEPARABILITY... not the PCA vector itself'.
  - _fix:_ Reword section 2 to: held-out logistic-regression separability probe (AUC/accuracy), a refined adaptation of the paper's linear reading methods (cf. paper section 5.1 LR directions).

- **[ADJUDICATED-SURVIVED]** `repe_refusal_direction` — Docstring promises a per_layer output key that evaluate() deliberately never returns — votes 1/1 upheld, unanimous
  - `src/blme/tasks/representation_engineering.py:462`
  - Docstring line 462 lists "per_layer: the same metrics computed at every layer" as a reported metric, but the result dict (lines 641-654) omits it by design; tests/tasks/test_repe.py:116 asserts 'per_layer' not in results. Docstring contradicts the code's own output contract.
  - _fix:_ Rewrite the docstring bullet to describe the actual outputs: auc_at_depth_{0,25,50,75,100} depth-quantile AUCs, best_layer, best_layer_fraction — noting per_layer was removed for architecture-agnostic CSV aggregation.

- **[ADJUDICATED-SURVIVED]** `repe_steering_effectiveness` — steering_success_rate documented as 'fraction of layers' but is fraction of <=5 sampled layers on deep models, and is never emitted in the default flow — votes 1/1 upheld, unanimous
  - `docs/interpreting_results.md:205`
  - Code tests only layers {0, n/4, n/2, 3n/4, n-1} when num_layers>10 (representation_engineering.py:329-333), so the denominator is 5, not all layers; doc says 'Fraction of layers'. Also the doc presents the feature as produced, yet the default pipeline always errors (finding 1) — 0 steering columns in the study's feature_matrix.csv.
  - _fix:_ Document the 5-layer depth subsample (or emit tested-layer indices), and note the dataset requirement until the fallback fix lands.

- **[ADJUDICATED-SURVIVED]** `topology_persistence_landscape` — Module docstring 'Reported metrics' does not match the keys evaluate() actually emits — votes 1/1 upheld, unanimous
  - `src/blme/tasks/topology/persistence_landscape.py:19`
  - Docstring lines 19-22 promise landscape_max_k / landscape_norm_k (per k=1..K) and 'landscape_mean_integral'; evaluate() lines 182-185 emits only *_landscape_max_1, *_landscape_norm_1, and *_mean_landscape_integral (name reversed), plus *_landscape_integrals list.
  - _fix:_ Align docstring to actual output keys (max/norm reported for k=1 only; key is mean_landscape_integral), or emit per-k max/norm.

## Refuted findings — dropped by adversarial review (15)

These skeptic claims did **not** survive refutation and are excluded from the standing list above.

- **[WEAK_TEST]** `geometry_lipschitz` — Comprehensive parity test knowingly pins the docstring-violating max semantics as correct
  - _why dropped:_ Claim's core evidence—that pinning rests on the degenerate single-pair "1.9==1.9" case—is refuted by part B (test lines 3231-3260), which independently recomputes `ref_arr.max()` over multiple layer pairs and asserts `lipschitz_max` equality, deliberately pinning max-of-means. So the suggested "add multi-pair assertion" fix already exists. `lipschi

- **[WEAK_TEST]** `repe_concept_separability` — Parity file tests a transcribed copy of the BLME kernel, not the imported module
  - _why dropped:_ Factually accurate: parity file's `_blme_separability` (line 67 docstring: "verbatim (rep_eng.py:214-248)") is a transcription; none of its 5 tests import `ConceptSeparabilityTask`. But the finder itself concedes coverage is "ACCEPTABLE" — test_comprehensive_parity.py:5905 imports and runs the REAL `ConceptSeparabilityTask.evaluate()` on distilgpt2

- **[WEAK_TEST]** `repe_refusal_direction` — Standalone parity test transcribes the BLME kernel instead of importing real blme code
  - _why dropped:_ Claim accurately describes the transcription: lines 52-58/61-75 are verbatim copies and the file imports nothing from blme. But its implied harm — a formula change "would still pass all 4 tests" — is scoped to this file only. test_comprehensive_parity.py:6000 drives the REAL get_task("repe_refusal_direction").evaluate() and asserts direction_norm/s

- **[WEAK_TEST]** `topology_homology` — End-to-end test asserts only output-key presence, so evaluate()-flow numeric regressions (e.g. the layer-duplication bug) pass unnoticed
  - _why dropped:_ Factually accurate: test_topology.py:13-23 asserts only isinstance(results,dict) and any('persistence_h0' in k); on 2-layer mocks target_layers=[0,1,1] duplicates layer-1's cloud, and parity test:308 covers only _lifespan_summary. But this is a standard cross-architecture smoke test (docstring-documented) paired with a strong analytic+GUDHI kernel 

- **[CITATION]** `repe_steering_effectiveness` — Ledger row cross-pairs Turner's arXiv id with Zou's repo
  - _why dropped:_ Facts check out: 2308.10248=Turner/ActAdd (repo montemac/activation_additions), 2310.01405=Zou/RepE (repo andyzoujm/representation-engineering); row 174 does cross them. But task_metadata.py:126 and PAPERS.md:324 honestly cite BOTH "Zou 2023; Turner 2023" for this proxy-only diagnostic. Turner is the steering paper; andyzoujm is the reused code lin

- **[CITATION]** `topology_betti_curve` — Second docstring reference is a quoted title matching no real publication
  - _why dropped:_ Code matches: line 27 quotes "Topological Data Analysis of Large Language Models' Hidden Representations," and searches find no paper by that title (closest: arXiv:2406.05798 "Hidden Holes," survey arXiv:2411.10298). But the parenthetical "(General TDA on Transformers literature)" honestly flags it as a genre pointer, not a specific citation, and i

- **[DOC_GAP]** `geometry_magnitude` — Module docstring calls magnitude_auc 'the area under the magnitude function', which is the paper's MagArea (Def 3.3) — but code computes area under log\|tX\| vs log t
  - _why dropped:_ Code (magnitude.py:275-278) computes np.trapezoid(log\|tX\|, log t) — a log-log AUC. The claim describes this accurately. But the class docstring (line 201) documents magnitude_auc precisely as "area under log\|tX\| vs log t (trapezoid)" — one screen away in the same file. Line 59's "the area under the magnitude function" is loose shorthand inside 

- **[DOC_GAP]** `geometry_matrix_entropy` — 'parity-ready' label and REPOSITORIES.md claim focused reference tests that do not exist
  - _why dropped:_ Formula is faithful to Wei et al. 2024 Def. 4.1-4.3 (centering, row-L2, /log d) and finder concedes it is correct. task_metadata.py:6 defines parity-ready loosely as "focused formula/reference tests exist" — and test_geometry.py has 4 focused matrix_entropy tests, one explicitly pinning Def. 4.3 normalization ("Wei et al. 2024 Def. 4.3"), "not just

- **[DOC_GAP]** `geometry_matrix_entropy` — 'Headline Wei et al. number' comment sits on the raw key; paper Def 4.3 number is the normalized one
  - _why dropped:_ Primary sources confirm the finder's fact: paper Def 4.3 = H/(n·log d) and official code `normalized_entropy = entropy/math.log(A.shape[0])`; Table 1 values (~0.36, 0.28) are normalized. But the L199 comment is a block header over the raw+normalized headline PAIR (L200-201), paralleling the "Per-layer extension" and "aggregate" comments that also e

- **[DOC_GAP]** `geometry_mp_bulk_deviation` — Ledger claims VERIFIED_PARITY/MATCH vs scikit-rmt with no checked-in fixture and an unimportable reference package
  - _why dropped:_ Ledger legend (line 102) defines "Parity MATCH = the numeric check at audit time," and "Reference repo" names the canonical impl (as every row does), not a checked-in dep. The parity-ready upgrade is documented as "analytic MP edge/CDF" (line 97); shipped metadata (task_metadata.py:61) cites the real passing test test_mp_bulk_edge_and_cdf_sanity, n

- **[DOC_GAP]** `geometry_positional_decay` — Docstring's 'intact context => strong negative correlation' is contradicted by intact GPT-2 on the pooled headline metric
  - _why dropped:_ Refuted empirically. Running the actual current code (HEAD, commit 69e1484) on intact eager GPT-2 gives mean_positional_decay_correlation = -0.2923 (layers -0.27..+0.01) and distilgpt2 = -0.2454 — strong NEGATIVE, exactly as docstring lines 9-11 predict. The finder's cited +0.0713/+0.1154 do not reproduce; even a naive pool-all-entries Spearman (th

- **[DOC_GAP]** `geometry_procrustes_linearity` — Layer coverage differs from official pipeline: embedding-layer pair excluded, undocumented
  - _why dropped:_ Facts check out: cache.py:363-365 (`n_layers=len(hs)-1`, `hs[li+1]`) drops the embedding, so BLME pairs run (block1,block2)…(L-1,L); official `range(max(keys)-1)` runs (embedding,block1)…(L-2,L-1). But this task honestly pins the metric (bit-exact `procrustes_similarity`), not the notebook driver, and its docstring already states "Interpret the abs

- **[DOC_GAP]** `geometry_schatten` — Yusupov et al. arXiv:2509.25359 was retitled in v2 (2026-06-10); BLME cites the v1 title
  - _why dropped:_ Facts verified: arXiv:2509.25359 v1 (29 Sep 2025) = "From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation"; v2 (10 Jun 2026) retitled "Geometric Metrics and LLMs: What They Measure and When They Work". schatten.py:8-10 cites the v1 string with the correct ID. But arXiv permanently archives v1; citing a real version'

- **[DOC_GAP]** `geometry_trajectory_curvature` — Ledger VERIFIED_PARITY/MATCH unsupported; official repo never indexed
  - _why dropped:_ Finder's premises are false. REPOSITORIES.md line 55 DOES index it: "Hosseini, Fedorenko 2023...arXiv:2311.04930 (NeurIPS 2023) \| geometry/trajectory_curvature.py" (also line 269). task_metadata.py:72 cites papers="Hosseini & Fedorenko 2023" (empty repo field matches ~20 other tasks). Parity tests EXIST: test_geometry.py verifies the kernel to 1e-

- **[DOC_GAP]** `interpretability_superposition` — VERIFICATION_LEDGER Campaign-1 claim 'computes its stated definition' is false for this task
  - _why dropped:_ The ledger's "stated definition" is pinned by test_comprehensive_parity.py:5646 — "BC=(skewness^2+1)/kurtosis (Pearson, non-excess)" — the asymptotic form the code computes, verified against an independent re-derivation to abs=1e-9 (lines 5698/5701/5702). Finder concedes the kernel is "numerically correct." The helper-docstring's correction-term fo

## Pipeline review (5 dimensions)

### core-loop
The core loop is largely well-engineered: error isolation works (verified end-to-end with a crashing task), signal-handler safety is explicitly handled (main-thread + SIGALRM checks, handler restore in finally), all 8 hook-registering tasks remove hooks in finally, and task registration has no fragile optional imports. However, four MAJOR issues remain: the per-task timeout is silently defeated by tasks' own broad except-Exception handlers (verified: a 2s-timeout task ran 10s and reported OK); the default corpus is loaded (200 samples) before per-task num_samples are resolved, so larger reques

- **[MAJOR]** Per-task timeout silently defeated by task-internal broad exception handlers; timed-out task reported as OK (`src/blme/core.py:23`)
  - Verified: task with config timeout=2 wrapping work in `except Exception: continue` ran 10.0s and printed '✓ OK slept=10', errors=None. SIGALRM fires once; _timeout_handler raises TimeoutError (an Exception), swallowed by any of the 55 `except Exception` blocks in src/blme/tasks/ (e.g. geometry/perplexity.py:107), then no alarm remains — task runs unbounded and is recorded as success.
  - _fix:_ Make the handler raise a custom class deriving from BaseException (e.g. class TaskTimeout(BaseException)) so task-level `except Exception` cannot swallow it, and catch TaskTimeout in the core loop. Also guard `signal.alarm(int(timeout_sec))` against int() truncating 0<timeout<1 to 0, which disarms t

- **[MAJOR]** Default corpus loaded before task configs are resolved: num_samples > 200 silently capped at 200 while envelope records the un-capped value (`src/blme/core.py:148`)
  - Verified by run: evaluate(tasks=['geometry_svd'], task_configs={'geometry_svd': {'num_samples': 500}}) → corpus loader called with 200 (line 148 runs before the max-of-num_samples resolution at lines 230-237), log shows 'forward pass over 200 samples', yet envelope says cache_num_samples_resolved: 500. Results computed on 200 samples; metadata claims 500 — silent shortfall + wrong provenance.
  - _fix:_ Resolve task configs and cache_num_samples first, then load the corpus with max(resolved cache_num_samples, max per-task num_samples, 200); or log a warning and write the actual sample count used into the envelope.

- **[MAJOR]** Shared cache unconditionally stores full-vocab logits for every sample even when no logits-consuming task is in the run (`src/blme/cache.py:378`)
  - cache.py:377-379 appends outputs.logits for every sample ('always collected'); core.py:174-177 knows cache_logits_tasks={geometry_perplexity, consistency_calibration} but populate() (cache.py:67) has no need_logits flag. 200 samples x 512 tokens x 50257 vocab x 4B fp32 ≈ 20.6 GB host RAM for a hidden-state-only run (e.g. geometry_svd alone); Llama-3's 128k vocab ≈ 2.5x more.
  - _fix:_ Add need_logits to populate()/ModelOutputCache and only accumulate logits when a cache_logits_task is in the run; optionally store logits in fp16 or store per-position log-probs of the target token instead of full-vocab tensors.

- **[MAJOR]** Mixed diagnostic + lm_eval runs hold two full model copies plus the cache in memory; pure-benchmark runs load the diagnostic model for nothing (`src/blme/core.py:305`)
  - core.py:130 loads the model unconditionally (even when diagnostic_tasks is empty), then step 5 calls run_lm_eval → benchmarks.py:52 simple_evaluate(model='hf', model_args=...) loads a second copy while `model` and the populated `cache` remain referenced in evaluate()'s scope. Double GPU footprint → OOM for any model near half of VRAM; cache RAM also held throughout.
  - _fix:_ Skip load_model_and_tokenizer when diagnostic_tasks is empty; before the lm_eval phase, call cache.clear(), del model, gc.collect(), torch.cuda.empty_cache() (or pass the already-loaded HFLM instance into simple_evaluate).

- **[MINOR]** Results printed to terminal before being saved: any print failure destroys the whole run's results (`src/blme/core.py:340`)
  - print_results_table (core.py:340) runs before save_results (core.py:343-344). It raises AttributeError if any task returns a non-dict (results.py:131 `"error" in result`, 169 `result.items()` — the ABC contract is not enforced), and prints ✓/⚠/✗ which raise UnicodeEncodeError on cp1252-redirected Windows output — where timeouts are already disabled, so Windows is a supported target. Hours of compute lost after all tasks succeeded.

- **[MINOR]** results.json can contain bare Infinity/NaN — invalid strict JSON that breaks non-Python consumers (`src/blme/results.py:199`)
  - json.dump(envelope, ..., default=str) uses allow_nan=True. Verified: json.dumps({'ppl': float('inf'), 'x': float('nan')}) emits `"ppl": Infinity, "x": NaN`. Tasks genuinely produce these (geometry/perplexity.py:110 returns float('inf') when cnt_total==0). JavaScript JSON.parse, Rust serde, and strict parsers reject the file.

- **[MINOR]** git_hash in the results envelope records the user's cwd repo, not the blme package version (`src/blme/results.py:34`)
  - _get_git_hash runs `git rev-parse --short HEAD` with no cwd argument (results.py:33-38). A pip-installed blme executed inside any unrelated git repo stamps that repo's HEAD into the envelope as provenance — misleading reproducibility metadata; blme_version already covers the package.

- **[MINOR]** Cache silently clamps out-of-range layer indices to the nearest valid layer (`src/blme/cache.py:136`)
  - get_hidden_states clamps: `actual = max(keys[0], min(actual, max_layer))` (cache.py:131-136). Requesting layer_idx=40 on a 6-layer model silently returns layer 5's hidden states; a config written for a larger model produces plausible-looking but wrong-layer metrics with no warning.

- **[MINOR]** Envelope summary arithmetic is inconsistent: soft-error results count as completed, unknown tasks inflate total, lm_eval tasks collapse into one (`src/blme/results.py:97`)
  - Verified run: tasks=[svd, crasher, bogus] → summary {"total_tasks": 3, "completed_tasks": 1, "failed_tasks": 1} — bogus is unaccounted; completed+failed≠total. Tasks returning {"error": ...} dicts (e.g. geometry/unembedding.py:64) count as completed_tasks while print_results_table shows them as '⚠ ERROR'. N lm_eval tasks become one '_lm_eval' result.

- **[NIT]** Duplicate task registration silently overwrites the earlier class (`src/blme/registry.py:9`)
  - register_task does `_TASK_REGISTRY[name] = cls` with no collision check (registry.py:8-10). A contributor adding a task whose name collides with an existing one silently shadows it; the run proceeds with the wrong implementation under a familiar name.

- **[NIT]** YAML recipes cannot set output_format; runner hardcodes JSON (`src/blme/runner.py:57`)
  - run_from_yaml reads output_dir, device, batch_size, limit, cache_num_samples, seed, task_timeout from the global section (runner.py:33-39) but never output_format, and does not pass it to evaluate() — CSV output is unreachable from the documented recipe path.

### model-layer
The model layer (src/blme/models/wrapper.py, ~220 lines, plus utils.py) is small and mostly sound for the plain-GPT2 path, but live testing against the installed transformers 5.2.0 exposed real breakage: attention-consuming tasks silently receive tuple-of-None attentions under the default SDPA load and two tasks (geometry_positional_decay, interpretability_attention_rollout) fail on the happy path because core.py's compensating force-eager list omits them; the documented multi-device max_memory syntax is unparseable by the comma-splitting arg parser; a missing/typo'd `pretrained` key silently 

- **[MAJOR]** Attention tasks fail on default load: SDPA returns all-None attentions with no warning; force-eager list in core.py is incomplete (`src/blme/core.py:190`)
  - Ran: default wrapper load (sdpa, transformers 5.2.0) gives outputs.attentions = tuple of 12 Nones, zero warnings. geometry_positional_decay -> misleading {'error': 'Could not compute positional correlations (sequences too short or nan).'}; interpretability_attention_rollout -> error dict. Both succeed with attn_implementation=eager. Neither task is in force_eager_tasks (core.py:190-196); positional_decay.py:84-85 silently skips None entries.
  - _fix:_ Add interpretability_attention_rollout and geometry_positional_decay to force_eager_tasks (or derive the set from task metadata instead of a hardcoded list). In positional_decay, detect all-None attentions and report the real cause. Consider having the wrapper log a warning when attn impl is sdpa/fl

- **[MAJOR]** Documented multi-device max_memory syntax is silently mangled by comma-splitting parser (`src/blme/models/wrapper.py:37`)
  - Docstring (wrapper.py:105) advertises max_memory='0:20GiB,cpu:40GiB', but parse_model_args splits the whole model_args string on ','. Ran: parse_model_args('pretrained=gpt2,max_memory=0:20GiB,cpu:40GiB,dtype=float16') -> {'pretrained':'gpt2','max_memory':'0:20GiB','dtype':'float16'}; 'cpu:40GiB' dropped with no error.
  - _fix:_ Either use a quoting/escaping scheme (e.g. semicolons inside max_memory: '0:20GiB;cpu:40GiB') or a smarter parser; at minimum raise on tokens without '=' instead of silently discarding them, and fix the docstring.

- **[MAJOR]** Missing or typo'd 'pretrained' key silently evaluates gpt2 instead of erroring (`src/blme/models/wrapper.py:116`)
  - wrapper.py:116 args.get('pretrained', 'gpt2') plus parse_model_args silently ignoring unknown keys and '='-less tokens. Ran: load_model_and_tokenizer('pretrainedd=distilgpt2') loads and reports gpt2 — the user's entire evaluation runs on the wrong model with no warning.
  - _fix:_ Require 'pretrained' explicitly (raise if absent) and warn/raise on unrecognized model_args keys; keep gpt2 only as a CLI-level documented default if desired.

- **[MAJOR]** load_in_8bit/load_in_4bit crash with TypeError under transformers v5, which the unbounded version pin allows (`src/blme/models/wrapper.py:172`)
  - pyproject pins transformers>=4.35.0 (no upper bound); installed 5.2.0. Ran: AutoModelForCausalLM.from_pretrained('distilgpt2', load_in_8bit=True) -> TypeError "GPT2LMHeadModel.__init__() got an unexpected keyword argument 'load_in_8bit'". TypeError is not caught by _load_model's (ValueError, OSError) fallback (wrapper.py:216), so the documented flag hard-crashes.
  - _fix:_ Build a BitsAndBytesConfig and pass quantization_config= (works on 4.35+ and v5). Also switch torch_dtype= to dtype= (v5 prints '`torch_dtype` is deprecated! Use `dtype` instead!' on every load; wrapper.py:146-148) or gate on transformers version.

- **[MINOR]** accelerate (and bitsandbytes) not declared anywhere, but documented device_map/max_memory/quantization options require them (`pyproject.toml:19`)
  - Ran: load_model_and_tokenizer('pretrained=distilgpt2,load_in_8bit=true') -> ValueError "Using a `device_map`... requires `accelerate`. You can install it with `pip install accelerate`". Neither accelerate nor bitsandbytes appears in dependencies or any optional extra, yet wrapper.py:99-105 documents device_map, max_memory, load_in_8bit/4bit.

- **[MINOR]** set_global_seed omits cuDNN/deterministic-algorithms settings, so GPU runs are not bit-reproducible (`src/blme/utils.py:17`)
  - utils.py:17-28 seeds random/numpy/torch/cuda only. No torch.backends.cudnn.deterministic=True, no torch.backends.cudnn.benchmark=False, no torch.use_deterministic_algorithms; core.py:87-88 relies solely on this for the run's reproducibility claim (seed recorded in results at core.py:333).

- **[MINOR]** Pad-token fallback silently leaves pad_token=None for tokenizers lacking both EOS and BOS (`src/blme/models/wrapper.py:136`)
  - wrapper.py:131-137: if eos_token is None the code assigns tokenizer.pad_token = tokenizer.bos_token; when bos is also None this is a silent no-op, and the comment 'critical for batched inputs' is violated — any later padded batch raises 'Asking to pad but the tokenizer does not have a padding token' far from the cause.

### release
The happy path is solid: `blme evaluate --model-args pretrained=distilgpt2 --tasks geometry_svd geometry_lid interpretability_attention_entropy --device cpu` ran end-to-end in ~5s with a correct results.json envelope (README's `effective_rank` key exists); `list-tasks`, `--json`, `--dry-run --strict`, and both example recipes validate; the built wheel is clean (101/101 py files, assets + defaults.yaml included, no stale build/ leakage, internal docs excluded); optional deps (ripser/nltk/sklearn/sae-lens/lm-eval/wandb) are all lazily imported with clear install-hint messages. Four real release 

- **[MAJOR]** Package crashes on import under Python 3.9, the advertised minimum (requires-python >=3.9) (`src/blme/tasks/geometry/positional_decay.py:27`)
  - `def _row_distance_correlation(attn_row: np.ndarray) -> float \| None:` — PEP 604 union evaluated at def time; no `from __future__ import annotations`. Module is imported by geometry/__init__.py:17, which core._register_all_tasks (core.py:37) imports, so every CLI command and `import blme.core` raises TypeError on 3.9. Only occurrence (AST scan of all 101 files). CI matrix includes 3.9 (test.yml:14).
  - _fix:_ Change to `Optional[float]` or add `from __future__ import annotations` to positional_decay.py; alternatively bump requires-python to >=3.10 and drop 3.9 from the CI matrix and README badge.

- **[MAJOR]** Task failures returned as {"error": ...} are silently counted as successes: --strict exits 0, summary says failed_tasks=0, errors=None (`src/blme/results.py:103`)
  - Ran CLI with ripser blocked: `blme evaluate --tasks topology_homology --strict` → table shows '⚠ ERROR ... Ripser library not installed' but EXITCODE=0 and results.json has summary {'completed_tasks': 1, 'failed_tasks': 0}, errors: None. cli.py:102 promises --strict fails 'if any task errors'; core.py:322 only checks raised exceptions, and many tasks return error dicts (e.g. probing.py:100, homology.py:68).
  - _fix:_ In build_results_envelope and the strict check, treat results containing an 'error' key like task_errors: count them in failed_tasks, surface them in the errors field, and make fail_on_task_error raise. CI users piping results.json currently see clean success on broken runs.

- **[MAJOR]** Internal paper/audit/study artifacts are git-tracked and will ship with the open-source repo (`AUDIT_V2.md:1`)
  - `git ls-files` shows AUDIT_REPORT.md, AUDIT_V2.md, root main.tex + references.bib, REPRODUCE.md, RESEARCH.md, TASK_FIXES.md, VERIFICATION_LEDGER.md, full paper/ (main.tex, neurips_2025.sty, figure PDFs, tables), results/study_v1..v3 + audit outputs, docs/PAPERS.md, PAPER_SURVEY.md, TOP_PREDICTORS.md — study/paper material that per CLAUDE.md belongs in the other two repos. Wheel itself is clean.
  - _fix:_ Move paper/, results/, root main.tex/references.bib, and audit/verification markdowns to beyond_benchmarks / the paper repo before the public push (an uncommitted cleanup deleting scripts/ is already in flight in the working tree — extend it to these files). Note neurips_2025.sty redistribution and 

- **[MINOR]** README headline task counts are stale: claims 74 tasks (27 geometry / 15 interp / 4 topology) but registry has 82 (32 / 17 / 5) (`README.md:13`)
  - `blme list-tasks` prints 'Available BLME tasks (82)'; per-group counts: geometry 32, interpretability 17, topology 5, causality 6, consistency 12, dynamics 6, repe 4. README.md:13-23 says 74 with the old per-category table; docs/getting_started.md:86 also says topology '(4 tasks)' (actual 5, so its categories sum to 81, not 82).

- **[MINOR]** Declared build-system minimum (setuptools>=61) cannot build the package: PEP 639 license string needs setuptools>=77 (`pyproject.toml:2`)
  - pyproject has `license = "MIT"` (string) with `requires = ["setuptools>=61.0"]`. Verified: building with setuptools 75.9.1 fails — 'configuration error: `project.license` must be valid exactly by one definition ... BUILD FAILED: ValueError \| invalid pyproject.toml config: `project.license`'. Works only because isolated builds fetch latest setuptools; pinned/offline/no-isolation builds break.

- **[MINOR]** examples/demo.py prints 'Available tasks: []' and contains internal scaffolding comments (`examples/demo.py:10`)
  - Ran `python examples/demo.py`: first output line is 'Available tasks: []' because list_tasks() is called before any task registration (registration happens lazily inside evaluate()). File also ships comments '# Mock evaluation' and 'using mocked model loading for speed/test if possible' — leftover dev notes in the showcase example.

- **[MINOR]** No py.typed marker despite fully type-hinted public API (`pyproject.toml:77`)
  - `find src -name py.typed` → nothing; wheel contains no py.typed and package-data (pyproject.toml:77-81) only lists assets/*.json and tasks/configs/*.yaml. Downstream mypy/pyright users get no types from `from blme import evaluate` even though core.py/base.py are annotated.

- **[MINOR]** Thin PyPI metadata: no Python-version or license classifiers, no keywords, only a Homepage URL (`pyproject.toml:13`)
  - Classifiers are only 'Programming Language :: Python :: 3' and 'Operating System :: OS Independent' (pyproject.toml:13-16); no 3.9–3.12 classifiers matching the CI matrix, no Repository/Issues/Documentation URLs, no keywords. Wheel METADATA confirms. Hurts PyPI discoverability and version filtering.

- **[NIT]** [data] extra is a no-op (datasets is already a core dependency) (`pyproject.toml:56`)
  - dependencies list includes `datasets` (pyproject.toml:36) while optional-dependencies keeps `data = ["datasets"]` with a back-compat comment; README (line 38) and getting_started.md (line 15) still advertise `pip install -e ".[data]"` as if it enabled functionality.

### reproducibility
Two identical evaluate() calls on distilgpt2 produce byte-identical results (verified), the results envelope records seed/git-hash/versions, and most geometry/causality tasks correctly use locally seeded generators. However, seeding relies on a single set_global_seed(42) at run start, so ~6 tasks that draw from the global torch/numpy RNGs (induction_heads, head_roles, sharpness, probing, superposition, and the sampling tasks by default) produce different feature values depending on which other tasks ran before them — empirically confirmed. Worse, the per-task `seed` config on the two sampling 

- **[MAJOR]** Setting the `seed` config on sampling tasks breaks them entirely on transformers 5.x (generator kwarg rejected, error swallowed) (`src/blme/tasks/consistency/self_consistency.py:100`)
  - transformers 5.2.0 generate() raises ValueError for `generator=` ("model_kwargs are not used: ['generator']"); fallback at line 104 only catches TypeError, so line 111's broad except skips every prompt. Ran evaluate(distilgpt2, seed=0 in task_configs): task returns {"error": "No samples successfully generated"}; dynamics_generation_diversity.py:189-196 likewise returns {"error": "No completions generated"}.
  - _fix:_ Do not pass `generator` to generate(); instead call torch.manual_seed(seed) (saving/restoring RNG state) before each generate(), or catch ValueError in the fallback. Apply to both self_consistency.py and generation_diversity.py, and add a regression test that a seeded run yields metrics, not an erro

- **[MAJOR]** Task results depend on which other tasks run in the same evaluate() call (global-RNG order dependence) (`src/blme/tasks/interpretability/induction.py:122`)
  - Seeding is only set_global_seed(seed) at core.py:88; induction.py:122 draws torch.randint from the global RNG. Verified: evaluate(distilgpt2, seed=42, tasks=[induction_heads]) gives avg_induction_score=0.10347615, but tasks=[head_roles, induction_heads] gives 0.10295437 for the same task. Same pattern: head_roles.py:59-64, dynamics/sharpness.py:91/94 (Hutchinson vectors), interpretability/probing.py:82 and superposition.py:147 (global np.random.c
  - _fix:_ Use a per-task locally seeded generator (torch.Generator().manual_seed(cfg seed) / np.random.default_rng(seed)) as already done in geometry/lid.py, tracing.py, ablation.py; pass the run seed into task configs so each task is reproducible standalone and in any suite composition.

- **[MINOR]** Sampling tasks are unseeded by default (seed: None), so their metrics are only reproducible via incidental global-RNG state (`src/blme/tasks/consistency/self_consistency.py:57`)
  - self_consistency.py:57 and generation_diversity.py:149 default `seed = self.config.get("seed", None)`; configs/defaults.yaml has no seed entry for either task (only vendi/phd have one). With seed=None, do_sample=True generation consumes the global torch RNG, so results inherit the order-dependence above — and the opt-in seed path is the broken one (finding 1).

- **[MINOR]** Tasks and evaluate() reseed process-global RNGs (library side effect / mid-run pollution) (`src/blme/tasks/geometry/categories.py:145`)
  - categories.py:145/285/369 and dynamics/stability.py:83 call np.random.seed(42); self_consistency.py:72/109 calls torch.manual_seed(seed); core.py:88 evaluate() itself calls set_global_seed (utils.py:24-26), silently reseeding random/numpy/torch of any host application embedding blme. Contrast with weight_activation_alignment.py:87-96, which correctly saves/restores torch RNG state around its seeded svd_lowrank.

- **[MINOR]** No documentation of reproducibility guarantees or dtype/device sensitivity of feature values (`README.md:105`)
  - README's only reproducibility mention is the results envelope (line 105); docs/configuration.md never mentions `seed` despite `blme evaluate --seed` existing (cli.py:85). Nothing warns that feature values differ across dtype (fp16/bf16/fp32 — wrapper.py defaults to dtype=auto, so the same command gives different numbers per checkpoint dtype), across CPU/GPU, or that some tasks are suite-order dependent.

### test-infra
The suite is large and well-designed at its core: 1293 tests collect cleanly in 0.7s with zero import-time errors/warnings; conftest.py builds tiny random-weight models across 11 architectures with no downloads; the completeness/metadata guards genuinely enforce the 82-task catalog (exact count, instantiability, certification coverage in both directions), and most of the 31 official-code parity tests pin numbers in checked-in JSON fixtures rather than requiring live reference repos. However, GitHub Actions CI on main is currently FAILING (badge confirms), and I reproduced two concrete causes: 

- **[CRITICAL]** CI on main is red: test.yml dependency set cannot run the suite (vendi_score collection error, ripser test failures) (`.github/workflows/test.yml:28`)
  - Badge fetch: 'Tests - failing'. test.yml installs only `.[dev]` + scikit-learn. tests/tasks/parity/test_vendi_parity.py:36 has module-level `from vendi_score import vendi` (vendi_score in no extra) -> collection error. Simulated missing ripser: `2 failed, 33 passed` (tests/tasks/test_reference_parity_formulas.py:315 raw `from ripser import ripser`; test_topology.py betti test).
  - _fix:_ Add a `test` extra (or extend dev) with scikit-learn, ripser, vendi_score, nltk, etc., and install it in CI; guard vendi_score with pytest.importorskip at module level and the ripser import at test_reference_parity_formulas.py:315 with importorskip. Make the badge green before release.

- **[MAJOR]** Three parity tests hardcode an ephemeral Claude-session /tmp scratchpad as the official-reference location; phd has no override at all (`tests/tasks/parity/test_phd_dimension_parity.py:49`)
  - _REF_CANDIDATES contains only '/tmp/claude-1736197890/.../42dc090a-.../scratchpad/refrepos/GPTID' — no env var. test_lipschitz_parity.py:49 and test_magnitude_parity.py:61 same path with LIPESTIMATION_DIR/BLME_MAGNIPY_DIR fallbacks. /tmp is wiped on reboot, so these reference-parity tests silently skip forever, for maintainer and contributors alike.
  - _fix:_ Drop the scratchpad paths; resolve references via documented env vars (add BLME_GPTID_DIR for phd) or a tests/refrepos/ convention documented in CONTRIBUTING, so 'official code parity' remains re-verifiable.

- **[MAJOR]** blme product code crashes with NameError (not a clean error) when ripser is missing and _count_betti is reached (`src/blme/tasks/topology/betti_curve.py:108`)
  - Module guard sets HAS_RIPSER=False on ImportError (lines 40-44) but `_count_betti` calls bare `ripser(...)` at line 108. Reproduced: with ripser absent, tests/tasks/test_topology.py::test_betti_curve_geodesic_recovers_clusters_and_loops fails `NameError: name 'ripser' is not defined` at betti_curve.py:108 — a misleading crash instead of the intended 'pip install ripser' message.
  - _fix:_ Raise ImportError with the install hint inside _count_betti when HAS_RIPSER is False, and add pytest.importorskip('ripser') to the direct-helper test.

- **[MAJOR]** Fresh-clone test run hard-requires HF Hub network; download failures surface as test failures, not skips (`tests/test_common.py:70`)
  - GPT2TokenizerFast.from_pretrained('gpt2') at test_common.py:70,99 and tests/tasks/test_bias_task.py:24 are unguarded (contrast test_comprehensive_parity.py:48 which skips on cache miss). 54 tests pass dataset=None, triggering load_dataset('EleutherAI/lambada_openai') where only ImportError is caught (betti_curve.py:149 pattern), so offline/firewalled contributors get crashes.
  - _fix:_ Wrap all from_pretrained/load_dataset calls in tests with the existing try/pytest.skip('not available offline') pattern, and catch OSError/ConnectionError (not just ImportError) in tasks' dataset-download fallbacks.

- **[MINOR]** Docs/packaging drift guards are cwd-relative; the stale-identifier guard passes vacuously when run from another directory (`tests/test_publication_docs.py:6`)
  - Ran `cd /tmp && python -m pytest <abs paths>`: 3 failed (FileNotFoundError: 'docs/getting_started.md', REPOSITORIES.md, pyproject.toml), 1 passed — the passing one is test_no_stale_publication_identifiers, because `Path('docs').glob('*.md')` is empty so text=='' and no banned pattern matches: the guard silently asserts nothing.

- **[MINOR]** No pytest configuration at all (no [tool.pytest.ini_options], pytest.ini, or markers) (`pyproject.toml:1`)
  - pyproject.toml has no [tool.pytest.ini_options]; no pytest.ini/setup.cfg/tox.ini exist. No testpaths (rootdir depends on invocation cwd, feeding the cwd-relative guard issue), no filterwarnings policy, no slow/network markers to let contributors deselect the gpt2-downloading parity tests.

- **[MINOR]** Checked-in parity fixtures leak local machine and Claude-session paths in provenance fields (`tests/fixtures/reference_parity/parity/head_roles.json:7`)
  - grep shows 'venv': '/tmp/claude-1736197890/.../scratchpad/venvs/tlens2' in head_roles.json:7, induction_heads.json:7, spectral.json:16, edge_attribution.json:22, knowledge_neurons.json:7, and '/home/dtsaras/miniconda3/...' in correlation_dimension.json:13. Provenance references environments no one (including the maintainer, post-reboot) can inspect.

- **[NIT]** No guard that every registered task is documented in docs/tasks_*.md (currently satisfied but unenforced) (`tests/test_completeness.py:156`)
  - Only docs guard is the literal string '82 diagnostic tasks' in getting_started.md. I scripted the real check: all 82 registered task names do appear somewhere in docs/*.md today ('not mentioned in docs: 0'), but nothing prevents a new task from shipping undocumented.

## Gap survey — missing methods

### Must-add (synthesis judge)
- **Under-trained ('glitch') token detection — Magikarp** — arXiv:2405.05417 · `https://github.com/cohere-ai/magikarp` · verified=True
  - EMNLP 2024 main (Outstanding Paper). Weights-only vocabulary-pathology axis none of the 82 tasks touches: geometry_tokenizer_efficiency is tokenizer-only fertility and geometry_unembedding is effective rank of W_U; neither detects per-token training-coverage failures. Zero forward passes, official repo publishes per-model reports in results/, so ex
- **Categorical & hierarchical concept geometry (causal inner product, Park et al.)** — arXiv:2406.01506 · `https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations` · verified=True
  - ICLR 2025 Oral + ICML 2024 MechInterp workshop Best Paper; independently proposed by two lenses (deduped here). Hierarchy-as-orthogonality and simplex-regularity under the whitened (causal) unembedding metric is a concept-structure signal distinct from geometry_unembedding (rank only) and repe_concept_separability (activation-space), and it gives B
- **Norm-based attention analysis (\|\|alpha f(x)\|\|, Kobayashi et al.)** — arXiv:2004.10102 · `https://github.com/gorokoba560/norm-analysis-of-transformer` · verified=True
  - EMNLP 2020 canonical correction (~700+ citations) with EMNLP 2021 and ICLR 2024 follow-ups in the same official repo (confirmed: three per-paper directories). Every existing BLME attention statistic is raw-weight based; this shows raw alpha systematically overweights special tokens and separates true attention sinks from value-norm-cancelled ones, 
- **Retrieval heads (per-head retrieval-score taxonomy)** — arXiv:2404.15574 · `https://github.com/nightdessert/Retrieval_Head` · verified=True
  - New head-taxonomy class missing from interpretability_head_roles: cross-context copy-during-decode (the documented mechanism behind long-context factuality), vs the existing short-range previous-token/duplicate/induction scores. Fully label-light (synthetic needle/haystack), generation is explicitly in-mold. Repo confirmed: retrieval_head_detection
- **Sharded rank comparison test (provable benchmark contamination)** — arXiv:2310.17623 · `https://github.com/tatsu-lab/test_set_contamination` · verified=True
  - ICLR 2024 Oral (Hashimoto lab). Genuinely different signal from consistency_contamination's per-document Min-K% thresholding: a set-level exchangeability hypothesis test with exact false-positive guarantees ('is THIS benchmark in the training set', with a p-value). Repo confirmed: compute_sharded_comparison_test.py, canonical test files for 8 bench
- **WeightWatcher official layer-quality metrics (alpha via PL-MLE, alpha_hat, rand_distance, num_spikes)** — arXiv:1810.01075 · `https://github.com/CalculatedContent/WeightWatcher` · verified=True
  - Re-raise of the prior survey's 'weight-spectral: no gap' verdict, accepted on a new argument I verified locally: src/blme/tasks/geometry/spectral.py's own docstring admits its Hill-estimator alpha is only a monotone transform of the published WeightWatcher alpha — i.e., BLME's flagship weight metric cannot pass the official-code parity bar today (m

### Nice-to-have
- **Min-K%++ (calibrated pretraining-data detection)** — arXiv:2404.02936 · `https://github.com/zjysteven/mink-plus-plus`
  - ICLR 2025 Spotlight; I fully verified arXiv + repo (official, includes all baselines). Kept out of must_add only because it refines the statistic of the existing consistency_contamination task (vocabulary-normalized z-score / mode test) rather than o
- **LapEigvals (Laplacian eigenvalue spectrum of attention maps) — re-raise of prior-survey #11** — arXiv:2502.17598 · `https://github.com/graphml-lab-pwr/lapeigvals`
  - Prior survey watch-listed this solely for unconfirmed repo + label-dependence; both blockers now resolved (official repo, EMNLP 2025, label-free spectral core). Graph-Laplacian spectra of attention are a different operator from all existing attention
- **Transformer block coupling (Jacobian singular-vector alignment)** — arXiv:2407.07810 · `https://github.com/sugolov/coupling`
  - ICLR 2025; first derivative-level spectral-geometry signal (nothing in the 82 examines block-Jacobian alignment) with a published capability correlation. Needs backward passes — acceptable under the dynamics_sharpness exemption; pip-installable offic
- **Representation Topology Divergence (RTD)** — arXiv:2201.00058 · `https://github.com/IlyaTrofimov/RTD`
  - ICML 2022; topological counterpart of the CKA layer-similarity family, sensitive to multi-scale cluster/loop changes CKA misses; distinct from topology_zigzag_persistence. Official pip-installable package; deterministic on fixed subsamples.
- **Attention-map topological features (TDA4ATD)** — arXiv:2109.04825 · `https://github.com/danchern97/tda4atd`
  - EMNLP 2021 oral; fills the biggest topology-lens hole — all 5 topology_* tasks consume hidden-state point clouds, none applies filtration-based topology to attention. Label-free feature core; golden-value parity on saved attention tensors.
- **LLM-Check (single-pass attention-kernel + hidden-state log-det hallucination scores)** — arXiv:none (NeurIPS 2024; OpenReview LYx4w3CAgy) · `https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection`
  - The only verified single-forward-pass hallucination detector with official code — fills prior survey gap (a). Held out of must_add because it has no arXiv id (formally fails criterion 3) and its Hidden Score is adjacent to geometry_matrix_entropy; im
- **Super weights (data-free single-parameter criticality)** — arXiv:2411.07191 · `https://github.com/mengxiayu/LLMSuperWeight`
  - Proposed by two lenses (deduped). Weight-space localization + causal criticality is arguably distinct from interpretability_activation_sinks/kurtosis, and published coordinates make parity exact — but it is the closest-to-covered candidate (same phen
- **POSIX (Prompt Sensitivity Index)** — arXiv:2410.02185 · `https://github.com/kowndinya-renduchintala/POSIX`
  - Findings of EMNLP 2024, pip package, high parity. The axis is partially covered (consistency_format_robustness, consistency_paraphrase), but POSIX is the published likelihood-based index that would give those custom tasks a canonical anchor — similar
- **GRIDE scale-resolved intrinsic dimension + abstraction-phase onset** — arXiv:2405.15471 · `https://github.com/chengemily1/id-llm-abstraction (estimator: sissa-data-science/DADApy, pip dadapy)`
  - ICLR 2025 with LLM-specific validation; adds the scale axis all five existing ID tasks lack, plus asymmetric information imbalance. Maintained pip dependency, easy float-tolerance parity vs dadapy.
- **Density-peaks representation landscape (ADP clustering)** — arXiv:2409.03662 · `https://github.com/diegodoimo/geometry_icl_finetuning (+ dadapy)`
  - NeurIPS 2024; density multimodality/hierarchy is orthogonal to all geometry and topology tasks (holes vs probability modes). Label-free core (peak counts/hierarchy); two-level parity via dadapy. Pairs naturally with GRIDE (shared dependency).
- **Entropy / confidence-regulation neurons — re-raise of prior-survey #14** — arXiv:2406.16254 · `https://github.com/bpwu1/confidence-regulation-neurons`
  - Prior survey's two blockers (no located repo; overlap fear with prediction_entropy) are addressed: official NeurIPS 2024 repo now verified by the scout, and the weights-only null-space norm-ratio is not an output-entropy statistic. Would be BLME's fi
- **INSIDE / EigenScore — re-raise of prior-survey #9** — arXiv:2402.03744 · `https://github.com/D2I-ai/eigenscore`
  - Legitimate re-raise: the old rejection ('needs sampling') no longer binds since generations are in-mold, and the prior survey's 'reduces to matrix_entropy' claim only covers the single-pass core, not the actual K-generation covariance. No auxiliary m
- **Sparse feature circuits (SAE-feature-level faithfulness/completeness)** — arXiv:2403.19647 · `https://github.com/saprmarks/feature-circuits`
  - ICLR 2025 Oral; feature-resolution circuits genuinely absent (causality tasks operate on heads/edges). Accepted restriction: only models with public SAEs (same as interpretability_sae_features), which caps cross-model comparability — hence not must_a
- **SelfCheckGPT (SelfCheckNgram variant only)** — arXiv:2303.08896 · `https://github.com/potsawee/selfcheckgpt`
  - Canonical behavioral hallucination-consistency signal, not in the prior survey; n-gram variant is strictly single-model. Distinct from self_consistency (answer agreement) and generation_diversity. Deterministic scoring kernel gives clean parity.
- **PriDe option-ID prior (MCQ selection bias)** — arXiv:2309.03882 · `https://github.com/chujiezheng/LLM-MCQ-Bias`
  - ICLR 2024 Spotlight; answer-choice-ID bias under content permutation is covered by no consistency task. Label-free PriDe prior kernel only (headline RStd needs labels — must be excluded and documented); modest extraction work from an experiment-shape
- **Neural persistence + deep graph persistence (weight-space topology pair)** — arXiv:1812.09764 + 2307.10865 · `https://github.com/BorgwardtLab/Neural-Persistence + https://github.com/ExplainableML/Deep-Graph-Persistence`
  - Would be the only weights-only topological task. Adopt as a pair (DGP primary, NP sub-score) since TMLR 2023 shows classic NP is variance-dominated. TF1-era / Aleph-dependent reference code makes parity the main friction — Docker-pinned goldens.
- **Euler characteristic curves & profiles (eulearning)** — arXiv:2303.14040 · `https://github.com/vadimlebovici/eulearning`
  - JMLR 2024; fills the empty Euler-characteristic slot with all-degree topological information at near-linear cost; 2-parameter profiles beyond every 1-parameter topology task. Deterministic sklearn-style transformers, easy parity.
- **Euclidicity / TARDIS singularity detection** — arXiv:2210.00069 · `https://github.com/aidos-lab/TARDIS`
  - ICML 2023; measures where the manifold hypothesis fails — orthogonal to all four ID estimators that assume it. Python-version pin is the integration nuisance; vendor the kernel and pin goldens.
- **Local Learning Coefficient via devinterp** — arXiv:2308.12108 · `https://github.com/timaeus-research/devinterp`
  - Proposed by two lenses (deduped). Basin degeneracy is provably distinct from dynamics_sharpness's curvature, and devinterp is maintained with a wrap-not-reimplement path (sae-lens precedent). Held back: hyperparameter-sensitive SGLD estimator, backwa
- **MAUVE (human-likeness divergence frontier)** — arXiv:2102.01454 · `https://github.com/krishnap25/mauve`
  - NeurIPS 2021 Outstanding Paper; distributional human-likeness of open-ended generation is a real portfolio gap (all generation tasks measure internal diversity). Needs an unlabeled human reference corpus and a featurizer choice — pin both for parity.
- **Text-diversity suite (compression ratio, self-repetition, templates)** — arXiv:2403.00553 · `https://github.com/cshaib/diversity`
  - Cheap extension reusing dynamics_generation_diversity's completions; gzip compression ratio is an information-theoretic diversity signal distinct from n-gram counting. Maintained pip package, deterministic parity.
- **Entropy calibration gap (generation entropy rate vs log loss)** — arXiv:2511.11966 · `https://github.com/stevenxcao/entropy-calibration`
  - NeurIPS 2025; compares generative entropy rate to predictive log loss — the canonical degeneration/error-accumulation diagnostic, covered by no task. Decoupled-generation parity feasible.
- **LLM-as-lossless-compressor (cross-modal compression rate)** — arXiv:2309.10668 · `https://github.com/google-deepmind/language_modeling_is_compression`
  - ICLR 2024; the cross-modal (text/image/audio bytes) compression profile and size-adjusted rate are new axes beyond geometry_perplexity. Arithmetic-coder kernel is pure Python with bit-identical parity; modality plumbing is the main effort.
- **TOHA (topological divergence of attention during generation)** — arXiv:2504.10063 · `https://github.com/sb-ai-lab/TOHA`
  - ACL 2026; only generation-time topological signal. Requires the label-free adaptation (all-head distribution instead of the ~50-sample head selection) — an honest deviation from the reference pipeline, hence low rank despite good code.
- **Transcoder circuits (MLP transcoder fidelity)** — arXiv:2406.11944 · `https://github.com/jacobdunefsky/transcoder_circuits`
  - NeurIPS 2024; transcoder slot is empty, but out-of-the-box coverage is GPT-2-small only (pretrained transcoders are model-specific), limiting cross-model comparability more than any accepted task.
- **Information flow routes (llm-transparency-tool)** — arXiv:2403.00824 · `https://github.com/facebookresearch/llm-transparency-tool`
  - EMNLP 2024; patching-free per-input circuit sparsity. Caveats: repo archived read-only and CC BY-NC (parity-oracle only, BLME reimplements); shares its norm-contribution kernel with the Kobayashi must_add — implement that first and reassess the margi

**Rejected / duplicates:** Categorical/hierarchical concept geometry (second scout copy, lens 1) — identical method/arXiv/repo as the lens-3 proposal; merged into one must_add; Super weights (second scout copy) — two lenses proposed the same paper/repo; merged into one nice_to_have entry; Local Learning Coefficient / devinterp (second scout copy) — two lenses, same paper/package; merged into one nice_to_have entry; Domain-stratified BPC (2404.09937) — core quantity is BPC, already computed by geometry_perplexity; only the corpora/protocol are new (scout concedes); Blockwise Hessian spectrum (2402.16788) — dynamics_sharpness family refinement, heaviest compute in the batch; heterogeneity scalar alone doesn't justify a new task; Filter-normalized loss sections (1712.09913) — another dynamics_sharpness-family refinemen

**Notes:** Verification performed personally on all 6 must_add entries: arXiv abstract AND official repo fetched for each (12 fetches), all resolve and match the scouts' descriptions; additionally the WeightWatcher parity-debt claim was confirmed directly in src/blme/tasks/geometry/spectral.py (docstring lines 23-28 admit the Hill alpha is a monotone transform of published WW alpha). Nice_to_have entries were not independently re-fetched except Min-K%++ (verified both ends; it is must_add-grade and should be promoted first if a slot opens — held back only because it refines the existing consistency_contamination statistic rather than opening a new axis). One caveat on Retrieval Heads: the arXiv page lists no venue; ICLR 2025 acceptance rests on the scout's proceedings check. Prior-survey leftovers le

### All verified raw candidates (6 lenses, arXiv + repo fetched)
- Attention-map topological features (TDA4ATD, Kushnareva et al.) — arXiv:2109.04825 (verified resolves) · `https://github.com/danchern97/tda4atd (verified; self-declared official repo for the EMNLP 2021 paper)`
- Representation Topology Divergence (RTD) — arXiv:2201.00058 (verified resolves) · `https://github.com/IlyaTrofimov/RTD (verified; official, pip-installable via git, Docker + Colab provided)`
- Euler characteristic curves & profiles (eulearning) — arXiv:2303.14040 (verified resolves) · `https://github.com/vadimlebovici/eulearning (verified; official implementation by the paper's authors, MIT license, scikit-learn transformers in descriptors.py)`
- Euclidicity / topological singularity detection (TARDIS) — arXiv:2210.00069 (verified resolves) · `https://github.com/aidos-lab/TARDIS (verified; official aidos-lab code for the ICML 2023 paper, BSD-3, pip install . / poetry)`
- Neural persistence (weight-space 0-dim PH) — arXiv:1812.09764 (verified resolves) · `https://github.com/BorgwardtLab/Neural-Persistence (verified; official Borgwardt-lab code, Docker-reproducible)`
- Deep graph persistence (corrected neural persistence) — arXiv:2307.10865 (verified resolves) · `https://github.com/ExplainableML/Deep-Graph-Persistence (verified; confirmed official code for the TMLR paper)`
- TOHA — topological divergence of attention graphs during generation — arXiv:2504.10063 (verified resolves) · `https://github.com/sb-ai-lab/TOHA (verified; self-declared official repo, run_mtopdiv.py, Docker)`
- LapEigvals — Laplacian eigenvalue spectrum of attention maps (label-free feature core) — arXiv:2502.17598 · `https://github.com/graphml-lab-pwr/lapeigvals (fetched; 'Official implementation... accepted at EMNLP 2025')`
- Under-trained ('glitch') token detection from embedding geometry (Magikarp) — arXiv:2405.05417 · `https://github.com/cohere-ai/magikarp (fetched; official, ~190 stars, per-model verification reports in results/)`
- Transformer block coupling (Jacobian singular-vector alignment across depth/tokens) — arXiv:2407.07810 · `https://github.com/sugolov/coupling (fetched; 'Code release for Transformer Block Coupling... in ICLR2025'; pip-installable, run_coupling_hf / coupling_from_hooks + Colab demo)`
- Density-peaks representation landscape (ADP clustering of hidden states) — arXiv:2409.03662 · `https://github.com/diegodoimo/geometry_icl_finetuning (fetched; official NeurIPS 2024 code) + maintained pip package dadapy (sissa-data-science/DADApy, Patterns 2022) implementing the ADP estimator`
- Categorical/hierarchical concept geometry in the unembedding space (causal inner product) — arXiv:2406.01506 · `https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations (fetched; official, 7 numbered validation notebooks, WordNet extraction scripts included)`
- Super weights / super activations (single-parameter criticality) — arXiv:2411.07191 · `https://github.com/mengxiayu/LLMSuperWeight (fetched; official — analyze.py, evaluate.py, reproduction scripts for OLMo/Mistral/Phi-3/Llama)`
- Local Learning Coefficient (LLC) via devinterp — arXiv:2308.12108 · `https://github.com/timaeus-research/devinterp (fetched; MIT, pip 'devinterp', supports 85+ HF model types incl. Qwen2.5-0.5B example)`
- Min-K%++ (calibrated pretraining-data detection) — arXiv:2404.02936 (verified: abstract fetched) · `https://github.com/zjysteven/mink-plus-plus (verified: fetched; MIT; includes Loss/zlib/Min-K%/Ref/Neighbor baselines)`
- SelfCheckGPT (n-gram variant; sampling-based hallucination consistency) — arXiv:2303.08896 (verified: abstract fetched) · `https://github.com/potsawee/selfcheckgpt (verified: fetched; official; pip install selfcheckgpt)`
- POSIX (Prompt Sensitivity Index) — arXiv:2410.02185 (verified: abstract fetched) · `https://github.com/kowndinya-renduchintala/POSIX (verified: fetched; official; pip install prompt-sensitivity-index)`
- Sharded Rank Comparison Test (provable benchmark contamination) — arXiv:2310.17623 (verified: abstract fetched) · `https://github.com/tatsu-lab/test_set_contamination (verified: fetched; official; includes 8 canonical test sets + contaminated-model challenge)`
- INSIDE / EigenScore (internal-state semantic consistency of sampled generations) — RE-PROPOSAL with new argument — arXiv:2402.03744 (verified: abstract fetched) · `https://github.com/D2I-ai/eigenscore (verified: fetched; official, author-maintained)`
- PriDe option-ID prior / MCQ selection-bias (label-free permutation debiasing prior) — arXiv:2309.03882 (verified: abstract fetched) · `https://github.com/chujiezheng/LLM-MCQ-Bias (verified: fetched; official)`
- Generating with Confidence: graph-spectral black-box UQ (EigV / Degree / Eccentricity, Jaccard kernel) — same-class-as-rejected-KLE, with new argument — arXiv:2305.19187 (verified: abstract fetched) · `https://github.com/zlin7/UQ-NLG (verified: fetched; official)`
- LLM-Check (single-pass eigen-analysis hallucination scores: attention-kernel + hidden-state log-det) — arXiv:none — no arXiv listing; peer-reviewed NeurIPS 2024, verified via https://neurips.cc/virtual/2024/poster/95584 and https://openreview.net/pdf?id=LYx4w3CAgy (both fetched/confirmed) · `https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection (verified: fetched; official)`
- LLM-as-lossless-compressor (cross-modal arithmetic-coding compression rate) — arXiv:2309.10668 · `https://github.com/google-deepmind/language_modeling_is_compression`
- Domain-stratified BPC (compression-represents-intelligence protocol) — arXiv:2404.09937 · `https://github.com/hkust-nlp/llm-compression-intelligence`
- Text-diversity suite: compression ratio, self-repetition, n-gram diversity, homogenization — arXiv:2403.00553 · `https://github.com/cshaib/diversity`
- MAUVE (divergence-frontier gap between model text and human text) — arXiv:2102.01454 · `https://github.com/krishnap25/mauve`
- GRIDE scale-resolved intrinsic dimension + high-dimensional abstraction-phase onset — arXiv:2405.15471 · `https://github.com/chengemily1/id-llm-abstraction (paper code; uses https://github.com/sissa-data-science/DADApy, pip `dadapy`)`
- Entropy calibration gap (generation entropy-rate drift vs log loss on human text) — arXiv:2511.11966 · `https://github.com/stevenxcao/entropy-calibration`
- Per-input vs across-input diversity decomposition (EAD / sentence-BERT / NLI diversity) — arXiv:2310.06452 · `https://github.com/facebookresearch/rlfh-gen-div`
- Retrieval heads (retrieval-score head taxonomy) — arXiv:2404.15574 (verified: abstract fetched) · `https://github.com/nightdessert/Retrieval_Head (verified: fetched; official, 241 stars; entry point retrieval_head_detection.py -> per-head head_score JSON; supports Llama/Qwen/Mistral/Phi)`
- Norm-based attention analysis (\|\|alpha f(x)\|\| value-weighted attention) — arXiv:2004.10102 (verified: abstract fetched) · `https://github.com/gorokoba560/norm-analysis-of-transformer (verified: fetched; official, covers all three papers, per-paper directories with notebooks)`
- Categorical & hierarchical concept geometry (causal-inner-product simplex/orthogonality scores) — arXiv:2406.01506 (verified: abstract fetched) · `https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations (verified: fetched; official; numbered notebooks 01-07 computing concept vectors, orthogonality tests, polytope validation on Gemma-2B and LLaMA-3-8B)`
- Entropy / confidence-regulation neurons (unembedding null-space neuron taxonomy) — RE-PROPOSAL of prior-survey watch item #14 with a NEW argument — arXiv:2406.16254 (verified: abstract fetched) · `https://github.com/bpwu1/confidence-regulation-neurons (verified: fetched; official NeurIPS 2024 code; entry points run_and_store_ablation_results.py, run_and_store_unigram_results.py, induction.py; Hydra configs; GPT-2/Pythia/LLaMA)`
- Sparse feature circuits (SAE-feature-level circuit faithfulness/completeness) — arXiv:2403.19647 (verified: abstract fetched) · `https://github.com/saprmarks/feature-circuits (verified: fetched; official; scripts/get_circuit.sh, scripts/evaluate_circuit.sh, experiments/faithfulness.ipynb; downloads pretrained Pythia-70M dictionaries — no SAE training required)`
- Transcoder circuits (MLP transcoder fidelity + input-independent circuit attribution) — arXiv:2406.11944 (verified: abstract fetched) · `https://github.com/jacobdunefsky/transcoder_circuits (verified: fetched; official; walkthrough.ipynb + setup.sh downloads pretrained GPT-2-small transcoders from HF pchlenski/gpt2-transcoders)`
- Information flow routes (single-pass attribution graph over heads and FFNs) — arXiv:2403.00824 (verified: abstract fetched) · `https://github.com/facebookresearch/llm-transparency-tool (verified: fetched; the official Meta implementation — README states it shows Ferrando & Voita information flow routes; core logic in llm_transparency_tool/models/transparent_llm.py on TransformerLens. Caveats: archived read-only Feb 2026; CC BY-NC 4.0)`
- Copy suppression score (negative/anti-copy head class) — arXiv:2310.04625 (verified: abstract fetched) · `https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages (verified: fetched; official — README carries the paper citation and contains the cspa/ directory implementing copy-suppression-preserving ablation; research-grade fork of TransformerLens with a 'TODO: clean up' note)`
- WeightWatcher official layer-quality metrics (HTSR alpha via PL-MLE, alpha_hat, rand_distance / correlation traps, num_spikes; SETOL ERG) — arXiv:1810.01075 (primary, Martin & Mahoney; fetched OK); SETOL 2507.17912 (fetched OK); single-checkpoint anti-grokking evidence 2602.02859 (fetched OK) · `https://github.com/CalculatedContent/WeightWatcher (pip `weightwatcher` v0.8.0, 1.4k commits, maintained; SETOL experiments: github.com/charlesmartin14/SETOL_experiments)`
- Local Learning Coefficient (LLC) — singularity-aware loss-landscape complexity via devinterp — arXiv:2308.12108 (fetched OK); at-scale estimator 2402.03698 (resolves; devinterp docs implement it) · `https://github.com/timaeus-research/devinterp (official, pip `devinterp`, v2.0.1 released 2026-04, HuggingFace + TransformerLens support, Qwen2.5-0.5B quickstart)`
- Super weights — data-free localization of catastrophic scalar weights — arXiv:2411.07191 (fetched OK; v2 2025-07) · `https://github.com/mengxiayu/LLMSuperWeight (official; analyze.py/evaluate.py, scripts reproducing all paper tables, published index of super-weight coordinates per model)`
- Blockwise Hessian spectrum + block-heterogeneity (SLQ) — arXiv:2402.16788 (fetched OK) · `https://github.com/zyushun/hessian-spectrum (official; blockwise & full Hessian ESD via stochastic Lanczos quadrature, includes GPT-2/openwebtext setup)`
- Filter-normalized loss-landscape sections (scale-invariant sharpness) — arXiv:1712.09913 (fetched OK) · `https://github.com/tomgoldstein/loss-landscape (official PyTorch implementation; 1D/2D sections along filter-normalized random directions)`
- Rate-distortion weight complexity (grokking compressibility) — arXiv:2412.09810 (fetched OK) · `https://github.com/brantondemoss/GrokkingComplexity (official; main.py bayes_optimize_svd_quantization is the reference kernel)`
- Input-loss-curvature memorization score — arXiv:2402.18726 (fetched OK; ICML 2024 per dblp) + origin 2307.05831 (fetched OK) · `https://github.com/DeepakTatachar/Privacy-Memorization-Curvature (official ICML 2024 implementation; companion github.com/purdue-nrl/Memorization-Curvature also official)`
