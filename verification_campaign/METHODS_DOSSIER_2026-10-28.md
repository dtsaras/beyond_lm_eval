# Methods dossier — synthesis (2026-10-28)

This applies the user's method-fidelity rule (see `DECISIONS_2026-10-26.md`,
"Resolution") to all 51 paper-core tasks. The rule has four parts:
1. use the paper or GitHub protocol;
2. otherwise, use the precedent of LLM papers that apply the method;
3. otherwise, write down the rationale;
4. provide size-independent versions.

**Evidence.** There is one JSON per task, with sources, a protocol
comparison per axis, divergences, token policy, executed size tests and
proposed normalised forms. Archive: `~/hara-artifacts/blme/dossier_2026-10-28/`.
That folder also holds `work_{A..F}/` (scripts and raw outputs), `digest.txt`
and the shared activation bank, `~/hara-artifacts/blme/actbank/`:
- gpt2;
- pythia-160m, 410m and 1b;
- TinyLlama-1.1B, which adds a BOS token;
- Qwen3-4B and Qwen3-8B.

Six independent agents produced it. Each was read-only on the repo, cited
`file:line` for BLME and repo@commit or paper section for the references, and
checked every arXiv id it cited against the fetched title.

**Orchestrator re-verification.** I checked ten load-bearing claims against
code or data. All ten were confirmed:
- steering's fp32 gate at 2e9 parameters (`representation_engineering.py:858-867`);
- concept_separability fits an unstandardised `LogisticRegression` (`:745`);
- weight_norms' headline is the refuted sr/min(m,n) (`weight_norms.py:204`);
- platonic-rep scores per-text masked means by default (`extract_features.py:81-84,180`);
- bf16 `perturbation_growth` on gpt2 is 4.560 against 2.754 in fp32 (golden snapshot);
- the study's `n_samples: 8` is never read (`tasks.yaml:252` vs `generation_diversity.py:201`);
- positional_decay's 2e9 gate (`positional_decay.py:107`);
- betti_curve is overridden to N=50 (`tasks.yaml:228`);
- GPTID slices `[1:-1]` while BLME does not (refuting the campaign's claim);
- the same-document neighbour effect on TwoNN, which I re-measured independently:
  - gpt2 L12: +43%, with the nearest neighbour in the same document for 28% of rows;
  - Qwen3-4B L18: +81%, 69%;
  - pythia-1b L16: +202%, 83%.

**Two earlier campaign records turned out wrong:**
- "phd follows GPTID's `[1:-1]`" is false;
- weight_norms ships as its headline a twin that `normalization.py` records as refuted.

Campaign history is therefore treated as claims, not facts.

---

## A. Policies applied under the rule (proceeding unless the user objects)

**P1 — Align to the reference (rule 1).** 56 divergences are marked
`align` in the JSONs. The main ones:
- **cknna:** per-text masked mean, one joint clamp, embedding layer included;
- **zigzag:** last token, at the post-attention-LayerNorm sites plus the final norm, with k=4;
- **concept_separability:** `rep_token=-1`, the LAT template, PCA of pair differences, and the layer chosen on a validation split;
- **refusal:** chat template and post-instruction tokens for instruct models; layers indexed by block input; the l<0.8L restriction;
- **phd:** `[1:-1]` slicing and the notebook's adaptive grid;
- **magnitude:** the t_conv scale and MagArea/t_conv, with cosine distance as the paper uses for text;
- **trajectory_curvature:** keep HF's post-norm final state, and no skip of the first token on BOS-less tokenizers;
- **sharpness:** PyHessian's stopping rule (tol 1e-3, max 100);
- **generation_diversity:** Self-BLEU on word units, and the study key corrected to `n_samples_per_prompt`;
- **perplexity:** `bits_per_byte` as the headline;
- **spectral:** WeightWatcher's `alpha_official` as the headline;
- **betti_curve:** the reference's k counts the point itself;
- **coe:** the release's arithmetic, with no extra floor;
- **fractal:** natural long documents instead of splices.

**P2 — Position 0.** Applied on the token axis only, per the 2026-10-26 decision:
- **collapse.** Queipo-de-Llano 2025 Eq. (1) is added as a faithful key: per sequence, uncentred, BOS kept. It reproduces their compression valley on the bank: minimum H of 0.017–0.31 at middle layers with position 0, against 1.3–2.9 without. The Jing / Roy-Vetterli keys drop position 0, as decided. Measured on 6 of 7 models, this flattens them (collapse_ratio 0.07–0.74 → 0.79–0.91), because the "valley" they detected was the sink. They therefore go to `_meta_` if they no longer discriminate between models.
- **contextualization.** Drop position 0 for **every** tokenizer. Ethayarajh drops special tokens; on BOS-less tokenizers position 0 is a sink playing the same role, and keeping it there makes the metric asymmetric across tokenizers. `mev_corrected_sizematched` flips sign on gpt2 and pythia-1b.
- **Tier-1 tasks whose reference keeps position 0** (procrustes, isoscore, matrix_entropy, cknna, effective_depth) **keep it**. The value without it goes to `_meta_`.

**P3 — Size-independent versions (rule 4).** The task's headline stays the
reference quantity, and the task also emits its own normalised version,
pinned by a direct test. These versions pass that test:
- **Already emitted**, need registering and pinning:
  - `isoscore_effective_dims`, re-verified at ≤2.6e-13 under zero-padding;
  - `matrix_entropy_raw` (×ln d), exact under zero-padding;
  - mahalanobis dist²/d, within ≤3%;
  - topology `/mpd` twins (mean pairwise distance);
  - debiased CKA, currently a `_meta_` key, which removes the d_eff/N floor;
  - mp `*_g25` (fixed γ=0.25), which is N-stable.
- **New:**
  - procrustes adjusted R² `1−(1−sim)(N−1)/(N−1−r)`. It removes the exact r/(N−1) chance term and cuts the N-drift 4–10×;
  - task_vectors `consensus_energy_rel`, an unbiased U-statistic that is flat in N;
  - refusal `rel_direction_energy`;
  - representation_sensitivity `S_unit`, exact under the final-norm/unembedding gauge;
  - sharpness: adaptive SAM (Kwon 2021), bit-identical under a function-preserving rescale;
  - gradient_flow: end-to-end log(g_emb/g_final);
  - lipschitz: `e2e_log_norm_growth` and `total_angular_path`;
  - positional_decay: the write-norm-weighted mean, invariant from L=12 to 24 and under ×4 residual rescale;
  - effective_depth: Σ BI;
  - `layer_index/(L−1)` for the absolute-index keys.
- **Size-linked by meaning**, disclosed and not normalised: perplexity/bpb, baseline loss, Hurst exponent.
- **Capacity ratios**, kept only under their own name: raw IsoScore, superposition_ratio, effective_depth_90, trace_per_param.

**P4 — Layer aggregates must pass the identity-block test.** Inserting
function-preserving identity blocks moves every uniform mean, slope or quartile
over layers. This holds for vendi, cka/hsic/rsa, lipschitz, gradient_flow,
zigzag, intrinsic_dim, trajectory_curvature, coe (exactly 1/L), positional_decay
and collapse `max_drop`. Endpoint, end-to-end and write-weighted forms are
invariant. Such aggregates leave the headline set: they become `_meta_` profile
summaries, and the P3 forms replace them. Where the paper defines the
aggregate (CoE's mean over L), the paper value stays the headline and the
normalised version is emitted alongside it.

**P5 — Precision. This revises decision 7.**
- Decision 7 promoted only the analysis tensors; that cannot fix the measured
  error, which comes from bf16 arithmetic.
- Promoting only the analysis tensors is not enough:
  - perturbation_growth: gpt2 +69%, pythia-160m +119%;
  - gradient_flow: pythia-160m slope −2.05 → −4.13;
  - interpolation: not reproducible across thread counts.
- In each case, keeping bf16 *weights* but doing the *arithmetic* in fp32
  reproduces fp32 to within 0.04%.
- Rule: tasks that measure small differences or derivatives
  (perturbation_growth, gradient_flow, interpolation, steering,
  positional_decay, sharpness, refusal and concept scoring) compute in fp32 at
  **every** size. The 2e9-parameter gates are removed; they made the dtype a
  function of model size. The study runner places these tasks across 2 GPUs
  when fp32 weights do not fit on one.
- Tasks that only read activations keep the native bf16 forward pass; the
  oracle (Phase 3) checks that this is sound.

**P6 — Corpus.** Strip each passage; this removes the trailing-newline sink
(26–32 rows → 0). phd (the paper uses texts of about 300 tokens) and fractal
take natural long documents (`wikitext_document_level`, cached).
`interpolation` cuts by tokens, not by 128 characters.

**P7 — Key hygiene, folded into the schema change.**
- **Demote to `_meta_`:**
  - config echoes and constants: `n_id_fit`, `num_samples_analyzed`, `n_category_tokens_mapped`, `tail_fraction`, `n_tokens_in_stream`, `*_concat_legacy`;
  - the persistence_entropy H0 family: E/log2(n) passes every direct test but does not discriminate (between/within SD 0.5–1.3);
  - every betti_curve key at N=50 (constant);
  - `lid_min`;
  - svd `cond_number` (no source; it depends on N and d);
  - unembedding `purity` (flagged non-comparable by the task itself);
  - perplexity rare/freq buckets;
  - growth_rate_per_block;
  - `hsic_compression_ratio_raw` (unbounded);
  - `margin_uniform`.
- **Drop exact duplicates:**
  - `max_offdiag_*` (equal to `max_adjacent_*` in 65 or 66 of 66 models);
  - `auc_at_depth_100` (equal to `separability_auc`);
  - the H0 landscape keys (a closed form of homology).
- **Replace absolute layer-index names** with first/mid/last plus `_frac` keys.
- **Swap the refuted weight_norms headline** for the raw stable rank, which is invariant under zero-padding.
- **Restrict vocabulary-row tasks** (hubness, superposition, categories, prediction_alignment) to the rows the tokenizer can emit.

---

## B. Decisions for the user

**Q1 — The unit of a point cloud for sequence-level estimators.**
- **Scope:** intrinsic_dim, lid, vendi, magnitude, correlation_dimension, and the four mean-pooled topology tasks.
- **Status quo:** BLME pools every token of 200 documents into one cloud.
- **Effect:** each token's nearest neighbour is in its own document 28–83% of the time. Restricting to other documents raises TwoNN by +43% to +202% and LID by +6% to +95%, by an amount that differs per model, so it does not cancel in comparisons.
- **LLM precedents** use one point per sequence:
  - Cheng 2024 and Yin 2024: last token;
  - Valeriani 2023: mean;
  - Fay 2025 and Gardinazzi 2025: last-token clouds.
- **Options:**
  - **(a) Recommended, rule 2:** one point per sequence, last token, with N pinned at ≥1000 sequences, the same for every model. The token-cloud value moves to `_meta_`.
  - (b) Keep token clouds but exclude same-document neighbours (a Theiler window, the standard dynamical-systems correction), with N pinned.

**Q2 — When the paper text and the released code disagree, or the code has a bug.**
- **Cases:**
  - CoE's release indexes `[0][0]`, which pools prompt position 0 into every layer's mean, while the paper text pools output tokens. The release also mean-pools *generated* tokens; BLME reads the last prompt token, and the two correlate at −0.30 / −0.05, not "highly".
  - fractal: eps 5e-2 in the notebook vs 5e-3 in the paper.
  - The fractal notebook scores tokens off by one (cell 27) and never resets the Joseph accumulator (cell 46).
  - Texygen's default gram=3 vs the 4 used here.
- **Recommended default:** the paper's stated definition wins. A demonstrable code bug is never reproduced. Code fills in what the paper leaves unspecified. Each case is recorded in the task's notes.
- **Consequence for CoE:** it would generate greedily and pool the generated tokens, as both the paper and the release do.

**Q3 — Restoring the reference sample sizes, at a compute cost.**
- **The study's reductions are material:**
  - sharpness, with 8 Hutchinson probes: relative standard error 79% on pythia-160m;
  - betti_curve at N=50: constant;
  - topology at N=50, against thousands of points in the precedents;
  - zigzag: 40 prompts and 1 subset, against 500 × 16;
  - generation_diversity: 36 samples, against 500;
  - cknna: 100 texts at k=10, against 1024;
  - mahalanobis: Ledoit-Wolf fit on 25 points in d=768–5120.
- **Recommended:** restore every reference size. Measure the wall-clock cost on Qwen3-8B first, then bring back for a decision only the tasks whose cost is prohibitive.

**Q4 — refusal_direction on base models.**
- Arditi's base-model protocol appends `\n` and reads that token. On the families where the newline is a sink (pythia, TinyLlama, probably Llama and Mistral), the metric collapses: pythia-1b AUC 0.979 → 0.688.
- **Options:**
  - **(a) Recommended:** the paper protocol on instruct models only, with their chat template; NaN plus a `_meta_` reason on base models;
  - (b) a last-raw-token read on base models, as a rule-3 adaptation;
  - (c) the paper's base protocol, as-is.

---

## C. Per-task change list (headline = reference quantity; ⇄ = normalised version emitted)

| task | rule | changes | ⇄ size-independent |
|---|---|---|---|
| coe | 1 | release arithmetic; Q2 (generated tokens) | paper mean over L kept; /L-free twin under study |
| fractal | 1 | natural long docs; demote legacy keys; Q2 (eps, notebook bugs) | Hurst size-linked by meaning |
| generation_diversity | 1 | word units; study key fix; Q2 (gram); Q3 (n) | — (undetermined) |
| gradient_flow | 2 | fp32 compute; final-residual endpoint; exclude pos0 | e2e log(g_emb/g_final) |
| interpolation | 3 | fp32 compute and head; cut by tokens | convexity_gap (depth-invariant) |
| perturbation_growth | 3 | fp32 compute; demote rate_per_block, slope | growth_log_ratio (invariant) |
| sharpness | 1 | PyHessian stopping rule; Q3 | adaptive SAM |
| stability | 3 | document k=50; drop non-emittable rows | — (V-dependent; undetermined) |
| categories | 3 | V-independent random draw; demote counts | — |
| cka | 1 | exclude pos0; drop max_offdiag; P4 | debiased CKA |
| cknna | 1 | per-text mean, joint clamp, embedding layer; Q3 | — |
| collapse | 2 | add QdL Eq.1; Jing/RV keys exclude pos0 | — |
| contextualization | 1 | drop pos0 for every tokenizer | — |
| correlation_dimension | 3 | fixed published fit window; exclude pos0; Q1 | — |
| effective_depth | 1 | none (paper keeps all tokens) | Σ BI; effective_depth_90 = capacity ratio |
| hsic | 1 | exclude pos0; demote raw ratio | debiased nCKA |
| hubness | 2 | unembedding (precedent) with input embeddings in `_meta_`; emittable rows only; fix zero-row ranking | pin the row count |
| intrinsic_dim | 2 | exclude pos0; Q1 | first/last invariant |
| isoscore | 1 | none | isoscore_effective_dims |
| lid | 2 | exclude pos0; pin N; Q1 | — |
| lipschitz | 3 | exclude pos0; embedding transition separate | e2e_log_norm_growth |
| magnitude | 1 | t_conv, MagArea/t_conv, cosine; demote non-paper keys | MagArea/t_conv |
| mahalanobis | 2 | exclude pos0; Q3 (n_fit) | dist²/d |
| matrix_entropy | 1 | none | matrix_entropy_raw |
| mp_bulk_deviation | 3 | exclude pos0; g25 headline (raise the token cap so d≥5000 is defined) | g25 |
| neural_collapse | 2 | next-token classes with token features (Wu & Papyan precedent) | NC1 / permutation null |
| perplexity | 1 | bpb headline; demote buckets | size-linked by meaning |
| phd_dimension | 1 | `[1:-1]`, adaptive grid, ≥300-token docs | — |
| positional_decay | 3 | nosink headline; fp32 at every size | write-norm-weighted mean |
| prediction_alignment | 3 | exclude pos0; emittable rows; demote margin_uniform | — |
| procrustes_linearity | 1 | rank-aware fit; BOS kept | adjusted R² |
| representation_sensitivity | 3 | keep all positions | S_unit |
| rsa | 2 | angular distance; exclude pos0 | — |
| schatten | 2 | MLP activations averaged over layers (Yusupov); full text (MNN) | — |
| spectral | 1 | alpha_official headline; demote tail_fraction | — |
| superposition_strength | 1 | emittable rows only | MSO invariant |
| svd | 2 | exclude pos0 (erank/PR); demote cond | — |
| tokenizer_efficiency | 3 | document the whitespace word unit | — |
| trajectory_curvature | 1 | keep HF post-norm final; no skip on BOS-less | straightening_ratio (invariant) |
| unembedding | 3 | demote purity; num_samples decoupled | erank (invariant) |
| vendi_score | 3 | exclude pos0; P4; Q1 | first/last/max |
| weight_norms | 3 | raw stable rank as headline | raw sr (invariant) |
| concept_separability | 1 | rep_token=-1, LAT, PCA, validation layer, fp32 | max_auc_layer/(L−1) |
| refusal_direction | 1 | chat template, post-instruction tokens, layer rule, fp32; Q4 | rel_direction_energy |
| steering_effectiveness | 3 | fp32 at every size; data-matched control; alpha relative to residual norm | — |
| task_vectors | 3 | replace mvnn headline | consensus_energy_rel |
| betti_curve | 2 | k counts self; exclude pos0; Q3 (N) | — |
| homology | 2 | exclude pos0; Q1 | /mpd twins |
| persistence_entropy | 2 | demote the H0 family; relative layer names; Q1 | E/log2 n (H1) |
| persistence_landscape | 2 | mean landscape over subsamples; fix 30 mis-prefixed keys | /mpd^p |
| zigzag_persistence | 1 | last token, post-attention-LN sites, k=4; Q3 | — (none passes the identity-block test) |

**Still open after the dossier:**
- the normalised versions passed their direct tests, but none has a measured cross-model reduction yet; that needs a zoo rerun after implementation;
- width tests for gradient_flow, perturbation_growth and sharpness, where zero-padding a pre-LN model is not function-preserving;
- the Phase 3 precision oracle.
