# Verification campaign — state at 2026-09-04

## The bottom line

**Nothing is signed off.** The bar requires BOTH Opus 5 and gpt-5.6-sol, and for
normalized variants both MUST sign off. GPT-5.6 is quota-blocked until
**Sep 7 2026 10:43 AM** (see `CODEX_BLOCKER.md`), so:

| axis | Opus 5 | both models | bar met? |
|---|---|---|---|
| implementation | **51 of 51 GREEN** | 17 | **no** |
| normalized variant | **1 registered** (10 removed on measurement) | 0 | **no** |

The Opus column is a single reviewer's judgement. Treat it as *evidence
gathered*, not as sign-off.

## What one reviewer has established

Independent of the sign-off bar, these are checkable facts about the code:

- **17 published numbers reproduced.** `geometry_correlation_dimension` alone
  hits three GP-1983 constants — Lorenz 2.05 → 2.0521 (+0.10%), Hénon 1.21 →
  1.2209 (+0.90%), logistic at the accumulation point 0.500 → 0.5005 (+0.11%) —
  and also agrees with `skdim.id.CorrInt` to ≤0.40%. Others include Razzhigaev
  et al.'s Procrustes 0.99, Wei et al.'s matrix entropy, and Tulchinskii et
  al.'s PHD 9.491 to 2.7%.
- **28 tasks matched against the authors' released code**, several bit-exactly:
  Procrustes vs `llm-microscope` at rel 0.0, CKNNA vs the official `cknna` at
  rel 0.0, refusal directions vs `andyrdt/refusal_direction` at cos 1.0000000000,
  RSA vs `rsatoolbox` at 2.2e-16, spectral α vs live weightwatcher 0.7.7 at
  mean |Δα| 3.5e-04.

## Read `RERUN_LEDGER.md` before trusting any archived result

Rounds 13–35 changed behaviour under nearly every task. Recorded there in full;
the load-bearing ones:

- A **forward hook** now recovers the deepest block's pre-norm output. HF's last
  `hidden_states` entry is `ln_f(block L-1)`; including it contaminated
  transition metrics, and truncating it deleted a block. Verified exactly:
  `ln_f(hooked)` reproduces HF's entry to **0.0e+00**.
- **Headline keys changed meaning** in `matrix_entropy` (raw → normalised),
  `prediction_alignment` (raw cosine → anisotropy-corrected margin) and
  `correlation_dimension` (percentile window → scaling region). Old and new
  results **must not be pooled**; the incompatibilities are recorded in
  `task_metadata.py`.
- Two tasks were **failing outright** on common configurations:
  `correlation_dimension` on any fp16 model (every pairwise distance non-finite)
  and `positional_decay` on a default-loaded gpt2 (SDPA returns no attention,
  reported as "sequences too short").
- Several columns were **dtype-dependent to the point of sign flips** on bf16 —
  the dtype `dtype="auto"` picks for Llama/Gemma/Qwen/SmolLM2.

## What normalization work established

**Eleven twins were registered on plausible mechanisms. Measurement removed
ten.** This is the campaign's clearest methodological result, and it is a
warning, not a milestone.

The four failure kinds, all measured:

1. **The divisor manufactures the dependence.** Zero-padding a fixed operator
   from d = 64 to 4096 leaves stable rank at **15.846333** while sr/d falls
   ×64 — so `avg_stable_rank/d` and `mean_stable_rank/d` FLIPPED THE SIGN and
   kept the magnitude (+0.793 → −0.575). Same for schatten's shipped
   `row_normalized_*`, where the raw norm is bit-identical (120.403159) across
   d = 16…4096 while raw/d falls ×256.
2. **The mechanism is false.** `lid_mean/log(d)` was justified as "an ID
   estimate carries a log(d) term". LID is **bit-identical (9.7467)** under
   isometric embedding at d = 64…4096. The twin converted a null vocabulary
   association into a significant one.
3. **Dead keys and near-constants.** Two twins targeted keys emitted nowhere;
   one targeted a key the study config never produces; one spans 0.913–0.982
   (cv 0.023). Two earlier ones were deleted for being identically 1.0.
4. **Duplicates of in-task columns**, one with the wrong denominator
   (Pearson 0.9991 with the column it shadows).

**The bar, now recorded in `normalization.py`:** a twin needs a mechanism that
survives a DIRECT TEST — isometric embedding, zero-padding, or a
function-preserving reparameterization — AND a measured reduction that does not
move the association to another architecture variable. **A good fit is not
evidence.** `vendi/n_layers` fit 66 models well and was still rejected for
having no mechanism; `lid/log d` fit worse and was registered anyway. That
inconsistency was caught by review, not by me.

**The one twin that survives** is `representation_sensitivity_scalefree` =
‖h‖²·‖∇_h log P‖², reproduced independently at **ratio 1.00000000** under a
reparameterization that leaves logits bit-identical while the raw value changes
by exactly 1/c². It is reparameterization-invariant, NOT size-free — it trades a
vocabulary association for a width one — and must be reported that way.
(`lipschitz × √L` is also kept, at YELLOW: it flattens depth −0.583 → −0.130 but
its stated mechanism is wrong in form, since the residual norm grows as l^+0.82,
not l^+0.5.)

**A correction to a headline this campaign made:** `matrix_entropy`'s raw column
was demoted on the argument that it is "bounded by log(min(N−1,d)) and therefore
a width proxy". At the shipped ~91-token protocol that is backwards — the cap is
log(N−1), constant in width, so dividing by log(d) INTRODUCES the dependence
(raw ρ(d) +0.033 vs divided −0.483). The headline stays normalised because it is
Wei et al.'s Def 4.3 and reproduces their table; `matrix_entropy_raw` is emitted
for cross-model work.

## Remaining work

1. **Run `run_codex_verification.sh` when quota resets.** Resumable; skips tasks
   already verified against current source. Start with the tasks where the two
   reviewers are most likely to disagree — the last YELLOW, and the normalized
   twins, which have no second-reviewer evidence at all.
2. **Regenerate the archived results.** `results/blme` and
   `results/blme_verified_a` were produced by pre-fix code and no longer match
   the shipped implementation.
3. **Open estimand questions**, recorded but not coded around: steering's sign
   family saturates because `fx_pos > 0` is forced by the residual stream (as
   true of a random-init model as a trained one) — it needs a yardstick
   independent of the model's own task vectors; and neural_collapse's three NC2
   statistics cannot beat their own label-permutation null at the shipped
   5-class × 8-sentence bundle.

---

# Round 12 (gpt-6-astra, max effort) — complete 2026-09-08

**51 of 51 reviewed: 21 GREEN / 20 YELLOW / 10 RED.** All 10 REDs were
`defect_location: CODE` and all 10 are now fixed, each against the reviewer's
own counterexample, each with a regression test validated against the
pre-fix code (a guard that does not fail on the known-bad state proves nothing).

| task | defect | fix | verified |
|---|---|---|---|
| `geometry_correlation_dimension` | C(r) dropped coincident pairs from BOTH numerator and denominator, so the headline was the CONDITIONAL integral (C−c₀)/(1−c₀) | pass the full pair population; radii still drawn from positive distances | μ = ½δ(−2)+½U[0,1], exact D2 = 0: **0.9873 → 0.00296** (nolds 5.35e-17). Cube controls bit-identical |
| `geometry_magnitude` | `np.linalg.solve` (LU) raises only on EXACTLY singular ζ; magnipy's `cho_factor` raises on any non-PD one, then perturbs by 0.01·I | use magnipy's `weights_cholesky` for detection, catching bare `Exception` as it does | invalid solve returned **−497.03** (residual 513) where the fix gives 32.7586; matches live magnipy to 5e-16; pinned duplicate-parity values bit-preserved |
| `geometry_intrinsic_dim` | rescale divided by an arbitrary max — the comment claimed it "only shifts exponents", true ONLY for a power of two — re-rounding Two-NN's exact-equality exclusions | divide by 2^ceil(log2(max)); stop narrowing to float32 | **constant 2.170755946319 across all 65 integer offsets** (was 2.1708 → 3.7654, +73.5%); scale ladder 1e-30…2.4e38 preserved |
| `geometry_svd` | `.double()` promotion ran BEFORE the stored dtype was recorded, so the rank tolerance used float64 eps for every torch input — τ shrank by 2²⁹ | capture the incoming dtype (via `torch.finfo`, so bf16 reports its own eps) before promoting | exact 100×5 fixture: **rank 5 → 3, cond 268435456 → 4.0**, matching the numpy route; erank/PR unmoved |
| `geometry_spectral` | exact positivity admitted SVD backward-error residue as eigenvalues: HDHᵀ's 9th singular value is 4.59e-15, not 0, so 64 numerical positives replaced the true 8 | resolve numerical zeros with the RELATIVE envelope eps·max(m,n)·S₀ before squaring | alpha now identical between D and HDHᵀ at ranks 2,3,8,19,20,32,48,63,64; rank 8 = **1.330045361632** (released WeightWatcher). 0 of 50 gpt2 matrices affected, min ratio 2.3215e7 |
| `geometry_perplexity` | byte ownership split shared character-offset spans EVENLY and labelled it exact; a byte-level BPE reports the same span for every piece of one character | derive ownership from the token PIECES, verified to rebuild the source UTF-8 exactly; withhold BPB when it cannot be established | '😀b' ownership **[3,1,1]**, 2 scored bytes (was 3): BPB 8.1436 → **12.2154** (ref 12.215388); sentence 50 → **49** bytes |
| `geometry_prediction_alignment` | the offloaded collector returned ONE tensor for the whole corpus while the consumer treats each element as a document — every within-document control became corpus-wide | retain per-document lengths and return per-document views of the memmaps | all **9 outputs now identical** across fresh/offload routes; concatenated tensors bit-identical, lengths equal |
| `geometry_vendi_score` | a cache populated for an attention-only consumer is populated but UNUSABLE; the task errored instead of self-collecting, so co-running another task deleted all 9 headlines | require usable hidden states before accepting the cache, else self-collect | attention-only cache now yields **15 headline keys** instead of an error |
| `repe_refusal_direction` | zero-direction layers were dropped with `continue`, and the headline is the last SURVIVING layer — so an earlier block's norm was published under the final layer's name | record a zero direction as 0.0 with undefined unit-direction quantities; depth denominator from the CANDIDATE set | zero-final fixture: **3.0 → 0.0**; pre-fix code published 0.0425 (gpt2) / 0.0471 (llama) |
| `topology_betti_curve` | duplicate rows were not quotiented, so 5 exact copies changed a 42-point circle's Betti numbers | `np.unique(X, axis=0)` before choosing n/k | **duplicate-invariant**; 13-row fixture beta_0 = 1, matching the released route and GUDHI |

## The betti_curve reversal, stated plainly

I reverted this same fix on 2026-09-07 with the condition "not reinstated until
someone runs the released `topnn_framework` loader and shows what it does" —
caution that was right for `geometry_magnitude`, where live magnipy turned out
to PERTURB rather than de-duplicate. That condition is now met by reading the
released source directly, `transformers/betti_calc_parallel.py`, `read_csv`:

    data = np.unique(np.transpose(np.array(data)), axis=0)

Two existing tests encoded BLME's multiplicity semantics rather than the
reference's and were updated, with the reversal recorded in both.

## Sibling sweeps run after these fixes

- LU-as-singularity-detector: population 2, both in `magnitude.py`, both fixed.
- Unregistered hidden-state consumers: population **6**, not 1 — five already
  handled a `None` cache read; `magnitude.py` did not and was the second live
  site of the Vendi defect.
- Divide-by-arbitrary-max: `hubness.py` carries the same false "exact" claim.
  Measured: 0 of 2000 unit rows bit-identical, but **2000 of 2000 identical
  k=10 neighbour lists** — no consumer-visible effect, so the comment was
  corrected and the arithmetic left alone rather than risking pinned parity.
- `_byte_mode` string consumers: 1, fixed — introducing a second exact mode
  silently demoted every exact BPB to a lower bound until it was taught the
  new name.

## Sign-off state

**0 of 51 dual sign-off.** All 51 tasks carry both reviewers' verdicts, and all
51 are voided: today's fixes touched `utils.py`, `common.py` and other shared
dependencies, so under the Pink–Ponk protocol every prior verdict on those
tasks falls. This is the protocol working, not a regression — round 13 must
re-review from the frozen post-fix state.
