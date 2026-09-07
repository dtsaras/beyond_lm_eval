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
