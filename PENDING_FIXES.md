# Batched fixes — pending the gpt-5.6-sol pass (2026-09-06)

The tree is FROZEN. Every item below is a confirmed, measured defect that will
be applied in ONE batch once the codex pass completes, then re-reviewed. Editing
now would void codex's in-flight verdicts for all 39 tasks, including the ~30
these findings do not touch.

Ordered by severity. Items marked **MINE** are defects introduced by this
session's own fixes — the majority, which is the honest shape of round 8.

---

## 1. `topology_zigzag_persistence` — the fix never reached production **MINE**

`defaults.yaml` says `persistence_frac: 0.1`; `beyond_benchmarks/configs/tasks.yaml:231-232`
still says `0.3`, and user overrides win. Measured on the merged PRODUCTION
config over 5 models: `betti1_fraction_layers`, `h1_num_bands`,
`h1_max_band_length`, `h1_total_layer_persistence` all **exactly 0.0**;
`_meta_h1_family_is_identically_zero` True everywhere; **9 of 14 columns sd = 0**.

FIX: change the study config too. Then re-measure that the H1 family is alive.

LESSON, and it is the reason this file exists: the identical defect was
diagnosed in `dynamics_fractal` the same morning, written up, and then committed
again hours later. A default is not a fix when an override shadows it.

## 2. `repe_task_vectors` — the sixth reference case **MINE**

`representation_engineering.py:68` hooks `layers[-1]`, against the same
`repe/rep_reading_pipeline.py:22-28` line that `repe_concept_separability` was
reverted for — 500 lines away in the same file. With the hook disabled:
`mean_vector_norm_normalized` +25.8% (gpt2) / -30.3% (pythia-70m) / +63.1%
(distilgpt2); `pair_direction_cos_mean` SIGN FLIP on distilgpt2 (+0.006941 ->
-0.029364).

FIX: revert to the reference read. CHECK FIRST whether the seventh candidate is
real — `geometry_cknna` vs `minyoungg/platonic-rep` `extract_features.py:82`,
which stacks raw `llm_output["hidden_states"]`.

## 3. `cache.py:485` — the 11th hook leak, in the most shared path

`ModelOutputCache._run_forward_pass` removes its hook with a bare post-loop
`if _hook is not None: _hook.remove()`, outside any try/finally. Raise on the
second forward during `populate()` and `layers[-1]._forward_hooks` goes
**0 -> 1 permanently**. `core.py` populates ONE shared cache per run, so every
later task pays it. Exposed via `geometry_procrustes_linearity`,
`geometry_trajectory_curvature`, `geometry_vendi_score`.

FIX: try/finally. AND fix the discovery gap that hides it — see item 8.

## 4. Two provenance flags that report the opposite of the truth **MINE**

* `dynamics_coe`: `_meta_last_entry_is_block_output` is True on 3/3 models while
  the headline chain's deepest entry is `ln_f`. When the chain was reverted to
  the reference, the flag was left tracking whether the HOOK FIRED rather than
  whether the HEADLINE USED IT.
* `geometry_schatten`: structurally always False on the cache route — the
  post-norm `get_hidden_states` call the split fix added resets
  `_last_block_substituted` before it is read.

These are worse than a missing flag. A missing flag says nothing; these assert
something false about which representation produced the number.

## 5. `geometry_mahalanobis` — split 1 duplicates split 0 **MINE**

`np.random.default_rng(seed)` is re-seeded inside the split loop, so its first
permutation reproduces `_perm` exactly (verified at seeds 42/0/7; 4 distinct fit
signatures across 5 calls). `n_splits: 5` averages FOUR splits, one twice.

Also: the headline AUROC got **noisier**, sd 0.0241 -> 0.0358 (**+49%**), because
it is computed on distances POOLED across splits rather than averaged over them.
FIX both. Pairing survives and is fine.

## 6. `geometry_magnitude` — the guard is one-sided **MINE**

Refuses an undersized cache, silently accepts an oversized one. Production
`cache_num_samples` = 200 gives `magnitude_area` **5.689075** against the
self-collected 5.097373 (**+11.6%**), and `_meta_magnitude_n_points` reads 100 in
every case, so the provenance added to catch this cannot.

FIX: refuse ANY cache whose document count differs from `num_samples`, and make
the provenance report the true point count.

## 7. `geometry_mp_bulk_deviation` — wrong knob, revert **MINE**

At `num_samples: 200` the corpus supplies **23,272 tokens**, so `max_tokens`
20000 -> 32000 changed a cap that no longer binds. yi1.5-34b (needs 28,673) is
**still NaN**. Cost: 72 of 80 columns moved (gamma +14.06%) for **+0.22%** on the
`_g25` family it targeted.

FIX: revert to 20000 and solve it where it lives — raise `num_samples`, or
accept the NaN and document which models the width-comparable family covers.

## 8. Both guards, again

**`test_post_final_norm_declaration.py`: 15 known evasions.** Ways 8-15:
`hs: tuple = ...` (AnnAssign), `keep = hs = ...` (multi-target),
`list(reversed(...))` + `[0]`, `first, *rest = ...` (Starred),
**`out["hidden_states"]`** (the idiom this repo quotes from RepE),
`... if x else []` (IfExp), `... or []` (BoolOp), a closure over an
enclosing-scope alias, and a dict comprehension + `hs[len(hs)-1]`.
Also: per-function marker scope lets ONE marker blanket-cover a long function.

**`test_forward_hook_leaks.py`: three structural gaps.**
(a) discovery maps `inspect.getfile(cls)` over task classes, so a task hooking
only through `cache.py` or `geometry/utils.py` is invisible;
(b) `_hook_counts` walks only top-level `get_layers(model)` modules, so hooks on
`layers[0].mlp` or the embedding are invisible — **9 repo sites** target those;
(c) 6 of 29 discovered tasks register ZERO hooks under the test's conditions, so
their coverage is vacuous — including one of the ten wrapped last round.

FIX: scan shared helper modules; count hooks recursively over `model.modules()`;
assert each discovered task actually registered at least one hook, or skip it
loudly rather than passing silently.

## 9. Smaller, confirmed

* `repe_steering_effectiveness`: the "32 draws lands at 2.56 and is stable"
  comment is **still there verbatim** — recorded as fixed, was not. Measured:
  32 draws -> 0.9523 +/- 0.2285 (CV 24.0%), and the MEAN drifts monotonically
  1.0573 / 0.9523 / 0.8284 at 8/32/128, so the shipped value is ~15% biased
  high, not merely noisy.
* `repe_task_vectors`: `mean(layer_task_vector_norms)` recovers the demoted
  `_meta_mean_vector_norm` identically, and `is_excluded_predictor` returns
  False for it. The sqrt(d_model) normalization does essentially nothing
  (pearson 0.9992 with the raw over 22 models).
* `geometry_phd_dimension`: a fifth drop channel sits UPSTREAM of the bucket
  invariant — 20 blank docs in 100 give `seen = 80` with buckets summing to 80.
  Also 32% of contributing docs are two-point "fits"; the comment claims 12 max,
  the measured max is 8.
* `geometry_isoscore`: the 40M-cell cap in `_isoscore_null_floor` now binds —
  0.985425757 reported at both N=1000 and N=2000 against true 0.994294470 /
  0.996808505.
* `topology_homology`: `*_h1_fine_persistence_frac` exactly 0.0 on 3/3 models,
  still feature-eligible.
* `geometry_matrix_entropy`: the per-layer PROFILE emits no provenance flag,
  though both sibling split-flag tasks do.
* `task_metadata.py`: still says `contextualization`'s "Last block now hooked"
  and quotes the retracted +1513% — describing a REVERTED change as current.
* `repe_refusal_direction`: reference prunes the last 20% of layers
  (`prune_layer_percentage=0.20`); BLME's headline is the deepest layer.

## Not a defect — recorded so it is not re-litigated

`repe_refusal_direction`'s hook is CORRECT. `andyrdt/refusal_direction`'s
`generate_directions.py` uses forward PRE-hooks on `block_modules` caching
`input[0]` — the pre-norm residual stream — and never reads
`hidden_states[-1]`. The raw-tuple clause does not apply.

---

# From the gpt-5.6-sol pass (in flight) — four REDs Opus marked GREEN

This is what the two-reviewer protocol is for. All four carry
`defect_location: CODE`, so none is axis leakage, and three are the
scale-invariance class I had recorded as closed — `test_scale_invariance.py`
pins specific kernels and evidently does not reach these.

## 10. `geometry_cka` — linear CKA is not isotropically scale-invariant **RED**

Matches the released fixture at 0.557613279271 vs 0.557613256612 (rel 4.06e-08),
but rescaling one fixed pair of finite, nonconstant matrices from max magnitude
1 to **1e-13** takes BLME from 0.408370281528 to **NaN**, while the defining
formula still gives 0.408370278617. Kornblith Eq. 4 is exactly invariant to a
positive isotropic rescale, so this is a value-changing violation on the
paper's own domain.

**STATUS: CONFIRMED independently.** `_linear_cka` on a seeded (200, 24) pair,
column-centered, float32: 0.3499494344 at scale 1, 0.3499493650 at 1e-6, then
**NaN at 1e-13 and NaN at 1e-20**.

FIX: `safe_unit_rows_torch`/max-factored arithmetic through `_linear_cka`, and
add CKA to the scale ladder — it is not there.

## 11. `geometry_intrinsic_dim` — Two-NN is scale-invariant, the estimator is not **RED**

Every `mu = r2/r1` is unchanged by a global rescale, yet the estimator returns
**24.233986884 at scale 1, 16.927948590 at 1e-22 (-30.15%), and NaN at 1e-23
and 2e19**. Power-of-two-rescaled calculations on the same represented
coordinates give 24.233985027 / 24.233985278, so the information is present and
the arithmetic loses it. Ordinary-scale slope still agrees with the released
Ansuini formula to 2.67e-15.

**STATUS: REPRODUCED AND FIXED (2026-09-07).** The non-reproduction below
was MY error: every check ran on a GPU host, so `_compute_id` took the
CUDA branch and the sklearn CPU fallback -- where the defect lives -- was
never exercised. With the branch forced, the reported 16.927948590 at
1e-22 reproduces to nine digits and four more scales return NaN. Fixed by
rescaling to unit max at the top of `_compute_id`. Original note kept
below for the record:

~~STATUS: NOT REPRODUCED — do not fix until it is.~~ Checked independently on
four fixtures (2000x16, 2000x64, 5000x128, 20000x768 Gaussians, unit max-abs so
the rescale is exact), driving `_compute_id` directly over the same ladder
1e-23 ... 2e19:

```
n= 2000 d=  16   ID@1= 14.7428   all finite   max rel dev 6.8e-09
n= 2000 d=  64   ID@1= 37.7225   all finite   max rel dev 1.0e-08
n= 5000 d= 128   ID@1= 58.5636   all finite   max rel dev 1.0e-08
n=20000 d= 768   ID@1=175.0684   all finite   max rel dev 3.7e-09
```

No NaN anywhere and nothing resembling -30.15%. None of these clouds gives the
ID ~24.23 the report quotes at scale 1, so the fixture is in a regime these do
not reach — plausibly duplicate-heavy data interacting with the `mu <= 1`
pre-filter, which is exactly where this estimator's documented deviation lives.

The claim is specific and may well hold on that input. ASK FOR THE FIXTURE
before changing anything: on this campaign's own standard a measurement has to
be reproducible to act on, and that standard has to apply to measurements that
arrive from a reviewer as much as to my own.

## 12. `geometry_correlation_dimension` — wrong on measures with atoms **RED**

For `mu = 0.04*delta_0 + 0.96*U([3,4]^3)` the exact small-r correlation integral
has a positive constant floor `0.04^2`, so **D2 = 0**. A fresh end-to-end
`evaluate()` at N=1000 returned **2.8835033699291728** with `status: ok`,
`zero_pair_frac = 0.00156`, fallback `atoms_removed`. Below the minimum positive
separation, C(r) is constant at 0.00156 with slope 2.6e-16 — the estimator sees
the floor and reports a dimension anyway. Pure one- and two-atom clouds return
an **error** rather than 0.

FIX: this is an estimand question, not just arithmetic — decide what the task
reports on a measure with atoms, and make the atom-removal fallback disclose
rather than silently substitute.

## 13. `geometry_magnitude` — fails the paper's own twin example **RED**

On distinct point sets BLME is faithful: 3 seeded clouds x 97 scales, at most
1.421085e-14 absolute / 8.978436e-16 relative against released Magnipy Cholesky.
But the paper's explicit example requires `X = {(0),(1)}` and `Z = X` plus a
DUPLICATE of 0 to agree. Magnipy collapses Z from 3 rows to 2 and matches
exactly (0.000e+00); BLME keeps the singular 3x3 system and at `t = ln(19)`
returns **1.886569920263 against 1.900000000000 (-0.706846%)**.

FIX: de-duplicate points before forming the similarity matrix, matching the
reference. This is separate from item 6 (the one-sided cache guard) and both
apply.

---

**Consequence for the goal.** These four tasks cannot reach dual sign-off until
fixed and re-reviewed by both. Opus returned GREEN on all four; codex returned
RED with a measurement each. On the campaign's own standard the measurement
wins, and Opus's GREEN on these is now suspect for the same reason my sweep was:
a check that did not probe the regime where the property breaks.

---

# The gpt-5.6-sol pass completed — 38 of 39, and the asymmetry is the finding

```
OPUS   50 GREEN,  1 YELLOW
CODEX  29 GREEN, 12 RED, 9 YELLOW
DUAL GREEN: 29 / 51
DISAGREEMENTS: 20 — every one Opus GREEN, codex non-GREEN. None the other way.
AXIS LEAKAGE: 0 (all 21 non-GREENs carry defect_location: CODE)
```

A 20-0 split in one direction is not noise. Either codex is systematically
over-strict, or Opus is systematically lax. I adjudicated four of the REDs
myself, and the answer is mostly the second.

## Adjudicated independently

| task | codex claim | my check |
|---|---|---|
| `geometry_cka` | NaN at max-magnitude 1e-13 where Kornblith Eq.4 is exactly scale-invariant | **CONFIRMED** — 0.3499494 at 1, 0.3499494 at 1e-6, **NaN** at 1e-13 and 1e-20 |
| `geometry_weight_norms` | stable rank returns 0.0 at 1e-25 and 10.0 at 1e30 | **CONFIRMED** — exactly those values; the estimand is a ratio of squared norms and is scale-invariant by construction |
| `topology_homology` | lifetimes collapse to 0 with 0 H1 bars at 2^±100 | **CONFIRMED** — mean_h0 1.618e+00 with 9 bars at scale 1; **0.0 with 0 bars** at both 2^-100 and 2^+100, through the task's own float32 cast |
| `geometry_intrinsic_dim` | Two-NN -30.15% at 1e-22, NaN at 1e-23 | **NOT REPRODUCED** on four fixtures (max rel dev 1e-8, no NaN). Needs their fixture. |

**My first two spot-checks were too shallow and I nearly recorded these as
over-strict.** I tested `weight_norms` only at scale 1 — where the discrepancy
is float32 rounding at 7e-8 — and tested `homology` by calling `ripser` directly
on float64, bypassing the float32 cast the task performs. Both look clean that
way. The defect is entirely in the regime the reviewer named and I did not
probe. That is the same error as the sweep: checking the place the property
holds rather than the place it breaks.

## What this means for the goal

**Opus returned GREEN on all four**, including the three now confirmed by direct
measurement. Opus's scale-invariance checking has the same blind spot mine did:
it exercises named kernels at ordinary magnitudes. The dual-GREEN count of 29
should therefore be read as an UPPER bound on what is actually verified, not a
floor — some of those 29 may be GREEN on both sides for the same reason.

## Revised priority

The dominant defect class across codex's 12 REDs is **an invariance the
reference guarantees, violated on the reference's own domain** — scale for
`cka` / `weight_norms` / `homology` / `intrinsic_dim` / `spectral`, translation
for `vendi_score`, permutation for `rsa`, isometry for `trajectory_curvature`.
`tests/test_scale_invariance.py` pins named kernels one at a time; it needs to
become a PROPERTY SWEEP over every task's public numeric outputs, the same move
that turned the post-final-norm patching into a population count.

Two are plain formula defects and are separate:
* `geometry_representation_sensitivity` — the value path omits a term of
  `grad_h log p(y|h) = s[W_y - sum_k softmax(s(Wh+b))_k W_k]`.
* `repe_task_vectors` — reading-direction definition vs Zou et al. Appendix B.3.

---

## 14. `repe_steering_effectiveness` — unconverged estimator (NOT a correction; a redesign)

**Status: OPEN. Do not quote `steering_beats_random` as a directional claim.**

The control-seed defect is fixed (2026-09-07): the configured `seed` now reaches
the control RNG, which it previously did not (12 seeds → sd exactly 0.0000).

The remaining problem is not a bug to correct. The statistic does not converge:

| draws | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ratio | 0.766340 | 0.753817 | 1.201288 | 1.052713 | 0.704796 | 0.801071 |

It crosses 1.0 **three times**, so 32 draws say "beats random" and 64/128 say
"loses". Mechanism (isolated by Opus round 9): it is the **median of 5 layer
ratios** sharing identical numerators, in which layer 3 is pinned at exactly
0.0 by the `max(0, ·)` clamp — a clamp artefact, not a measurement — and layer
0's denominator (5.5-9.8 against the others' 87-247) swings which value lands
in the middle. Increasing the draw count cannot fix a median whose middle
element is decided by a clamped zero.

**What a fix requires:** drop the clamped layer from the median (or replace the
median over layers with a per-layer reported profile), then re-establish
convergence against draw count. That is an estimator redesign and it changes
the meaning of the key, so it is deliberately not being done under a
correctness pass.

---

## 15. Scale-fragility population: 34 unprotected square-before-sum functions

**Status: OPEN, and it is a POPULATION, not a list of bugs to sample.**

Round 9 produced scale-invariance REDs on `geometry_collapse`,
`geometry_contextualization`, `geometry_effective_depth`,
`geometry_matrix_entropy`, `geometry_lipschitz`, `geometry_neural_collapse`,
`dynamics_interpolation` and `dynamics_perturbation_growth` — eight tasks, one
defect class, reported one task at a time. That is the exact shape the `ln_f`
defect had before anyone counted it (24 of 39 sites unfixed after eleven
"fixes"), so it is being counted now instead of sampled.

An AST sweep for functions that square before summing (`.norm()`, `**2 .sum()`,
`cosine_similarity`, `pdist`/`cdist`) with **no** scale factoring anywhere in
the function body finds **34**, in 19 files:

```
dynamics/stability.py, dynamics/trajectories.py (x2), geometry/collapse.py,
geometry/contextualization.py (x2), geometry/correlation_dimension.py (x3),
geometry/intrinsic_dim.py, geometry/lid.py, geometry/lipschitz.py,
geometry/magnitude.py, geometry/phd_dimension.py,
geometry/procrustes_linearity.py, geometry/rmt_bulk.py (x3),
geometry/rsa.py (x2), geometry/trajectory_curvature.py, geometry/vendi.py,
interpretability/activation_kurtosis.py, interpretability/attention_graph.py,
interpretability/superposition.py, representation_engineering.py (x2),
topology/persistence_landscape.py
```

The sweep is deliberately conservative and **contains false positives**: some
of these are correct as written (`_pairwise_distance_matrix` is *supposed* to
scale linearly; `rmt_bulk`'s `_mp_cdf` works on O(1) eigenvalue ratios; `lid`
already factors scale through a verified escalation path). The list is a
*population to adjudicate*, not a defect count.

**Why a generic harness is not the answer:** the invariant differs per
function. Magnitude scales, D2 does not, a distance matrix scales linearly and
a cosine does not. So each entry needs its own stated invariant before it can
be tested — which is the work, and it is why this is filed rather than
batch-patched.

**How to close it:** for each of the 34, state the invariant in the docstring
(scale-invariant / scale-equivariant / scale-dependent-by-definition), then add
one ladder test per invariant class over 1e-30 … 2.4e38. Sites already carrying
`safe_unit_rows` / `safe_norm` / `pow2_rescale` / `ldexp` are excluded by
construction, so the list shrinks as the helpers spread.

### 15b. Sub-population: centring performed in float32

Found twice in one hour, in two different files, by two different routes:

- `matrix_entropy.py` — cast to float32 **before** subtracting the mean, so a
  common offset was quantised away before centring could cancel it: entropy
  0.467692434788 → 0.092731282115 at +1e6 (**−80.17%**), NaN at +1e8. Wei et al.
  centre, so the statistic is translation-invariant by definition.
- `collapse.py` — centred in float32, and the SUBTRACTION ITSELF overflows near
  the top of the range (a unit-max cloud at 2.4e38 has centred entries past
  3.4e38), returning inf. Wiring `safe_svdvals` did **not** fix this, because
  the overflow was upstream of the SVD. This is worth remembering: the helper
  was correct, the call site was correct, and the defect was in the line before.

Both now promote to double, centre, then rescale, then come back down — one
ordering that satisfies the offset case and the denormal case together, so the
two earlier fixes no longer have to be traded against each other.

A grep finds **24 further centring sites** in 12 files (`consistency.py`,
`mutual_info.py`, `isotropy.py`, `unembedding.py`,
`representation_engineering.py`, `activation_sinks.py`,
`procrustes_linearity.py` ×4, `rmt_bulk.py` ×4, `schatten.py`, `zigzag.py`,
`bias.py`). Each needs the same question asked — *is the statistic
translation-invariant by definition, and does the subtraction survive the
scale ladder?* — before it is either patched or documented as scale-dependent
on purpose. Not batch-patched, for the reason in item 15: the invariant
differs per site.

---

## 16. Round-9 REDs queued behind the full-suite run (2026-09-07)

Read and triaged; deliberately NOT patched while the full suite is mid-run,
because editing modules that pytest has not yet imported produces failures that
belong to neither the old code nor the new.

- **`dynamics_interpolation`** — SLERP agrees with the released formula to
  4.392e-15 on ordinary input, but the homogeneity transform `h→c·h, W→W/c`
  (which preserves endpoint logits to 1.073e-7) takes the required gap
  0.131697065 to **0.0 at c=1e-25**, 5.960e-8 at 1e-13, and NaN at 1e25. Also
  flags an omitted native Gemma-2 softcap. Same class as the fixes already
  landed today. `trajectories.py::_slerp` is on the item-15 sweep list.
- **`dynamics_perturbation_growth`** — Li et al. Eq. 9 is homogeneous of degree
  zero; `_growth_profile` returns the exact −0.693147180560 at 1e-13/1/1e13 but
  **emits no metric at all** at 1e-30, 1e-25, 1e25 and 2.406e38. Separately a
  common +1e6 translation moves an exact 0.139761942549 to 0.110812307057
  (**−20.71%**) — the translation half is the same defect fixed in
  `matrix_entropy` today.
- **`geometry_neural_collapse`** — NC1 is invariant under appending zero
  coordinates and under global rescaling. BLME gives 0.6666666666666666 at
  d=47 and **0.0** after appending ONE zero coordinate at d=48; that d=48 cloud
  then gives 0.0 at six of seven scales and 0.6666666666666669 at 1e25. Points
  at the top-(K−1) eigenvalue selection, which the file already rewrote once to
  avoid a pinv blowup — so check what that earlier fix was protecting before
  changing it (the `matrix_entropy` lesson).
- **`geometry_lipschitz`** — **REFUTED. Do not "fix" this.** The RED reports
  that the source documents `L_l = ||Δh|| / (||h|| + ε)` over all n rows while
  the implementation drops zero-norm rows and divides by `||h||` alone. Both
  halves are true, and the divergence is **deliberate, documented, and already
  adjudicated** at `lipschitz.py:270-283`: the paper's literal ε destroys the
  scale invariance the same paper claims for the quantity (a 1e-10 rescale of
  an unchanged representation moves the ratio 0.8679 → 0.0554), while the
  ε-free form is exactly constant. The paper's number is emitted under its own
  name as `_meta_relative_change_mean_paper_eps`, so both exist.

  The divergence being documented is what distinguishes this from the hubness
  zero-row note and the Tomašev/Gini citation corrected today: those were
  comments describing code that did not exist, this is a comment describing
  code that does, together with the reason. On the two-axis standard this is a
  `comparable`/`paper` matter, not a `verdict` one.

  **I queued this as "decide which is intended" before reading the file** —
  i.e. I nearly manufactured work by trusting a reviewer over the code's own
  record. The one real follow-up is in the PAPER repo, not here:
  `sections/metrics_geometry.tex:101` still prints the ε form, and the code
  comment has been saying it needs correcting since 2026-09-05.

### 16b. Two more of the same class (verdicts landed 2026-09-07 ~04:00)

- **`geometry_prediction_alignment`** — the margin is exactly 4/3 by
  construction on a centred four-row simplex. Returns 1.3333332538604736
  (5.96e-08) at 1e-30 … 1e25 and **NaN at 2.406159650500376e38**. The reviewer
  states the repair explicitly: *factor out the finite maximum before the
  existing centring*. That is verbatim the fix applied to `matrix_entropy`,
  `collapse` and `contextualization` today.
- **`geometry_representation_sensitivity`** — the repaired logit-scale path
  agrees with an independent score-gradient calculation to 9.225e-08, but a
  binary anchor at weight magnitude 4.156000412762293e-39 returns
  2.159042718946516e-78 instead of 1.727233943088035e-77 — a ratio of **exactly
  1/8**, which is a lost power of two, not noise — and NaN at 2.406e38.

**This makes FOUR tasks with the identical centring/reduction defect**
(`matrix_entropy`, `collapse`, `contextualization`, `prediction_alignment`),
plus `representation_sensitivity`, `perturbation_growth` and
`dynamics_interpolation` in the same family. Per item 15 this is now firmly a
population: it should be closed by making the safe form the DEFAULT — a shared
`centre_and_rescale()` helper that promotes, centres, power-of-two rescales and
returns — so that writing the unsafe form requires deliberately not calling it,
rather than by patching the seventh site and waiting for the eighth.

### 16c. The class now spans NINE tasks — treat the count, not the members

Adding the two YELLOWs (correctly filed: both report the implementation as
faithful on ordinary inputs — `dynamics_coe` matches the released formulas to
2.13e-14 and its headline to an exact 0.000e+00; `geometry_cknna` matches the
clean dcd76ba checkout bit-for-bit in 240/240 kernel cases, 40/40 preprocessing
tensors and 20/20 pipeline scores — and put the defect on the implementation
axis only because it is genuinely value-changing at scale, **not** axis
leakage):

| task | symptom |
|---|---|
| `matrix_entropy` | −80.17% at +1e6 — **FIXED** |
| `collapse` | erank NaN at 2.4e38 — **FIXED** |
| `contextualization` | 0.0/NaN at 1e-30 and 1e25 — **FIXED** |
| `prediction_alignment` | NaN at 2.4e38 — **FIXED via helper** |
| `representation_sensitivity` | ratio exactly 1/8 at 4.156e-39; NaN at 2.4e38 |
| `perturbation_growth` | no metric at 1e-30/1e-25/1e25/2.4e38; −20.71% at +1e6 |
| `dynamics_interpolation` | gap 0.1317 → 0.0 at c=1e-25; NaN at c=1e25 |
| `dynamics_coe` | NaN at 5 of 7 rescalings; 0.5/0.5 vs 3.14e-14/0.4938 |
| `geometry_cknna` | +23.484% at 2.4e38 vs a stable 0.129550017 |

Nine tasks, one class, reported across two review rounds one task at a time —
and four of the nine had been individually "fixed" for a scale bug before.
This is the `ln_f` shape exactly (24 of 39 sites unfixed after eleven fixes).

**The remaining five are not five tickets.** They close by wiring
`centre_and_rescale` / `safe_unit_rows` / `safe_svdvals` at each site and then
adding ONE ladder test per invariant class, so the unsafe form becomes the one
that takes extra effort to write. The helper is in place and pinned
(`tests/tasks/test_centre_and_rescale.py`, 18 tests, each verified to fail on
the unsafe orderings); adopting it cannot move a shipped number because the
rescale is gated to outside 1e-6..1e6 and is a pure `ldexp`.

## 17. Verdicts 41-44 (2026-09-07, queued behind the definitive suite run)

Codex round 9 now stands at 27 RED / 6 YELLOW / 11 GREEN of 44. Two shapes in
the newest batch, both actionable, neither patched yet because the definitive
full-suite run is in flight.

**`geometry_svd` — a THIRD-CLAUSE case, not a scale bug.** `isotropy.py:520-540`
replaces the cited raw-HF tuple's final post-`ln_f` state with the pre-norm
last-block output and promotes the substituted entry to the headline, while the
released Ethayarajh extractor scores its tuple's layer 12 — the post-norm entry.
gpt2 `avg_cosine_similarity` 0.984922922817 -> 0.834701676247 (**−15.25%**).

This is the rule the campaign already settled seven times: the per-layer profile
wants homogeneity, a single-layer estimand may want post-norm, **and the
released reference wins**. If it holds here, `geometry_svd` is the eighth task
needing the post-norm entry, and the fix is a `replace_final_norm=False` at that
site plus a declaration — NOT a new argument about which representation is
"better". Verify against the released extractor before changing anything; the
−15.25% is large enough that getting the direction wrong is expensive.

**`repe_task_vectors` — the scale class again.** Extraction and
`v = mean(h_pos) − mean(h_neg)` now agree with the released reader at **maximum
absolute error 0.0**, so the port is faithful; the reported summaries are not.
A scale-equivariant norm whose exact values are [9.2865494e-28, 1.8573099e-27]
at scale 1e-25 is returned as **[0.0, 0.0]**, and [9.2865494e22, 1.8573099e23]
at 1e25 becomes [Inf, ...]. Same `.norm()`-squares-in-float32 defect closed in
nine tasks today; fix with the same float32-first structure, since this task's
value rests on that 0.0 agreement with the released reader.

## 18. Round 9 COMPLETE (51/51). Topology cluster: translation invariance + duplicate quotienting

Final: **31 RED / 8 YELLOW / 11 GREEN** of 50 written (`geometry_categories`
timed out at 90 min, twice, and still has no verdict — re-run it alone).
**Zero axis leakage** across the whole round: no non-GREEN verdict cites a
CONFIG defect location.

My same-day topology fixes are confirmed by this round:
- `zigzag` — max absolute error **0.0** for all 33 beta values and all 32
  forward-adjacent Z values against the authors' 233586-effective-H1-bar
  artifact, and 0 mismatches on 94 trajectories. YELLOW only for a tie boundary.
- `betti_curve` — matched an independent GF(2) clique-complex derivation on 9/9
  analytic and adversarial clouds (circle (1,1), figure-eight (1,2), two
  disjoint circles (2,2)).
- `homology` — the rescale-and-restore path now holds all 7 scale rungs (9 H1
  bars at every rung, worst normalised disagreement with GUDHI 1.49e-7).

**New: TRANSLATION invariance in the topology cluster.** Vietoris-Rips
persistence is exactly translation-invariant, and it is not implemented that way:
- `topology_homology` — translating the seeded cloud by +1e6 gives **14 H1 bars,
  mean H1 0.07011418**, against 9 bars and 0.1... at offset 0.
- `topology_persistence_entropy` — a radius-1e-5 circle gives 19 H0 bars and
  4.247892868881332 bits at offset 0, and **0 bars / 0.0 bits at +1e3 and
  +1e6**, while an independent distance-matrix evaluation keeps 19 bars and
  4.2479...

Cause is the same everywhere: `pdist` on a float32 cloud sitting far from the
origin cancels catastrophically, so genuinely distinct points collapse.
**`centre_and_rescale` solves this directly** — Rips persistence is
translation-invariant, so subtracting the centroid is mathematically a no-op
and numerically restores every digit. This is the helper's centre=True path,
already pinned by `test_centring_is_exact_under_a_common_offset`.

**`topology_betti_curve` — duplicate quotienting, same as `magnitude`.** The
implementation does not quotient duplicate rows the way the paper's point-set X
and the released loader do; adding 5 exact copies to a 42-point circle changes
the answer. `magnitude` was fixed for exactly this on 2026-09-07 (|{0,1}| must
equal |{0,0,1}|). Reuse that reasoning: a metric space is a SET.

## 19. Round 9 CLOSED (51/51): 32 RED / 8 YELLOW / 11 GREEN

`geometry_categories` finished in 18.9 min alone (it had timed out twice at
90 min under the 16-worker sweep — starvation, not difficulty).

**Queued, not patched** (the final clean suite is mid-flight, and after today's
five self-inflicted failures I am not editing modules while it runs):

- **`geometry_categories`** — kernels match independent definitions to
  6.505e-09 at every rung and exact axis anchors hold at 1.000000000, but at
  2.406159650500376e38 the relation arithmetic gives NaN/0.0/1.0 where the
  definition gives 1.0/1.0/1.0. All six lower rungs exact. Same class as the
  nine already closed.

Still unadjudicated from the completed round, all likely the same class or the
`.norm()`-squares family:
`geometry_spectral`, `geometry_trajectory_curvature`, `geometry_weight_norms`,
`repe_concept_separability`, `repe_refusal_direction`,
`topology_persistence_landscape`, `geometry_unembedding` (YELLOW),
`geometry_svd` (third-clause, see item 17), `repe_task_vectors` (item 17).

**Approach for this batch — do NOT repeat the one-at-a-time pattern.** Nine
tasks of one class were closed today by fixing them individually, and the
tenth (`categories`) arrived the same evening. The durable move is the one
already started: `centre_and_rescale` exists and is pinned, `safe_unit_rows` /
`safe_svdvals` / `safe_norm` exist. Wire the remaining sites to them, then add
one ladder test per invariant class, so the unsafe form is what takes effort to
write.

**And carry today's two hard-won cautions into it:**
1. Where a task's verdict rests on bit-exact parity with a released reference
   (`cknna`, `coe`, `schatten`, `task_vectors`), structure the repair
   **float32-first** — a strictly more accurate computation is a REGRESSION
   there.
2. Before implementing any claim about what a reference implementation does,
   **run the reference**. `magnitude` cost a revert, a withdrawn doc edit, and
   a broken pinned parity because two reviewers and I agreed without doing so.
