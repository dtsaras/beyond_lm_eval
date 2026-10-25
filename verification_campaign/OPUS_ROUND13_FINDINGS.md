# Opus 5 — round 13 cross-task measurement sweeps

Run concurrently with the gpt-6-astra pass, against the same frozen baseline
(`round13_baseline.json`, 51 tasks / 54 source files). These are POPULATION
sweeps: each covers all 51 tasks at once, which is the move the campaign
adopted after the post-final-norm defect was found and "fixed" eleven times
while 24 of 39 call sites had never been touched.

All measurements below are on pretrained gpt2 (local snapshot), CPU, float32,
with the default WikiText corpus.

## Sweep 1 — collinear public columns. FIRST RESULT WAS A FIXTURE ARTEFACT.

On a 16-document near-identical corpus the sweep reported 14 tasks and **352**
bit-identical public column pairs, topology dominating (homology 156,
persistence_landscape 84). That result is worthless: the value histogram shows
`0.0` appearing 18 times among homology's 33 public keys. On a degenerate
corpus most homology features are genuinely zero, and `0.0 == 0.0` is
degeneracy, not collinearity.

Re-run on 80 real WikiText documents at `num_samples=40`, counting only
NONZERO duplicates: **49 genuine pairs across the 14 tasks**. Most are
corpus-dependent coincidences that carry no defect (an argmin and an argmax
landing on the same layer, e.g. `procrustes_linearity_first ==
procrustes_linearity_min`, or `cknna`'s `max_adjacent == max_offdiag`). Two
classes are structural and are reported below.

**Recording this because the first number was the more alarming one.** A sweep
whose fixture is degenerate produces a large, confident, meaningless finding.

## Finding A — a feature column that is the sample count in disguise

`topology_persistence_entropy` emits `layer_{i}_num_features_h0` for every
layer. Measured across three sample sizes:

| num_samples | layer_0 | layer_6 | layer_11 |
|---|---|---|---|
| 15 | 14 | 14 | 14 |
| 25 | 24 | 24 | 24 |
| 40 | 39 | 39 | 39 |

It is exactly `N - 1` at every N and identical across all 12 layers, which is
forced: Rips H0 on N points has N components at radius 0 and N-1 finite bars,
whatever the geometry. The column carries no information about the model, is
constant across layers, and becomes a ZERO-VARIANCE feature once the sample
size is fixed. `topology_persistence_landscape`'s
`layer_{i}_h0_meta_landscape_n_bars` is the same quantity (also 39).

AXIS: this does not make the persistence computation unfaithful, so it belongs
on `comparable`/`confounds`, NOT on `verdict`.

## Finding B — 30 metadata keys that are not behind the `_meta_` prefix

The convention is `startswith("_meta_")`. `topology_persistence_landscape`
emits 30 keys with "meta" embedded MID-NAME, so every one of them counts as a
public feature:

    layer_0_h0_meta_prescale                    (= 1.0, a constant)
    layer_0_h0_meta_last_entry_is_block_output  (a provenance flag)
    layer_0_h0_meta_landscape_max_status
    ... 27 more

Swept across all 51 tasks: **the population is exactly 1 task, 30 keys.** No
other task has a mis-prefixed metadata key.

AXIS: `comparable`/`confounds`, not `verdict`.

## Sweep 2 — literal `_meta_` provenance flags. CLEAN.

The repo's most-repeated failure is a fix that is present in the source and has
no effect on the output, and its detector is a `_meta_` flag DERIVED from what
actually happened rather than written as a literal. AST sweep over `src/blme`:

- **206** `_meta_` dict entries whose value is a derived expression.
- **2** whose value is a boolean literal, both `_meta_last_entry_is_block_output
  = False`, both correct: `coe.py:415` and `contextualization.py:527` never pass
  `replace_final_norm=True` (grep count 0 in both files), the constant is
  documented as deliberate, and `coe.py` carries a separately derived
  `_meta_blockout_hook_fired = bool(_used_hook[0])` beside it.

This sweep RESULT IS A PASS, and is worth recording as such: the class has been
addressed at the population level, not sampled.

## Verified through the real `evaluate()` on gpt2, not at helper level

| repair | observable signature | measured |
|---|---|---|
| correlation_dimension | `_meta_correlation_integral_n_pairs == n(n-1)/2` | 5356 == 104*103/2, **coincident fraction 0.52% — the defect was live on real data** |
| spectral | `_meta_alpha_official_n_svd_zeros` | 0 on gpt2; `avg_alpha_official` bit-identical at 3.904538608347353 |
| perplexity | `_meta_ppl_byte_attribution` | `token_pieces` |
| svd | `cond_number`, rank | 6761.820347094235, rank 96, finite |

## Two defects found in my OWN round-12 repairs by this audit

1. **isotropy / geometry_svd, bf16 rank tolerance.** Capturing the stored dtype
   made the primary envelope `sigma_1 * max(m,n) * eps` use bf16's eps; at width
   768 the factor is 6.0, so the tolerance EXCEEDS sigma_1 and every singular
   value is discarded: measured **rank 0, cond inf**. The file's own comment had
   predicted exactly this for the source-precision variant. Fixed with the
   `sqrt(max(m,n))` fallback the file already established, gated so float32 --
   every shipped route -- keeps its tolerance bit-identical.

   **The regression test I wrote for that repair asserted `rank <= 3` and
   PASSED.** It encoded the bug. A test written from the same misunderstanding
   as the fix confirms the misunderstanding; the independent bf16 measurement
   caught it, the suite did not.

2. **spectral, provenance flag never reached the caller.** The derived
   `_meta_alpha_official_n_svd_zeros` was written into the per-layer dict, which
   the aggregation drops -- so the flag whose entire job is to say whether the
   repair fired was absent from `evaluate()`'s output. Now accumulated across
   modules and emitted; `avg_alpha_official` verified bit-identical.

## What this pass does NOT establish

These are population sweeps, not 51 per-task Opus verdicts. No `*.opus5.json`
has been written for round 13. Dual sign-off remains **0 of 51** and requires
both a completed astra pass and a per-task Opus pass on this same frozen state.

---

# Sweep 3 — determinism, document order, sample stability (per task)

For each task, on pretrained gpt2 with the WikiText default corpus:
same input twice (DETERMINISM), the same documents permuted (ORDER), and
N vs 2N (SAMPLE STABILITY).

## A REFUTED FINDING, AND WHY IT MATTERS MORE THAN THE CONFIRMED ONES

The first run of this sweep reported `dynamics_generation_diversity` as
NON-DETERMINISTIC: `distinct_1` differed by 1.92% between two identical runs.
That is a serious charge -- an unreproducible metric cannot rank models -- and
the code appears to support it: `generation_diversity.py:205` reads
`seed = self.config.get("seed", None)`, and line 233 only calls
`torch.manual_seed` when the seed is not None.

It is wrong. `defaults.yaml` supplies `seed: 42`, and the merge happens in
`resolve_task_config`, NOT in the task constructor. My harness built every task
with a bare dict, so it measured a configuration that no study has ever run.
Re-measured through `resolve_task_config`: **0 of 12 float keys differ between
identical runs.**

Two consequences, both recorded rather than quietly fixed:

1. Every number the first version of this sweep produced was about an
   unshipped configuration. Order and sample-stability figures moved too --
   `dynamics_interpolation`'s order sensitivity fell from 13.9% to 5.6%, and
   `generation_diversity` went from 4.3%/8.3% to 0.0%/0.0%.
2. This is the third promising finding this session to dissolve under
   measurement, after the topology "352 collinear columns" (degenerate fixture)
   and, earlier in the campaign, magnipy de-duplication. **A reviewer's harness
   is code, and it needs the same adversarial treatment as the code under
   review.**

## Confirmed, on the SHIPPED configuration

| task | determinism | order (same docs permuted) | N 20 -> 40 |
|---|---|---|---|
| dynamics_coe | 0.0 | 0.0 | 3.70% |
| dynamics_fractal | 0.0 | **3.95%** | 0.0 |
| dynamics_gradient_flow | 0.0 | 0.0 | 2.51% |
| dynamics_interpolation | 0.0 | **5.59%** | 2.23% |
| dynamics_stability | 0.0 | 0.0 | 1.85% |
| dynamics_generation_diversity | 0.0 | 0.0 | 0.0 |

Every task is deterministic on the shipped config. Two carry a real document-
ORDER dependence on an identical document set: `dynamics_fractal` (3.95%) and
`dynamics_interpolation` (5.59%). A model-level metric that moves when the
corpus is shuffled is measuring the shuffle as well as the model; this belongs
on `confounds`, not on `verdict`, since neither is an unfaithful implementation
of its estimand.

`dynamics_stability` deserves credit rather than a finding: it emits its own
standard error and refuses to be over-read, printing that its SEM at
num_samples=20 is 0.0069 while the gpt2-vs-distilgpt2 difference it would be
used to rank is 0.0063 -- i.e. inside its own noise. That is the disclosure
other tasks should imitate.

---

# Sweep 4 — reference-parity COVERAGE. Gap found and CLOSED.

Coverage question: which of the 51 tasks have an EXECUTED reference-parity
check, as opposed to implementation faithfulness argued from code reading?

Measured by mapping each registered task to its module and grepping the test
tree: **45 of 51 had one, 6 did not** — `dynamics_coe`,
`dynamics_generation_diversity`, `geometry_contextualization`,
`geometry_mp_bulk_deviation`, `topology_persistence_entropy`,
`topology_persistence_landscape`.

`tests/tasks/parity/test_coverage_gap_parity.py` closes all six, with 13 tests.
Every reference is derived independently from the cited definition, not
transcribed from BLME:

| task | independent reference | result |
|---|---|---|
| topology_persistence_entropy | `scipy.stats.entropy(l, base=2)` (giotto-tda's own route), plus analytic extremes log2(m) and 0 | matches to 1e-12 over 20 random barcodes; scale-invariant over 1e-14..1e14 |
| geometry_mp_bulk_deviation | analytic MP edge (1+sqrt(gamma))^2, AND the sampled largest eigenvalue of white Wisharts at gamma = 0.1, 0.25, 0.5 | edge matches to 1e-15; brackets the sampled spectra |
| dynamics_generation_diversity | Li et al. 2016 Distinct-n, reimplemented | matches to 1e-15 over 30 random token strings and n = 1,2,3; the collection-vs-mean distinction reproduced (0.1 vs 1.0 on ten duplicate completions) |
| geometry_contextualization | Ethayarajh 2019 IntraSim, reimplemented | matches to 1e-10; invariant over 1e-30..2.4e38 |
| dynamics_coe | Wang et al. 2025 CoE-R/CoE-C in the official normalized/normalized form, reimplemented | matches to 1e-9 over 15 random chains; dimensionless over 1e-30..2.4e38 |
| topology_persistence_landscape | Bubenik 2015 tent definition, AND GUDHI `representations.Landscape` | see below |

**Reference-parity coverage is now 51 of 51.**

## The landscape sqrt(2), caught by checking reference against reference

The landscape test compares BLME against the tent definition AND GUDHI against
the same definition. GUDHI disagreed with the definition — by exactly
sqrt(2), ratio 1.414214 on every nonzero sample, max|gudhi - sqrt2*bubenik| =
1.67e-16. GUDHI uses the ROTATED-frame convention.

BLME matches Bubenik's original lambda_k(t) = k-max_i max(0, min(t-b_i, d_i-t))
with a difference of **exactly 0.0**, i.e. it is faithful to its own cited
paper. Both conventions are pinned in the test, because a BLME landscape number
and a GUDHI-derived one differ by sqrt(2) and someone will eventually compare
them. **Had the test only compared BLME to GUDHI it would have reported a
defect that does not exist.**

# Sweep 5 — determinism / order / sample stability, all 51

On the shipped configuration (`resolve_task_config`), 36 tasks measured so far:

**Non-deterministic tasks: 0.** Every task reproduces its headline exactly.

Document-ORDER sensitivity above 2%, on a PERMUTATION OF THE IDENTICAL
document set:

| task | key | order sensitivity |
|---|---|---|
| geometry_lid | `lid_max` | **101.05%** |
| geometry_magnitude | `magnitude_area` | 9.30% |
| geometry_mahalanobis | `auroc` | 8.00% |
| dynamics_interpolation | `convexity_gap` | 5.59% |
| geometry_contextualization | `mean_anisotropy_baseline` | 4.37% |
| dynamics_fractal | `hurst_exponent` | 3.95% |

## Finding C — `lid_max` doubles when the corpus is shuffled

`geometry_lid` on 20 documents, forward vs reversed:

    lid_max     25.357403  ->  50.981638   (x2.01)
    lid_mean     8.965920  ->   9.039726   (+0.8%)
    lid_median   8.590945  ->   8.608436   (+0.2%)
    lid_std      4.456568  ->   4.600298   (+3.2%)

Mechanism, at `lid.py:88`: `query_indices = rng.choice(len(X), size=n_queries,
replace=False)` picks query points BY INDEX under a fixed seed, while `X` is
the concatenation of document token vectors IN CORPUS ORDER. Permuting the
documents therefore leaves the indices identical and the points they select
different. The mean and median absorb that; `lid_max`, an extreme order
statistic over a subsample, does not.

AXIS: the LID estimator itself is unchanged and faithful, so this is
`confounds`, NOT `verdict`. `lid_max` should not be read as a model descriptor
at this sample size — it is a property of the corpus ordering as much as of the
model.

---

# CORRECTION to Finding C, and a caveat that applies to all of Sweep 5

Sweep 5 ran every task at `num_samples` 20 and 40. Several tasks ship a LARGER
default -- `geometry_lid` ships 50, `geometry_correlation_dimension` ships 100 --
so those measurements were taken BELOW the shipped configuration, where noise is
larger. Every order-sensitivity number in the Sweep 5 table is therefore an
UPPER BOUND on shipped behaviour, not a description of it.

Re-measured at the shipped default:

| task | key | order sensitivity at N=20 | at shipped N=50 |
|---|---|---|---|
| geometry_lid | `lid_max` | 101.05% | **4.00%** |
| geometry_lid | `lid_mean` | 0.82% | 1.72% |
| geometry_lid | `lid_median` | 0.20% | 2.14% |

The headline claim -- "lid_max doubles when the corpus is shuffled" -- is true
at N=20 and NOT true at the configuration that ships, where lid_max's 4.00%
sits alongside lid_median's 2.14%. The verdict has been downgraded from
`comparable: RED / confounds: CONFIRMED` to `comparable: YELLOW /
confounds: NONE`, with the bound stated in both fields.

This is the campaign's own reachability rule turned on the reviewer: a finding
must name the magnitude at the configuration that actually runs, and the
released/shipped setting is the one that counts. It is also the FOURTH finding
this session to shrink or vanish under proper measurement, after the
generation_diversity determinism claim, the topology collinearity artefact, and
the GUDHI sqrt(2) trap.

Similarly, `geometry_correlation_dimension` returning NaN in Sweep 5 is NOT a
defect: at N=20 (its default is 100) the plateau kernel correctly reports
`status='insufficient'` and refuses to emit a number, which is the designed
behaviour. Its repaired provenance key is correct there:
`_meta_correlation_integral_n_pairs = 190 = 20*19/2`, coincident fraction 0.

---

# Self-audit of the seven Opus verdicts, and two withdrawals

Having established that Sweep 5 ran below several tasks' shipped defaults, I
re-checked every quantitative claim in the seven verdicts I had written against
the sample size that actually ships:

| task | shipped num_samples | claim as first written | status |
|---|---|---|---|
| geometry_lid | 50 | `lid_max` order 101.05% | CORRECTED to 4.00% at N=50; comparable RED -> YELLOW |
| geometry_contextualization | **1000** | anisotropy baseline order 4.37% | **WITHDRAWN** -- measured at N=20, fifty times below shipped |
| dynamics_coe | **5** | 3.70% N-sensitivity | RE-ATTRIBUTED: measured between N=20 and N=40, both ABOVE the shipped 5 |
| topology_persistence_entropy | n/a | `num_features_h0 == N-1` | STANDS -- structural, verified at N=15/25/40 |
| topology_persistence_landscape | n/a | 30 mis-prefixed keys | STANDS -- structural, config-independent |
| geometry_mp_bulk_deviation | n/a | analytic MP edge | STANDS -- closed form, no sampling involved |
| dynamics_generation_diversity | n/a | deterministic | STANDS -- measured on the resolved shipped config |

Two claims withdrawn or corrected out of seven verdicts, both for the same
reason: a sampling-noise figure measured below the shipped configuration is not
evidence about the shipped configuration. The parity-based claims all stand,
because a helper-level parity test supplies its own inputs and does not depend
on `num_samples` at all -- which is an argument for grounding verdicts in
executed parity rather than in end-to-end variability wherever possible.

`geometry_contextualization` is now `comparable: GREEN / confounds: NONE` with
the open question recorded in `fix` rather than dressed as a finding.

# Sweep 5b — at each task's own shipped default (in progress)

| task | shipped N | determinism | order |
|---|---|---|---|
| dynamics_coe | 5 | 0.0 | 2.08e-16 |
| dynamics_fractal | 100 | 0.0 | 2.23% (was 3.95% at N=20) |

Both confirm the direction: the Sweep 5 figures are upper bounds, and shrink
toward the shipped configuration.

---

# Opus round-13 pass COMPLETE: 51 of 51 verdicts

verdict (implementation faithfulness): **51 GREEN**
comparable: 1 RED, 10 YELLOW, 40 GREEN
confounds: recorded per task

## Why 51 GREEN is not as reassuring as it looks, stated plainly

A reviewer who finds nothing on the verdict axis is usually not looking hard
enough. Three things make this particular result less alarming than it sounds,
and one makes it weaker than it sounds:

1. The implementation defects on this tree were ALREADY found -- by astra in
   round 12, which returned 10 REDs, every one `defect_location: CODE`. All ten
   were repaired and each repair was verified against the reviewer's own
   counterexample before this pass began. A second reviewer arriving after that
   should find the verdict axis clean; that is what a working protocol looks
   like.
2. The findings did not vanish, they MOVED AXIS. This pass produced a RED and
   ten YELLOWs on comparability, plus two structural defects
   (persistence_entropy's N-1 columns, persistence_landscape's 30 mis-prefixed
   keys). Under the two-axis rule those must not be laundered into `verdict`,
   and they have not been.
3. 401 executed reference assertions pass on the frozen tree, coverage 51/51 --
   up from 45/51 at the start of this pass.

**The weakness:** a majority of the 51 verdicts rest on EXECUTED PARITY --
I ran the reference checks and confirmed they pass -- rather than on an
independent reimplementation of each reference. Where I did reimplement
independently (the six coverage-gap tasks) the checks were far more searching,
and one of them immediately caught a real trap (GUDHI's sqrt(2) convention).
A pass built on running existing tests inherits whatever those tests fail to
ask. Every verdict says so in its `github_reason` and `fix` fields rather than
implying more.

## Methodological defects found in MY OWN review process this session

Five, all caught by measurement rather than by reasoning:

1. Harness bypassed `defaults.yaml` -> `dynamics_generation_diversity` looked
   non-deterministic (1.92%). It is not (0 of 12 keys differ).
2. Degenerate fixture -> "352 collinear column pairs" in topology, which were
   `0.0 == 0.0`.
3. Reference-vs-reference disagreement -> GUDHI's landscape is sqrt(2) x
   Bubenik's. Comparing BLME only to GUDHI would have filed a defect that does
   not exist.
4. Sample size below the shipped default -> `lid_max` "doubles" (101%) at N=20,
   4.00% at the shipped N=50. Verdict corrected RED -> YELLOW.
5. Document SELECTION mistaken for document ORDER, twice: passing 20 documents
   with num_samples=10 makes a reversal choose a different 10.
   `dynamics_interpolation` read 30.85% that way and 0.62% on the identical
   set. The honest figure is the 12.85% spread over six permutations.

A single reversal is ONE DRAW, not a bound. Every order-sensitivity number in
Sweep 5 is a single draw at a sample size I chose, and is labelled as such in
the verdicts that cite it.

## What remains before certainty

Dual sign-off is still **0 of 51**: astra's round-13 pass is at 31/51. No pair
has been cross-checked, deliberately -- reading astra's verdicts before
finishing mine would have destroyed the independence the protocol exists to
provide. When astra completes, the disagreements are the interesting output,
particularly on `topology_betti_curve`, where I changed two tests to match the
released loader and asked the second reviewer to challenge exactly that.

---

# Sweep 6 — extreme scale, and a sixth harness error

Most verdicts in this pass carry the scope limit "extreme-scale behaviour was
not swept". This closes part of that for the kernels whose estimand is
scale-FREE, over the ladder 1e-30, 1e-22, 1e-8, 1, 1e8, 1e22, 2.406159650500376e38.

| kernel | result |
|---|---|
| `geometry_isoscore` `_isoscore` | **holds across the whole ladder**, base 0.7647032732503589, no movement to 1e-6 relative including at 2.4e38 |
| `geometry_cka` `_linear_cka` | holds to 1e22; **NaN at 2.406159650500376e38** |
| `geometry_hsic` `_debiased_hsic` | moves at every scale -- SEE BELOW, this was my error |

## CKA's NaN at 2.4e38 is documented and SHARED WITH THE REFERENCE

This is not a new finding. The campaign already established that the NaN at
2.406159650500376e38 reproduces in Kornblith's own released code on the same
float32 array, and disappears only after a float64 promotion that no shipped
route performs. My sweep reproduces exactly that boundary: fine at 1e22, NaN at
2.4e38. "Wrong where no model goes, where the reference is also wrong" is not a
defect, and gpt2's largest hidden magnitude is ~3.0e3.

## Sixth harness error: HSIC is not scale-free, it is scale-EQUIVARIANT

My sweep flagged `_debiased_hsic` as moving at every scale, which looks
alarming. The premise was wrong. HSIC takes KERNEL matrices: K = (sX)(sX)^T =
s^2 XX^T, so HSIC scales as s^2 by definition. It is CKA -- the normalised
version -- that is scale-free.

Measured, and it is exact:

    scale      measured              base * s^2            ratio
    1e-08     -1.552372397597e-17   -1.552372397597e-17    1.000000000000
    1e+08     -1.552372397597e+15   -1.552372397597e+15    1.000000000000
    1e+22     -1.552372397597e+43   -1.552372397597e+43    1.000000000000

Ratio exactly 1.000000000000 at every scale, over 30 orders of magnitude. That
is not a defect; it is the correct behaviour of an unnormalised dependence
measure, and the cleanest confirmation of it I could have asked for.

**Sixth methodological error in my own review process this session**, and the
same shape as the other five: I asserted an invariance the estimand does not
have, then read the resulting movement as a defect. The invariance a metric
should satisfy has to be derived from its DEFINITION, not assumed from the
family it belongs to.

---

# THE CACHE CLASS: read-only sweep said 2 sites, execution said 15

This is the single largest finding of the campaign, and the method is the whole
point.

## What the defect is

`cache.is_populated` says a cache HAS been filled, not that it was filled with
the observations a given task needs. A cache populated for an attention-only
consumer returns `None` from `get_hidden_states`. A task that dereferences that
None RAISES; a task that returns an error instead of self-collecting silently
loses every headline. Either way, merely CO-RUNNING another task destroys the
measurement.

## How the population was mis-counted, twice

- **Round 12.** I swept this class BY READING the code and reported 2 sites
  (`geometry_vendi_score`, `geometry_magnitude`). I listed `geometry_isoscore`
  and `geometry_svd` among the candidates and cleared them by reading the
  surrounding lines.
- **Round 13.** astra RED-flagged both of those, plus
  `geometry_prediction_alignment`. `X.float()` on a None cache read raises
  `AttributeError`. I then claimed the population was 5.
- **The execution sweep.** Running every one of the 51 tasks against a REAL
  attention-only cache found **15 sites**:
  - 4 that RAISED: `geometry_cka`, `geometry_collapse`, `geometry_lid`,
    `geometry_hsic` (`AttributeError: 'NoneType' object has no attribute
    'keys'`, and a `TypeError` in lid).
  - 6 that returned an ERROR while running perfectly with NO cache at all:
    `geometry_cknna`, `geometry_lipschitz`, `geometry_mp_bulk_deviation`,
    `geometry_procrustes_linearity`, `geometry_rsa`,
    `geometry_trajectory_curvature`.
  - the 5 already repaired.

Sweep before: 4 raised / 9 errored / 38 produced results.
Sweep after:  **0 raised / 3 errored / 48 produced results**, and all 3 remaining
declines are legitimate (attention backend, or fewer than 20 points supplied).

**Two of the 15 -- `geometry_collapse` and `geometry_lid` -- were DUAL-SIGNED at
the time.** Both reviewers had passed them while they carried a defect that an
executed check surfaced in minutes.

## The discriminating test, and why the guard has no allow-list

A task is at fault only if it WORKS with no cache and FAILS with an unusable
one. My first guard used a hardcoded allow-list of "legitimate" declines, which
is exactly how a real defect gets excused. It now compares each task against its
own no-cache behaviour, so the guard justifies its own exclusions:
`geometry_phd_dimension` declines identically either way (a fixture-size limit,
n=90 < min_points=50) and is correctly not counted, with no entry in any list.

`tests/test_cache_population_guard.py` enforces this across all 51 tasks, plus a
second test asserting the fixture really does produce an unusable cache, so the
guard cannot pass vacuously.

## The rule this establishes

**A sweep that has not been EXECUTED is not evidence.** The repo already carried
this lesson for AST sweeps ("validate the sweep against a known positive before
trusting a zero"). This extends it: for a defect that manifests at RUNTIME, the
only valid sweep is one that runs every member of the population and observes
what happens. Reading 51 call sites produced the wrong count twice.

---

# Round 15 (PARTIAL — both reviewer models are now quota-blocked)

gpt-6-astra hit its usage limit mid-pass: 23 of 51 verdicts written, the other
28 failing in 0.3 min with "You've hit your usage limit ... try again at
Sep 15th, 2026 9:31 AM". gpt-5.6-sol remains blocked until Sep 11 11:22, so it
returns FIRST — but `~/.codex/config.toml` pins `model = "gpt-6-astra"`, so
using sol on Sep 11 needs that switched or a `-m` flag added to the runner.

The 23 verdicts that landed: 2 GREEN / 5 YELLOW / **16 RED**.

## A DATA-CORRUPTION REGRESSION I INTRODUCED, caught by two of those REDs

`geometry_cka` and `geometry_collapse` both reported that the DISK COLLECTOR
corrupts the representation matrix: an exact [4, 2, 1] profile became
[1.994847434735, 2, 1], collapse_ratio moved 0.124905430864 -> 0.358592800794
(+187.09%) on unchanged GPT-2, and CKA's GPT-2 alignment became NaN in all 10
headlines.

Cause: MY dtype-preserving fix to `collect_hidden_states` from the previous
turn. The offload path writes raw bytes with `h_flat.numpy().tobytes()` and the
reader hard-codes `np.memmap(..., dtype=np.float32)`. Preserving float64 in the
writer therefore wrote 8-byte floats that were read back as 4-byte ones.

Fixed by recording `np_dtype` in the offload metadata and reading with it.
Verified: a float64 cloud now round-trips through disk offload identically to
the in-memory route (same dtype, same 42x768 shape, same values). 305 tests pass.

**The full suite had passed at 2635 green WITH this corruption present**,
because no test drove float64 through the offload path. A green suite bounds
the defects the tests ask about and nothing else.

## `geometry_hubness` RED — NOT REPRODUCED

Claim: "occurrence accumulation corrupts the public maximum: [1,2,1,0] becomes
[2,4,2,0], emitting hubness_k1_max=4 instead of 2 (+100%)", with the same
doubling on real models (46 -> 92 on GPT-2, 29 -> 58 on Pythia-70m).

Measured against an independently computed occurrence vector (unit-normalise,
full pairwise distances, mask the diagonal, argsort, bincount):

| fixture | k | batch_size | BLME max | independent max |
|---|---|---|---|---|
| 4 points, 2-D | 1 | default | 2 | 2 |
| 40 points, 8-D | 1 | 64 (one batch) | 3 | 3 |
| 40 points, 8-D | 1 | 7 (six batches) | 3 | 3 |
| 40 points, 8-D | 3 | 64 | 6 | 6 |
| 40 points, 8-D | 3 | 7 | 6 | 6 |

The 4-point case is astra's own: BLME returns exactly [1,2,1,0] -> max 2, not
[2,4,2,0] -> max 4. Self-distance IS masked (`sims[rows, cols] = inf`), the
batched and unbatched paths agree, and the occurrence sums equal n*k exactly.

An exact 2x on two different models suggests a doubled INPUT (e.g. an embedding
matrix returned with the vocabulary concatenated twice) rather than a defect in
the accumulation. FILED AS NEEDING THE REVIEWER'S FIXTURE, not dismissed: I
cannot reproduce it on any fixture I can construct, and their real-model numbers
are too clean to ignore.

---

# The narrowing class, counted three times and still growing

Round 15's REDs kept resolving to one shape: a value narrowed to float32 before
the arithmetic that needs the precision. Each sweep I wrote found a subset,
because each query could only express one spelling of it.

| sweep | query | found | missed |
|---|---|---|---|
| 1 | `X.double().mean(dim=0).float()` | 7 | the `.to(dtype)` spelling |
| 2 | + `.to(<dtype>)` after a double reduction | 5 more (12) | collectors, raw clouds |
| 3 | `.detach().float().cpu().numpy()` on a cloud | 17 candidates | — |

Fixed and verified so far:

- **12 sites**, promotion immediately narrowed back after a reduction. On native
  Pythia-70m `geometry_correlation_dimension` read 7.486064047571674 against an
  independently reconstructed Grassberger-Procaccia 15.295160200027059 (-51.06%),
  and a LOSSLESS translation collapsed it to 0.0. Removing only the cast gives
  15.295160200026809.
- **`dynamics_perturbation_growth`**, 4 collector conversions narrowing BEFORE
  the difference is taken -- and the difference is the whole measurement. On a
  ten-block affine contraction `h -> 1000 + (h-1000)/2` (exact rate log(1/2) =
  -0.693147180560), narrowing drove the final divergence to 0.0 and the task
  refused outright; preserving float64 gives -0.6931471804667221. At 8 blocks
  narrowing emits -0.7241 instead of -0.6931.
- **`geometry_magnitude:531`**, the tensor branch narrowed to float32 while the
  numpy branch beside it used float64, so the same cloud gave different answers
  depending on how it arrived. Magnitude is translation-invariant, and a
  lossless +1 shift made evaluate() report "Degenerate cloud (no positive
  distances)". Measured on a 32-point cloud with a 1e5 offset and 1e-4
  structure: float32 resolves **0 of 496** pairwise distances, float64 resolves
  **496 of 496**.

## The 17 candidates are NOT being blanket-fixed

Sweep 3 lists 17 sites narrowing a tensor on its way to numpy. They are
CANDIDATES, not confirmed defects: several are attention probabilities or
sparsity masks where float32 is the honest representation and the estimand is
not precision-sensitive.

A blanket edit across a class is exactly how the disk-offload corruption was
introduced two days ago -- that change was correct in isolation and broke a
byte-format contract two hundred lines away. Each remaining site needs its own
counterexample before it is touched, and the ones that matter will surface the
same way these three did: as a reviewer's analytic anchor that the code misses.

Recorded for the next pass:
`edge_attribution` (3), `knowledge_neurons` (2), `bias` (1),
`phd_dimension` (2), `procrustes_linearity` (2), `trajectory_curvature` (1),
`unembedding` (1), `vendi` (1), `activation_kurtosis` (1), `induction` (1),
`sparsity` (1), `zigzag` (1).

---

# Round 15's REDs — final disposition (23 of 51 reviewed before quota died)

| task | cause | status |
|---|---|---|
| cka, cknna, collapse, hsic, intrinsic_dim, isoscore, lid, lipschitz | ONE cause: the disk-offload dtype corruption I introduced | fixed + population guard |
| correlation_dimension | `.double().mean(...).to(dtype)` narrowing, 12-site class | fixed |
| perturbation_growth | 4 collector conversions narrowing before the difference | fixed |
| magnitude | tensor branch float32 while the numpy branch beside it used float64 | fixed |
| gradient_flow | `F.cross_entropy` on bf16 logits without the promotion the native loss uses | fixed |
| categories | relation pairs skipped by the ambiguity census | fixed |
| hubness | occurrence doubling | NOT REPRODUCED on any fixture I can build |

## categories — the census ignored list-valued entries

`categories.py:166` read `if not isinstance(_w, str): continue`, so a relation
category written as pairs (`plural = [[man, men], [woman, women], ...]`)
contributed NOTHING to the multiply-claimed census and its tokens were labelled
first-wins. The census exists precisely so the answer does not depend on asset
key order, and it did: reversing the two keys moved plural purity
0.083333333333 -> 0.357142857143 and people purity 0.25 -> 0.083333333333.

After flattening one level, the census is ORDER-INDEPENDENT: both key orders now
return the identical ambiguous set {582, 805, 1048, 1200, 2415, 6259, 8580, 9410}.

One honest discrepancy: astra reports the correct counts as 4 ambiguous / 8
mapped, and this gives 8 / 8. The shipped census tests BOTH surface forms of
each word (`man` and `<space>man`), so each of the 4 overlapping words
contributes two ambiguous ids; their census evidently counts one variant per
word. That is a convention difference in the census, not a disagreement about
the defect or its fix, and it is recorded rather than tuned to match.

## gradient_flow — half precision under the loss

`F.cross_entropy` ran on bf16/fp16 logits without the promotion the released
model's own `forward(labels=...)` performs. The seed derivative of this entire
measurement is `(softmax(z) - one_hot(y))/(T-1)`, so the error enters at the
first backward step and propagates through every layer norm reported: on bf16
Pythia-70m layer 1 read 162.675113677979 against the native 53.704253345424
(+202.9%), and a 256-class constant-loss anchor whose true norms are exactly
[0,0,0,0] emitted four positive values.

Measured on that anchor: bf16 CE is 1.393e-02 away from the exact log(256);
promoted, 1.524e-08. float16: 1.698e-03 -> 1.524e-08. float32 unchanged.

---

# The 17 narrowing candidates, resolved by counterexample (2026-09-10)

Rather than blanket-fix the class, each PAPER-CORE candidate was probed on a
40x16 (120x16 for PHD) cloud carrying 1e-4 structure on a 1e5 offset -- the
shape real hidden states have, and the regime where float32's ULP (~0.0078 at
1e5) swallows the signal.

| candidate | float64 | narrowed | disposition |
|---|---|---|---|
| `procrustes_linearity` | 0.9999999999999806 | **NaN** | DEFECT — fixed (2 sites) |
| `trajectory_curvature` | 2.0871608250868765 | **NaN** | DEFECT — fixed |
| `vendi_score` | 1.0000000000000526 | 1.00000000000005 | not a defect (rel 2.665e-15) |
| `unembedding` | 480.0 | 480.0 | not a defect (rel 0.0) |
| `zigzag` | (1, 0) | (1, 0) | not a defect (identical) |
| `phd_dimension` | NaN | NaN | NO COUNTEREXAMPLE FOUND |

**Three sites carry the identical spelling as the two defects and are provably
unaffected.** A class-wide edit would have changed all six; the two that matter
were separated from the four that do not by running them, and the reasoning is
recorded in a comment at each fixed site so the next reviewer can see why
`vendi` was skipped.

`phd_dimension` is left alone deliberately: the regime where narrowing bites and
the regime where PHD is defined do not overlap on this construction, so there is
no counterexample to justify a change. Absence of a counterexample is not proof
of absence — it is recorded as an open question, not as a clearance.

The remaining candidates (`edge_attribution` x3, `knowledge_neurons` x2, `bias`,
`activation_kurtosis`, `induction`, `sparsity`) are outside the 51 paper-core
set and were not probed.

## Why this matters more than the two fixes

The disk-offload corruption earlier in this campaign came from exactly the edit
this pass declined to make: a correct-looking change applied across a class
without checking each member's contract. It broke a byte-format contract two
hundred lines away, produced EIGHT of round 15's sixteen REDs from one cause,
and passed a 2635-green suite because no test drove float64 through offload.

---

# The study runs a DIFFERENT configuration from the one I verified (2026-09-10)

A sweep of `beyond_benchmarks/configs/tasks.yaml` against
`src/blme/tasks/configs/defaults.yaml` over the 51 paper-core tasks:
**46 tasks carry a study override, 77 keys are shared with a shipped default,
and 34 of those diverge.**

## The first version of this sweep was VACUOUS

It read the override file's top level and compared task names there, finding
"0 divergences". The overrides actually live under a `task_configs:` key, so
the loop compared nothing at all. The file has 9 top-level entries, of which
exactly one is a dict and none is a task name — so the zero was an artefact of
a query that could not reach the data.

Caught only by asking how many keys the sweep had actually compared. **A sweep
must report its denominator**: "0 divergent" is meaningless without "out of 77
compared". This is the fourth vacuous or undercounting sweep of the campaign.

## Most divergences raise the sample size. Six LOWER it.

| task | key | shipped | study | reduction |
|---|---|---|---|---|
| topology_betti_curve | num_samples | 1000 | 50 | **20.0x** |
| topology_persistence_entropy | num_samples | 1000 | 50 | **20.0x** |
| dynamics_sharpness | n_power_iter | 64 | 10 | 6.4x |
| geometry_contextualization | num_samples | 1000 | 200 | 5.0x |
| dynamics_sharpness | n_hutchinson | 24 | 8 | 3.0x |
| dynamics_stability | num_samples | 5000 | 2000 | 2.5x |

The two `dynamics_sharpness` keys are not sample sizes — they control ESTIMATOR
ACCURACY directly. `n_power_iter` governs convergence of the top Hessian
eigenvalue and `n_hutchinson` the variance of the trace estimator, so the study
computes both at a materially coarser setting than the package ships.

`dynamics_stability` ships 5000 for a documented reason: the task emits its own
standard error and warns that at small samples its noise exceeds the
between-model differences it would be used to rank. The study runs 2000.

`geometry_contextualization` ships 1000 because IsoScore-style estimands need
N >> d; the study runs 200 against d_model 768.

## A third value exists for sharpness

`sharpness.py:355-356` falls back to `n_hutchinson=5, n_power_iter=8` when the
key is absent, `defaults.yaml` ships `24, 64`, and the study passes `8, 10`.
Three different values for the same knob, and which one applies depends on how
the task is constructed.

## What this does to my own review pass

Every determinism, order and sample-stability figure I measured used
`resolve_task_config` WITHOUT the study overrides — i.e. the shipped defaults.
For these six tasks that is not the configuration the paper's numbers come from,
so those measurements do not describe the published runs. Recorded rather than
quietly re-run: the fix is to re-measure under the study's own overrides, which
needs the study repo's config in the loop.

AXIS: this is `comparable`/`confounds`, not `verdict` — no implementation is
unfaithful. It is exactly the config-vs-code separation the brief demands, and
it is the same class as the `dynamics_fractal` `holder_eps` override that once
collapsed a headline by -92.3%.

## Quantifying the sharpness config reduction — the task is RIGHT, the config is not

Measured on pretrained gpt2 (eager attention), same seed and corpus, varying
only `n_hutchinson`:

| n_hutchinson | `hutchinson_trace_per_param` | `_meta_hutchinson_trace_rel_se` |
|---|---|---|
| 1 | -0.000015 | (undefined, n=1) |
| 4 | -0.000001 | **19.56** |
| **8 — the study's setting** | +0.000005 | **1.11** |
| 16 | +0.000008 | 0.38 |
| **24 — shipped default** | +0.000009 | 0.34 |

**At the study's setting the standard error EXCEEDS the estimate** (relative SE
1.11), which is exactly why the sign flips between n=4 and n=8. At the shipped
24 the relative SE is 0.34 — large, but the quantity is at least
distinguishable from zero.

This is NOT a code defect, and the distinction matters: the task computes the
Hutchinson estimator correctly AND emits
`_meta_hutchinson_trace_rel_se` alongside the headline, so it discloses its own
uncertainty. It is the same pattern as `dynamics_stability`, which emits its
SEM and warns when the noise exceeds the between-model difference. The
instrument works; the configuration renders the number unusable and the
disclosure is in a `_meta_` key that a results table is unlikely to read.

`n_power_iter` is the opposite case and should be left alone: 64 vs 10 vs 8
gives top eigenvalue -31.7842292786 / -31.7871170044 / -31.7873458862, a
relative spread of ~1e-4, and `_meta_top_eigenvalue_sign_stable` is True at all
three. The study's reduction there costs nothing measurable.

RECOMMENDATION (config, not code): raise the study's `n_hutchinson` to at least
16 for any table that reports a trace, or report the trace with its relative SE
attached. Leave `n_power_iter` at 10.

## topology_persistence_entropy — the whole H0 family is a function of N, not the model

Following Finding A (`num_features_h0` is structurally N-1), I tested the
prediction that the H0 ENTROPY must then be approximately log2(N-1). Measured on
pretrained gpt2, varying only the sample size:

| N | `layer_0_pe_h0` | log2(N-1) | ratio |
|---|---|---|---|
| 50 | 5.575624 | 5.614710 | 0.9930 |
| 100 | 6.589503 | 6.629357 | 0.9940 |
| 200 | 7.589562 | 7.636625 | 0.9938 |
| 400 | 8.589905 | 8.640245 | 0.9942 |

The ratio is constant to within 0.12% over an 8x range of N. **The H0
persistence entropy is log2 of the sample size, scaled by 0.994.** The full key
set at N=100 shows why:

    layer_0_num_features_h0   99        <- exactly N-1
    layer_0_pe_h0             6.5895    <- ~log2(99)
    layer_0_pe_h0_norm        0.99399   <- saturated at its own ceiling
    layer_0_num_features_h1   32        <- genuinely geometry-dependent
    layer_0_pe_h1             4.4543
    layer_0_pe_h1_norm        0.89086   <- not saturated; carries information

Rips H0 on N points always yields N-1 finite bars, and on real hidden states
their lifespans are near-uniform, so the entropy saturates at its ceiling
log2(N-1) and the normalised twin pins at ~0.994 regardless of the model.

**Three of the seven per-layer keys — `num_features_h0`, `pe_h0`,
`pe_h0_norm` — cannot discriminate between models.** In a feature table they are
near-constant columns; across configurations they move only because N moved. The
H1 family is unaffected and is the part of this task that measures something.

Consequence for the study's 20x reduction (N 1000 -> 50): the H0 headline drops
from ~9.9 to ~5.6 purely from the sample size. Within-study comparisons are not
corrupted, because N is held constant there — but the quantity being compared is
essentially a constant, so it cannot separate models either way.

AXIS: `comparable`/`confounds`. The entropy is computed correctly (it matches
scipy.stats.entropy base 2 to 1e-12); the estimand simply has no model-dependent
content in H0.

---

# Sweep 7 — which keys can tell two models apart? (2026-09-10)

The persistence-entropy result generalises into the question a comparative
benchmark actually rests on: a key that returns the same value on two different
architectures carries no model-discriminative content, however well-founded its
source paper. Every task run on gpt2 and distilgpt2, same corpus, same N=20.

**49 tasks compared, 386 public float keys, 35 flat (rel < 1e-3) across models.**

The 35 are NOT 35 defects. They separate into four kinds, and only the first is
unambiguous.

## A. Five keys are CONFIG ECHOES — inputs published as measurements

| task | key | value | in defaults.yaml |
|---|---|---|---|
| dynamics_generation_diversity | `temperature` | 0.8 | **0.8** |
| dynamics_perturbation_growth | `rel_eps` | 0.01 | **0.01** |
| geometry_spectral | `tail_fraction` | 0.2 | **0.2** |
| topology_persistence_landscape | `layer_0_h0_meta_prescale` | 1.0 | (constant) |
| topology_persistence_landscape | `layer_0_h1_meta_prescale` | 1.0 | (constant) |

These are settings the caller supplied, echoed back without the `_meta_` prefix,
so `is_feature()` treats them as measurements. They cannot discriminate between
models under ANY configuration — they are constant by construction. The two
`prescale` keys are the same 30 mis-prefixed landscape keys reported earlier,
now confirmed to be inert in the feature table as well as mis-named.

## B. A TRUE NEGATIVE that validates the sweep

`geometry_tokenizer_efficiency`'s `fertility` (1.2157143) and `token_entropy`
(5.5462214) are bit-identical across the two models. That is CORRECT: distilgpt2
uses gpt2's tokenizer, and the vocabularies are identical (both 50257, dicts
equal). A tokenizer metric SHOULD be identical for models sharing a tokenizer.

This is the known-positive check the campaign's own rules demand before trusting
a sweep: the method correctly flags a pair it should flag, for the right reason.

## C. Saturated or barely-separated genuine measurements

`geometry_cka`'s `max_adjacent_cka` / `max_offdiag_cka` both sit at 0.9999 (a
maximum over layer pairs saturates near 1); `geometry_collapse`'s `max_erank`
reads 474.09603 vs 474.16214; `geometry_spectral`'s `median_alpha_official`
3.6419853 vs 3.642498; several `geometry_categories` purity/inter/intra keys
agree to 3-4 digits. These measure something real but do not separate THESE two
models — distilgpt2 is a distillation of gpt2, so genuine similarity is expected
and is not evidence of a defect.

## D. Degenerate at N=20, inconclusive

The topology zeros (`betti_curve`'s six h1 keys, `zigzag`'s h0 components and
ranges, `homology`'s fine-persistence fraction) and `geometry_lid`'s `lid_min`
are 0 or 1 because a 20-document cloud carries no H1 structure. Not judged here;
they need the shipped N before any claim.

## Recommendation

Demote the three config echoes (`temperature`, `rel_eps`, `tail_fraction`) to
`_meta_` and rename the landscape `*_meta_prescale` keys to carry the prefix.
That is a rename, not a computation change, and it removes five columns that can
only ever add noise to a regression over the feature table.

## Sweep 7, category D resolved at N=150

The topology zeros were inconclusive at N=20. Re-run on gpt2 vs distilgpt2 with
N=150, they split:

**Became discriminative — they were sample-size artefacts, not flat keys:**

| key | gpt2 | distilgpt2 |
|---|---|---|
| `betti_curve.h1_max_persistence_mean` | 0.164616 | 0.150397 |
| `betti_curve.h1_total_persistence_mean` | 0.690410 | 0.767460 |
| `zigzag.h0_mean_components` | 1.25 | 1.33333 |
| `zigzag.zz_true_full_depth` | 0.047619 | 0.043478 |

**Still flat at N=150 — and for a structural reason:**

| key | value (both models) |
|---|---|
| `betti_curve.simplification_ratio` | 1 |
| `betti_curve.max_betti_1_layer_frac` | 0 |
| `betti_curve.h1_max_persistence` | 0.25 |
| `zigzag.h0_first_layer_components` | 1 |
| `zigzag.h0_persistence_range` | 1 |
| `geometry_lid.lid_min` | 0 |

The three `beta_0`-derived ones share one cause: on real hidden states the kNN
graph is CONNECTED at every layer, so beta_0 = 1 throughout, and any quantity
built from it — the first-layer component count, the last/first ratio, the range
— is pinned at 1 by construction. `simplification_ratio` in particular can only
depart from 1 on a cloud whose graph fragments, which real representations do
not do at this k.

That is worth stating precisely because `simplification_ratio` was the key astra
reported in round 14 as "2.0 instead of 1.0". The repair made it 1.0, which was
correct — but the quantity is 1.0 for every model regardless, so the repair
fixed a WRONG value into a CONSTANT one. It is right, and it is also uninformative.

`lid_min = 0` is a floor: the minimum over query points picks up whichever point
has a degenerate local estimate, and there is always one.

## Standing count after Sweep 7

Of 386 public float keys across 49 tasks, **11 cannot distinguish two models**
for structural reasons (5 config echoes + 6 above), and 2 more are correctly
identical because the models share a tokenizer. The remaining 22 flagged keys
are saturated or genuinely-similar measurements on a distilled pair, which is
expected and not a defect.
