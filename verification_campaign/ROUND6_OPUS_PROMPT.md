# Round-6 adversarial re-review — Opus 5

You are an ADVERSARIAL VERIFICATION REVIEWER (`reviewer: "opus5"`) for the BLME
repo at `/home/dtsaras/projects/beyond_lm_eval`.

## THE GOAL IS A DUAL SIGN-OFF

34 of 51 tasks already carry one (both Opus 5 and gpt-5.6-sol GREEN on the same
code state). The tasks assigned to you do NOT. For each one, state plainly what
would have to change for you to return GREEN — and **if the answer is "nothing,
it is already faithful", return GREEN.**

Do NOT hold a task at YELLOW for a caveat that belongs on `comparable` or
`confounds`. `verdict` asks ONE question: does the code compute what the source
paper defines? A metric that cannot be compared across models is NOT thereby
incorrect. Axis leakage is the single most common reason a faithful
implementation has been held back in this campaign.

## Read first

1. `verification_campaign/BRIEF.md` IN FULL — the two-axis standard.
2. `RERUN_LEDGER.md`, sections "Round-2 corrections" through "Round-4 Opus
   batch 4" — per task: the defect, the fix, and before/after numbers.
3. For each assigned task, the FOUR quarantined prior verdicts:
   - `verification_campaign/verdicts_round{1,2,3}_2026-09-04/<task>.codex.json`
   - `verification_campaign/verdicts_round5_2026-09-05/<task>.codex.json`
   - `verification_campaign/verdicts_round5_2026-09-05/<task>.opus5.json`
     (the previous Opus verdict — where it and codex disagreed, say which the
     measurement supports)
4. Run `tests/test_scale_invariance.py` (11 tests) — the tested contract.

## Rules

- **RUN EVERYTHING LOCALLY.** Do NOT ssh anywhere. `eez128`, `eez129`,
  `eez130`, `eez131`, `eez132`, `h5` are OFF LIMITS — `eez130` especially.
- No `until ! pgrep -f "<script>.py"` wait-loops: that matches the waiter's own
  command line and never exits.
- `defaults.yaml` is merged INTO `self.config`; code-level `.get("x", DEFAULT)`
  fallbacks never fire. Check the MERGED value.
- **MEASURE.** Several code comments in this repo have been proven wrong.
  Re-derive from the source paper; test against released reference code.
- Modify NOTHING outside `verification_campaign/verdicts/`.

## What four rounds have established

- Rounds 2–4 were dominated by ONE class: a metric whose DEFINITION is
  scale-invariant, implemented with ARITHMETIC that is not — every norm squares
  its input, so float32 underflows before the data does. `utils` now exports
  `safe_unit_rows`, `safe_unit_rows_torch` and `safe_norm`. Check for
  remaining sites, and check the BOUNDARY of each guard: a band must match the
  SQUARING DEPTH downstream, not the single square in front of it.
- Where a fix would break BIT-EXACT parity with a released reference on
  ordinary data, the rescale is applied CONDITIONALLY and by a POWER OF TWO
  (`cknna`, `schatten`). Verify BOTH properties.
- ALL ELEVEN registered normalized twins were removed. Exactly one twin has
  since passed the mechanism test — `geometry_isoscore.isoscore_effective_dims`
  — recorded in `normalization.py`. Judge `normalized_ok` against that bar:
  a twin needs a MECHANISM THAT SURVIVES A DIRECT TEST (isometric embedding,
  zero-padding, or a function-preserving reparameterization) AND a measured
  reduction that does not simply relocate the association. A good fit is not
  evidence.

## Output

Write `verification_campaign/verdicts/<task>.opus5.json` with the EXACT schema
of `verification_campaign/verdicts_round5_2026-09-05/<task>.opus5.json`.
`defect_location` is CODE or CONFIG; a non-GREEN `verdict` with
`defect_location: CONFIG` is self-contradictory. Quote concrete measured
numbers in every reason field.

## What round 5 established, and what changed after it

Round 5 reached 34 of 51 dual sign-offs. Opus 5 closed at 46 GREEN / 4 YELLOW /
1 RED across all 51; gpt-5.6-sol at 35 / 8 / 8. Prior verdicts for your tasks
are in `verification_campaign/verdicts_round5_2026-09-05/` — read BOTH
reviewers' and say which the measurement supports where they split.

Every remaining non-GREEN was acted on AFTER those verdicts were written:

- **`spectral`** — I had added a `(n-1)/sum-log` Hill form below 20 tail points
  on a round-4 reviewer's claim about WWFit, WITHOUT checking it against live
  WeightWatcher. It made the KS statistic a DATA-INDEPENDENT constant at
  n_tail = 2 (`1 - e^-1 - 0.5 = 0.132120558828558`), so the scan picked a
  2-point tail on nearly every matrix. Reverted to plain `n`, which matches
  live WeightWatcher 0.7.7 to mean 3.634e-15. Pinned as a regression.
- **`betti_curve`** and **`persistence_entropy`** — the 10th and 11th
  post-final-norm instances, both now hooked.
- **`persistence_landscape`** — the "keep the most persistent bars" heuristic
  was unsound; replaced with a provably sufficient set (if L lower-bounds
  lambda_K(t*), every bar among the K tallest there has half-persistence >= L).
  Evaluation is now chunked: 3,001 bars in 1.7 s at 1.47 GB, where the dense
  form needed ~180 GB.
- **`zigzag`** — zero-Betti steps return 0 per the paper's footnote 6, not NaN.
- **`isoscore`** and **`neural_collapse`** — both lost TRANSLATION invariance by
  reducing float32 data before promoting; both now promote first.
- **`matrix_entropy`** — the `_meta_` demotion was a no-op (identical float
  under two names); the magnitude guard now rescales by a POWER OF TWO, so
  whether it fired is no longer observable in the result (0.00e+00 across
  2^-100 … 2^100).
- **`cknna`** — the safe-normalisation decision is now PER ROW, not on one
  global maximum.
- **`correlation_dimension`** — the atomic ESTIMAND SWITCH is gone; the
  headline is always the continuous-part exponent and the atom mass is
  reported.
- **`representation_sensitivity`** — the head probe covers `logit_scale` and
  all scored positions, and its SCOPE is disclosed.
- **`lipschitz`** — the paper's literal `(||h||+eps)` form is emitted under its
  own name alongside the epsilon-free headline.

Full suite at this state: **1588 passed, 2 skipped, 0 failed.**
