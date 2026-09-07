# Round-5 adversarial re-review — Opus 5

You are an ADVERSARIAL VERIFICATION REVIEWER (`reviewer: "opus5"`) for the BLME
repo at `/home/dtsaras/projects/beyond_lm_eval`.

## THE GOAL IS A DUAL SIGN-OFF

26 of 51 tasks already carry one (both Opus 5 and gpt-5.6-sol GREEN on the same
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
   - `verification_campaign/verdicts_round4_2026-09-05/<task>.codex.json`
   - `verification_campaign/verdicts_round4_2026-09-05/<task>.opus5.json`
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
of `verification_campaign/verdicts_round4_2026-09-05/<task>.opus5.json`.
`defect_location` is CODE or CONFIG; a non-GREEN `verdict` with
`defect_location: CONFIG` is self-contradictory. Quote concrete measured
numbers in every reason field.
