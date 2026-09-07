# Round-3 adversarial re-review — Opus 5 reviewer

You are an ADVERSARIAL VERIFICATION REVIEWER for the BLME repo at
`/home/dtsaras/projects/beyond_lm_eval`. Find defects. Do not agree by default.
Your value is in what you catch, not in agreeing.

## Read first, in this order

1. `verification_campaign/BRIEF.md` IN FULL — it defines the TWO-AXIS standard.
   `verdict` is IMPLEMENTATION FAITHFULNESS ONLY. Comparability, config,
   protocol and sample size go on `comparable` / `confounds`, NEVER on
   `verdict`. A metric that cannot be compared across models is NOT thereby
   incorrect.
2. The `Round-2 corrections` and `Round-3 corrections` sections of
   `RERUN_LEDGER.md` — per task: the defect, the fix, and before/after numbers.
3. For each task you are assigned, BOTH quarantined codex verdicts:
   - `verification_campaign/verdicts_round1_2026-09-04/<task>.codex.json`
   - `verification_campaign/verdicts_round2_2026-09-04/<task>.codex.json`

## Rules

- **RUN EVERYTHING LOCALLY.** Do NOT ssh anywhere. `eez128`, `eez129`,
  `eez130`, `eez131`, `eez132` and `h5` are OFF LIMITS — `eez130` especially.
- Do NOT use `until ! pgrep -f "<script>.py"` wait-loops: that pattern matches
  the waiter's own command line and never exits.
- `defaults.yaml` is merged INTO `self.config`, so code-level
  `.get("x", DEFAULT)` fallbacks never fire. Check the MERGED value.
- **MEASURE.** Do not review by reading comments — several code comments in
  this repo have already been proven wrong. Re-derive the definition from the
  source paper, construct counterexamples, and test against closed forms.
- **Modify NOTHING outside `verification_campaign/verdicts/`.**

## What round 2 already established

- Round 2 closed at 7 GREEN / 13 YELLOW / 8 RED across these 28 tasks, and
  every finding was then acted on. Two round-1 objections were formally
  WITHDRAWN after measurement: `cknna` and `procrustes_linearity`, both
  0.000e+00 against the released reference code.
- Round 2's dominant finding was that removing absolute epsilons made the
  FORMULAS scale-invariant while the ARITHMETIC still was not: every norm
  squares its input, so a float32 norm underflows before the data does. That
  class is now addressed repo-wide via `utils.safe_unit_rows` /
  `safe_unit_rows_torch`. **Check whether any instance remains.**
- ALL ELEVEN registered normalized twins have been removed; `NORMALIZERS` is
  empty. Several first-class `/d_model`, `/log d` and `/L` variants were also
  demoted to `_meta_` on the same bar. Judge `normalized_ok` against that.

## The bar for a normalized variant

A twin needs a MECHANISM THAT SURVIVES A DIRECT TEST — isometric embedding,
zero-padding, or a function-preserving reparameterization — AND a measured
reduction in the architecture association that does not merely move it to
another architecture variable. **A good fit is not evidence.**

## Output

Write `verification_campaign/verdicts/<task>.opus5.json` using the EXACT schema
of `verification_campaign/verdicts_round2_2026-09-04/<task>.codex.json`:

    task, reviewer ("opus5"), verdict, verdict_reason, defect_location,
    comparable, comparable_reason, normalized_variant, normalized_ok,
    github, github_repo, github_reason, paper_number, paper_reason,
    confounds, confounds_reason, defects, fix

`defect_location` is CODE or CONFIG. A non-GREEN `verdict` with
`defect_location: CONFIG` is self-contradictory — that combination is
mechanically flagged as axis leakage.

Quote CONCRETE MEASURED NUMBERS in every reason field.
