# The Pink-Ponk protocol — convergence, not a single pass

A metric is signed off ONLY when **both** Opus 5 and gpt-5.6-sol return GREEN on
the **same** code state. One reviewer's GREEN is evidence, never a sign-off, and
a fix applied after a sign-off voids it.

## The loop

1. **Opus reviews** → verdict written to `verdicts/<task>.opus5.json`
2. **Codex reviews** the same code → `verdicts/<task>.codex.json`
3. **If they disagree**, or either is non-GREEN:
   a. Read BOTH reasons. Do not assume the harsher one is right.
   b. **Challenge**: if a reviewer withholds GREEN for a reason that belongs on
      `comparable`/`confounds`, or disputes a metric that reproduces its paper's
      number or matches the released reference code, push back with evidence.
      Axis leakage has occurred in this campaign in BOTH directions.
   c. Where the finding is real, **fix it** and record the measurement.
   d. **Send it back to BOTH** — a fix invalidates both prior verdicts, not just
      the one that objected.
4. Repeat until both are GREEN on the same code state.

## Why both directions of challenge matter here

- Codex's first pass returned 26 RED / 8 YELLOW / 17 GREEN and it later
  self-diagnosed the cause: *"I applied a broader standard… the zero-GREEN
  pattern was caused partly by my axis leakage."* Those verdicts are quarantined
  in `verdicts_stale_2026-09-03/`.
- Opus caught five separate errors in fixes made during this campaign — a
  non-permutation derangement, a CUDA run-killer, a no-op dtype fix, a vacuous
  test anchor, and two gates that censored the configurations they were meant to
  protect.
- Opus also *withdrew its own* recommendation twice on re-measurement, and
  removed 10 of 11 normalized twins it had earlier accepted.

Neither reviewer is a rubber stamp for the other, and neither is presumed
correct. The convergence is the evidence.

## The same bar applies to `normalized_ok`

A normalized variant needs BOTH sign-offs too. The twin bar is stricter than a
correlation: a MECHANISM THAT SURVIVES A DIRECT TEST (isometric embedding,
zero-padding, or a function-preserving reparameterization) AND a measured
reduction that does not merely move the association to another architecture
variable. A good fit is not evidence.
