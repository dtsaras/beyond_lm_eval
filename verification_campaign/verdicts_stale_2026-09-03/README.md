# STALE codex verdicts — 2026-09-03. DO NOT READ AS CURRENT.

Moved here 2026-09-04 before the fresh gpt-5.6-sol pass.

Void for two independent reasons:

1. **They predate Rounds 13-39**, which changed behaviour under nearly every
   task — the post-final-norm hook, the cache hook, the steering estimand
   replacement, headline-key changes in `matrix_entropy`,
   `prediction_alignment` and `correlation_dimension`, the fp16 fix in
   `correlation_dimension`, and the removal of 10 of 11 normalized twins.
2. **They came from a miscalibrated pass** — 26 RED / 8 YELLOW / 17 GREEN,
   which codex itself later diagnosed: *"I applied a broader standard… the
   zero-GREEN pattern was caused partly by my axis leakage"* — folding config,
   protocol and comparability into the correctness axis, which `BRIEF.md` now
   explicitly forbids.

Retained only as an audit trail of how the standard was calibrated.
