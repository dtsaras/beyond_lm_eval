# GPT-5.6 (codex) verification — BLOCKED ON QUOTA

## Status

`codex exec` returns, verbatim:

```
ERROR: You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage
to purchase more credits or try again at Sep 7th, 2026 10:43 AM.
```

Checked 2026-09-04 05:29. The account is rate-limited until **Sep 7 2026,
10:43 AM** — roughly three days out. This is an external quota limit, not a
configuration problem: `~/.codex/config.toml` still pins
`model = "gpt-5.6-sol"` with `model_reasoning_effort = "max"`, and the CLI
authenticates and reaches the model before erroring.

## What this blocks

The campaign goal requires **both** Opus 5 and gpt-5.6-sol to sign off, and
specifies that for normalized variants both MUST sign off. Neither can be
satisfied until quota returns. Concretely:

- Implementation axis: **47/51 Opus GREEN**, but `both-models impl` is 15.
- Normalized axis: **8 twins registered with Opus evidence only**; `both-models
  norm` is 2. Under the stated bar, **no normalized variant is signed off.**

## The 51 codex verdicts on disk are STALE — do not read them as current

They were written 2026-09-03 (newest 20:21) and are stale in two independent
ways:

1. **They predate essentially all of this work.** Rounds 13–31 changed the code
   under nearly every task — the post-final-norm hook, the cache hook, the
   estimand replacement in steering, the headline changes in matrix_entropy and
   prediction_alignment, the fp16 fix in correlation_dimension, and more.
2. **They were produced under a miscalibrated standard.** That pass returned
   26 RED / 8 YELLOW / 17 GREEN, and codex itself later self-diagnosed the
   cause: *"I applied a broader standard… the zero-GREEN pattern was caused
   partly by my axis leakage"* — folding config, protocol and comparability into
   the correctness axis, which `BRIEF.md` now explicitly forbids.

`refresh.py` already gates on Opus alone, so these do not inflate any headline
number. They are retained only as an audit trail.

## To run when quota returns

`verification_campaign/run_codex_verification.sh` re-reviews all 51 tasks
against the CURRENT code under the two-axis standard in `BRIEF.md`, writing
`verdicts/<task>.codex.json`. It processes tasks one at a time so a mid-run
quota exhaustion loses at most one task, and skips any task whose codex verdict
is already newer than its source file.
