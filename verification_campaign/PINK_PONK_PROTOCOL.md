# The Pink-Ponk protocol — convergence, not a single pass

**Revised 2026-10-26 (user decision):** the Codex reviewer (gpt-5.6-sol /
gpt-6-astra) is retired. The two reviewers are now two **independent,
fresh-context Claude reviewers**, `claude_a` and `claude_b`. Everything else
that made the old bar meaningful is kept: convergence on the same code state,
and a fix voids every prior verdict on what it touched.

A metric is signed off ONLY when **both** `claude_a` and `claude_b` return GREEN
on every gating axis against the **same fingerprint**. One reviewer's GREEN is
evidence, never a sign-off.

## Independence is structural, not requested

- Each reviewer works from a `git archive` export of the frozen tree, which has
  **no `.git`** and no `verification_campaign/` directory. Verdicts sit in
  history, so a checkout with `.git` would let a reviewer read them via
  `git show`.
- Neither reviewer sees the other's verdict, any earlier verdict, or the
  maintainer's hypothesis about the task. Each gets the same task brief
  (`BRIEF.md`) and the list of changes already planned for the task.
- The two slots use DIFFERENT strategies, because two reviewers on the same
  model share blind spots (lid was dual-GREEN while 50% wrong):
  - `claude_a` works **derivation first**: re-derive the estimator from the
    paper, then check the code against that derivation, then against the
    released reference.
  - `claude_b` works **counterexample first**: it starts from
    `tests/adversarial_fixtures.py` (degenerate spectra, scale ladders, on-grid
    translations, duplicates, N ≲ d) and from the released reference's
    behaviour on them, and only then reads the paper.
- Reviewers are read-only on the package source. They may write only in their
  scratch directory, and they must EXECUTE every counterexample they cite.

## Fingerprint, not mtime

Each verdict records `fingerprint_at_launch`, produced by
`python source_hash.py --launch <task>` at the START of the round, against
the tree the reviewer is about to read. The fingerprint covers the task's
import closure, the runtime modules, the resolved config (defaults merged with
the study overrides) and the library versions. `python source_hash.py`
reports every verdict as VERIFIED, STALE, or UNVERIFIED (no fingerprint).
Stamping after the fact is retired: it once turned "0 of 102 verified" into
"102 of 102".

## Gating axes (all must be GREEN, from BOTH reviewers)

| axis | GREEN means — operationally |
|---|---|
| `verdict` | The code computes what the source paper defines. It is checked against the reviewer's own derivation and the released reference. Config, sample-size and comparability concerns do NOT belong here (see AXIS LEAKAGE in `BRIEF.md`). |
| `precision` | The precision oracle (`tests/test_precision_oracle.py`) passes for the task's keys, and the reviewer found no executed counterexample of arithmetic error above the task's declared contract. |
| `llm_adaptation` | The task's DECLARED token, layer and pooling policy (`TOKEN_POLICY`, the post-final-norm declaration) is implemented exactly as declared, and the declaration matches the policy the user decided (position 0 excluded for non-LLM-origin metrics; LLM-native metrics follow their paper). Disagreeing with the policy itself is recorded in `confounds`, not here. |
| `comparable` | As before: whether the number can be compared across models of different size, family and tokenizer. |
| `normalized_ok` | As before: a twin needs a mechanism that survives a direct test AND a measured reduction. |

`simplification` is recorded but never gates.

## The loop

1. Freeze the tree, export it, and stamp `fingerprint_at_launch` for every task in the round.
2. Run `claude_a` and `claude_b` on the export in parallel, writing `verdicts/<task>.claude_a.json` and `verdicts/<task>.claude_b.json`.
3. If they disagree, or either is non-GREEN:
   a. Read BOTH reasons. Do not assume the harsher one is right.
   b. **Challenge.** If a reviewer withholds GREEN for a reason that belongs on `comparable` or `confounds`, or disputes a metric that reproduces its paper's number or matches the released reference code, push back with evidence. Axis leakage has occurred in this campaign in both directions.
   c. Where the finding is real, **fix it** under the fix rules: a regression test that fails on the pre-fix code, a derived `_meta_` flag, `sweep_siblings.py` over the class, and reference-parity gating.
   d. A fix to a shared helper is applied only BETWEEN rounds, because it voids nearly every task.
   e. **Send it back to BOTH.** A fix invalidates both prior verdicts, not just the one that objected.
4. Repeat until both are GREEN on the same fingerprint. A task that has not converged after 3 signing rounds goes to the user with both positions and the executed evidence.

Round 0 is a discovery round and does NOT sign. It covers the kernel and paper axes only, with one reviewer per task, and its findings feed the fix batches.

## Why both directions of challenge matter (history)

- The first Codex pass returned 26 RED / 8 YELLOW / 17 GREEN, and the reviewer later self-diagnosed axis leakage. Those verdicts are quarantined in `verdicts_stale_2026-09-03/`.
- Opus caught five separate errors in fixes made during the campaign, and twice withdrew its own recommendation on re-measurement.
- Neither reviewer is a rubber stamp for the other, and neither is presumed correct. The convergence is the evidence.

## The same bar applies to `normalized_ok`

A normalized variant needs BOTH sign-offs too. The twin bar is stricter than a correlation. It needs:

1. a MECHANISM THAT SURVIVES A DIRECT TEST (isometric embedding, zero-padding, or a function-preserving reparameterization); and
2. a measured reduction that does not merely move the association to another architecture variable.

A good fit is not evidence.
