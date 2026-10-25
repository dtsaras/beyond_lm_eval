# Campaign state — 2026-09-11

## One-line status

**Dual sign-off: 0 of 51.** Both reviewer models are quota-blocked account-wide
until **2026-09-15 09:31**. Suite green: 2637 passed, 158 skipped, 0 failed.

## What blocks sign-off

Nothing technical. It needs a reviewer model, and the quota is ACCOUNT-LEVEL:
probing `gpt-5.6-sol` by name returns the identical error and identical reset
time as `gpt-6-astra`, so switching `~/.codex/config.toml` would NOT get a
reviewer back sooner. The earlier "sol returns 2026-09-11 11:22" figure is
superseded.

## What the watcher will do on 2026-09-15

`auto_resume_codex.sh` is NOT a passive poller. On its first successful probe it
QUARANTINES every live verdict into a dated directory and launches a run over
all tasks not dual-signed. It already did this once, at 2026-09-10 01:21:31,
moving 23 verdicts into `verdicts_round6_2026-09-10/` before dying on quota.

Every round-15 verdict is preserved in `verdicts_round15_partial_2026-09-09/`
(23 files, verified to be a superset of the quarantine). Treat that directory as
the durable record, not `verdicts/`.

## Round 15 (partial: 23 of 51 before quota died) — all adjudicated

2 GREEN / 5 YELLOW / 16 RED. Disposition:

| cause | tasks | status |
|---|---|---|
| disk-offload dtype corruption (self-inflicted) | cka, cknna, collapse, hsic, intrinsic_dim, isoscore, lid, lipschitz | fixed + guard |
| narrowing after a double reduction (12 sites) | correlation_dimension + 11 | fixed |
| collector narrowing before subtraction | perturbation_growth | fixed |
| tensor branch float32, numpy branch float64 | magnitude | fixed |
| bf16 logits under cross-entropy | gradient_flow | fixed |
| ambiguity census skipped list-valued entries | categories | fixed |
| occurrence doubling | hubness | NOT REPRODUCED — needs their fixture |

28 tasks were never reviewed in round 15.

## Guards added this campaign (run the population, don't read it)

- `tests/test_cache_population_guard.py` — every task against a real
  attention-only cache. No allow-list: a task is at fault only if it works with
  NO cache and fails with an unusable one.
- `tests/test_offload_parity_guard.py` — every task's headline must be identical
  between the in-memory and disk-offloaded routes, plus a float64 round-trip.
- `tests/tasks/parity/test_coverage_gap_parity.py` — the 6 tasks that had no
  executed reference check (coverage is now 51/51).
- `tests/tasks/parity/test_independent_reimplementation.py` — CKA, HSIC and
  IsoScore re-derived from their papers rather than run through existing tests.

## Open items that need a DECISION, not more analysis

1. **Config echoes in the feature table.** `dynamics_generation_diversity.temperature`
   (0.8), `dynamics_perturbation_growth.rel_eps` (0.01),
   `geometry_spectral.tail_fraction` (0.2) are settings echoed back as public
   features, plus 30 mis-prefixed `*_meta_prescale`-style keys in
   `topology_persistence_landscape`. Demoting them is a RENAME of the results
   schema and will break anything in `beyond_benchmarks` that reads those keys.
2. **The study runs a different configuration from the shipped defaults.** 34 of
   77 shared keys diverge; six are REDUCTIONS, and one is costly:
   `dynamics_sharpness` `n_hutchinson` 24 -> 8 puts the trace's relative standard
   error at **1.11 — larger than the estimate**. (`n_power_iter` 64 -> 10 costs
   nothing: top eigenvalue moves ~1e-4.)
3. **Keys that cannot discriminate.** 11 of 386 public float keys return the same
   value on gpt2 and distilgpt2 for structural reasons — including
   `topology_persistence_entropy`'s entire H0 family, which is log2(N-1) x 0.994,
   i.e. the sample size rather than the model.

## Open technical items

- `geometry_phd_dimension`: the narrowing class was probed and NO counterexample
  found; recorded as an open question, not a clearance.
- 11 narrowing candidates outside the paper-core 51 were never probed
  (`edge_attribution` x3, `knowledge_neurons` x2, `bias`, `activation_kurtosis`,
  `induction`, `sparsity`).
- Every determinism/order/stability figure in the findings was measured at the
  SHIPPED defaults, not the study's overrides, so for the six reduced tasks it
  does not describe the published runs.

## The methodological record

Full detail in `OPUS_ROUND13_FINDINGS.md`. The short version: **four sweeps in
this campaign undercounted or were vacuous**, and each was caught only by
checking the sweep itself — cache (read-only said 2 sites, executing found 15),
narrowing (7 -> 12 -> 17 as each query expressed one more spelling), the
study-config sweep (reported 0 divergences because it compared nothing), and the
order-sensitivity measurements (taken below the shipped sample size). A sweep
must report its denominator, and a sweep that has not been EXECUTED is not
evidence.
