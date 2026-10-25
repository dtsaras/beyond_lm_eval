# RESUME HERE — hardening session 2026-10-26 (pod1, paused by the user)

**Plan (approved):** `~/.claude/plans/i-previously-asked-another-tender-brooks.md`.
Read its "Amendments" section first; it overrides the phase text. User
decisions are recorded in the plan and in memory (`hardening-plan-2026-10`).
**Open decisions for the user:** `verification_campaign/DECISIONS_PENDING_2026-10-26.md`.

## Where things stand
- Branch `hardening/2026-10` (main tree):
  - `d106ef9` is the checkpoint of the rounds 12–15 state;
  - `bd3b9b7` holds the Phase 0 tooling;
  - `5f6f36d` holds the crash fixes (made on `hardening/2026-10-crashfix`);
  - `b6980a9` merges them.
- `main` is untouched, and nothing has been pushed. Ask before any push.
- Golden snapshots are in `~/hara-artifacts/blme/golden/`:
  - `d106ef9_shipped_cache_run{A,B}` is the pre-CBWR pair (20 nondeterministic leaves);
  - `b6980a9_shipped_cache_run{C,D}` is the post-merge CBWR pair, which is the **Phase 2 baseline**. Check it with `compare C D`; whether it was bit-identical is recorded below once known.
- **Env:** venv `~/hara-venvs/beyond-lm-eval`. Never put "blme" in a venv path. pod1 has no git identity, so commits pass `GIT_AUTHOR_*`/`GIT_COMMITTER_*` for Dimitrios Tsaras.

## Done this session
- Phase 0:
  - `.gitignore` covers the 1.8 GB reviewer scratch;
  - checkpoint commit;
  - sharded runner (scratch `shard_pytest.py`, 16 workers, CUDA hidden);
  - golden tool `tests/golden/golden_snapshot.py`, with snapshots in `~/hara-artifacts/blme/golden/`;
  - `source_hash.py` rewritten: import closure + runtime modules + resolved config + library versions, `--launch`, `--self-test`, `--write` retired;
  - `PINK_PONK_PROTOCOL.md` moved to the claude_a/claude_b bar.
- Phase 1 (partial):
  - `_is_blme_hook` decides ownership by package path, with a known-negative test;
  - `requires-python>=3.10`, scikit-learn made core, CI matrix 3.10–3.12;
  - the 2 guards read `tests/fixtures/paper_core_tasks.json` instead of a cwd-relative path;
  - the reference guard uses `~/.cache/blme-refrepos` and honours `BLME_REQUIRE_REFS=1`. The schatten MNN pin was SKIPPING on pod1 and now passes at 0.0; its reference was saved from scratch with a sha256 check.
- Phase 3 (early, crash-only, no shipped number moves): bf16 `.numpy()` crashes (offload, lipschitz, rsa), the phd `pooling: mean` crash, hubness duplicate `k_values`, and the refusal sklearn guard. Each test fails on pre-fix code for the original reason.

**Correction:** Llama-3.1-8B is NOT cached on pod1 (no config, no weights). The cached modern models are Qwen3-4B and Qwen3-8B, and neither adds a BOS. Download a BOS model (Llama/Gemma) before measuring the BOS case.

## Findings this session that change the plan
- **Golden runs are not bit-deterministic as first built.** Two runs of `d106ef9` with fresh processes, 2 pinned threads, CPU and a fixed corpus differed in 20 of 6,184 values (3 of 258 entries, relative 3e-8..2e-6): `topology_persistence_entropy` and `consistency_logical` on gpt2, and `geometry_svd` on tiny-random-gpt2. The likely cause is MKL's alignment-dependent kernels. The golden tool now sets `MKL_CBWR=COMPATIBLE`, `MKL_DYNAMIC=FALSE` and `OMP_DYNAMIC=FALSE`, and `compare --tolerate-from BASE0 BASE1` tolerates only leaves that differ between two baseline runs. That was checked: a 1-ulp change on a deterministic leaf is flagged. **The CBWR setting has not been validated with a fresh pair yet.**
- **Golden coverage gaps.** On gpt2, `dynamics_sharpness` and `geometry_intrinsic_dim` hit the 30-min timeout at 2 threads in both dtypes, so they are locked on the tiny model only. Errors are only the expected `sae_lens`-missing and `consistency_knowledge_capacity` dataset cases.
- **Sink census (decision #6):** on Qwen3-4B and Qwen3-8B, position 0 carries 91–99% of the centred sum of squares and delimiters do not. See the decision packet.
- **Hidden CUDA changes code paths.** The sharded runner hides CUDA, so 5 tasks take their CPU paths and the 51-task guards run more than 45 min each. The handoff's ~1 h serial baseline evidently ran with GPU0 visible. Compare like with like.

## Not done yet (next steps, in order)
1. Merge the crashfix branch, then run the full sharded suite on the merged state.
2. `refresh.py` rewrite for claude_a/claude_b and the new verdict axes; it still hard-codes opus5/codex and an old /tmp HTML path.
3. Rest of Phase 1:
   - campaign moves to beyond_benchmarks, then the scratch tar (after the user confirms);
   - delete the stale paper copy;
   - packaging extras (dionysus → `[topology]` with a loud error);
   - `normalization.py` out of the package;
   - point the 5 library-free parity files at the real kernels.
4. Golden: run the study profile and the nocache/offload routes on `d106ef9` as the Phase 2 baseline.
5. Phases 2–5 per the plan.

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
