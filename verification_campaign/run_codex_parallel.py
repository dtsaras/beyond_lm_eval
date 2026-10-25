#!/usr/bin/env python3
"""Parallel gpt-5.6-sol verification over the paper-core tasks.

Resumable: skips any task that already has a `.codex.json`. Each task is an
independent `codex exec`, so a mid-run quota exhaustion costs at most the
in-flight tasks rather than the whole pass.

    CODEX_WORKERS=6 python3 verification_campaign/run_codex_parallel.py

Serially the pass took ~11 min/task on the first sample, i.e. ~9 hours for 51.
Concurrency brings that to roughly an hour and a half; the host has 224 cores
and was at load ~27 when this was written, so the constraint is the account,
not the machine.
"""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.expanduser("~/projects/beyond_lm_eval")
CAMP = os.path.join(ROOT, "verification_campaign")
VERD = os.path.join(CAMP, "verdicts")
WORKERS = int(os.environ.get("CODEX_WORKERS", "6"))
PER_TASK_TIMEOUT = int(os.environ.get("CODEX_TASK_TIMEOUT", "5400"))

PROMPT = """You are an adversarial verification reviewer for the BLME repo at {root}.

FIRST read {camp}/BRIEF.md IN FULL. It defines the TWO-AXIS standard -- `verdict`
is IMPLEMENTATION FAITHFULNESS ONLY, while comparability, config, protocol and
sample size go on `comparable`/`confounds` -- and the AXIS LEAKAGE rule.

RUN EVERYTHING LOCALLY. Do NOT ssh to any host; eez128/eez129/eez130/eez131/
eez132/h5 are OFF LIMITS, eez130 especially. Do NOT use
`until ! pgrep -f "<script>.py"` wait-loops: that pattern matches the waiter's
own command line and never exits.

ROUND 15. FULL RE-REVIEW AFTER A 15-SITE CLASS REPAIR.

Your round-13 and round-14 passes were right where the Opus pass was too
lenient: 8 REDs then 3 more, all real code defects, all now repaired and
verified against your own counterexamples.

The largest repair since then was NOT one you reported. The shared-cache class
you RED-flagged on `geometry_isoscore`, `geometry_svd` and
`geometry_prediction_alignment` -- a populated cache that carries no hidden
states, so the task raises or refuses instead of self-collecting -- was swept
BY READING in round 12 and reported as 2 sites. EXECUTING that check against a
real attention-only cache across all 51 tasks found **15**:

  raised outright : geometry_cka, geometry_collapse, geometry_lid, geometry_hsic
  refused, though they run fine with NO cache at all :
                    geometry_cknna, geometry_lipschitz, geometry_mp_bulk_deviation,
                    geometry_procrustes_linearity, geometry_rsa,
                    geometry_trajectory_curvature
  plus the 5 already repaired.

Two of those fifteen -- geometry_collapse and geometry_lid -- were DUAL-SIGNED
by both reviewers at the time. All 51 verdicts are therefore void and you are
re-reviewing the whole set.

Also repaired since round 14:
  geometry_perplexity   the normalisation guard now compares the decoded STRING,
                        not its byte length, because byte-PRESERVING
                        normalisation ('a' + U+0323) slipped through.
  geometry_spectral     the Hill MLE is withheld when the top-k tail is
                        numerically flat; rotation-split equal singular values
                        had produced alpha = 1.2e15 where none exists.
  geometry_intrinsic_dim  the static route and collect_hidden_states no longer
                        narrow to float32; they preserve the incoming float dtype.

Full suite: 2635 passed, 158 skipped, 0 failed. Reference-parity coverage 51/51,
401+ executed assertions. `tests/test_cache_population_guard.py` now enforces
the cache class across all 51 tasks by RUNNING them, with no allow-list -- a
task is at fault only if it works with no cache and fails with an unusable one.

Assume the same failure mode as before: repairs applied at the reported site and
not swept. A defect you find will be one the tests do not ask about.

Also check the two tests that were CHANGED rather than fixed --
`test_betti_zero_distance_edges_do_not_inflate_beta0` (now
`test_betti_duplicate_rows_are_quotiented_like_the_reference`) and the
`_count_betti(np.zeros((2,5)))` degenerate guard. Both were updated on the
grounds that they encoded BLME's own semantics rather than the released
loader's, whose `read_csv` does `np.unique(..., axis=0)`. If that reasoning
is wrong, say so: changing a test to match the code is exactly how a real
defect gets ratified.

Your round-10 pass closed at 37 RED / 12 GREEN / 2 YELLOW with zero axis
leakage. Opus 5 then scored the SAME code at 28 GREEN / 22 YELLOW / 1 RED. That
gap was not disagreement about facts — on nearly every task both of you measured
the same behaviour. It was disagreement about SEVERITY, and Opus supplied the
thing your pass lacked:

**GIVE EVERY EXTREME-SCALE FINDING A REACHABILITY BOUND.** State the magnitude a
real model actually reaches, and whether the released reference survives there.
Examples from round 11: `schatten`'s guard fires only above |X|max 1.6641e17
while gpt2's largest hidden magnitude is 3054.12 — 5.4e13x below — and the
released MNN.py itself returns inf at 1e-30 and nan at 8.5e37; `cka`'s NaN at
2.406e38 reproduces in Kornblith's own released code on the same float32 array,
and only disappears after a float64 promotion no shipped route performs.
"Wrong at 1e37" and "wrong where no model goes, where the reference is also
wrong" are different claims. A RED without a reachability bound will be
downgraded.

FOUR OF YOUR ROUND-10 CLAIMS WERE REFUTED BY MEASUREMENT. Do not re-file without
NEW evidence addressing the counter-measurement:
 * `geometry_rsa` — your permutation RED was a cherry-pick. Over ALL 5040
   coordinate permutations of your own fixture, BLME's max error is 0.1369018129
   and live rsatoolbox's is LARGER at 0.1433226690; BLME is exact on 24/5040,
   the reference on 8/5040. Your cited 0.9235-vs-1.0 pair is one of the 8 where
   the reference happens to land on 1.0.
 * `geometry_collapse` — your +1e6 translation quantised the cloud. float32's
   ULP at 1e6 is 0.0625; on a lossless (on-grid) translation BLME's centred
   erank is bit-identical, rel 0.000e+00, including at erank 1.4476 where your
   fixture sat.
 * `geometry_magnitude` — magnipy does NOT de-duplicate; it perturbs the
   singular zeta. BLME matched it bit-for-bit before the "fix", which was
   reverted.
 * `geometry_neural_collapse` — zero-padding invariance does not reproduce; the
   mechanism is confined to an eigenvalue window of relative width 2.2e-16, and
   numpy.linalg.pinv uses the same max(shape)*eps convention.

**FIXTURE INTEGRITY.** `(P + 1e6).astype(np.float32)` quantises a unit-scale
cloud onto a 0.0625 lattice, so the task is handed destroyed data and correctly
reports garbage. This produced five false findings in round 10, one of them
yours. Use `tests/adversarial_fixtures.py`: `on_grid_translation` returns
`(baseline, translated, ok)` — compare against `baseline`, NOT your original
array, because it snaps to the grid first. Also `scale_ladder` (which ends at the
last value whose REDUCTIONS survive, not a round number — `dynamics_stability`
held to 1e38 and broke at 2.406e38), `degenerate_spectra`, `wide_dynamic_range`,
`centring_is_safe`, `narrowing_is_safe`. Say in the verdict that you checked.

WHAT CHANGED SINCE YOUR PASS — re-measure everything:
 * `apply_lm_head` now applies Gemma-2's `final_logit_softcapping`. It did not,
   which flipped `dynamics_interpolation`'s `convexity_gap` SIGN (+0.1347 ->
   -0.1424, 194.6%). Three tasks share that helper.
 * A 13-site token-pooling population now pools in double.
 * `schatten`, `isotropy`, `rmt_bulk` fixed for narrow-before-centre.
   `consistency` and `superposition_strength` were checked and deliberately NOT
   changed — the first is lossless on the dtypes it receives, the second would
   lose its 2.07e-09 agreement with the official overlap-0.py.
 * `intrinsic_dim`'s CPU fallback is float64 + ball_tree (auto/brute retains
   1.7e-12 of Gram residue even in float64; only a tree is exact).
 * `isotropy`'s two `except np.linalg.LinAlgError` clauses could never fire —
   `safe_svdvals` raises `torch._C._LinAlgError`, which derives from
   RuntimeError, not ValueError. Both widened.
 * `correlation_dimension`'s atom branch was DEAD CODE (stored to a variable
   read earlier); the D2=0 logic now lives where the headline is produced.
 * `mahalanobis` fits its covariance in float64 — float32 REORDERED the scores
   at +1e6 (spearman 0.998123827), and AUROC depends only on order.
 * `sharpness` wraps its model mutations in the try/finally that restores them,
   and emits `_meta_top_eigenvalue_sign_stable` derived from a second power
   iteration at 4x the count — the sign was an artefact of `n_power_iter`
   (-104737 at 10, +111530 at 64 on pythia-70m).
 * `representation_sensitivity`'s bf16 linearity guard now probes at 1x/2x/4x
   and takes the worst; a real softcap passed the single-rung 3*eps test at
   0.02309 vs 0.02344 and cost +8.57%.

ONE CLAIM I COULD NOT REPRODUCE — bring the fixture if you still believe it:
`superposition_strength` emitting a `superposition_ratio_centered` BELOW the
Welch bound. On the real gpt2 unembedding it is 1.910674; on random V/m from
1.06 to 3.12 it never drops below 1 and rises to 16.6 as V approaches m. State
model, config and V/m if you re-file.

The full suite is green (2535 passed, 158 skipped, 0 failed). NOT evidence of
correctness: it was green while `geometry_schatten` emitted nothing at all.

TEST THE INVARIANCES AT THE EXTREMES, as you did last round: 1e-30, 1e-25,
1e-13, 1, 1e13, 1e25, 2.406159650500376e38, and translations of 1e3/1e6. That is
exactly where the other reviewer was not looking.

Note `defaults.yaml` is merged INTO self.config, so code-level
`.get("x", DEFAULT)` fallbacks never fire. A defect in
`~/projects/beyond_benchmarks/configs/tasks.yaml` is `defect_location: CONFIG`
and does NOT make `verdict` non-GREEN. Where BLME matches a RELEASED reference
bit-for-bit, that is faithfulness even if the paper's prose differs. A metric
that cannot be compared across models is NOT thereby incorrect.

WRITE your verdict to {verd}/{task}.codex.json using the EXACT schema of the
existing verdict files (task, reviewer "codex", verdict, verdict_reason,
defect_location, comparable, comparable_reason, normalized_variant,
normalized_ok, github, github_repo, github_reason, paper_number, paper_reason,
confounds, confounds_reason, defects, fix). Quote concrete measured numbers in
every reason field. Modify NOTHING outside {verd}, and do not touch
`*.opus5.json` -- a second reviewer is running concurrently.
"""


def main():
    _list = os.environ.get("CODEX_TASK_LIST", "tasks.json")
    tasks = [t["task"] if isinstance(t, dict) else t
             for t in json.load(open(os.path.join(CAMP, _list)))]
    todo = [t for t in tasks
            if not os.path.exists(os.path.join(VERD, f"{t}.codex.json"))]
    print(f"[{time.strftime('%H:%M:%S')}] {len(todo)} of {len(tasks)} tasks "
          f"to verify, {WORKERS} workers", flush=True)

    lock = threading.Lock()
    done = {"ok": 0, "fail": 0}

    def run(task):
        t0 = time.time()
        out = os.path.join(VERD, f"{task}.codex.json")
        try:
            subprocess.run(
                ["codex", "exec", "--skip-git-repo-check",
                 PROMPT.format(root=ROOT, camp=CAMP, verd=VERD, task=task)],
                cwd=ROOT, capture_output=True, text=True,
                timeout=PER_TASK_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:  # pragma: no cover - defensive
            print(f"    {task}: {type(e).__name__}: {e}", flush=True)
        dt = time.time() - t0
        ok = os.path.exists(out)
        with lock:
            done["ok" if ok else "fail"] += 1
            n = done["ok"] + done["fail"]
            print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)} {task:34s} "
                  f"{'WROTE' if ok else 'NO VERDICT'} ({dt / 60:.1f} min)",
                  flush=True)
        return task, ok

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run, t) for t in todo]
        for _ in as_completed(futures):
            pass

    print(f"[{time.strftime('%H:%M:%S')}] complete: {done['ok']} written, "
          f"{done['fail']} missing", flush=True)


if __name__ == "__main__":
    main()
