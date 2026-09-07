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

ROUND 9. YOUR LAST PASS WAS THE SHARPER OF THE TWO, AND IT WAS RIGHT.

You and Opus 5 scored the same frozen tree. You disagreed on 20 tasks and
**every single disagreement was Opus GREEN / you non-GREEN**. The maintainer
adjudicated four of your REDs by direct measurement:

  * `geometry_weight_norms` stable rank 0.0 at 1e-25 and 10.0 at 1e30 against an
    exact 3.85 -- CONFIRMED, and now fixed.
  * `geometry_cka` NaN at magnitude 1e-13 -- CONFIRMED, and now fixed.
  * `topology_homology` 0.0 with 0 H1 bars at 2^±100 -- CONFIRMED, and now fixed.
  * `geometry_intrinsic_dim` Two-NN -30.15% at 1e-22 and NaN at 1e-23 -- **NOT
    REPRODUCED** on four fixtures (2000x16, 2000x64, 5000x128, 20000x768
    Gaussians; all invariant to 1e-8, no NaN, and none giving the ID~24.23 you
    quote at scale 1). If that claim holds, SAY WHAT THE FIXTURE IS -- the
    `mu <= 1` pre-filter on duplicate-heavy data is the likely regime. If it
    does not hold, withdraw it. Your zero axis-leakage record last round is why
    this is being asked rather than dismissed.

So do not soften. Your REDs were acted on. Bring the same standard.

WHAT CHANGED SINCE YOUR LAST PASS -- 63 source files, and RE-MEASURE EVERY CLAIM:

  * Five fidelity fixes: weight_norms stable rank and cka now factor the max out
    before norms/SVD (the ratio cancels it exactly); topology_homology rescales
    to unit max before ripser and scales the barcode back; a new `safe_svdvals`
    in `tasks/geometry/utils.py` because numpy CASTS singular values back to
    float32 and overflows to inf at 2.4e38; and
    `geometry_representation_sensitivity` now applies `config.logit_scale` to
    BOTH the softmax input and the gradient -- it was applied only in the
    linearity guard, so both halves were wrong (-4.23% at logit_scale 1.02,
    -59.9% at 1.5, verified against float64 autograd).
  * SEVEN tasks were reverted to read the RAW HF hidden_states tuple, because
    their released references do: matrix_entropy, contextualization, coe,
    schatten, concept_separability, task_vectors and cknna. Each was confirmed
    by cloning the reference repo and reading the line. VERIFY THE REVERTS --
    if any is now on the wrong side, that is a fidelity defect.
  * `cache.py`'s forward hook is wrapped in try/finally (it leaked into every
    later task in a run); mahalanobis averages 5 DISTINCT splits (split 1 used
    to duplicate split 0 bit-for-bit); magnitude's cache guard is now an exact
    match (it accepted an oversized cache, +11.6% in production); mp_bulk's
    `max_tokens` was REVERTED to 20000 (32000 never bound -- the corpus supplies
    23,272 tokens); zigzag's `persistence_frac` reached the STUDY config at last
    (the harness default had been changed and the override left at 0.3, so the
    whole H1 family stayed exactly 0.0 with 9 of 14 columns at sd 0).
  * Two provenance flags that reported the OPPOSITE of the truth are corrected.

STILL OPEN -- your REDs from last round that are NOT fixed. Re-state them with
current numbers so they are not lost:
  `geometry_correlation_dimension` (D2 must be 0 on a measure with an atom;
  returns 2.8835), `geometry_magnitude` (the paper's duplicate-point twin
  example: 1.886569920263 vs 1.900000000000), `geometry_rsa` (coordinate
  permutation must not change RSA), `geometry_vendi_score` (RBF is
  translation-invariant), `geometry_spectral`, `geometry_trajectory_curvature`.

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
