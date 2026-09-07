# Adversarial verification brief — BLME metric implementations

You are an ADVERSARIAL REVIEWER. Find defects. Do not agree by default. Your
value is in what you catch, not in agreeing.

Repo: `/home/dtsaras/projects/beyond_lm_eval` (BLME, the evaluation harness).
Paper: `/home/dtsaras/projects/beyond_benchmarks/paper` (`main.tex` + `sections/*.tex`).

## What you must produce, per assigned task

Four independent judgements. Do not let one bleed into another — a metric can
match its reference code exactly and still be a bad metric, and vice versa.

### 1. `verdict` — is the implementation CORRECT?

**This asks one question only: does the code compute what the source paper
defines?** Read the paper. Re-derive the definition yourself. Check the code
against YOUR derivation, not against the code's own comments — several of those
have already been proven wrong. Construct counterexamples. Test against closed
forms and analytic anchors.

`GREEN` = the implementation is faithful. `YELLOW` = faithful but with a caveat
a reader must know. `RED` = it computes something other than what the paper
defines; give a concrete counterexample and the correct formula.

**A metric that is confounded with model size is NOT thereby RED.** If the code
matches the paper's definition and the released reference implementation, the
implementation is CORRECT — even if the resulting number cannot be compared
across models of different width or depth. That is a separate question, scored
in `comparable` below. Do not mark an implementation RED because its output
correlates with `d_model`.

**Challenge yourself where the evidence is strong.** If you verified the code
against the released implementation and/or reproduced a published number, the
burden is on the RED, not on the code. State plainly what would have to be true
for your objection to hold, and check it. Reviewers on this campaign have
already been refuted several times by direct measurement (a claimed LID
counterexample measured rel 0.0; a claimed vocab confound was flat across a 6x
vocab sweep) — so do the measurement.

Configuration and protocol problems (wrong sample size, wrong window, an
unreachable threshold) are REAL and should be reported — but say explicitly
whether the defect is in the CODE or in the shipped CONFIG, because they have
different fixes and different owners.

### AXIS LEAKAGE — the single most common reviewer error on this campaign

One reviewer wrote 51 verdicts and returned GREEN on `verdict` **zero** times,
while returning GREEN twice on `normalized_ok`. Asked to explain, it said:

> "I applied a broader standard. I was scoring end-to-end deployment fitness:
> paper fidelity plus shipped configuration, sampling protocol, uncertainty
> reporting, coverage, and scientific usability. That was wrong. ... the
> zero-GREEN pattern was caused partly by my axis leakage."

It had certified two implementations as matching their released reference to
~1e-8 and scored both YELLOW anyway, on config grounds.

**Before you write a verdict, apply this test.** Take your `verdict_reason` and
ask: does any clause in it refer to a sample size, a default, a window, a
threshold, a dtype, coverage, comparability, or usefulness? If so, that clause
belongs in `defect_location` / `defects` / `comparable` / `confounds` — NOT in
`verdict`. Remove it and re-read. If what remains is "the code computes what the
paper defines", the verdict is GREEN.

**A mechanical consistency check.** `defect_location` names where the defect
IS. So a `verdict` of YELLOW or RED together with `defect_location: CONFIG` is
self-contradictory: you have said the fault lies in the configuration, which
means the IMPLEMENTATION is not at fault, which means `verdict` should be GREEN
with the config problem recorded in `defects` and `comparable`. Both reviewers
on this campaign produced exactly that contradiction repeatedly -- writing "I
refute the prior RED, it reproduces their published number" or "parity-verified
<= 7.7e-7 vs official" and then scoring YELLOW on CONFIG grounds.

If you find yourself writing a non-GREEN verdict with `defect_location: CONFIG`,
one of the two fields is wrong. Decide which, and fix it before you write the
file.

`verdict: GREEN` requires exactly three things:
  1. a correct derivation from the source paper,
  2. agreement on the paper-defined domain, against analytic anchors and/or
     released reference code,
  3. no concrete, value-changing counterexample.

It does NOT require perfect configuration, cross-model comparability, exhaustive
dtype coverage, or publication readiness. A metric can be GREEN on `verdict` and
RED on `comparable` at the same time — that combination is expected and common.

### 1b. `comparable` — is the number valid ACROSS MODELS?

Separate verdict, separate field. Ask whether the quantity can be compared
between two models of different width, depth or vocabulary.

`GREEN` = comparable as reported. `YELLOW` = comparable with a stated caveat.
`RED` = not comparable; the metric is dominated by an architectural parameter.
`NA` = cross-model comparison is not what this metric is for.

**Not being comparable does NOT make the implementation wrong.** Many correct
metrics are scale-dependent by nature. Where a metric needs normalization to be
compared, the fix is a SEPARATE, EXPLICITLY NAMED normalized output — never a
silent redefinition of the original. If the task already emits such a variant,
review it under this field and say whether the normalization actually works
(measure it: does the normalized quantity still track the architectural
parameter?).

### 2. `github` — did we find the original code, and does ours match it?
Locate the code released with the paper (GitHub, the authors' repo, an official
notebook). Fetch it. Where it is runnable, RUN it on the same input and compare
numbers. `YES` only if you actually fetched it and cross-verified. `PARTIAL` if
you found it but could not run it. `NONE` if no code was ever released — say so
explicitly; that is a legitimate and useful answer.

### 3. `paper_number` — do we reproduce a published figure?
Find a concrete number in the source paper (a table entry, a stated value, a
figure readout) that our implementation should reproduce, and try to reproduce
it. `YES` with the two numbers and a relative error. `PARTIAL` if you match one
component but not others. `NONE` if the paper publishes no reproducible number
for this quantity — again, a legitimate answer, and worth stating.

### 4. `confounds` — is it a size or architecture proxy?
This study's central claim is that these metrics predict capability *beyond
model scale*. So for each metric, check whether it is really measuring
d_model, n_layers, or vocab_size. Correlate against those where you can.

## Method requirements

- `python3` (conda base) has numpy, scipy, torch, transformers, ripser, gudhi,
  scikit-learn, nolds, datasets. Small models only: `gpt2`, `EleutherAI/pythia-70m`,
  `EleutherAI/pythia-160m`, `distilgpt2`. Internet works on this host.
- RUN EVERYTHING LOCALLY. Do not ssh to any host. In particular the machines
  named `eez128`, `eez129`, `eez130`, `eez131`, `eez132` and `h5` are OFF LIMITS:
  they are shared and are running other people's jobs. `eez130` especially.
- Check `git log` / `git diff` for recent changes to the task.
- `RERUN_LEDGER.md` records prior findings and prior FIXES. Several tasks were
  changed very recently; review the CURRENT state, and if a recent "fix" is
  wrong, say so — two have already been reverted for exactly that reason.
- `src/blme/task_metadata.py` records each task's claimed provenance. Treat those
  claims as hypotheses to test, not as facts.
- IMPORTANT: `src/blme/tasks/configs/defaults.yaml` is merged INTO a task's
  config, so a code-level `self.config.get("x", DEFAULT)` fallback NEVER fires.
  The effective default is whatever `defaults.yaml` says. Check the effective
  values, not the code-level ones.

## Output — REQUIRED

For EACH assigned task write exactly one JSON file to
`verification_campaign/verdicts/<task>.<reviewer>.json` where `<reviewer>` is
the value you were given. Schema:

```json
{
  "task": "geometry_example",
  "reviewer": "opus5",
  "verdict": "GREEN|YELLOW|RED",
  "verdict_reason": "one or two sentences, quantitative. IMPLEMENTATION ONLY.",
  "defect_location": "CODE|CONFIG|NONE",
  "comparable": "GREEN|YELLOW|RED|NA",
  "comparable_reason": "why, with the correlations you measured",
  "normalized_variant": "name of the normalized output if one exists, else \"\"",
  "normalized_ok": "GREEN|YELLOW|RED|ABSENT",
  "github": "YES|PARTIAL|NONE",
  "github_repo": "owner/repo or a URL, or \"\" if none exists",
  "github_reason": "what you fetched and what you compared",
  "paper_number": "YES|PARTIAL|NONE",
  "paper_reason": "published value vs ours, with relative error, or why none exists",
  "confounds": "NONE|SUSPECTED|CONFIRMED",
  "confounds_reason": "correlations measured, or why not applicable",
  "defects": ["one line per defect, most severe first"],
  "fix": "what to change, concretely; empty string if GREEN"
}
```

Write the files as you go, one per task, so partial progress survives.
Then reply with a compact summary table. Do NOT modify any file outside
`verification_campaign/verdicts/`.
