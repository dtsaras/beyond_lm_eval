# Maintaining the metric reviews

The [dashboard](../VALIDATION.md) covers every registered task. The [readout table](READOUTS.md) covers every declared library headline, including the study’s frozen confirmatory keys. Each task has an editable scientific assessment, exact pytest selectors, an outstanding check and a dated receipt ID in [metrics.json](metrics.json). Each receipt preserves actual case outcomes from its tested Transformers stacks, input hashes, package versions and the underlying JUnit hashes. A passing case can check a kernel, a component or a narrower operational definition. Read its scope before using the result. Dated scientific findings are in [findings.md](findings.md).

## Checking the current checkout

With BLME’s test dependencies installed, run these commands from the library root.

```bash
python scripts/metric_status.py --check
python scripts/metric_status.py --json
python scripts/metric_status.py --check --study ../beyond_benchmarks
```

The release check runs the inventory and dashboard check: `pytest -m release` and `scripts/metric_status.py --check`, in `.github/workflows/release-check.yml` (run it by hand before tagging; it runs again on every `v*` tag). The per-commit suite deselects it, since it fails on every commit that changes a pinned input until a receipt is recorded. A new registry task appears as unreviewed. Editing a reviewed implementation, a shared runtime dependency, a linked test, a pinned fixture, the task’s metadata or its shipped defaults marks affected rows for review. A removed task remains visible. Changing the tests linked to a readout also requires a new receipt. Shared test support is pinned conservatively, so editing a shared fixture can flag several tasks. The optional study check additionally checks the recorded study inputs, including the confirmatory key manifest and study overrides.

`--write` renders both Markdown tables from the current checkout. It preserves all dated receipts. A row with changed inputs keeps its historical test counts alongside **REVIEW DUE**. An unexecuted stack, missing case, skip, error or failure has its own status. `--check` verifies that the tracker reports its evidence accurately. The dated CPU environments, external reference installations, independent study results and paper reproductions have separate provenance.

## Recording a scientific review

1. Choose a new receipt ID. Retain earlier receipt files. Update the applicable task assessments, named selectors and readout comparisons in `metrics.json`. Use an immutable export of the intended source and tests. Record its base commit and any working-tree changes included.
2. Pin the inputs on that export **before** collecting the test results. The recorder uses the companion study’s existing `verification/source_hash.py` to trace in-package imports and the runtime path. Include the study inputs used to make the assessment.
3. Run the linked cases on the pinned export on both supported stacks with reference dependencies required. Save the full JUnit XML, log, environment, command and any observed failures. Additional suites on the same stack must have disjoint test IDs.
4. Import those runs into a new receipt. Set the reviewed rows’ `receipt` fields to its ID, review the measured quantities and limitations, then render and check the tables. Adding a published-value or predictive claim requires the corresponding scientific result.

For example, with `export_dir` set to the frozen library checkout, `study_dir` to the companion study, and `art_dir` to the run directory, the recorder accepts the following commands. Supply the actual commit IDs and new receipt ID in the indicated variables. Omit `--tasks` to review the entire registry.

```bash
python scripts/record_metric_review.py pin \
  --id "$review_id" --base "$library_commit" --study-base "$study_commit" \
  --source "$export_dir" --study "$study_dir" --reviews docs/validation/metrics.json \
  --study-evidence configs/analysis_keys.yaml configs/tasks.yaml \
  --tasks geometry_domain_bpb --out "$art_dir/pin.json"

python scripts/record_metric_review.py record \
  --source "$export_dir" --pin "$art_dir/pin.json" --execution "$art_dir/execution.json" \
  --junit 5.2.0 "$art_dir/cpu-5.2.0.xml" --junit 5.10.2 "$art_dir/cpu-5.10.2.xml" \
  --out "docs/validation/receipt-$review_id.json"

python scripts/metric_status.py --write
python scripts/metric_status.py --check --study "$study_dir"
```

The execution metadata is a JSON object identifying the host, source export, environment, commands and raw artifact location. The [18 November receipt](receipt-2026-11-18.json) includes the worked example. Raw JUnit and logs for that review are on pod1 under `~/hara-artifacts/blme/metric-status-20261118/`. The 18 November per-readout mapping starts from the study’s coverage evidence, uses the exact 204 keys in the frozen analysis manifest, and incorporates later named tests for seven readouts. Descriptions of reference and analytic checks retain their individual tolerances and component boundaries.

The metric-fix campaign reruns the four affected tasks on the project verification stacks and on Transformers 5.22.2 and 5.23.0. The dashboard shows “No run” when a task has no receipt on a newer version. A separate regression receipt covers the shared runtime changes on the two project stacks. [Four-task receipt](receipt-2026-11-18-metric-fixes.json) · [Shared-runtime receipt](receipt-2026-11-18-regression.json)

The input-boundary review records nine regressions that failed against the pre-fix source. All nine passed in the 100-case focused run on each of the four Transformers versions. The shared-runtime review reran 791 additional cases per project stack. The focused receipt also pins the public before-and-after measurements from the companion study. [Input-boundary receipt](receipt-2026-11-18-input-boundaries.json) · [Shared-runtime receipt](receipt-2026-11-18-input-regression.json)

The 19 November review covers the held-out Procrustes rank selection and generation-diversity entropy. It records the four software stacks, the pre-fix failures, the separate fresh-process entropy runs, and hashes of the companion study’s document-panel and cutoff-grid comparisons. [Numerical repair receipt](receipt-2026-11-19-numerical-fixes.json)


### Runner and architecture verification

The 19 November study-runners receipt covers the two Procrustes solvers, domain BPB and LiDAR inputs, position-wise attention ranks and the other metrics with four-version coverage. It includes 110 CPU checks on each of Transformers 5.2.0, 5.10.2, 5.22.2 and 5.23.0. Gemma-4 perturbation growth has a separate receipt that pins the token-identity regression and records the four execution environments. The project-stack receipt links the other numerical reviews to their raw JUnit files. [Four-version checks and frozen study rules](receipt-2026-11-19-study-protocol.json) · [Gemma-4 repair](receipt-2026-11-19-architecture.json) · [Project-stack checks](receipt-2026-11-19-library-suite.json)

The companion study keeps model scores, train and test article IDs, fold errors and checkpoint revisions under `verification/runner_readiness_2026-11-19/`. Full post-fix tests and multi-model GPU runs have separate artifacts there. A linked numerical test supports only the estimator or component described in its review.

The final 19 November receipt covers all 109 registered tasks and their linked readout tests. The same 801 cases passed on each of Transformers 5.2.0, 5.10.2, 5.22.2 and 5.23.0, with zero skips. It includes the PyHessian and other pinned references, Gemma’s logit softcap and byte accounting, Pythia’s published all-row overlap calculation, and a check that vocabulary padding changes the active NeoX head while preserving every original logit. [All-task receipt](receipt-2026-11-19-verification.json)

### 0.2.7

The 0.2.7 receipts cover all 111 registered tasks on commit 07887c3. The same 1,065 cases passed on each of Transformers 5.2.0, 5.10.2, 5.22.2 and 5.23.0 on CPU, with 8 skips (the GPU tests). They include the covariance-conditioning tests of `geometry_readout_dual_spectrum` and the regression test for tokenizers that append EOS in `dynamics_context_influence`. The dtype contract, precision provenance and float32 attention scores change shared runtime inputs (`core.py`, `models/wrapper.py`, `results.py`), so every task needed a new receipt. Their own tests are in the per-commit suite. A second receipt repeats `geometry_cka`'s cases with one GPU visible, so `evaluate()` takes the GPU path (20 cases per version). Raw JUnit, logs and the execution record are on pod1 under `artifacts/pod1/blme/review-20261121k` and `review-20261121l`. [All-task receipt](receipt-2026-11-21-v027.json) · [GPU CKA receipt](receipt-2026-11-21-cka-gpu-v027.json)
