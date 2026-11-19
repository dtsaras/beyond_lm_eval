"""The published status must fail closed on registry drift and incomplete evidence."""
import importlib.util
from pathlib import Path

import pytest


def tool(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status = tool("metric_status")


def test_published_reviews_cover_current_registry_and_inputs():
    rows, receipts = status.load_status()
    drift = {name: row["issues"] for name, row in rows.items() if row["issues"]}
    assert not drift, drift
    assert status.render(rows, receipts) == status.REPORT.read_text()
    assert status.render_readouts(rows) == status.READOUT_REPORT.read_text()


def test_unexecuted_failed_skipped_and_parameterized_cases_remain_visible():
    s = "tests/test_metric.py::test_value"
    runs = {"stack-a": {"cases": {s + "[1]": "passed", s + "[2]": "failure", s + "[3]": "skipped"}},
            "stack-b": {"cases": {s + "[1]": "error", s + "[3]": "passed"}}}
    out = status.case_results([s, s + "_unknown"], runs, ("stack-a", "stack-b", "stack-c"))
    assert out["stack-a"]["passed"] == 1
    assert out["stack-a"]["total"] == 3
    assert out["stack-a"]["exceptions"][s + "[2]"] == "failure"
    assert out["stack-a"]["exceptions"][s + "[3]"] == "skipped"
    assert out["stack-b"]["exceptions"] == {s + "[1]": "error", s + "[2]": "missing"}
    assert out["stack-b"]["missing_selectors"] == [s + "_unknown"]
    assert status.case_label(out["stack-c"]) == "No run"


def test_source_drift_and_new_or_removed_tasks_do_not_inherit_a_review():
    case = "tests/demo.py::test_demo"
    live = {n: {"source": n + ".py", "metadata_sha256": n, "defaults_sha256": n, "headlines": [], "declaration": "formula-faithful"} for n in ("a", "b", "new")}
    reviews = {n: {"evidence": "equation", "scope": "Known value", "next_check": "Repeat",
                   "selectors": [case], "receipt": "r", "tier": "sample"} for n in ("a", "b", "removed")}
    tasks = {n: {**live.get(n, {}), "inputs": [n + ".py"], "selectors": [case], "readout_selectors": {}}
             for n in reviews}
    receipt = {"tasks": tasks, "hashes": {n + ".py": n for n in reviews}, "common_inputs": [],
               "runs": {v: {"cases": {case: "passed"}} for v in status.STACKS}}
    local_hash = {"a.py": "changed", "b.py": "b", "removed.py": "removed"}.get
    rows = status.assess(reviews, {"r": receipt}, live, local_hash)
    assert rows["a"]["issues"] == ["Changed or missing a.py"]
    assert not rows["b"]["issues"]
    current_release = "5.23.0"
    updated = {**reviews, "b": {**reviews["b"], "required_stacks": [*status.STACKS, current_release]}}
    expected = status.assess(updated, {"r": receipt}, live, local_hash)["b"]
    assert "Missing suite on " + current_release in expected["issues"]
    assert status.case_label(expected["runs"][current_release]) == "No run"
    assert rows["new"]["issues"] == ["Unreviewed registry task"]
    assert rows["removed"]["issues"] == ["Reviewed task absent from registry"]
    # Rendering preserves the old numerical receipt even after a source edit.
    assert rows["a"]["runs"]["5.2.0"]["passed"] == 1
    assert "REVIEW DUE" in status.render(rows, {"r": receipt})


def test_junit_import_keeps_class_parameter_and_skip_and_checks_header(tmp_path):
    recorder = tool("record_metric_review")
    path = tmp_path / "example.xml"
    case = '<testcase classname="tests.test_m.TestEstimator" name="test_score[3]"><skipped/></testcase>'
    xml = '<testsuite tests="1" errors="0" skipped="1" failures="0">' + case + '</testsuite>'
    path.write_text(xml)
    assert recorder.junit(path, {"tests/test_m.py"})["cases"] == {
        "tests/test_m.py::TestEstimator::test_score[3]": "skipped"}
    path.write_text(xml.replace('tests="1"', 'tests="2"'))
    with pytest.raises(ValueError, match="counts disagree"):
        recorder.junit(path, {"tests/test_m.py"})


def test_study_measurements_and_extra_stack_columns(tmp_path):
    status = tool("metric_status")
    (tmp_path / "run.json").write_bytes(b"{}")
    import hashlib
    digest = hashlib.sha256(b"{}").hexdigest()
    receipt = {"id": "run", "study_sources_sha256": {}, "execution": {"study_outputs_sha256": {"run.json": digest}}}
    assert status.study_drift({"run": receipt}, tmp_path) == []
    (tmp_path / "run.json").write_bytes(b"changed")
    assert "run.json" in status.study_drift({"run": receipt}, tmp_path)[0]
    rows = {"older": {"runs": {s: {} for s in status.STACKS}},
            "recent": {"runs": {"5.22.2": {}, "5.23.0": {}}}}
    assert status.reported_stacks(rows) == (*status.STACKS, "5.22.2", "5.23.0")
