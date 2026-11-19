"""Render the task-by-task validation dashboard and detect stale scientific reviews."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/validation"
REPORT = ROOT / "docs/VALIDATION.md"
READOUT_REPORT = DATA / "READOUTS.md"
EVIDENCE = {
    "release": "Released estimator or component executed",
    "published": "Released dataset and published numerical anchor",
    "fixture": "Release transcription, recorded output or reimplementation",
    "independent": "Independent implementation or scientific library",
    "equation": "Equation, analytic anchor or controlled recomputation",
    "proxy": "Operational proxy or study-defined diagnostic",
}
STACKS = ("5.2.0", "5.10.2")


def value_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def inventory():
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from blme.core import _register_all_tasks
    from blme.registry import get_task, list_tasks
    from blme.task_metadata import TASK_CERTIFICATION

    _register_all_tasks()
    defaults_file = ROOT / "src/blme/tasks/configs/defaults.yaml"
    defaults = yaml.safe_load(defaults_file.read_text()) if defaults_file.is_file() else None
    tasks = {}
    for name in list_tasks():
        entry = TASK_CERTIFICATION.get(name)
        metadata = entry.to_dict() if entry else {}
        tasks[name] = {
            "source": Path(inspect.getfile(get_task(name))).resolve().relative_to(ROOT).as_posix(),
            "display_name": metadata.get("display_name"),
            "declaration": metadata.get("status", "undeclared"),
            "headlines": sorted(getattr(get_task(name), "HEADLINE_KEYS", ()) or ()),
            "metadata_sha256": value_digest(metadata),
            "defaults_sha256": value_digest(defaults.get(name) or {}) if defaults is not None else None,
        }
    return tasks


def match(selector, case):
    if selector.endswith("*"):
        return case.startswith(selector[:-1])
    if "::" not in selector:
        return case.split("::")[0] == selector
    return case == selector or case.startswith(selector + "[")


def case_results(selectors, runs, required_stacks=STACKS):
    all_cases = {case for run in runs.values() for case in run["cases"]}
    by_selector = {s: {c for c in all_cases if match(s, c)} for s in selectors}
    expected = set().union(*by_selector.values())
    summaries = {}
    for stack in dict.fromkeys([*required_stacks, *runs]):
        cases = runs.get(stack, {}).get("cases", {})
        counts = Counter(cases.get(c, "missing") for c in expected)
        unresolved = [s for s, selected in by_selector.items() if not selected]
        summaries[stack] = {"passed": counts["passed"], "total": len(expected),
                            "exceptions": {c: cases.get(c, "missing") for c in sorted(expected)
                                           if cases.get(c) != "passed"},
                            "missing_selectors": unresolved, "stack_recorded": stack in runs}
    return summaries


def assess(reviews, receipts, live, file_hash):
    rows = {}
    for name in sorted(set(reviews) | set(live)):
        review = reviews.get(name, {})
        current = live.get(name, {})
        receipt = receipts.get(review.get("receipt"), {})
        pinned = receipt.get("tasks", {}).get(name, {})
        issues = []
        runs = {}
        named_readouts = {**review.get("study_readouts", {}), **review.get("other_readouts", {})}
        readouts = {key: {**entry, "runs": {}} for key, entry in named_readouts.items()}
        if not review:
            issues.append("Unreviewed registry task")
        if not current:
            issues.append("Reviewed task absent from registry")
        if review:
            if review.get("evidence") not in EVIDENCE or not all(review.get(k) for k in ("scope", "next_check", "selectors", "tier")):
                issues.append("Incomplete scientific assessment")
            if not pinned:
                issues.append("Missing dated task receipt")
            else:
                if current:
                    for key in ("source", "metadata_sha256", "defaults_sha256", "headlines"):
                        if pinned[key] != current[key]:
                            issues.append("Changed " + key.removesuffix("_sha256"))
                if review["selectors"] != pinned["selectors"]:
                    issues.append("Test selection changed since review")
                for path in sorted(set(receipt["common_inputs"]) | set(pinned["inputs"])):
                    expected_hash = receipt["hashes"].get(path)
                    if not expected_hash or file_hash(path) != expected_hash:
                        issues.append("Changed or missing " + path)
                required = tuple(dict.fromkeys([*STACKS, *review.get("required_stacks", ())]))
                runs = case_results(review["selectors"], receipt["runs"], required)
                key_selectors = {k: v["selectors"] for k, v in named_readouts.items()}
                if key_selectors != pinned["readout_selectors"]:
                    issues.append("Readout selection changed since review")
                for key, detail in named_readouts.items():
                    checks = case_results(detail["selectors"], receipt["runs"], required)
                    readouts[key] = {**detail, "runs": checks}
                    if any(s["missing_selectors"] or s["total"] == 0 or "missing" in s["exceptions"].values()
                           for s in checks.values()):
                        issues.append("Incomplete case evidence for readout " + key)
                for stack, summary in runs.items():
                    if not summary["stack_recorded"]:
                        issues.append("Missing suite on " + stack)
                    if summary["missing_selectors"] or "missing" in summary["exceptions"].values():
                        issues.append("Incomplete case evidence on " + stack)
        rows[name] = {**review, **current, "issues": issues, "runs": runs,
                      "study_readouts": {k: readouts[k] for k in review.get("study_readouts", {})},
                      "other_readouts": {k: readouts[k] for k in review.get("other_readouts", {})},
                      "registered": bool(current)}
        if current and set(named_readouts) - set(current.get("headlines", ())):
            issues.append("Mapped readout absent from current headline declarations")
    return rows


def load_status():
    reviews = json.loads((DATA / "metrics.json").read_text())
    receipts = {}
    for key in {r["receipt"] for r in reviews.values()}:
        path = DATA / f"receipt-{key}.json"
        if path.is_file():
            receipt = json.loads(path.read_text())
            if receipt["id"] != key:
                raise ValueError(f"Mismatched receipt ID in {path}")
            receipts[key] = receipt

    @lru_cache(maxsize=None)
    def file_hash(relative):
        path = ROOT / relative
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    return assess(reviews, receipts, inventory(), file_hash), receipts


def cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def case_label(summary):
    if not summary or not summary["stack_recorded"]:
        return "No run"
    text = f'{summary["passed"]}/{summary["total"]}'
    flags = Counter(summary["exceptions"].values())
    if summary["missing_selectors"]:
        flags["unmatched selector"] += len(summary["missing_selectors"])
    return text + "".join(f" · {n} {status}" for status, n in sorted(flags.items()))


def reported_stacks(rows):
    return tuple(dict.fromkeys([*STACKS, *(s for row in rows.values() for s in row['runs'])]))


def render(rows, receipts):
    stacks = reported_stacks(rows)
    present = [r for r in rows.values() if r["registered"]]
    counts = Counter(r.get("evidence") for r in present)
    declarations = Counter(r["declaration"] for r in present)
    outstanding = [n for n, r in rows.items() if r["issues"]]
    out = [
        "# Metric validation status", "",
        f"**{len(present)} registered tasks.** Each row describes the quantity checked, the strongest numerical evidence selected for it, and the next scientific check. A public task can expose multiple readouts with different evidence. A passing comparison applies to the stated inputs and conventions.", "",
        f"**{sum(bool(r.get('runs')) for r in present)} tasks have linked numerical receipts. {len(outstanding)} rows flagged by the freshness check.** Changes to reviewed inputs or the registry are flagged by the regular test suite. [Maintenance and evidence format](validation/README.md) · [Machine-readable assessments](validation/metrics.json)", "",
        "## Evidence across the library", "", "| Primary evidence | Tasks |", "|---|---:|",
    ]
    out.extend(f"| {label} | {counts[code]} |" for code, label in EVIDENCE.items())
    out += ["", "The primary categories count each task once. Published-value tests also occur in other categories. Library declarations are "
            + ", ".join(f"{declarations[k]} `{k}`" for k in sorted(declarations)) + ". The rows below use the evidence from the review.", "",
            "## Numerical runs", "", "| Dated receipt | Transformers | Passed | Failed | Errors | Skipped |", "|---|---|---:|---:|---:|---:|"]
    for key, receipt in sorted(receipts.items()):
        for stack, run in receipt["runs"].items():
            c = Counter(run["cases"].values())
            out.append(f"| [{key}](validation/receipt-{key}.json) | {stack} | {c['passed']} | {c['failure']} | {c['error']} | {c['skipped']} |")
    for key, receipt in sorted(receipts.items()):
        for stack, run in receipt["runs"].items():
            for case, result in run["cases"].items():
                if result != "passed":
                    out.extend(["", f"**{stack} · {result}** `{case}` in the {key} run."])
    out += ["", "Counts above are unique executed cases per run. Row counts below are *passed / linked cases* per stack. Some tests serve multiple metrics. These are selected CPU numerical and reference tests. Whole-suite testing, architecture coverage, document reliability, and published-result reproduction each require their own evidence.", "",
            "## Scientific findings", "", (DATA / "findings.md").read_text().strip(), ""]
    nkeys = sum(len(r["study_readouts"]) for r in rows.values())
    ntasks = sum(bool(r["study_readouts"]) for r in rows.values())
    declared = sum(len(r.get("headlines", ())) for r in present)
    with_keys = sum(bool(r.get("headlines")) for r in present)
    other = sum(len(r["other_readouts"]) for r in rows.values())
    out.extend([f"The library declares {declared} headline keys or key templates across {with_keys} tasks. The remaining {len(present) - with_keys} tasks have task-level reviews. The frozen study uses {nkeys} confirmatory keys from {ntasks} tasks. The ledger maps tests individually for those keys and {other} other declared headlines. The other {declared - nkeys - other} headlines have task-level evidence. [Readout-level evidence](validation/READOUTS.md)", ""])
    if outstanding:
        out.extend(["## Reviews needing attention", ""])
        out.extend(f"- `{n}` — " + ", ".join(rows[n]["issues"]) for n in outstanding)
        out.append("")
    families = sorted({n.partition("_")[0] for n in rows})
    out.extend(["## Individual metrics", "", " · ".join(f"[{f.title()}](#{f})" for f in families), "",
                "Task links open the implementation. Numbered links open the test files. The machine-readable assessment lists the exact test selectors. The last column identifies an outstanding check, including where a comparison with the original paper is still needed.", ""])
    for family in families:
        out.extend([f"### {family.title()}", "", "| Task, tests and library declaration | Primary evidence | " + " | ".join(stacks) + " | Verified scope and next check |",
                    "|---|---|" + "---:|" * len(stacks) + "---|"])
        for name, row in rows.items():
            if name.partition("_")[0] != family:
                continue
            source = row.get("source")
            task = f"[{name}](../{source})" if source else f"`{name}`"
            if row.get("display_name"):
                task = row["display_name"] + "<br>" + task
            files = dict.fromkeys(s.split("::")[0] for s in row.get("selectors", []))
            links = " ".join(f"[{i}](../{f})" for i, f in enumerate(files, 1))
            if row["issues"]:
                task += " **REVIEW DUE**"
            detail = cell(row.get("scope", "Awaiting scientific review."))
            if "application" in row:
                detail += " " + cell(row["application"])
            if row.get("next_check"):
                detail += " **Next.** " + cell(row["next_check"])
            label = EVIDENCE.get(row.get("evidence"), "Pending")
            results = " | ".join(case_label(row["runs"].get(s)) for s in stacks)
            declaration = row.get("declaration", "undeclared")
            out.append(f"| {task} {links}<br>{declaration} | {label} | {results} | {detail} |")
        out.append("")
    return "\n".join(out)


def render_readouts(rows):
    stacks = reported_stacks(rows)
    columns = " | ".join(stacks)
    divider = "|---|---|" + "---:|" * len(stacks) + "---|"
    blanks = " | ".join("—" for _ in stacks)
    out = ["# Library headline and study readout evidence", "", "## Frozen confirmatory readouts", "",
           "This table follows the exact task and key names in the study's frozen analysis manifest. Each entry gives the reviewed comparison and its linked CPU test cases. A kernel check covers a component. A model or corpus change can affect the end-to-end measurement. [All registered tasks](../VALIDATION.md) · [Exact selectors](metrics.json)", "",
           f"| Task and readout | Comparison | {columns} | Tested scope |", divider]
    for name, row in rows.items():
        for key, entry in sorted(row["study_readouts"].items()):
            links = dict.fromkeys(s.split("::")[0] for s in entry["selectors"])
            code = " ".join(f"[{i}](../../{f})" for i, f in enumerate(links, 1))
            result = " | ".join(case_label(entry["runs"].get(stack)) for stack in stacks)
            status = " **REVIEW DUE**" if row["issues"] else ""
            out.append(f"| `{name}.{key}`{status} {code} | {cell(entry['evidence'])} | {result} | {cell(entry['scope'])} |")
    out.extend(["", "## Other library headlines", "",
                "Each declared key or template outside the frozen study set appears here. A task-level entry still has a scientific review in the main dashboard. Its exact headline lacks a separate comparison in this ledger. Component and aggregation checks identify the part exercised.", "",
                f"| Task and headline | Evidence | {columns} | Scope |", divider])
    for name, row in rows.items():
        for key in sorted(set(row.get("headlines", ())) - row["study_readouts"].keys()):
            status = " **REVIEW DUE**" if row["issues"] else ""
            entry = row["other_readouts"].get(key)
            if not entry:
                out.append(f"| `{name}.{key}`{status} | Task-level only | {blanks} | Key-specific comparison pending. |")
                continue
            files = dict.fromkeys(s.split("::")[0] for s in entry["selectors"])
            links = " ".join(f"[{i}](../../{f})" for i, f in enumerate(files, 1))
            result = " | ".join(case_label(entry["runs"].get(stack)) for stack in stacks)
            out.append(f"| `{name}.{key}`{status} {links} | {cell(entry['evidence'])} | {result} | {cell(entry['scope'])} |")
    return "\n".join([*out, ""])


def study_drift(receipts, study):
    issues = []
    for receipt in receipts.values():
        evidence = {**receipt["study_sources_sha256"],
                    **receipt.get("execution", {}).get("study_outputs_sha256", {})}
        for path, sha in evidence.items():
            source = study / path
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                issues.append(f"Study source changed since {receipt['id']}: {path}")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--check", action="store_true", help="check inventory, reviewed inputs and rendered Markdown (default)")
    flags.add_argument("--write", action="store_true", help="render the dashboard, preserving old receipts")
    flags.add_argument("--json", action="store_true", help="emit live per-task status, including changed inputs")
    parser.add_argument("--study", type=Path, help="also verify the dated study inputs against this checkout with --check")
    args = parser.parse_args()
    rows, receipts = load_status()
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    reports = {REPORT: render(rows, receipts), READOUT_REPORT: render_readouts(rows)}
    if args.write:
        for target, rendered in reports.items():
            with TemporaryDirectory(dir=target.parent) as directory:
                tmp = Path(directory) / target.name
                tmp.write_text(rendered, encoding="utf-8")
                tmp.replace(target)
        return
    errors = [f"{name}: {', '.join(row['issues'])}" for name, row in rows.items() if row["issues"]]
    if args.study:
        errors.extend(study_drift(receipts, args.study))
    for target, rendered in reports.items():
        if not target.exists() or rendered != target.read_text():
            errors.append(f"{target.relative_to(ROOT)} is out of date. Run scripts/metric_status.py --write.")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validation tracker matches {len(rows)} registered tasks and their recorded evidence.")


if __name__ == "__main__":
    main()
