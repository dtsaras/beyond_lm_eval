"""Pin a review export before testing it, then import fresh pytest JUnit results.

Run with --help or see docs/validation/README.md. Pinning and importing never edit
the scientific assessment, its active receipt ID, or the published dashboard.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def pin(args):
    from metric_status import value_digest

    root = args.source.resolve()
    sys.path.insert(0, str(root / "src"))
    from blme.task_metadata import task_metadata_dict
    from blme.tasks.config_loader import resolve_task_config

    spec = importlib.util.spec_from_file_location("source_hash", args.study / "verification/source_hash.py")
    sources = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sources)
    for name in ("module_file", "imports_of", "_registration_only", "_pkg_root"):
        setattr(sources, name, lru_cache(maxsize=None)(getattr(sources, name)))
    registry = sources._registry()
    reviews = json.loads(args.reviews.read_text())
    selected = sorted(args.tasks or reviews)
    # Include shared test support and reference fixtures conservatively. A change
    # to a fixture can require multiple rows to be rechecked.
    support = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")
               if not p.name.startswith("test_") and p.name != "__init__.py"}
    support.update(p.relative_to(root).as_posix() for p in (root / "tests/fixtures").rglob("*.json"))
    tasks = {}
    for name in selected:
        files = {Path(p).relative_to(root).as_posix() for p in sources.task_files(name, registry)}
        declared = sorted(getattr(registry[name], "HEADLINE_KEYS", ()) or ())
        readouts = {**reviews[name].get("study_readouts", {}), **reviews[name].get("other_readouts", {})}
        key_selectors = {key: readout["selectors"] for key, readout in readouts.items()}
        selectors = reviews[name]["selectors"] + [s for named in key_selectors.values() for s in named]
        files |= {s.split("::")[0] for s in selectors}
        if name == "geometry_categories":
            files.add("src/blme/assets/categories.json")
        own = Path(inspect.getfile(registry[name])).relative_to(root).as_posix()
        metadata = task_metadata_dict(name)
        tasks[name] = {
            "source": own, "inputs": sorted(files | support), "selectors": reviews[name]["selectors"],
            "readout_selectors": key_selectors, "headlines": declared,
            "declaration": metadata["status"], "metadata_sha256": value_digest(metadata),
            "defaults_sha256": value_digest(resolve_task_config(name)),
        }
    common = sorted(set.intersection(*(set(entry["inputs"]) for entry in tasks.values())))
    all_inputs = sorted({p for entry in tasks.values() for p in entry["inputs"]})
    for entry in tasks.values():
        entry["inputs"] = sorted(set(entry["inputs"]) - set(common))
    frozen = {
        "id": args.id, "base_commit": args.base, "study_base": args.study_base,
        "study_sources_sha256": {p: digest(args.study / p) for p in sorted(args.study_evidence)},
        "configuration_sources": {p: digest(root / p) for p in
                                  ("src/blme/tasks/configs/defaults.yaml", "src/blme/task_metadata.py")},
        "hashes": {p: digest(root / p) for p in all_inputs}, "common_inputs": common, "tasks": tasks,
    }
    write_new(args.out, frozen)


def junit(path, sources):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    results = {}
    for case in root.iter("testcase"):
        parts = case.get("classname", "").split(".")
        while parts:
            filename = "/".join(parts) + ".py"
            if filename in sources:
                break
            parts.pop()
        if not parts:
            raise ValueError(f"Cannot resolve JUnit case {case.attrib}")
        qualname = case.get("classname").split(".")[len(parts):] + [case.attrib["name"]]
        name = "::".join([filename, *qualname])
        if name in results:
            raise ValueError(f"Duplicate JUnit case {name}")
        state = next((tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None), "passed")
        results[name] = state
    expected = {"error": totals["errors"], "failure": totals["failures"], "skipped": totals["skipped"]}
    if len(results) != totals["tests"] or any(list(results.values()).count(k) != v for k, v in expected.items()):
        raise ValueError(f"JUnit case counts disagree with suite header in {path}")
    return {"junit_sha256": digest(path), "cases": dict(sorted(results.items()))}


def record(args):
    frozen = json.loads(args.pin.read_text())
    root = args.source.resolve()
    changed = [name for name, expected in {**frozen["hashes"], **frozen["configuration_sources"]}.items()
               if not (root / name).is_file() or digest(root / name) != expected]
    if changed:
        raise ValueError(f"Export changed since pinning: {changed}")
    sources = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
    runs = {}
    for stack, path in args.junit:
        result = junit(Path(path), sources)
        run = runs.setdefault(stack, {"junit_sha256": {}, "cases": {}})
        duplicates = result["cases"].keys() & run["cases"].keys()
        if duplicates or not result["cases"] or Path(path).name in run["junit_sha256"]:
            raise ValueError(f"Empty or overlapping JUnit suite {path}: {sorted(duplicates)}")
        run["junit_sha256"][Path(path).name] = result["junit_sha256"]
        run["cases"].update(result["cases"])
    frozen["runs"] = runs
    frozen["execution"] = json.loads(args.execution.read_text())
    write_new(args.out, frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pin", help="freeze reviewed dependencies on the exact checkout to be tested")
    for opt in ("source", "study", "reviews", "out"):
        p.add_argument("--" + opt, type=Path, required=True)
    for opt in ("id", "base", "study-base"):
        p.add_argument("--" + opt, required=True)
    p.add_argument("--study-evidence", nargs="+", default=[])
    p.add_argument("--tasks", nargs="+", help="review just these registry names")
    r = commands.add_parser("record", help="import the test runs on that pinned checkout")
    for opt in ("source", "pin", "execution", "out"):
        r.add_argument("--" + opt, type=Path, required=True)
    r.add_argument("--junit", nargs=2, metavar=("STACK", "FILE"), required=True, action="append")
    args = parser.parse_args()
    (pin if args.command == "pin" else record)(args)


if __name__ == "__main__":
    main()
