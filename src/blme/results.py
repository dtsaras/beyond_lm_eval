"""
Standardised results formatting and output for BLME evaluations.

Provides:
    - build_results_envelope() — wraps raw task results with metadata
    - print_results_table()    — pretty-print a summary table to the terminal
    - save_results()           — write JSON / CSV to disk
"""

import json
import csv
import os
import logging

import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from .task_metadata import task_display_name

logger = logging.getLogger("blme")

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def _item_set() -> int:
    from .cache import item_set
    return item_set()


def _get_blme_version() -> str:
    """The imported source tree's version (``blme._version``), not installed metadata."""
    try:
        from ._version import __version__
        return __version__
    except Exception:
        return "dev"


def _get_git_hash() -> Optional[str]:
    """The commit of the git checkout the imported ``blme`` package lives in.

    Run in the package's own directory, not the caller's: a run launched from
    another repository recorded that repository's commit. None for a source
    tree outside git (an export).
    """
    try:
        pkg = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "-C", pkg, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except Exception:
        return None


def _dependency_versions() -> Dict[str, Optional[str]]:
    packages = ["torch", "transformers", "numpy", "scipy", "datasets", "lm_eval"]
    versions: Dict[str, Optional[str]] = {}
    try:
        from importlib.metadata import version
    except Exception:
        return versions
    for package in packages:
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = None
    return versions


# ---------------------------------------------------------------------------
# Results envelope
# ---------------------------------------------------------------------------

def build_results_envelope(
    model_args: str,
    tasks_requested: List[str],
    task_results: Dict[str, Any],
    task_errors: Dict[str, str],
    device: str,
    task_timings: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    task_configs_resolved: Optional[Dict[str, dict]] = None,
    cache_settings: Optional[Dict[str, Any]] = None,
    unknown_tasks: Optional[List[str]] = None,
    experts_fallback: Optional[Dict[str, Any]] = None,
    attn_implementation: Optional[Dict[str, Any]] = None,
    precision: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Wrap raw per-task results into a structured envelope with metadata.

    ``experts_fallback`` is the record ``ensure_runnable_experts`` left on the
    model; ``config.experts_implementation_fallback`` appears only when the MoE
    experts were switched. ``precision`` is ``config.precision``: the requested,
    native, loaded and effective dtypes and any cast (``blme.models.dtype_contract``).
    """
    timings = task_timings or {}
    total_elapsed = round(sum(timings.values()), 2) if timings else None

    # Tasks that returned {"error": ...} did not raise but did not succeed —
    # count them as failures (not completed) and surface them in the errors
    # field, alongside tasks that raised (task_errors).
    errored_results = {
        name: r["error"] for name, r in task_results.items()
        if isinstance(r, dict) and "error" in r
    }
    completed_names = [name for name in task_results if name not in errored_results]
    all_errors = {**task_errors, **errored_results}

    envelope = {
        "blme_version": _get_blme_version(),
        "blme_source": os.path.dirname(os.path.abspath(__file__)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
        "environment": {
            "dependencies": _dependency_versions(),
            "item_set": _item_set(),
        },
        "config": {
            "model_args": model_args,
            "device": device,
            "tasks_requested": tasks_requested,
            "task_labels": {k: task_display_name(k) for k in tasks_requested if task_display_name(k) != k},
            "seed": seed,
            "task_configs_resolved": task_configs_resolved or {},
            "cache_settings": cache_settings or {},
            "unknown_tasks": unknown_tasks or [],
        },
        "summary": {
            "total_tasks": len(tasks_requested),
            "completed_tasks": len(completed_names),
            "failed_tasks": len(all_errors),
            "total_elapsed": total_elapsed,
        },
        "results": task_results,
        "errors": all_errors if all_errors else None,
        "task_timings": timings if timings else None,
    }
    if experts_fallback:
        envelope["config"]["experts_implementation_fallback"] = dict(experts_fallback)
    if attn_implementation is not None:
        # The attention kernel the tasks ran with (core.evaluate switches the
        # whole model to eager when a co-run task needs attention weights).
        envelope["config"]["attn_implementation"] = dict(attn_implementation)
    if precision is not None:
        envelope["config"]["precision"] = dict(precision)
    return envelope


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_results_table(
    task_results: Dict[str, Any],
    task_errors: Dict[str, str],
    task_timings: Optional[Dict[str, float]] = None,
) -> None:
    """Print a compact summary table to stdout."""
    timings = task_timings or {}
    has_timings = bool(timings)

    print()
    if has_timings:
        print("=" * 82)
        print(f"{'Task':<42} {'Status':<10} {'Time':>7}  {'Key Metric'}")
        print("-" * 82)
    else:
        print("=" * 72)
        print(f"{'Task':<42} {'Status':<10} {'Key Metric'}")
        print("-" * 72)

    for task_name, result in sorted(task_results.items()):
        label = task_display_name(task_name)
        if "error" in result:
            status = "⚠ ERROR"
            metric = result["error"][:30]
        else:
            status = "✓ OK"
            metric = _pick_summary_metric(result)
        time_str = f"{timings[task_name]:>6.1f}s" if task_name in timings else ""
        if has_timings:
            print(f"  {label:<40} {status:<10} {time_str:>7}  {metric}")
        else:
            print(f"  {label:<40} {status:<10} {metric}")

    for task_name, err in sorted(task_errors.items()):
        label = task_display_name(task_name)
        time_str = f"{timings[task_name]:>6.1f}s" if task_name in timings else ""
        if has_timings:
            print(f"  {label:<40} {'✗ FAIL':<10} {time_str:>7}  {err[:30]}")
        else:
            print(f"  {label:<40} {'✗ FAIL':<10} {err[:30]}")

    total = len(task_results) + len(task_errors)
    passed = sum(1 for r in task_results.values() if "error" not in r)
    warned = sum(1 for r in task_results.values() if "error" in r)
    failed = len(task_errors)
    sep_width = 82 if has_timings else 72
    print("-" * sep_width)
    summary = f"  Total: {total}  |  ✓ Passed: {passed}  |  ⚠ Warned: {warned}  |  ✗ Failed: {failed}"
    if has_timings:
        total_elapsed = sum(timings.values())
        summary += f"  |  Total time: {total_elapsed:.1f}s"
    print(summary)
    print("=" * sep_width)
    print()


def _pick_summary_metric(result: Dict[str, Any]) -> str:
    """Pick the most representative numeric value from a result dict."""
    for key, val in result.items():
        if isinstance(val, (int, float)):
            return f"{key}={val:.4g}"
    return "(no scalar)"


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_results(
    envelope: Dict[str, Any],
    output_dir: str,
    output_format: str = "json",
) -> str:
    """
    Save the results envelope to disk.

    Args:
        envelope: The full results envelope from build_results_envelope().
        output_dir: Directory to write the output file.
        output_format: 'json' or 'csv'.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if output_format == "json":
        path = os.path.join(output_dir, "results.json")
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
        return path

    elif output_format == "csv":
        path = os.path.join(output_dir, "results.csv")
        results = envelope.get("results", {})
        rows = []
        for task_name, metrics in results.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float, str, bool)):
                        rows.append({"task": task_name, "metric": k, "value": v})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Results saved to {path}")
        return path

    else:
        raise ValueError(f"Unknown output format: {output_format}")
