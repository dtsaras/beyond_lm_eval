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

logger = logging.getLogger("blme")

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def _get_blme_version() -> str:
    try:
        from importlib.metadata import version
        return version("blme")
    except Exception:
        return "dev"


def _get_git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Results envelope
# ---------------------------------------------------------------------------

def build_results_envelope(
    model_args: str,
    tasks_requested: List[str],
    task_results: Dict[str, Any],
    task_errors: Dict[str, str],
    device: str,
) -> Dict[str, Any]:
    """
    Wrap raw per-task results into a structured envelope with metadata.
    """
    return {
        "blme_version": _get_blme_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _get_git_hash(),
        "config": {
            "model_args": model_args,
            "device": device,
            "tasks_requested": tasks_requested,
        },
        "summary": {
            "total_tasks": len(tasks_requested),
            "completed_tasks": len(task_results),
            "failed_tasks": len(task_errors),
        },
        "results": task_results,
        "errors": task_errors if task_errors else None,
    }


# ---------------------------------------------------------------------------
# Terminal table
# ---------------------------------------------------------------------------

def print_results_table(
    task_results: Dict[str, Any],
    task_errors: Dict[str, str],
) -> None:
    """Print a compact summary table to stdout."""
    print()
    print("=" * 72)
    print(f"{'Task':<42} {'Status':<10} {'Key Metric'}")
    print("-" * 72)

    for task_name, result in sorted(task_results.items()):
        if "error" in result:
            status = "⚠ ERROR"
            metric = result["error"][:30]
        else:
            status = "✓ OK"
            # Pick the first numeric metric as the summary value
            metric = _pick_summary_metric(result)
        print(f"  {task_name:<40} {status:<10} {metric}")

    for task_name, err in sorted(task_errors.items()):
        print(f"  {task_name:<40} {'✗ FAIL':<10} {err[:30]}")

    total = len(task_results) + len(task_errors)
    passed = sum(1 for r in task_results.values() if "error" not in r)
    warned = sum(1 for r in task_results.values() if "error" in r)
    failed = len(task_errors)
    print("-" * 72)
    print(f"  Total: {total}  |  ✓ Passed: {passed}  |  ⚠ Warned: {warned}  |  ✗ Failed: {failed}")
    print("=" * 72)
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
