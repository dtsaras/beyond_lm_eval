"""
Re-run a specific subset of tasks on already-finished models and
patch the existing results.json files in place. Much faster than a
full Phase 1 re-run because it only runs 4 tasks per model.

Usage:
    python scripts/patch_failed_tasks.py --input-dir results/study_v1 \
                                          --models gpt2-small,pythia-70m,...
    python scripts/patch_failed_tasks.py --input-dir results/study_v1 --all

Each model runs on a single GPU by default; use --n-gpus N to run N
models in parallel.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("patch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import MODELS, build_model_args

# Default tasks to re-run (the 4 that were failing due to the fixed bugs).
# Override with --tasks on the command line.
DEFAULT_TASKS_TO_PATCH = [
    "geometry_svd",
    "geometry_prediction_alignment",
    "causality_tracing",
    "repe_task_vectors",
]


def _patch_model(entry, input_dir: Path, gpu_id: str, tasks_to_patch,
                  task_timeout: int = 600, subprocess_timeout: int = 7200):
    """Run the subset of ``tasks_to_patch`` that this model actually needs.

    For each model we look at its existing results.json and select only the
    tasks that are missing or recorded as errored — this avoids re-running
    tasks that already succeeded for a particular model when the same
    ``tasks_to_patch`` list is shared across heterogeneous failures.
    """
    name = entry["name"]
    model_dir = input_dir / "blme" / name
    results_path = model_dir / "results.json"
    if not results_path.exists():
        logger.info(f"[SKIP] {name} — no existing results.json")
        return name, "skipped"

    # Compute the per-model patch list: tasks in the requested list that
    # this model is actually missing or that errored last time.
    with open(results_path) as f:
        envelope = json.load(f)
    existing_results = envelope.get("results", {})
    existing_errors = envelope.get("errors", {})

    def _needs(t: str) -> bool:
        if t in existing_errors:
            return True
        if t not in existing_results:
            return True
        v = existing_results[t]
        if isinstance(v, dict) and (len(v) == 0 or "error" in v):
            return True
        return False

    per_model_tasks = [t for t in tasks_to_patch if _needs(t)]
    if not per_model_tasks:
        logger.info(f"[SKIP] {name} — none of the requested tasks need patching")
        return name, "already_done"

    # Run just the per-model patch list into a temp output dir
    tmp_dir = model_dir / "_patch_tmp"
    tmp_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        sys.executable, "-m", "blme.cli", "evaluate",
        "--model-args", build_model_args(entry),
        "--tasks", *per_model_tasks,
        "--output-dir", str(tmp_dir),
        "--task-timeout", str(task_timeout),
    ]

    logger.info(f"[PATCH] {name} (GPU {gpu_id}, {len(per_model_tasks)} tasks, "
                f"task_timeout={task_timeout}s): {per_model_tasks}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=subprocess_timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {name} (subprocess > {subprocess_timeout}s)")
        return name, "timeout"

    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error(f"[FAIL] {name} (returncode {result.returncode}): {result.stderr[-300:]}")
        return name, "failed"

    # Load the patched results
    patched_path = tmp_dir / "results.json"
    if not patched_path.exists():
        logger.error(f"[FAIL] {name} — no results.json in tmp dir")
        return name, "failed"
    with open(patched_path) as f:
        patched_env = json.load(f)
    patched_results = patched_env.get("results", {})
    patched_errors = patched_env.get("errors", {})

    # Merge into the original envelope
    envelope["results"].update(patched_results)
    # Remove from errors any task that now succeeded; record any new errors
    # for tasks that still failed.
    envelope_errors = envelope.get("errors", {})
    for t in per_model_tasks:
        if t in patched_results and isinstance(patched_results[t], dict) and "error" not in patched_results[t] and len(patched_results[t]) > 0:
            envelope_errors.pop(t, None)
        elif t in patched_errors:
            envelope_errors[t] = patched_errors[t]
    envelope["errors"] = envelope_errors
    # Update summary counts
    results = envelope.get("results", {})
    total = len(envelope.get("config", {}).get("tasks_requested", results))
    completed = sum(
        1 for v in results.values()
        if isinstance(v, dict) and "error" not in v and len(v) > 0
    )
    envelope["summary"]["completed_tasks"] = completed
    envelope["summary"]["failed_tasks"] = total - completed
    envelope["patched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Append, not overwrite — multiple patch passes accumulate.
    prior_patched = envelope.get("patched_tasks", [])
    if not isinstance(prior_patched, list):
        prior_patched = []
    envelope["patched_tasks"] = sorted(set(prior_patched) | set(per_model_tasks))

    # Write back
    with open(results_path, "w") as f:
        json.dump(envelope, f, indent=2)

    # Clean tmp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    newly_ok = sum(1 for t in per_model_tasks
                    if t in patched_results and isinstance(patched_results[t], dict)
                    and "error" not in patched_results[t] and len(patched_results[t]) > 0)
    logger.info(f"[DONE] {name} ({elapsed:.0f}s, {newly_ok}/{len(per_model_tasks)} tasks patched OK)")
    return name, f"patched {newly_ok}/{len(per_model_tasks)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="results/study_v1")
    ap.add_argument("--models", default=None, help="Comma-separated model names")
    ap.add_argument("--all", action="store_true", help="Run on all completed models")
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task names to patch. "
                         f"Defaults to: {','.join(DEFAULT_TASKS_TO_PATCH)}")
    ap.add_argument("--gpus", type=str, default=None,
                    help="Comma-separated physical GPU IDs to use (e.g. '3,4,5,6,7'). "
                         "Defaults to 0..n-gpus-1.")
    ap.add_argument("--n-gpus", type=int, default=4, help="Parallel GPUs (if --gpus not set)")
    ap.add_argument("--task-timeout", type=int, default=600,
                    help="Per-task timeout in seconds passed through to blme.cli "
                         "(default: 600). Bump for slow tasks on large/wide models.")
    ap.add_argument("--subprocess-timeout", type=int, default=7200,
                    help="Hard wall-clock cap on each model's whole patch run "
                         "(default: 7200s = 2h). Should comfortably exceed "
                         "task-timeout × number of patched tasks.")
    args = ap.parse_args()

    tasks_to_patch = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else list(DEFAULT_TASKS_TO_PATCH)
    )

    input_dir = Path(args.input_dir)
    if args.all:
        candidates = [m for m in MODELS if (input_dir / "blme" / m["name"] / "results.json").exists()]
    elif args.models:
        names = set(args.models.split(","))
        candidates = [m for m in MODELS if m["name"] in names]
    else:
        print("Specify --all or --models <list>")
        return

    # Resolve physical GPU ids first so we can decide how to dispatch each
    # model: small (n_gpus=1) models run as parallel workers across the
    # gpu_ids list; multi-GPU models reserve the entire gpu_ids list
    # sequentially via CUDA_VISIBLE_DEVICES="<list>".
    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(args.n_gpus)]

    multi_gpu_candidates = [m for m in candidates if m["n_gpus"] > 1]
    candidates = [m for m in candidates if m["n_gpus"] == 1]
    logger.info(f"Patching {len(candidates)} models across GPUs {gpu_ids}")
    logger.info(f"  tasks: {tasks_to_patch}")

    # Important: do NOT inherit CUDA_VISIBLE_DEVICES from parent so that
    # child processes see the physical GPU numbering and set_device works.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"  (unsetting parent CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
        del os.environ["CUDA_VISIBLE_DEVICES"]

    results = {}
    gpu_queue = list(gpu_ids)
    max_workers = len(gpu_ids)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        idx = 0
        while idx < len(candidates) or futures:
            while gpu_queue and idx < len(candidates):
                gpu_id = gpu_queue.pop(0)
                fut = executor.submit(_patch_model, candidates[idx], input_dir,
                                      str(gpu_id), tasks_to_patch,
                                      args.task_timeout, args.subprocess_timeout)
                futures[fut] = (candidates[idx]["name"], gpu_id)
                idx += 1
            if futures:
                done = next(as_completed(futures))
                name, gpu_id = futures.pop(done)
                gpu_queue.append(gpu_id)
                try:
                    _, status = done.result()
                    results[name] = status
                except Exception as e:
                    results[name] = f"error: {e}"

    # Multi-GPU models: each one gets the entire gpu_ids list passed as a
    # comma-joined CUDA_VISIBLE_DEVICES string, run sequentially. The model's
    # device_map=auto in build_model_args spreads the weights across them.
    if multi_gpu_candidates:
        joined = ",".join(gpu_ids)
        logger.info(f"Patching {len(multi_gpu_candidates)} multi-GPU models on GPUs {joined}")
        for m in multi_gpu_candidates:
            try:
                _, status = _patch_model(m, input_dir, joined, tasks_to_patch,
                                          args.task_timeout, args.subprocess_timeout)
                results[m["name"]] = status
            except Exception as e:
                results[m["name"]] = f"error: {e}"

    n_ok = sum(1 for v in results.values() if "patched" in str(v))
    n_skip = sum(1 for v in results.values() if v == "already_done")
    n_fail = len(results) - n_ok - n_skip
    logger.info(f"Patch complete: {n_ok} patched, {n_skip} already OK, {n_fail} failed")


if __name__ == "__main__":
    main()
