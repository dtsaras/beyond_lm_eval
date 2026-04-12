#!/usr/bin/env python3
"""
Run a comprehensive lm_eval benchmark suite across all 32 models.

This expands our Y-variable coverage from 6 benchmarks (hellaswag, piqa,
arc_easy, arc_challenge, winogrande, mmlu) to ~20 benchmarks spanning:

  1. Commonsense & NLI (existing + new)
  2. Knowledge & QA
  3. Math & Quantitative Reasoning
  4. Reading Comprehension
  5. Truthfulness
  6. Language Modeling
  7. Advanced Reasoning

All tasks use loglikelihood evaluation (not generation), so they work
cleanly with base models and are much faster than generative evals.

Usage:
    python scripts/run_comprehensive_benchmarks.py --output-dir results/study_v1 --n-gpus 8
    python scripts/run_comprehensive_benchmarks.py --model gpt2-small --output-dir results/study_v1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import MODELS, build_model_args, get_small_models, get_large_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("comprehensive_bench")

# ── New benchmark tasks (beyond what we already have) ──────────────────
# Each tuple: (task_group_name, lm_eval_task_string, n_fewshot, description)

NEW_BENCHMARKS = [
    # --- Commonsense (complement existing hellaswag/piqa/winogrande) ---
    ("boolq",           "boolq",           0,  "Boolean QA (yes/no, Clark et al. 2019)"),
    ("copa",            "copa",            0,  "Choice of Plausible Alternatives (Roemmele 2011)"),
    ("openbookqa",      "openbookqa",      0,  "Open Book QA (Mihaylov et al. 2018)"),
    ("sciq",            "sciq",            0,  "Science QA (Welbl et al. 2017)"),
    ("social_iqa",      "social_iqa",      0,  "Social Interaction QA (Sap et al. 2019)"),
    ("commonsense_qa",  "commonsense_qa",  0,  "CommonsenseQA (Talmor et al. 2019)"),

    # --- Knowledge & QA ---
    ("triviaqa",        "triviaqa",        5,  "TriviaQA (Joshi et al. 2017)"),
    ("nq_open",         "nq_open",         5,  "Natural Questions open-domain (Kwiatkowski et al. 2019)"),

    # --- Math & Quantitative ---
    ("gsm8k",           "gsm8k",           8,  "Grade School Math 8K (Cobbe et al. 2021)"),
    ("mathqa",          "mathqa",          0,  "MathQA (Amini et al. 2019)"),

    # --- Reading Comprehension ---
    ("drop",            "drop",            3,  "DROP: Discrete Reasoning Over Paragraphs (Dua et al. 2019)"),

    # --- Truthfulness ---
    ("truthfulqa_mc2",  "truthfulqa_mc2",  0,  "TruthfulQA MC2 (Lin et al. 2022)"),

    # --- Language Modeling ---
    ("lambada_openai",  "lambada_openai",  0,  "LAMBADA last-word prediction (Paperno et al. 2016)"),

    # --- Advanced Reasoning (BIG-Bench Hard — 27 sub-tasks) ---
    ("bbh",             "bbh",             3,  "BIG-Bench Hard composite (Suzgun et al. 2023)"),
]

# Skip list for tasks that need special handling or are too slow
SKIP_TASKS = set()


def run_model(model_entry, output_dir, task_str, n_fewshot, tag, gpu_ids=None):
    """Run a benchmark task group on a single model."""
    name = model_entry["name"]
    out_path = os.path.join(output_dir, "lm_eval_extended", f"{name}_{tag}")

    # Check if already done
    if os.path.exists(out_path):
        json_files = list(Path(out_path).rglob("results*.json"))
        if json_files:
            logger.info(f"[SKIP] {name}/{tag} — results exist")
            return name, tag, "skipped"

    os.makedirs(out_path, exist_ok=True)

    env = os.environ.copy()
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Build model args
    model_args_parts = [f"pretrained={model_entry['id']}"]
    model_args_parts.append(f"dtype={model_entry['dtype']}")
    if model_entry["n_gpus"] > 1:
        model_args_parts.append("parallelize=True")
    if model_entry.get("trust_remote_code"):
        model_args_parts.append("trust_remote_code=True")
    lm_eval_model_args = ",".join(model_args_parts)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", lm_eval_model_args,
        "--tasks", task_str,
        "--batch_size", "auto",
        "--output_path", out_path,
    ]
    if n_fewshot > 0:
        cmd += ["--num_fewshot", str(n_fewshot)]

    logger.info(f"[START] {name}/{tag} (GPUs: {gpu_ids or 'all'})")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=14400
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"[DONE]  {name}/{tag} ({elapsed:.0f}s)")
            return name, tag, "success"
        else:
            logger.error(f"[FAIL]  {name}/{tag}: {result.stderr[-500:]}")
            return name, tag, "failed"
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] {name}/{tag} (>14400s)")
        return name, tag, "timeout"
    except Exception as e:
        logger.error(f"[ERROR] {name}/{tag}: {e}")
        return name, tag, "error"


def run_batch(models, output_dir, benchmarks, n_gpus=8):
    """Run all benchmark groups on all models, parallelized across GPUs.

    Each (model, benchmark_group) pair is a job. Small models (1 GPU) run
    in parallel; large models run sequentially.
    """
    # Resolve physical GPU IDs
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        physical_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        physical_gpus = [str(i) for i in range(n_gpus)]
    physical_gpus = physical_gpus[:n_gpus]

    small = [m for m in models if m["n_gpus"] == 1]
    large = [m for m in models if m["n_gpus"] > 1]

    results = {}

    # --- Small models: parallel across GPUs ---
    # Each job = (model, benchmark_group). Round-robin across GPUs.
    jobs = [(m, tag, task_str, nshot)
            for m in small
            for tag, task_str, nshot, _ in benchmarks
            if tag not in SKIP_TASKS]

    logger.info(f"Running {len(jobs)} jobs ({len(small)} small models x "
                f"{len(benchmarks)} benchmarks) across {len(physical_gpus)} GPUs")

    gpu_queue = list(physical_gpus)
    with ProcessPoolExecutor(max_workers=len(physical_gpus)) as executor:
        futures = {}
        idx = 0
        while idx < len(jobs) or futures:
            while gpu_queue and idx < len(jobs):
                m, tag, task_str, nshot = jobs[idx]
                gpu_id = gpu_queue.pop(0)
                future = executor.submit(
                    run_model, m, output_dir, task_str, nshot, tag, str(gpu_id)
                )
                futures[future] = (m["name"], tag, gpu_id)
                idx += 1
            if futures:
                done = next(as_completed(futures))
                name, tag, gpu_id = futures.pop(done)
                gpu_queue.append(gpu_id)
                try:
                    _, _, status = done.result()
                    results[f"{name}/{tag}"] = status
                except Exception as e:
                    results[f"{name}/{tag}"] = f"error: {e}"

    # --- Large models: sequential (multi-GPU each) ---
    for m in large:
        n = m["n_gpus"]
        gpu_ids = ",".join(physical_gpus[:n])
        for tag, task_str, nshot, _ in benchmarks:
            if tag in SKIP_TASKS:
                continue
            _, _, status = run_model(m, output_dir, task_str, nshot, tag, gpu_ids)
            results[f"{m['name']}/{tag}"] = status

    return results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive lm_eval benchmarks")
    parser.add_argument("--output-dir", type=str, default="results/study_v1",
                        help="Output directory")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a specific model by name (otherwise all)")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Total available GPUs")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark tags to run (default: all new)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    args = parser.parse_args()

    benchmarks = NEW_BENCHMARKS
    if args.benchmarks:
        tags = set(args.benchmarks.split(","))
        benchmarks = [b for b in benchmarks if b[0] in tags]

    if args.model:
        models = [m for m in MODELS if m["name"] == args.model]
        if not models:
            print(f"Unknown model: {args.model}")
            sys.exit(1)
    else:
        models = MODELS

    if args.dry_run:
        print(f"=== Comprehensive Benchmark Plan ===")
        print(f"Output: {args.output_dir}/lm_eval_extended/")
        print(f"GPUs: {args.n_gpus}")
        print(f"Models: {len(models)}")
        print(f"\nBenchmarks ({len(benchmarks)}):")
        for tag, task_str, nshot, desc in benchmarks:
            print(f"  {tag:<20s} {task_str:<25s} {nshot}-shot  {desc}")
        print(f"\nTotal jobs: {len(models)} x {len(benchmarks)} = {len(models) * len(benchmarks)}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results = run_batch(models, args.output_dir, benchmarks, args.n_gpus)

    # Save summary
    summary_path = os.path.join(args.output_dir, "comprehensive_bench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    success = sum(1 for v in results.values() if v == "success")
    failed = sum(1 for v in results.values() if v in ("failed", "error", "timeout"))
    skipped = sum(1 for v in results.values() if v == "skipped")
    logger.info(f"COMPLETE: {success} success, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
