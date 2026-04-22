#!/usr/bin/env python3
"""
BLME benchmark study driver — runs all models through both BLME intrinsic
diagnostics and lm_eval benchmarks.

Parallelization strategy (8x RTX 3090):
  Phase 1: Small models (1 GPU each) — run 8 in parallel
  Phase 2: Medium models (1 GPU, but memory-heavy) — run 4 in parallel
  Phase 3: Large models (3 GPUs each) — run 1 at a time

Usage:
    python scripts/run_study.py --phase all --output-dir results/study_v1
    python scripts/run_study.py --phase small --output-dir results/study_v1
    python scripts/run_study.py --phase large --output-dir results/study_v1
    python scripts/run_study.py --model gpt2-small --output-dir results/study_v1
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_zoo import MODELS, build_model_args, get_small_models, get_large_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_study")

# ── Task lists ──────────────────────────────────────────────────────────

# Research tasks (used as X-variables in the correlation study).
# Excludes: library-only tasks, Y-variable tasks, tasks needing ripser/sae-lens.
BLME_RESEARCH_TASKS = [
    # Geometry (Tier 1)
    "geometry_spectral", "geometry_hubness", "geometry_unembedding",
    "geometry_weight_norms", "geometry_tokenizer_efficiency",
    # Geometry (Tier 2)
    "geometry_svd", "geometry_isoscore", "geometry_lid", "geometry_collapse",
    "geometry_lipschitz", "geometry_intrinsic_dim", "geometry_matrix_entropy",
    "geometry_hsic", "geometry_rsa", "geometry_cka",
    "geometry_correlation_dimension", "geometry_positional_decay",
    "geometry_prediction_alignment", "geometry_contextualization",
    "geometry_neural_collapse",
    # Interpretability
    "interpretability_logit_lens", "interpretability_attention_entropy",
    "interpretability_attention_rank", "interpretability_induction_heads",
    "interpretability_head_roles", "interpretability_prediction_entropy",
    "interpretability_sparsity", "interpretability_superposition",
    "interpretability_waa", "interpretability_attention_graph",
    # Causality
    "causality_tracing", "causality_attention_knockout",
    "causality_circuit_quality", "causality_knowledge_neurons",
    "causality_edge_attribution",
    # Consistency & Dynamics
    "dynamics_gradient_flow", "dynamics_sharpness",
    "consistency_position_sensitivity", "consistency_format_robustness",
    "consistency_icl_slope",
    # RepE
    "repe_task_vectors", "repe_concept_separability",
    "repe_refusal_direction",
    # Additional dynamics / causality (previously registered but not
    # wired into the pipeline — the full rewrites landed in the round-3
    # audit so they're paper-faithful now).
    "dynamics_coe",
    "causality_ablation",
    "geometry_mahalanobis",
    # Round-7: 2025-2026 literature additions. Schatten-p norms +
    # Matrix Nuclear-Norm + RankMe (Wei et al. 2025, Li et al. 2024,
    # Garrido et al. 2023 — all new spectral representations of
    # capability.
    "geometry_schatten",
    # Round-8: attention-sink + massive-activation + compression-
    # valley (Gu et al. ICLR 2025, Sun et al. 2024, Pedrotti & Guo
    # arXiv:2510.06477) — three linked phenomena the 2024-2025
    # literature unifies as the BOS-as-bias-channel mechanism.
    "interpretability_activation_sinks",
]

# Y-variable tasks (perplexity, calibration, prediction entropy)
BLME_Y_TASKS = [
    "geometry_perplexity", "consistency_calibration",
    "interpretability_prediction_entropy",
]

# lm_eval benchmarks
LM_EVAL_TASKS = "hellaswag,piqa,arc_easy,arc_challenge,winogrande"
LM_EVAL_MMLU = "mmlu"  # Run separately (5-shot)


def run_blme_model(model_entry, output_dir, gpu_ids=None):
    """Run all BLME tasks for a single model."""
    name = model_entry["name"]
    model_args = build_model_args(model_entry)
    model_out_dir = os.path.join(output_dir, "blme", name)

    # Check if results already exist (the CLI writes results.json in the output dir)
    if os.path.exists(os.path.join(model_out_dir, "results.json")):
        logger.info(f"[SKIP] {name} — BLME results already exist")
        return name, "skipped"

    os.makedirs(model_out_dir, exist_ok=True)

    all_tasks = BLME_RESEARCH_TASKS + BLME_Y_TASKS
    task_str = " ".join(all_tasks)

    env = os.environ.copy()
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Each model gets its own output subdirectory
    model_out_dir = os.path.join(output_dir, "blme", name)
    cmd = [
        sys.executable, "-m", "blme.cli", "evaluate",
        "--model-args", model_args,
        "--tasks", *all_tasks,
        "--output-dir", model_out_dir,
    ]

    logger.info(f"[START] BLME {name} (GPUs: {gpu_ids or 'all'})")
    t0 = time.time()

    # BLME timeout: 2h default, 8h for 70B+ models. Pipeline parallel across
    # 8 GPUs is ~4-6× slower than tensor parallel per-task because only one
    # GPU computes at a time, and the 50-task suite aggregates to 5-6h on
    # Llama-3.1-70B in practice.
    blme_timeout = 28800 if model_entry["n_gpus"] >= 8 else 7200
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=blme_timeout
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"[DONE]  BLME {name} ({elapsed:.0f}s)")
            return name, "success"
        else:
            logger.error(f"[FAIL]  BLME {name}: {result.stderr[-500:]}")
            # Save error log
            err_path = os.path.join(model_out_dir, "error.log")
            with open(err_path, "w") as f:
                f.write(result.stderr)
            return name, "failed"
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] BLME {name} (>{blme_timeout}s)")
        return name, "timeout"
    except Exception as e:
        logger.error(f"[ERROR] BLME {name}: {e}")
        return name, "error"


def run_lmeval_model(model_entry, output_dir, gpu_ids=None):
    """Run lm_eval benchmarks for a single model."""
    name = model_entry["name"]
    out_path = os.path.join(output_dir, "lm_eval", f"{name}.json")

    if os.path.exists(out_path):
        logger.info(f"[SKIP] {name} — lm_eval results already exist")
        return name, "skipped"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = os.environ.copy()
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Build lm_eval model args
    model_args_parts = [f"pretrained={model_entry['id']}"]
    model_args_parts.append(f"dtype={model_entry['dtype']}")
    if model_entry["n_gpus"] > 1:
        model_args_parts.append("parallelize=True")
    if model_entry.get("trust_remote_code"):
        model_args_parts.append("trust_remote_code=True")
    lm_eval_model_args = ",".join(model_args_parts)

    # Run main benchmarks (0-shot)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", lm_eval_model_args,
        "--tasks", LM_EVAL_TASKS,
        "--batch_size", "auto",
        "--output_path", os.path.join(output_dir, "lm_eval", name),
    ]

    logger.info(f"[START] lm_eval {name} (GPUs: {gpu_ids or 'all'})")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=14400
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"[DONE]  lm_eval(basic) {name} ({elapsed:.0f}s)")

            # Also run MMLU 5-shot
            cmd_mmlu = [
                sys.executable, "-m", "lm_eval",
                "--model", "hf",
                "--model_args", lm_eval_model_args,
                "--tasks", LM_EVAL_MMLU,
                "--num_fewshot", "5",
                "--batch_size", "auto",
                "--output_path", os.path.join(output_dir, "lm_eval", f"{name}_mmlu"),
            ]
            logger.info(f"[START] mmlu {name} (GPUs: {gpu_ids or 'all'})")
            t_m = time.time()
            r_m = subprocess.run(cmd_mmlu, env=env, capture_output=True, text=True, timeout=14400)
            m_elapsed = time.time() - t_m
            if r_m.returncode == 0:
                logger.info(f"[DONE]  mmlu {name} ({m_elapsed:.0f}s)")
            else:
                logger.error(f"[FAIL]  mmlu {name}: {r_m.stderr[-500:]}")
            return name, "success"
        else:
            logger.error(f"[FAIL]  lm_eval {name}: {result.stderr[-500:]}")
            return name, "failed"
    except subprocess.TimeoutExpired:
        logger.error(f"[TIMEOUT] lm_eval {name}")
        return name, "timeout"
    except Exception as e:
        logger.error(f"[ERROR] lm_eval {name}: {e}")
        return name, "error"


def run_single_gpu_batch(models, output_dir, n_gpus=8, run_type="blme"):
    """Run multiple 1-GPU models in parallel across available GPUs.

    Respects the parent's CUDA_VISIBLE_DEVICES: if the parent restricted
    the visible GPUs (e.g. CUDA_VISIBLE_DEVICES=3,4,5,6), child workers
    are assigned the PHYSICAL GPU ids from that list, not 0..n-1.
    Otherwise, when the child sets CUDA_VISIBLE_DEVICES="0" it would be
    interpreted as physical GPU 0, colliding with other phases.
    """
    runner = run_blme_model if run_type == "blme" else run_lmeval_model
    results = {}

    # Resolve physical GPU IDs from parent's CUDA_VISIBLE_DEVICES
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        physical_gpus = [g.strip() for g in parent_cvd.split(",") if g.strip()]
        # Unset parent's CUDA_VISIBLE_DEVICES so child sees all physical
        # GPUs and can pick the one we assign by physical id.
        del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        physical_gpus = [str(i) for i in range(n_gpus)]
    physical_gpus = physical_gpus[:n_gpus]

    # Queue physical GPU ids
    gpu_queue = list(physical_gpus)
    with ProcessPoolExecutor(max_workers=len(physical_gpus)) as executor:
        futures = {}
        model_idx = 0

        while model_idx < len(models) or futures:
            # Submit new jobs for free GPUs
            while gpu_queue and model_idx < len(models):
                gpu_id = gpu_queue.pop(0)
                m = models[model_idx]
                future = executor.submit(runner, m, output_dir, str(gpu_id))
                futures[future] = (m["name"], gpu_id)
                model_idx += 1

            # Wait for any completion
            if futures:
                done = next(as_completed(futures))
                name, gpu_id = futures.pop(done)
                gpu_queue.append(gpu_id)
                try:
                    _, status = done.result()
                    results[name] = status
                except Exception as e:
                    results[name] = f"error: {e}"
                    logger.error(f"[ERROR] {name}: {e}")

    return results


def run_multi_gpu_sequential(models, output_dir, total_gpus=8, run_type="blme"):
    """Run multi-GPU models sequentially (they need >1 GPU each)."""
    runner = run_blme_model if run_type == "blme" else run_lmeval_model
    results = {}

    for m in models:
        n = m["n_gpus"]
        gpu_ids = ",".join(str(i) for i in range(n))
        _, status = runner(m, output_dir, gpu_ids)
        results[m["name"]] = status

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BLME benchmark study")
    parser.add_argument("--phase", choices=["small", "large", "all", "lm_eval"],
                        default="all", help="Which phase to run")
    parser.add_argument("--model", type=str, default=None,
                        help="Run a specific model by name")
    parser.add_argument("--output-dir", type=str, default="results/study_v1",
                        help="Output directory")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Total available GPUs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model:
        # Run a single model
        entry = next((m for m in MODELS if m["name"] == args.model), None)
        if entry is None:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(m['name'] for m in MODELS)}")
            sys.exit(1)
        if args.dry_run:
            print(f"Would run: {entry['name']} ({build_model_args(entry)})")
            return
        run_blme_model(entry, args.output_dir)
        run_lmeval_model(entry, args.output_dir)
        return

    small = get_small_models()
    large = get_large_models()

    if args.dry_run:
        print(f"=== Study Plan ===")
        print(f"Output: {args.output_dir}")
        print(f"GPUs: {args.n_gpus}")
        print(f"\nPhase 1 — Small models ({len(small)} models, {args.n_gpus} parallel):")
        for m in small:
            print(f"  {m['name']:<25s} {m['dtype']}")
        print(f"\nPhase 2 — Large models ({len(large)} models, sequential):")
        for m in large:
            print(f"  {m['name']:<25s} {m['n_gpus']} GPUs, {m['dtype']}")
        print(f"\nPhase 3 — lm_eval benchmarks (same model list)")
        print(f"\nTotal unique models: {len(MODELS)}")
        return

    all_results = {}

    if args.phase in ("small", "all"):
        logger.info(f"═══ Phase 1: BLME on {len(small)} small models ({args.n_gpus} parallel) ═══")
        r = run_single_gpu_batch(small, args.output_dir, args.n_gpus, "blme")
        all_results.update(r)

    if args.phase in ("large", "all"):
        logger.info(f"═══ Phase 2: BLME on {len(large)} large models (sequential) ═══")
        r = run_multi_gpu_sequential(large, args.output_dir, args.n_gpus, "blme")
        all_results.update(r)

    if args.phase in ("lm_eval", "all"):
        logger.info(f"═══ Phase 3: lm_eval on {len(small)} small models ({args.n_gpus} parallel) ═══")
        r = run_single_gpu_batch(small, args.output_dir, args.n_gpus, "lm_eval")
        all_results.update(r)

        logger.info(f"═══ Phase 4: lm_eval on {len(large)} large models (sequential) ═══")
        r = run_multi_gpu_sequential(large, args.output_dir, args.n_gpus, "lm_eval")
        all_results.update(r)

    # Save summary
    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Report
    success = sum(1 for v in all_results.values() if v == "success")
    failed = sum(1 for v in all_results.values() if v in ("failed", "error", "timeout"))
    skipped = sum(1 for v in all_results.values() if v == "skipped")
    logger.info(f"═══ COMPLETE: {success} success, {failed} failed, {skipped} skipped ═══")


if __name__ == "__main__":
    main()
