"""
Core evaluation dispatcher for BLME.

Loads the model, resolves tasks from the registry, runs each task with error
isolation, and produces structured results with metadata.
"""

import logging
import signal
import time
from typing import List, Optional, Dict, Any, Union

import torch
from tqdm import tqdm

from .registry import get_task, list_tasks
from .models.wrapper import load_model_and_tokenizer
from .results import build_results_envelope, print_results_table, save_results

logger = logging.getLogger("blme")


def _timeout_handler(signum, frame):
    raise TimeoutError("Task execution timed out")


# ---------------------------------------------------------------------------
# Task registration — import all task modules so @register_task fires
# ---------------------------------------------------------------------------

_TASKS_REGISTERED = False

def _register_all_tasks():
    global _TASKS_REGISTERED
    if _TASKS_REGISTERED:
        return
    from .tasks import geometry, dynamics, consistency, interpretability  # noqa: F401
    from .tasks import causality, topology  # noqa: F401
    from .tasks import representation_engineering  # noqa: F401
    _TASKS_REGISTERED = True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    model_args: str = "",
    tasks: List[str] = None,
    limit: Optional[float] = None,
    output_dir: Optional[str] = None,
    output_format: str = "json",
    task_configs: Optional[Dict[str, dict]] = None,
    batch_size: Union[int, str, None] = None,
    device: Optional[str] = None,
    cache_num_samples: Optional[int] = None,
    seed: int = 42,
    task_timeout: int = 600,
) -> dict:
    """
    Unified evaluation entry point.

    Args:
        model_args: Comma-separated key=value string for model loading.
        tasks: List of task names to evaluate.
        limit: Sample limit (passed to lm_eval benchmark tasks).
        output_dir: Directory to write results file.
        output_format: 'json' or 'csv'.
        task_configs: Per-task config dicts {task_name: {config_key: value}}.
        batch_size: Batch size (passed to lm_eval benchmark tasks).
        device: Target device (e.g., 'cuda', 'cpu').
        cache_num_samples: Optional global sample count for shared cache.
        seed: Random seed for reproducibility (default: 42).
        task_timeout: Per-task timeout in seconds (default: 600). Unix only.

    Returns:
        Full results envelope dict.
    """
    # Set global seed for reproducibility
    from .utils import set_global_seed
    set_global_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Ensure all tasks are registered
    _register_all_tasks()

    # 2. Load Model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_args, device=device)

    # 3. Resolve tasks
    if tasks is None:
        tasks = []

    diagnostic_tasks = []
    lm_eval_tasks = []

    for task_name in tasks:
        if get_task(task_name):
            diagnostic_tasks.append(task_name)
        else:
            # Check if it's an lm_eval task
            try:
                from .tasks.benchmarks import is_lm_eval_task
                if is_lm_eval_task(task_name):
                    lm_eval_tasks.append(task_name)
                    continue
            except Exception:
                pass
            logger.warning(f"Task '{task_name}' not found in registry or lm_eval. Skipping.")

    # 4. Run diagnostic tasks with error isolation + shared cache
    task_results: Dict[str, Any] = {}
    task_errors: Dict[str, str] = {}
    task_timings: Dict[str, float] = {}

    if diagnostic_tasks:
        # Resolve configs once (defaults + user overrides)
        from .tasks.config_loader import resolve_task_config
        resolved_task_configs: Dict[str, dict] = {}
        for task_name in diagnostic_tasks:
            user_override = task_configs.get(task_name, {}) if task_configs else {}
            resolved_task_configs[task_name] = resolve_task_config(task_name, user_override)

        # Shared cache: only tasks known to consume cached hidden states or
        # logits are listed here so we avoid the memory cost of caching when
        # no requesting task is in the run.
        cache = None
        cache_hidden_tasks = {
            "geometry_svd",
            "geometry_lid",
            "geometry_lipschitz",
            "geometry_collapse",
            "geometry_mutual_info",
            "geometry_intrinsic_dim",
            "geometry_consistency",
            "geometry_rsa",
            "geometry_matrix_entropy",
            "geometry_cka",
        }
        cache_logits_tasks = {
            "geometry_perplexity",
            "consistency_calibration",
        }
        cache_candidates = set(cache_hidden_tasks) | set(cache_logits_tasks)

        cache_tasks = []
        for task_name, cfg in resolved_task_configs.items():
            if not cfg.get("use_cache", True):
                continue
            if task_name not in cache_candidates:
                continue
            if task_name == "geometry_intrinsic_dim" and not cfg.get("layerwise", False):
                continue
            cache_tasks.append(task_name)

        if cache_tasks:
            if cache_num_samples is None:
                sample_counts = [
                    resolved_task_configs[t].get("num_samples")
                    for t in cache_tasks
                    if resolved_task_configs[t].get("num_samples") is not None
                ]
                cache_num_samples = max(sample_counts) if sample_counts else 100

            if cache_num_samples and cache_num_samples > 0:
                need_hidden = any(t in cache_hidden_tasks for t in cache_tasks)
                need_attn = False

                from .cache import ModelOutputCache
                cache = ModelOutputCache(model, tokenizer, dataset=None, num_samples=cache_num_samples)
                cache.populate(need_hidden=need_hidden, need_attentions=need_attn)

        if not hasattr(signal, "SIGALRM"):
            logger.warning("signal.SIGALRM not available (non-Unix). Per-task timeouts disabled.")

        logger.info(f"Running {len(diagnostic_tasks)} diagnostic tasks...")
        for task_name in tqdm(diagnostic_tasks, desc="BLME tasks", unit="task"):
            task_cls = get_task(task_name)
            t_config = resolved_task_configs[task_name]
            timeout_sec = t_config.get("timeout", task_timeout)

            t_start = time.perf_counter()
            try:
                if hasattr(signal, "SIGALRM"):
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_sec)

                task = task_cls(config=t_config)
                result = task.evaluate(model, tokenizer, dataset=None, cache=cache)
                task_results[task_name] = result
            except TimeoutError:
                logger.error(f"Task '{task_name}' timed out after {timeout_sec}s")
                task_errors[task_name] = f"Timeout after {timeout_sec}s"
            except torch.cuda.OutOfMemoryError:
                logger.error(f"Task '{task_name}' ran out of GPU memory")
                torch.cuda.empty_cache()
                task_errors[task_name] = "CUDA out of memory"
            except Exception as e:
                logger.error(f"Task '{task_name}' failed: {e}")
                task_errors[task_name] = str(e)
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                elapsed = time.perf_counter() - t_start
                task_timings[task_name] = round(elapsed, 2)

    # 5. Run lm_eval benchmark tasks (optional)
    if lm_eval_tasks:
        logger.info(f"Running {len(lm_eval_tasks)} lm_eval benchmark tasks...")
        try:
            from .tasks.benchmarks import run_lm_eval
            lm_results = run_lm_eval(
                model="hf",
                model_args=model_args,
                tasks=lm_eval_tasks,
                batch_size=batch_size,
                device=device,
                limit=limit,
            )
            task_results["_lm_eval"] = lm_results
        except Exception as e:
            logger.error(f"lm_eval tasks failed: {e}")
            task_errors["_lm_eval"] = str(e)

    # 6. Build results envelope
    envelope = build_results_envelope(
        model_args=model_args,
        tasks_requested=tasks,
        task_results=task_results,
        task_errors=task_errors,
        task_timings=task_timings,
        device=device,
        seed=seed,
    )

    # 7. Print summary table
    print_results_table(task_results, task_errors, task_timings=task_timings)

    # 8. Save to disk
    if output_dir:
        save_results(envelope, output_dir, output_format)

    return envelope
