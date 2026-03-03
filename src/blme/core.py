"""
Core evaluation dispatcher for BLME.

Loads the model, resolves tasks from the registry, runs each task with error
isolation, and produces structured results with metadata.
"""

import logging
from typing import List, Optional, Dict, Any, Union

import torch
from tqdm import tqdm

from .registry import get_task, list_tasks
from .models.wrapper import load_model_and_tokenizer
from .results import build_results_envelope, print_results_table, save_results

logger = logging.getLogger("blme")

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

    Returns:
        Full results envelope dict.
    """
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

    if diagnostic_tasks:
        # Determine what the tasks need so We can populate the cache once
        _HIDDEN_TASK_PREFIXES = (
            "geometry_", "topology_", "repe_",
            "interpretability_probing",
            "consistency_calibration", "consistency_paraphrase",
        )
        _ATTN_TASK_PREFIXES = (
            "interpretability_attention", "interpretability_induction",
            "geometry_positional_decay",
        )
        need_hidden = any(t.startswith(_HIDDEN_TASK_PREFIXES) for t in diagnostic_tasks)
        need_attn = any(t.startswith(_ATTN_TASK_PREFIXES) for t in diagnostic_tasks)

        # Create shared cache
        from .cache import ModelOutputCache
        cache = ModelOutputCache(model, tokenizer, dataset=None, num_samples=100)
        if need_hidden or need_attn:
            cache.populate(need_hidden=need_hidden, need_attentions=need_attn)

        logger.info(f"Running {len(diagnostic_tasks)} diagnostic tasks...")
        for task_name in tqdm(diagnostic_tasks, desc="BLME tasks", unit="task"):
            task_cls = get_task(task_name)

            # Merge defaults.yaml + user overrides
            from .tasks.config_loader import resolve_task_config
            user_override = task_configs.get(task_name, {}) if task_configs else {}
            t_config = resolve_task_config(task_name, user_override)

            try:
                task = task_cls(config=t_config)
                result = task.evaluate(model, tokenizer, dataset=None, cache=cache)
                task_results[task_name] = result
            except Exception as e:
                logger.error(f"Task '{task_name}' failed: {e}")
                task_errors[task_name] = str(e)

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
        device=device,
    )

    # 7. Print summary table
    print_results_table(task_results, task_errors)

    # 8. Save to disk
    if output_dir:
        save_results(envelope, output_dir, output_format)

    return envelope
