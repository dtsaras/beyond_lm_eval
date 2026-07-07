"""
Core evaluation dispatcher for BLME.

Loads the model, resolves tasks from the registry, runs each task with error
isolation, and produces structured results with metadata.
"""

import logging
import signal
import time
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from .registry import get_task, list_tasks
from .models.wrapper import load_model_and_tokenizer
from .results import build_results_envelope, print_results_table, save_results

logger = logging.getLogger("blme")


class TaskTimeout(BaseException):
    """Raised by the SIGALRM handler when a task exceeds its timeout.

    Derives from BaseException (NOT Exception) so a task's own broad
    ``except Exception`` cannot swallow the alarm and silently defeat the
    per-task timeout (a timed-out task would then be reported as OK).
    """


def _timeout_handler(signum, frame):
    raise TaskTimeout("Task execution timed out")


# Tasks that read raw attention weights (their own forward pass and/or the
# shared attention cache). On modern transformers, SDPA/Flash attention silently
# return None for ``outputs.attentions``, so these MUST run with eager attention
# or they fail on a default model load. Keep in sync with every task that reads
# ``outputs.attentions`` / ``cache.get_attentions()`` — enforced by
# test_infrastructure::test_force_eager_covers_all_attention_tasks.
_ATTENTION_TASKS = frozenset({
    "interpretability_attention_entropy",
    "interpretability_attention_graph",
    "interpretability_attention_rank",
    "interpretability_attention_rollout",
    "interpretability_head_roles",
    "interpretability_induction_heads",
    "interpretability_activation_sinks",
    "geometry_positional_decay",
})


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
    dataset: Optional[Any] = None,
    fail_on_task_error: bool = False,
    strict_task_validation: bool = False,
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
        dataset: Optional list of ``{"text": ...}`` dicts to use as
            the evaluation corpus.  Passed to both the shared cache and
            individual tasks.  When *None*, WikiText-103 validation is
            loaded automatically (see :func:`cache.load_default_corpus`).

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

    # 2. Resolve task names before model load so typos fail fast.
    if tasks is None:
        tasks = []

    diagnostic_tasks = []
    lm_eval_tasks = []

    unknown_tasks = []
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
            unknown_tasks.append(task_name)

    if unknown_tasks:
        msg = (
            "Unknown task(s): "
            + ", ".join(unknown_tasks)
            + ". Use `blme list-tasks --json` to inspect registered tasks."
        )
        if strict_task_validation:
            raise ValueError(msg)
        logger.warning(msg + " Skipping.")

    # 3. Init result accumulators
    task_results: Dict[str, Any] = {}
    task_errors: Dict[str, str] = {}
    task_timings: Dict[str, float] = {}
    resolved_task_configs: Dict[str, dict] = {}
    cache_settings: Dict[str, Any] = {
        "cache_num_samples_requested": cache_num_samples,
        "cache_tasks": [],
        "need_hidden": False,
        "need_attentions": False,
        "need_logits": False,
    }

    # 4. Run diagnostic tasks with error isolation + shared cache
    model, tokenizer, cache = None, None, None

    if diagnostic_tasks:
        # Load the model only when there are diagnostic tasks — a pure-benchmark
        # (lm_eval-only) run loads its own model inside run_lm_eval, so loading
        # it here as well would waste a whole model copy.
        logger.info("Loading model...")
        model, tokenizer = load_model_and_tokenizer(model_args, device=device)

        # Resolve configs FIRST (defaults + user overrides) so the shared corpus
        # can be sized to the largest per-task num_samples. Loading the corpus
        # before this capped it at the 200-sample default and silently starved
        # any task requesting more.
        from .tasks.config_loader import resolve_task_config
        for task_name in diagnostic_tasks:
            user_override = task_configs.get(task_name, {}) if task_configs else {}
            resolved = resolve_task_config(task_name, user_override)
            # Propagate the run seed so tasks that build local RNGs are seeded
            # from it (and stay reproducible / order-independent), unless a task
            # config overrides it explicitly.
            resolved.setdefault("seed", seed)
            resolved_task_configs[task_name] = resolved

        # Load a shared evaluation corpus if none was provided, sized to the
        # largest requested sample count (never below the 200-sample default).
        if dataset is None:
            from .cache import load_default_corpus
            task_sample_counts = [
                resolved_task_configs[t].get("num_samples")
                for t in diagnostic_tasks
                if resolved_task_configs[t].get("num_samples") is not None
            ]
            corpus_size = max([cache_num_samples or 0, 200] + task_sample_counts)
            dataset = load_default_corpus(corpus_size)

        # Shared cache: only tasks known to consume cached hidden states or
        # logits are listed here so we avoid the memory cost of caching when
        # no requesting task is in the run. (cache is initialised to None above.)
        cache_hidden_tasks = {
            "geometry_svd",
            "geometry_lid",
            "geometry_lipschitz",
            "geometry_collapse",
            "geometry_hsic",
            "geometry_intrinsic_dim",
            "geometry_prediction_alignment",
            "geometry_rsa",
            "geometry_matrix_entropy",
            "geometry_cka",
            "geometry_trajectory_curvature",
            "geometry_mp_bulk_deviation",
        }
        cache_logits_tasks = {
            "geometry_perplexity",
            "consistency_calibration",
            "geometry_prediction_alignment",
        }
        # Two separate sets:
        #   * ``cache_attn_tasks`` — tasks that actually consume
        #     ``cache.get_attentions()``. Only for these do we need to
        #     populate the expensive per-layer attention tensors.
        #   * ``force_eager_tasks`` — tasks that require attention
        #     weights at all (their own forward pass or the cache). On
        #     modern transformers SDPA/Flash silently return None for
        #     ``outputs.attentions``, so we must force eager even when
        #     the task does its own forward pass. Caching is not needed.
        cache_attn_tasks = {
            "interpretability_attention_entropy",
        }
        force_eager_tasks = _ATTENTION_TASKS
        cache_candidates = (
            set(cache_hidden_tasks)
            | set(cache_logits_tasks)
            | set(cache_attn_tasks)
        )

        cache_tasks = []
        for task_name, cfg in resolved_task_configs.items():
            if not cfg.get("use_cache", True):
                continue
            if task_name not in cache_candidates:
                continue
            if task_name == "geometry_intrinsic_dim" and not cfg.get("layerwise", False):
                continue
            cache_tasks.append(task_name)

        # Force eager attention for any task that needs raw attention weights,
        # whether or not that task consumes the shared attention cache.
        force_eager = any(
            t in force_eager_tasks for t in diagnostic_tasks
        )
        if force_eager:
            try:
                if hasattr(model, "config"):
                    model.config._attn_implementation = "eager"
                if hasattr(model, "set_attn_implementation"):
                    model.set_attn_implementation("eager")
            except Exception as e:
                logger.info(
                    f"Could not switch to eager attention: {e}. "
                    "Attention-consuming tasks may fail."
                )

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
                need_attn = any(t in cache_attn_tasks for t in cache_tasks)
                need_logits = any(t in cache_logits_tasks for t in cache_tasks)
                cache_settings.update({
                    "cache_num_samples_resolved": cache_num_samples,
                    "cache_tasks": list(cache_tasks),
                    "need_hidden": bool(need_hidden),
                    "need_attentions": bool(need_attn),
                    "need_logits": bool(need_logits),
                })

                from .cache import ModelOutputCache
                cache = ModelOutputCache(model, tokenizer, dataset=dataset, num_samples=cache_num_samples)
                cache.populate(
                    need_hidden=need_hidden,
                    need_attentions=need_attn,
                    need_logits=need_logits,
                )

        # SIGALRM-based per-task timeouts only work on the main thread of a Unix
        # process. Detect this once so the per-task loop never tries (and fails)
        # to install a handler off the main thread (Audit-V2 infra:7).
        import threading
        can_timeout = (
            hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        if not hasattr(signal, "SIGALRM"):
            logger.warning("signal.SIGALRM not available (non-Unix). Per-task timeouts disabled.")
        elif not can_timeout:
            logger.warning("evaluate() not on the main thread. Per-task timeouts disabled.")

        logger.info(f"Running {len(diagnostic_tasks)} diagnostic tasks...")
        for task_name in tqdm(diagnostic_tasks, desc="BLME tasks", unit="task"):
            task_cls = get_task(task_name)
            t_config = resolved_task_configs[task_name]
            timeout_sec = t_config.get("timeout", task_timeout)

            t_start = time.perf_counter()
            # Initialize before the try so the finally never raises
            # UnboundLocalError if signal.signal() itself fails (e.g. when
            # evaluate() runs off the main thread). See Audit-V2 infra:7.
            old_handler = None
            alarm_armed = False
            try:
                if can_timeout:
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(int(timeout_sec))
                    alarm_armed = True

                task = task_cls(config=t_config)
                result = task.evaluate(model, tokenizer, dataset=dataset, cache=cache)
                task_results[task_name] = result
            except TaskTimeout:
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
                if alarm_armed:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                elapsed = time.perf_counter() - t_start
                task_timings[task_name] = round(elapsed, 2)

    # 5. Run lm_eval benchmark tasks (optional)
    if lm_eval_tasks:
        # Free the diagnostic model + shared cache before lm_eval loads its own
        # HFLM copy, so a mixed diagnostic+benchmark run never holds two full
        # models (plus the cache) in memory at once.
        if model is not None:
            import gc
            model = tokenizer = cache = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    # Tasks that returned {"error": ...} are failures too (they didn't raise,
    # but they did not succeed), so --strict must fail on them as well.
    errored_results = {
        name: r["error"] for name, r in task_results.items()
        if isinstance(r, dict) and "error" in r
    }
    all_failures = {**task_errors, **errored_results}
    if fail_on_task_error and all_failures:
        raise RuntimeError(f"BLME task errors under strict mode: {all_failures}")

    # 6. Build results envelope
    envelope = build_results_envelope(
        model_args=model_args,
        tasks_requested=tasks,
        task_results=task_results,
        task_errors=task_errors,
        task_timings=task_timings,
        device=device,
        seed=seed,
        task_configs_resolved=resolved_task_configs,
        cache_settings=cache_settings,
        unknown_tasks=unknown_tasks,
    )

    # 7. Print summary table
    print_results_table(task_results, task_errors, task_timings=task_timings)

    # 8. Save to disk
    if output_dir:
        save_results(envelope, output_dir, output_format)

    return envelope
