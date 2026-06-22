try:
    from lm_eval import simple_evaluate
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False

from typing import List
import torch

_LM_EVAL_TASK_CACHE = None

def is_lm_eval_task(task_name: str) -> bool:
    """Return True when lm_eval knows about *task_name*.

    Falls back to a small allow-list only when lm_eval is unavailable, so
    CLI validation remains useful in minimal BLME installs.
    """
    global _LM_EVAL_TASK_CACHE
    if HAS_LM_EVAL:
        try:
            if _LM_EVAL_TASK_CACHE is None:
                from lm_eval.tasks import TaskManager
                manager = TaskManager()
                if hasattr(manager, "all_tasks"):
                    _LM_EVAL_TASK_CACHE = set(manager.all_tasks)
                elif hasattr(manager, "task_index"):
                    _LM_EVAL_TASK_CACHE = set(manager.task_index)
                else:
                    _LM_EVAL_TASK_CACHE = set()
            if task_name in _LM_EVAL_TASK_CACHE:
                return True
        except Exception:
            pass

    common = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "truthfulqa", "winogrande", "gsm8k", "mmlu"]
    return task_name in common or any(c in task_name for c in common)

def run_lm_eval(model, model_args, tasks, batch_size=1, device=None, limit=None):
    """
    Wrapper around lm_eval.simple_evaluate
    """
    if not HAS_LM_EVAL:
        raise ImportError(
            "lm_eval is required for benchmark tasks. Install with: pip install lm-eval"
        )

    # lm_eval expects "cuda" or "cuda:0"
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    
    results = simple_evaluate(
        model=model, # "hf"
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit
    )
    return results
