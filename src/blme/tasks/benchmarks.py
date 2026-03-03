try:
    from lm_eval import simple_evaluate
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False

from typing import List
import torch

def is_lm_eval_task(task_name: str) -> bool:
    # This is a heuristic. Ideally we query lm_eval.tasks.TaskManager
    # For now, if it's not a diagnostic task, we assume it is lm_eval
    # But strictly, we should check.
    # Allow-list or query lm_eval
    # Simple check: does it look like a benchmark?
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
