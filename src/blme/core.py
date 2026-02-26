import json
import os
from typing import List, Union, Optional, Any
import torch

from .registry import get_task
from .utils import setup_logging
from .models.wrapper import load_model_and_tokenizer

from .tasks import geometry, dynamics, consistency, interpretability, gem, steering  # noqa: F401

def evaluate(
    model: Union[str, Any] = "hf",
    model_args: str = "",
    tasks: List[str] = None,
    limit: Optional[float] = None,
    output_dir: Optional[str] = None,
    task_configs: Optional[dict] = None,
    batch_size: Union[int, str, None] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Unified evaluation entry point.
    """
    logger = setup_logging()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    logger.info(f"Loading model: {model} with args: {model_args}")
    # TODO: Use a robust model loader that handles 'hf' string parsing like lm_eval
    # For now, we assume model_args="pretrained=path"
    hf_model, tokenizer = load_model_and_tokenizer(model_args, device=device)
    
    results = {
        "config": {
            "model": model,
            "args": model_args,
            "tasks": tasks,
            "batch_size": batch_size,
            "device": device,
        }
    }
    
    # 2. Resolve Tasks
    if tasks is None:
        tasks = []
        
    # Group tasks
    diagnostic_tasks = []
    lm_eval_tasks = []
    
    from .tasks.benchmarks import is_lm_eval_task
    
    for task_name in tasks:
        if get_task(task_name):
            diagnostic_tasks.append(task_name)
        elif is_lm_eval_task(task_name):
            lm_eval_tasks.append(task_name)
        else:
            logger.warning(f"Task '{task_name}' not found in registry or lm_eval.")

    # 3. Run Diagnostic Tasks
    for task_name in diagnostic_tasks:
        logger.info(f"Running Diagnostic Task: {task_name}")
        task_cls = get_task(task_name)
        # Get specific config for this task if available
        t_config = task_configs.get(task_name, {}) if task_configs else {}
        task = task_cls(config=t_config)
        
        # pass appropriate dataset if needed
        task_results = task.evaluate(hf_model, tokenizer, dataset=None)
        results[task_name] = task_results

    # 4. Run lm_eval Tasks
    if lm_eval_tasks:
        logger.info(f"Running lm_eval tasks: {lm_eval_tasks}")
        from .tasks.benchmarks import run_lm_eval
        lm_results = run_lm_eval(
            model=model,
            model_args=model_args,
            tasks=lm_eval_tasks,
            batch_size=batch_size,
            device=device,
            limit=limit
        )
        results["lm_eval"] = lm_results

    # 5. Save Results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
    return results
