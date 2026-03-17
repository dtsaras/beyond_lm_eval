import yaml
import argparse
import sys
from typing import Optional
from .core import evaluate

def run_from_yaml(config_path: str):
    """
    Run evaluation based on a YAML recipe.
    """
    with open(config_path, 'r') as f:
        recipe = yaml.safe_load(f)
        
    print(f"Loaded recipe: {recipe.get('experiment_name', 'Unnamed')}")
    
    # Parse Model
    model_config = recipe.get("model", {})
    model_name = model_config.get("name", "hf")
    model_path = model_config.get("path")
    model_args = model_config.get("args", "")
    
    # If path is provided but args not, construct args for hf
    if model_path and not model_args:
        model_args = f"pretrained={model_path}"
        
    # Parse Tasks
    tasks_dict = recipe.get("tasks", {})
    task_list = list(tasks_dict.keys())
    
    # Global settings
    global_config = recipe.get("global", {})
    output_dir = global_config.get("output_dir", None)
    device = global_config.get("device", None)
    batch_size = global_config.get("batch_size", None)
    limit = global_config.get("limit", None)
    cache_num_samples = global_config.get("cache_num_samples", None)
    seed = global_config.get("seed", 42)
    task_timeout = global_config.get("task_timeout", 600)

    # Run
    return evaluate(
        model_args=model_args,
        tasks=task_list,
        task_configs=tasks_dict,
        batch_size=batch_size,
        device=device,
        limit=limit,
        output_dir=output_dir,
        cache_num_samples=cache_num_samples,
        seed=seed,
        task_timeout=task_timeout,
    )

def main():
    parser = argparse.ArgumentParser(description="Run BLME evaluation from YAML recipe")
    parser.add_argument("recipe", type=str, help="Path to YAML recipe file")
    args = parser.parse_args()
    
    run_from_yaml(args.recipe)
    
if __name__ == "__main__":
    main()
