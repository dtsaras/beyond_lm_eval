# Configuration System

BLME supports running complex evaluations using YAML recipes.

## Recipe Format

```yaml
experiment_name: "my_experiment"

global:
  device: "cuda"
  output_dir: "results/"
  batch_size: 1

model:
  name: "hf"
  path: "gpt2"
  args: "pretrained=gpt2" # Optional, override args

tasks:
  # Task Name: Task Config
  geometry_categories:
    categories_path: "assets/categories.json"
    projection_method: "umap"
    
  steering_concept:
    method: "hidden_normalized"
    scales: [0.5, 1.0]
```

## Running a Recipe

From Python:

```python
from blme import run_from_yaml
run_from_yaml("my_recipe.yaml")
```
