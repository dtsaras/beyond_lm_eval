# Getting Started with BLME

**Beyond LM Eval (BLME)** is a comprehensive library for analyzing Large Language Model geometry, interpretability, and dynamics.

## Installation

```bash
pip install -e .
```

## Comparisons Usage

Run the main evaluation script:

```bash
python -m blme.core \
    --model_path /path/to/checkpoint \
    --tasks geometry_isotropy,interpretability_logit_lens \
    --output_dir results/
```

## Module Overview

- **[Geometry](tasks_geometry.md)**: Manifold analysis, hubness, alignment residuals.
- **[Interpretability](tasks_interpretability.md)**: Logit lens, component attribution.
- **[Dynamics](tasks_dynamics.md)**: Trajectories, stability, multimodal tests.
- **[Steering](tasks_steering.md)**: Concept vectors, mixture editing.

## Adding New Tasks

1. Create a new task class inheriting from `blme.tasks.base.DiagnosticTask`.
2. Implement `evaluate(self, model, tokenizer, dataset)`.
3. Decorate with `@register_task("task_name")`.
