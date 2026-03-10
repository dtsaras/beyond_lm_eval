# Getting Started with BLME

**Beyond LM Eval (BLME)** is a diagnostic library for analyzing intrinsic properties of language models — geometry, topology, causality, interpretability, and more.

## Installation

```bash
# Core install
pip install -e .

# With all optional dependencies (topology, probing, SAE, etc.)
pip install -e ".[all]"
```

## CLI Quickstart

```bash
# Evaluate a model on specific tasks
blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd geometry_lid

# Run all tasks in a group
blme evaluate --model-args pretrained=gpt2 --task-group geometry

# Use a YAML recipe
blme evaluate --recipe examples/recipes/default_all.yaml

# List all available tasks
blme list-tasks

# Filter by group
blme list-tasks --group topology
```

### Model Arguments

The `--model-args` flag uses a comma-separated `key=value` format:

```bash
# Basic
--model-args pretrained=gpt2

# With options
--model-args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16,device_map=auto

# All supported keys
--model-args pretrained=...,dtype=...,device_map=...,trust_remote_code=true,attn_implementation=flash_attention_2,revision=main,load_in_8bit=true
```

### Output

Results are saved as structured JSON with metadata:

```bash
blme evaluate --model-args pretrained=gpt2 \
              --tasks geometry_svd \
              --output-dir results/ \
              --output-format json
```

## Python API

```python
from blme import evaluate

results = evaluate(
    model_args="pretrained=gpt2",
    tasks=["geometry_svd", "interpretability_attention_entropy"],
    output_dir="results/",
)

print(results["results"]["geometry_svd"]["effective_rank"])
```

## Module Overview

BLME organizes 51 diagnostic tasks across 7 categories:

- **[Geometry](tasks_geometry.md)** (20 tasks): Manifold structure — isotropy, intrinsic dimension, CKA, collapse, Lipschitz constants, layer change ratio
- **[Interpretability](tasks_interpretability.md)** (12 tasks): Internal mechanisms — logit lens, attention entropy, probing, sparsity, superposition index
- **[Topology](tasks_topology.md)** (3 tasks): Manifold shape — persistent homology, Betti curves, persistence entropy
- **[Causality](tasks_causality.md)** (4 tasks): Information flow — causal tracing, ablation robustness, attention knockout, circuit quality
- **[Consistency](tasks_consistency.md)** (6 tasks): Output reliability — calibration, paraphrase invariance, logical consistency, contamination detection, knowledge capacity
- **[Dynamics](tasks_dynamics.md)** (3 tasks): Temporal behavior — stability, interpolation, chain-of-embedding drift
- **[Representation Engineering](tasks_representation_engineering.md)** (3 tasks): Concept encoding — task vectors, concept separability, steering effectiveness

## Shared Cache

When running multiple tasks, BLME automatically creates a shared cache that runs **one forward pass** and serves hidden states to all tasks. This eliminates redundant computation:

```bash
# Without cache: 14 separate forward passes
# With cache (automatic): 1 forward pass → shared across all 14 tasks
blme evaluate --model-args pretrained=gpt2 --task-group geometry
```

## Adding New Tasks

```python
from blme.tasks.base import DiagnosticTask
from blme.registry import register_task

@register_task("my_custom_metric")
class MyMetric(DiagnosticTask):
    def evaluate(self, model, tokenizer, dataset, cache=None):
        # Use cache if available
        if cache is not None and cache.is_populated:
            X = cache.get_hidden_states(layer_idx=-1)
        else:
            # Fall back to manual collection
            ...

        return {"my_value": 42.0}
```

## Next Steps

- [Configuration](configuration.md) — YAML recipes and default configs
- [Interpreting Results](interpreting_results.md) — what each metric means
