# BLME — Beyond LM Eval

[![Tests](https://github.com/dtsaras/beyond_lm_eval/actions/workflows/test.yml/badge.svg)](https://github.com/dtsaras/beyond_lm_eval/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Intrinsic diagnostics for language models.** BLME measures *what your model has learned* — geometry, topology, causality, interpretability — rather than just benchmark scores.

## Why BLME?

Standard evaluation harnesses (lm-eval, HELM) measure **task performance**. BLME measures **intrinsic properties**: How isotropic are your representations? Which layers are redundant? Where does causal information flow? These diagnostics help you understand *why* a model performs the way it does.

**45 diagnostic tasks** across 7 categories:

| Category | Examples | What it measures |
|----------|----------|-----------------|
| **Geometry** (19) | SVD isotropy, LID, CKA, Lipschitz, collapse | Representation space structure |
| **Interpretability** (11) | Logit lens, attention entropy, probing, sparsity | Internal mechanisms |
| **Topology** (3) | Persistent homology, Betti curves, persistence entropy | Manifold shape |
| **Causality** (3) | Causal tracing, ablation robustness, attention knockout | Information flow |
| **Consistency** (4) | Calibration, paraphrase invariance, logical consistency | Output reliability |
| **Dynamics** (3) | Stability, interpolation, chain-of-embedding drift | Temporal behavior |
| **RepE** (2) | Task vectors, concept separability | Concept encoding |

## Installation

```bash
# Core
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Specific extras
pip install -e ".[topology]"   # ripser for persistent homology
pip install -e ".[benchmarks]" # lm-eval integration
pip install -e ".[probing]"    # scikit-learn for probing tasks
```

## Quick Start

### CLI

```bash
# Evaluate a model on specific tasks
blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd geometry_lid

# Run all tasks in a category
blme evaluate --model-args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16,device_map=auto \
              --task-group geometry

# From a YAML recipe
blme evaluate --recipe examples/recipes/default_all.yaml

# List available tasks
blme list-tasks
blme list-tasks --group topology
```

### Python API

```python
from blme import evaluate

results = evaluate(
    model_args="pretrained=gpt2",
    tasks=["geometry_svd", "interpretability_attention_entropy"],
    output_dir="results/",
)

# results["results"]["geometry_svd"]["effective_rank"] → e.g. 127.4
```

### YAML Recipe

```yaml
experiment_name: "llama2-diagnostics"

global:
  device: "cuda"
  output_dir: "results/"

model:
  path: "meta-llama/Llama-2-7b-hf"

tasks:
  geometry_svd:
    num_samples: 200
  geometry_lid:
    k: 30
  topology_homology:
    num_samples: 50
```

## Key Features

- **Shared forward-pass cache** — runs one forward pass, shares hidden states across all tasks (~10x faster for multi-task runs)
- **Structured JSON output** — results envelope with metadata, git hash, timestamps for reproducibility
- **Error isolation** — one failing task doesn't crash the entire run
- **Default configs** — every task has sensible defaults in `defaults.yaml`, override per-task in your recipe
- **HuggingFace-native** — supports `dtype`, `device_map`, quantization, `trust_remote_code`, `attn_implementation`

## Interpreting Results

See [**docs/interpreting_results.md**](docs/interpreting_results.md) for a comprehensive guide on what each metric means, expected ranges, and diagnostic patterns.

## Documentation

- [Getting Started](docs/getting_started.md)
- [Configuration](docs/configuration.md)
- [Interpreting Results](docs/interpreting_results.md)
- Task category docs: [Geometry](docs/tasks_geometry.md) · [Interpretability](docs/tasks_interpretability.md) · [Topology](docs/tasks_topology.md) · [Causality](docs/tasks_causality.md) · [Consistency](docs/tasks_consistency.md) · [Dynamics](docs/tasks_dynamics.md) · [RepE](docs/tasks_representation_engineering.md)

## License

MIT — see [LICENSE](LICENSE).
