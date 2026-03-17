# Configuration System

BLME supports two ways to configure evaluations: CLI arguments and YAML recipes.

## CLI Arguments

```bash
blme evaluate \
    --model-args pretrained=gpt2,dtype=bfloat16,device_map=auto \
    --tasks geometry_svd geometry_lid topology_homology \
    --output-dir results/ \
    --output-format json \
    --device cuda \
    --cache-samples 200 \
    --verbosity INFO
```

## YAML Recipes

For complex evaluations, use a YAML recipe:

```yaml
experiment_name: "llama2-full-diagnostics"

global:
  device: "cuda"
  output_dir: "results/"
  batch_size: 1
  cache_num_samples: 200

model:
  path: "meta-llama/Llama-2-7b-hf"
  args: "pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16,device_map=auto"

tasks:
  # Geometry
  geometry_svd:
    num_samples: 200
    use_cache: true
  geometry_lid:
    k: 30
    num_samples: 100
  geometry_cka:
    num_samples: 100

  # Topology (requires: pip install -e ".[topology]")
  topology_homology:
    num_samples: 50

  # Interpretability
  interpretability_attention_entropy:
    num_samples: 100

  # Causality
  causality_tracing:
    num_samples: 10
    noise_std: 0.1
  consistency_calibration:
    num_samples: 200
    use_cache: false
```

Run with:

```bash
blme evaluate --recipe my_recipe.yaml
```

## Default Task Configs

Every task has sensible defaults defined in [`src/blme/tasks/configs/defaults.yaml`](../src/blme/tasks/configs/defaults.yaml). When you run a task without specifying parameters, these defaults are used automatically.

## Cache Settings

BLME can precompute a shared cache of hidden states:

- `global.cache_num_samples`: global sample count for the shared cache (overrides per-task `num_samples` for cached tasks).
- If omitted, BLME uses the maximum `num_samples` among cacheable tasks.
- `use_cache`: per-task flag to opt out when a task needs its own dataset or sampling.

### Config Resolution Priority

```
User recipe overrides  →  defaults.yaml
```

For example, if `defaults.yaml` says `geometry_svd.num_samples: 100` and your recipe says `geometry_svd.num_samples: 500`, the value `500` is used.

### Viewing Defaults

You can inspect the full default configs:

```bash
cat src/blme/tasks/configs/defaults.yaml
```

## Model Arguments Format

The `--model-args` flag (or `model.args` in YAML) uses comma-separated `key=value` pairs:

| Key | Example | Description |
|-----|---------|-------------|
| `pretrained` | `gpt2` | HuggingFace model ID or local path |
| `dtype` | `bfloat16` | Model dtype (`float16`, `bfloat16`, `float32`, `auto`) |
| `device_map` | `auto` | HuggingFace device map strategy |
| `trust_remote_code` | `true` | Allow custom code in model repos |
| `attn_implementation` | `flash_attention_2` | Attention backend |
| `revision` | `main` | Model revision/branch |
| `load_in_8bit` | `true` | 8-bit quantization (requires bitsandbytes) |
| `load_in_4bit` | `true` | 4-bit quantization (requires bitsandbytes) |
| `max_memory` | `0:20GiB,cpu:40GiB` | Per-device memory limits |

## Output Formats

### JSON (default)

```bash
blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd --output-dir results/
```

Produces `results/results.json`:

```json
{
  "blme_version": "0.1.0",
  "timestamp": "2026-02-28T14:30:00",
  "config": {"model_args": "pretrained=gpt2", ...},
  "results": {
    "geometry_svd": {
      "effective_rank": 127.4,
      "avg_cosine_similarity": 0.23,
      ...
    }
  },
  "errors": {},
  "summary": {"total_tasks": 1, "completed_tasks": 1, "failed_tasks": 0}
}
```

### CSV

```bash
blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd --output-format csv --output-dir results/
```
