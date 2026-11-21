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
    # noise_std defaults to a data-driven value when omitted; do not force 0.1
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
| `attn_score_dtype` | `float32` | `model` (default) or `float32`: eager attention computes q k^T in float32 ([Numerical precision](#numerical-precision)) |

## Numerical precision

**The rule.** Run a checkpoint in a 16-bit dtype only if its mean next-token NLL in that dtype is at most 0.02 nats above its float32 NLL; otherwise run float32. The checkpoint's native dtype is a hint, not a test: a float16-native checkpoint can degrade badly in bfloat16, and a bfloat16-native one can still degrade.

**The check** is opt-in and measures the rule's quantity:

```bash
blme precision-check --model-args pretrained=<id-or-path>[,device_map=auto,...] \
    [--dtype bfloat16|float16] [--threshold 0.02] [--device cuda] [--output check.json]
```

```python
from blme.precision import check
rec = check("pretrained=<id-or-path>", dtype="bfloat16", threshold=0.02)
```

Method:

- Text: WikiText-103 validation (`Salesforce/wikitext`, `wikitext-103-raw-v1`), the non-empty lines of the first 4,000 rows joined with newlines.
- Rows: the first 2,044 tokens of that text, tokenized without special tokens, as 4 rows of 511. When the tokenizer prepends a BOS (`tokenizer("a")` starts with `bos_token_id`), each row is that BOS followed by its 511 tokens; otherwise the row is the 511 tokens.
- Model: loaded once per dtype with `attn_implementation=eager`, through BLME's loader, so under the dtype contract below. `dtype` and `attn_implementation` in `--model-args` are replaced; other keys, such as `device_map` and `revision`, are kept.
- Loss: one forward per dtype; logits cast to float32; per-token cross-entropy of every next token; the mean over finite values, with the non-finite count reported beside it.

The record gives `nll_float32`, `nll_<dtype>`, `delta` (16-bit minus float32), the non-finite counts, each load's parameter dtypes and dtype contract, the checkpoint's native dtype, and `recommendation` with its `reason`. float32 is recommended when `delta` exceeds the threshold, or when the 16-bit run has more non-finite values than float32. The float32 NLL agrees across devices to about 1e-4, but 16-bit kernels round differently on CPU and GPU, so the 16-bit NLL, and with it `delta`, can differ by several thousandths of a nat. On CPU, the 16-bit result also moves with the thread count (pythia-160m-deduped: 4.2631 with 8 threads, 4.2677 with 16). Run the check on the kind of device the evaluation will use.

**The dtype contract.** When `dtype` is given explicitly, the loader checks every floating parameter and buffer after the load. A tensor honours the request when it is in the requested dtype or a wider one: transformers itself keeps some tensors in float32, such as `_keep_in_fp32_modules` and RoPE `inv_freq` buffers, and those are left as they are. The contract is broken by a tensor in a narrower dtype or in the other 16-bit format. For a 16-bit request, it is also broken when float32 parameters hold most of the parameter elements. When the contract is broken, BLME logs a warning naming the model class. It then loads the checkpoint again from a config with the dtype pinned on the config and every sub-config, and casts whatever is still narrower: parameters, and buffers transformers builds in the model's dtype. It never casts rotary buffers (`inv_freq`, `original_inv_freq`, cos/sin caches), which transformers keeps in float32 and which move predictions when rounded; a whole-model `model.to(dtype)` would cast them. It never narrows a tensor that the loader returned wider. A loader that honours the request is not touched. Quantized models are recorded but neither reloaded nor cast. With `dtype=auto` (the default), nothing is checked.

Why: on transformers 5.10, `from_pretrained(path, dtype=torch.float32)` returns bfloat16 parameters for Qwen3.5. The config loader consumes the `dtype` keyword into the top-level config. The auto class then swaps in the text sub-config, whose own `dtype` in `config.json` is bfloat16, and the model loads with `dtype="auto"`. The same path exists for emu3, llama4, mllama and qwen3_5_moe. transformers 5.2 honours the request.

**Eager attention with float32 scores (`attn_score_dtype=float32`).** Recommended for bfloat16 runs that need eager attention. transformers' eager attention computes `matmul(q, k^T) * scaling` in the model's dtype, then applies `softmax(..., dtype=float32)`. In bfloat16 the scores are rounded first. Where they reach 1e3 to 1e5 (large k_proj biases, for example), bfloat16's spacing is 16 to 1024, and the token-dependent part of the score is lost. SDPA does not round the scores this way. BLME switches a run to eager attention whenever an attention-weight or double-backward task is in it, so such runs are exposed. With `attn_score_dtype=float32`:

- Whenever eager attention runs (requested with `attn_implementation=eager`, or forced by the run), BLME uses `blme_eager_fp32_scores`. This runs the model's own `eager_attention_forward` with only the score product q k^T computed in float32. The scaling, mask and float32 softmax then work on float32 scores, the probabilities are cast back to the model's dtype and multiplied with v as transformers does, and the attention weights are returned, so `output_attentions` and double backward work. Per-model details such as softcaps are kept, because the model's own function runs.
- A float32 model gives bit-identical results to plain eager. An attention module whose eager function cannot be run this way raises an error instead of falling back silently.
- SDPA runs are unchanged.
- The option is off by default. Without it, eager attention is transformers' own, and no number changes.
- The kernel is recorded in `config.attn_implementation` and in `config.precision.attn_implementation`, and the option in `config.precision.attn_score_dtype`.
- `blme precision-check --model-args pretrained=...,attn_score_dtype=float32` measures the 16-bit run with it.

**Provenance.** Every results envelope records `config.precision`. Task results never contain these fields.

| Field | Meaning |
|---|---|
| `requested_dtype` | The `dtype` model argument (`auto` when absent) |
| `native_dtype`, `native_dtype_source` | The checkpoint's dtype, from its config (`dtype`/`torch_dtype`, or the text sub-config's). If the config has none, the dtype holding most elements in its safetensors header. |
| `loaded_param_dtypes`, `loaded_buffer_dtypes` | Floating dtypes after the load and the contract |
| `dtype_contract`, `load_violation`, `reloaded` | `honoured`, `honoured after reload`, `reloaded, then cast`, or `not checked: ...`; what the first load got wrong; whether the checkpoint was loaded again |
| `cast_applied`, `casts` | Whether BLME changed any dtype after the load, and each cast. This includes the float16 to bfloat16 parameter cast that runs need when they force eager attention. |
| `effective_param_dtypes`, `effective_dtype` | What the tasks ran with |
| `attn_score_dtype`, `attn_implementation` | The `attn_score_dtype` option (`model` or `float32`) and the attention kernel the tasks ran with |
| `warnings` | Native-dtype hints: bfloat16 on a float16-native checkpoint, or 16-bit on a float32-native one |

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
  "config": {"model_args": "pretrained=gpt2", "precision": {"requested_dtype": "auto", ...}, ...},
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
