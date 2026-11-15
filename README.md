# BLME — Beyond LM Eval

[![Tests](https://github.com/dtsaras/beyond_lm_eval/actions/workflows/test.yml/badge.svg)](https://github.com/dtsaras/beyond_lm_eval/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Intrinsic diagnostics for language models.** BLME measures *what your model has learned* — geometry, topology, causality, interpretability — rather than just benchmark scores.

## Why BLME?

Standard evaluation harnesses (lm-eval, HELM) measure **task performance**. BLME measures **intrinsic properties**: How isotropic are your representations? Which layers are redundant? Where does causal information flow? These diagnostics help you understand *why* a model performs the way it does.

**107 diagnostic tasks** across 7 categories:

| Category | Count | Examples | What it measures |
|----------|-------|----------|-----------------|
| **Geometry** | 50 | SVD isotropy, LID, CKA/CKNNA, Schatten-p + MNN, Lipschitz, hubness, tokenizer efficiency, collapse, spectral, matrix entropy, trajectory curvature, MP bulk deviation, Vendi, magnitude, PHD, Procrustes linearity, superposition strength, effective depth | Representation space structure |
| **Interpretability** | 19 | Logit lens, attention entropy / rank, probing, sparsity, superposition, head roles, induction heads, activation sinks, WAA | Internal mechanisms |
| **Consistency** | 13 | Calibration, paraphrase / logical / contrastive invariance, ICL slope and score, format robustness, position sensitivity, contamination, knowledge capacity, bias | Output reliability |
| **Causality** | 7 | Causal tracing, ablation, attention knockout, edge attribution, circuit quality, knowledge neurons, layer skip | Information flow |
| **Dynamics** | 9 | Sharpness, gradient flow, stability, interpolation, chain-of-embedding, generation diversity, fractal likelihood, perturbation growth, entropy calibration | Temporal behavior |
| **RepE** | 4 | Task vectors, concept separability, steering effectiveness, refusal direction | Concept encoding |
| **Topology** | 5 | Persistent homology, Betti curves, persistence entropy / landscape, zigzag persistence | Manifold shape |

## Tasks by validation tier

The 107 tasks fall into three tiers: 51 + 21 + 35. How far each has been checked is in [**docs/VALIDATION.md**](docs/VALIDATION.md). It states what has been checked and how; none of it is a sign-off. Every task also carries a certification label (`parity-ready`, `formula-faithful`, `refined-adaptation` or `proxy-only`), and most carry a note on what they compute and which test pins it (`blme list-tasks --json`, or `src/blme/task_metadata.py`).

### 1. Paper-core tasks (51)

These are the tasks the accompanying study uses (`tests/fixtures/paper_core_tasks.json`). Each one declares its token policy, sources, adaptation note, headline keys and size invariances on its class, and `tests/test_task_declarations.py` checks those declarations. For each task, `docs/VALIDATION.md` lists:
- the released code it is compared with;
- the strongest parity test that runs against that code;
- how many of its confirmatory headline keys have a value test against released code or an analytic anchor.

- **Geometry (34):** categories, cka, cknna, collapse, contextualization, correlation_dimension, effective_depth, hsic, hubness, intrinsic_dim, isoscore, lid, lipschitz, magnitude, mahalanobis, matrix_entropy, mp_bulk_deviation, neural_collapse, perplexity, phd_dimension, positional_decay, prediction_alignment, procrustes_linearity, representation_sensitivity, rsa, schatten, spectral, superposition_strength, svd, tokenizer_efficiency, trajectory_curvature, unembedding, vendi_score, weight_norms
- **Dynamics (8):** coe, fractal, generation_diversity, gradient_flow, interpolation, perturbation_growth, sharpness, stability
- **Topology (5):** betti_curve, homology, persistence_entropy, persistence_landscape, zigzag_persistence
- **RepE (4):** concept_separability, refusal_direction, steering_effectiveness, task_vectors

### 2. Methods added in 0.2.0 (21)

Each follows its paper's released code. The parity test runs that code, or for the two methods without released code the paper's formula, against BLME, in most cases through `evaluate()`. Departures from the release are listed in the task's `ADAPTATION_NOTE` and its metadata note.

| task | released code | parity test |
|---|---|---|
| geometry_alpha_req | BARL-SSL/reptrix | `tests/tasks/parity/test_alpha_req_parity.py` |
| geometry_spectral_signature | Ingrid-505/Spectral_Signature | `tests/tasks/parity/test_spectral_signature_parity.py` |
| geometry_gride | sissa-data-science/DADApy | `tests/tasks/parity/test_gride_parity.py` |
| geometry_information_imbalance | sissa-data-science/DADApy | `tests/tasks/parity/test_gride_parity.py` |
| geometry_neighborhood_overlap | sissa-data-science/DADApy | `tests/tasks/parity/test_gride_parity.py` |
| geometry_farms_alpha | HUST-AI-HYZ/FARMS | `tests/tasks/parity/test_farms_parity.py` |
| geometry_ww_randomized | CalculatedContent/WeightWatcher | `tests/tasks/parity/test_ww_randomized_parity.py` |
| geometry_representation_dispersion | yanhong-lbh/rep_dispersion | `tests/tasks/parity/test_dispersion_parity.py` |
| geometry_predictive_information | chili-lab/InfoRidge | `tests/tasks/parity/test_predictive_information_parity.py` |
| geometry_rogue_dimensions | wtimkey/rogue-dimensions | `tests/tasks/parity/test_rogue_dimensions_parity.py` |
| geometry_lm_head_free_energy | josephwilsonmaths/heavytail | `tests/tasks/parity/test_lm_head_free_energy_parity.py` |
| geometry_dime | OFSkean/information_flow; uk-cliplab/representation-itl | `tests/tasks/parity/test_dime_parity.py` |
| geometry_zipfian_symmetry | cl-tohoku/zipfian-whitening | `tests/tasks/parity/test_zipfian_symmetry_parity.py` |
| geometry_delta_hyperbolicity | marlin-codes/HypLoRA | `tests/tasks/parity/test_delta_hyperbolicity_parity.py` |
| geometry_prompt_id | RitAreaSciencePark/token_geometry; DADApy | `tests/tasks/parity/test_prompt_id_parity.py` |
| geometry_ffn_spectral_utilization | none (formula and analytic anchors) | `tests/tasks/test_ffn_spectral_utilization.py` |
| interpretability_logit_lens_kl | jennhu/model-human-processing | `tests/tasks/parity/test_logit_lens_kl_parity.py` |
| interpretability_inactive_heads | psandovalsegura/inactive-heads | `tests/tasks/parity/test_inactive_heads_parity.py` |
| causality_layer_skip | robertcsordas/llm_effective_depth | `tests/tasks/parity/test_layer_skip_parity.py` |
| consistency_icl_score | none (formula: brute-force cross-entropy) | `tests/tasks/test_icl_score.py` |
| dynamics_entropy_calibration | stevenxcao/entropy-calibration | `tests/tasks/parity/test_entropy_calibration_parity.py` |

Parity tests load the released code from a local clone (`~/.cache/blme-refrepos`, or `$BLME_REFREPOS`) or a side install and skip without it; `BLME_REQUIRE_REFS=1` turns those skips into failures.

### 3. Experimental: interpretability, causality and consistency (35)

These tasks run, and they are tested for crashes and for leaks (forward hooks left on the model, global RNG state). They have **not** been validated against their papers. Several have a kernel test against a reference implementation, which their certification label reflects. A kernel test does not validate the task end to end, so treat their outputs as exploratory.

- **Interpretability (17):** activation_kurtosis, activation_sinks, attention_effective_rank, attention_entropy, attention_graph, attention_rank, attention_rollout, attribution, head_roles, induction_heads, logit_lens, prediction_entropy, probing, sae_features, sparsity, superposition, waa
- **Causality (6):** ablation, attention_knockout, circuit_quality, edge_attribution, knowledge_neurons, tracing
- **Consistency (12):** bias_weat, calibration, contamination, contrastive, format_robustness, icl_slope, knowledge_capacity, logical, membership_inference, paraphrase, position_sensitivity, self_consistency

## Installation

Requires Python ≥ 3.10 and `transformers>=5.2.0,<5.11` (5.2.0 and 5.10.2 are tested).

```bash
# Core
pip install -e .

# With all optional dependencies
pip install -e ".[all]"

# Specific extras
pip install -e ".[topology]"   # ripser for persistent homology
pip install -e ".[benchmarks]" # lm-eval integration
pip install -e ".[probing]"    # scikit-learn for probing tasks
pip install -e ".[data]"       # datasets for HF-backed task data
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

# Machine-readable task list with certification status
blme list-tasks --json

# Validate a run without loading a model
blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd --dry-run --strict
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

- **Shared forward-pass cache** — shares hidden states/logits for cache-aware tasks to avoid redundant forwards
- **Structured JSON output** — results envelope with metadata, git hash, dependency versions, resolved task configs, cache settings, and timestamps for reproducibility
- **Error isolation** — one failing task doesn't crash the entire run
- **Strict validation** — use `--strict` / `--dry-run` to catch task-name and recipe drift before model loading
- **Task certification metadata** — every task is labeled as `parity-ready`, `formula-faithful`, `refined-adaptation`, or `proxy-only`
- **Default configs** — every task has sensible defaults in `defaults.yaml`, override per-task in your recipe
- **HuggingFace-native** — supports `dtype`, `device_map`, quantization, `trust_remote_code`, `attn_implementation`

## Interpreting Results

See [**docs/interpreting_results.md**](docs/interpreting_results.md) for a comprehensive guide on what each metric means, expected ranges, and diagnostic patterns.

## Documentation

- [Getting Started](docs/getting_started.md)
- [Configuration](docs/configuration.md)
- [Interpreting Results](docs/interpreting_results.md)
- [Validation Status](docs/VALIDATION.md)
- Task category docs: [Geometry](docs/tasks_geometry.md) · [Interpretability](docs/tasks_interpretability.md) · [Topology](docs/tasks_topology.md) · [Causality](docs/tasks_causality.md) · [Consistency](docs/tasks_consistency.md) · [Dynamics](docs/tasks_dynamics.md) · [RepE](docs/tasks_representation_engineering.md)

## License

MIT — see [LICENSE](LICENSE).
