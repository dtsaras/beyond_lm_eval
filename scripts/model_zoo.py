"""
Model zoo definition for the BLME benchmark study.

Each entry specifies the HuggingFace model ID, dtype, number of GPUs
needed, and any special loading flags. Models are grouped by experimental
purpose (within-family scaling, cross-family, base-vs-instruct).

GPU budget: 8x RTX 3090 (24 GB each), 192 GB total.
All models loaded in bfloat16 (or float32 for GPT-2/Pythia which don't
support bf16). No quantization — we need clean intrinsic metrics.
"""

# Each entry: {
#   "id": HuggingFace model ID,
#   "name": short name for results,
#   "family": model family for grouping,
#   "dtype": "bfloat16" or "float32",
#   "n_gpus": number of GPUs needed (1 = single GPU, >1 = device_map=auto),
#   "attn": attention implementation ("eager" for autograd tasks, else default),
#   "trust_remote_code": bool,
#   "purpose": list of experimental purposes,
# }

MODELS = [
    # ── GPT-2 family (4 models, float32, 1 GPU each) ──────────────────
    {"id": "openai-community/gpt2", "name": "gpt2-small", "family": "gpt2",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "openai-community/gpt2-medium", "name": "gpt2-medium", "family": "gpt2",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "openai-community/gpt2-large", "name": "gpt2-large", "family": "gpt2",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "openai-community/gpt2-xl", "name": "gpt2-xl", "family": "gpt2",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},

    # ── Pythia family (8 models, float32, 1-2 GPUs) ───────────────────
    {"id": "EleutherAI/pythia-70m-deduped", "name": "pythia-70m", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-160m-deduped", "name": "pythia-160m", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-410m-deduped", "name": "pythia-410m", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-1b-deduped", "name": "pythia-1b", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-1.4b-deduped", "name": "pythia-1.4b", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-2.8b-deduped", "name": "pythia-2.8b", "family": "pythia",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "EleutherAI/pythia-6.9b-deduped", "name": "pythia-6.9b", "family": "pythia",
     "dtype": "float16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    {"id": "EleutherAI/pythia-12b-deduped", "name": "pythia-12b", "family": "pythia",
     "dtype": "float16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},

    # ── Llama 3 family (3 base + 1 instruct, bf16, 1 GPU each) ────────
    {"id": "meta-llama/Llama-3.2-1B", "name": "llama3-1b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "name": "llama3-1b-it", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["instruct"]},
    {"id": "meta-llama/Llama-3.2-3B", "name": "llama3-3b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "meta-llama/Llama-3.1-8B", "name": "llama3-8b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},

    # ── Qwen 3.5 family (5 base + 2 instruct, bf16) ───────────────────
    {"id": "Qwen/Qwen3.5-0.8B", "name": "qwen3.5-0.8b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-2B", "name": "qwen3.5-2b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-4B", "name": "qwen3.5-4b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "cross-family"]},
    {"id": "Qwen/Qwen3.5-4B-Instruct", "name": "qwen3.5-4b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-9B", "name": "qwen3.5-9b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-9B-Instruct", "name": "qwen3.5-9b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-27B", "name": "qwen3.5-27b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 3, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},

    # ── Gemma 4 family (3 base + 1 IT, bf16) ──────────────────────────
    {"id": "google/gemma-4-E2B", "name": "gemma4-e2b", "family": "gemma4",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "google/gemma-4-E4B", "name": "gemma4-e4b", "family": "gemma4",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "cross-family"]},
    {"id": "google/gemma-4-E4B-it", "name": "gemma4-e4b-it", "family": "gemma4",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "google/gemma-4-31B", "name": "gemma4-31b", "family": "gemma4",
     "dtype": "bfloat16", "n_gpus": 3, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},

    # ── Cross-family extras (1 GPU each) ──────────────────────────────
    {"id": "allenai/OLMo-1B-hf", "name": "olmo-1b", "family": "olmo",
     "dtype": "float32", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["cross-family"]},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "tinyllama-1.1b", "family": "tinyllama",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["cross-family"]},
    {"id": "microsoft/phi-2", "name": "phi-2", "family": "phi",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["cross-family"]},
]


def get_small_models():
    """Models that fit on 1 GPU (≤12B params)."""
    return [m for m in MODELS if m["n_gpus"] == 1]


def get_large_models():
    """Models that need multi-GPU (>12B params)."""
    return [m for m in MODELS if m["n_gpus"] > 1]


def get_all_model_names():
    """All unique model short names."""
    return [m["name"] for m in MODELS]


def build_model_args(model_entry):
    """Build a BLME model_args string from a model entry."""
    parts = [f"pretrained={model_entry['id']}"]
    parts.append(f"dtype={model_entry['dtype']}")
    if model_entry["n_gpus"] > 1:
        parts.append("device_map=auto")
    if model_entry.get("attn"):
        parts.append(f"attn_implementation={model_entry['attn']}")
    if model_entry.get("trust_remote_code"):
        parts.append("trust_remote_code=true")
    return ",".join(parts)


if __name__ == "__main__":
    print(f"Total models: {len(MODELS)}")
    print(f"  Small (1 GPU): {len(get_small_models())}")
    print(f"  Large (multi-GPU): {len(get_large_models())}")
    for m in MODELS:
        print(f"  {m['name']:<25s} {m['family']:<10s} {m['n_gpus']}GPU  {','.join(m['purpose'])}")
