"""
Model zoo definition for the BLME benchmark study.

Each entry specifies the HuggingFace model ID, dtype, number of GPUs
needed, and any special loading flags. Models are grouped by experimental
purpose — within-family scaling, cross-family, base-vs-instruct, and
the newer "generation" axis (Llama-2/3/3.1/3.3, Qwen-2/2.5/3/3.5,
Gemma-1/2/3/4) — to test whether intrinsic-metric profiles shift
systematically across a single lab's successive releases.

GPU budget: 8x RTX 3090 (24 GB each), 192 GB total. 70B-class bf16
weights fit (≈140 GB) with ~50 GB headroom for activations when
evaluated at 128-token context; the "scale-anchor" purpose tag marks
these large models used to extend the LOFO held-out regime beyond the
31B ceiling of the original study.

All models loaded in bfloat16 (or float32 for GPT-2/Pythia which don't
support bf16 reliably — Pythia-6.9B/12B need fp32 to avoid forward-pass
NaN on prediction-entropy / sharpness / gradient-flow tasks). No
quantization — we need clean intrinsic metrics.
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

    # ── Llama 2 family (2 sizes, older generation) ───────────────────
    # Gated on HF — needs Meta's license acceptance on the HF account
    # used by the machine running the study. Kept in the zoo so that the
    # "full" roster is visible; skipped automatically when the token does
    # not have access.
    {"id": "meta-llama/Llama-2-7b-hf", "name": "llama2-7b", "family": "llama2",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "meta-llama/Llama-2-70b-hf", "name": "llama2-70b", "family": "llama2",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "scale-anchor", "gated"]},

    # ── Llama 3 / 3.1 / 3.2 / 3.3 family (expanded generation axis) ──
    {"id": "meta-llama/Llama-3.2-1B", "name": "llama3-1b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "meta-llama/Llama-3.2-1B-Instruct", "name": "llama3-1b-it", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["instruct"]},
    {"id": "meta-llama/Llama-3.2-3B", "name": "llama3-3b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "cross-family"]},
    {"id": "meta-llama/Meta-Llama-3-8B", "name": "llama3-8b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling"]},
    # Meta-Llama-3-70B (base) is gated for matthieu637 — Llama-3.1-70B
    # below is the accessible scale-anchor substitute.
    {"id": "meta-llama/Meta-Llama-3-70B", "name": "llama3-70b", "family": "llama3",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "scale-anchor", "gated"]},
    {"id": "meta-llama/Llama-3.1-8B", "name": "llama3.1-8b", "family": "llama3.1",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation"]},
    {"id": "meta-llama/Llama-3.1-70B", "name": "llama3.1-70b", "family": "llama3.1",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "scale-anchor"]},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "name": "llama3.3-70b-it", "family": "llama3.3",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": False,
     "purpose": ["generation", "instruct", "scale-anchor"]},

    # ── Qwen 2 family (3 sizes, older generation) ────────────────────
    {"id": "Qwen/Qwen2-1.5B", "name": "qwen2-1.5b", "family": "qwen2",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen2-7B", "name": "qwen2-7b", "family": "qwen2",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen2-72B", "name": "qwen2-72b", "family": "qwen2",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "scale-anchor"]},

    # ── Qwen 2.5 family (4 sizes — 1.5B, 7B, 32B, 72B) ───────────────
    {"id": "Qwen/Qwen2.5-1.5B", "name": "qwen2.5-1.5b", "family": "qwen2.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen2.5-7B", "name": "qwen2.5-7b", "family": "qwen2.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen2.5-32B", "name": "qwen2.5-32b", "family": "qwen2.5",
     "dtype": "bfloat16", "n_gpus": 4, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "scale-anchor"]},
    {"id": "Qwen/Qwen2.5-72B", "name": "qwen2.5-72b", "family": "qwen2.5",
     "dtype": "bfloat16", "n_gpus": 8, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "scale-anchor"]},

    # ── Qwen 3 family (4 sizes — 1.7B, 8B, 14B, 32B) ─────────────────
    {"id": "Qwen/Qwen3-1.7B", "name": "qwen3-1.7b", "family": "qwen3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen3-8B", "name": "qwen3-8b", "family": "qwen3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen3-14B", "name": "qwen3-14b", "family": "qwen3",
     "dtype": "bfloat16", "n_gpus": 2, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation"]},
    {"id": "Qwen/Qwen3-32B", "name": "qwen3-32b", "family": "qwen3",
     "dtype": "bfloat16", "n_gpus": 4, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "scale-anchor"]},

    # ── Qwen 3.5 family (4 base + 4 instruct, bf16) ───────────────────
    # Note: For Qwen 3.5, bare ID = instruct (-It), -Base suffix = pretrained.
    # The 27B has only the bare (instruct) variant available.
    {"id": "Qwen/Qwen3.5-0.8B-Base", "name": "qwen3.5-0.8b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-0.8B", "name": "qwen3.5-0.8b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-2B-Base", "name": "qwen3.5-2b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-2B", "name": "qwen3.5-2b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-4B-Base", "name": "qwen3.5-4b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "cross-family"]},
    {"id": "Qwen/Qwen3.5-4B", "name": "qwen3.5-4b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-9B-Base", "name": "qwen3.5-9b", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},
    {"id": "Qwen/Qwen3.5-9B", "name": "qwen3.5-9b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["instruct"]},
    {"id": "Qwen/Qwen3.5-27B", "name": "qwen3.5-27b-it", "family": "qwen3.5",
     "dtype": "bfloat16", "n_gpus": 3, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling"]},

    # ── Gemma 1 family (2 sizes, oldest generation) ──────────────────
    # Gated for matthieu637 — needs either token swap on eez130 or
    # license acceptance. Runs fail with a 403 at config.json download.
    {"id": "google/gemma-2b", "name": "gemma1-2b", "family": "gemma1",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-7b", "name": "gemma1-7b", "family": "gemma1",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "gated"]},

    # ── Gemma 2 family (3 sizes) ─────────────────────────────────────
    # Gated for matthieu637 — same situation as Gemma-1.
    {"id": "google/gemma-2-2b", "name": "gemma2-2b", "family": "gemma2",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-2-9b", "name": "gemma2-9b", "family": "gemma2",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-2-27b", "name": "gemma2-27b", "family": "gemma2",
     "dtype": "bfloat16", "n_gpus": 3, "attn": "eager", "trust_remote_code": False,
     "purpose": ["scaling", "generation", "scale-anchor", "gated"]},

    # ── Gemma 3 family (4 sizes — pt = pretrained, base variant) ─────
    # Gated for matthieu637 — same situation.
    {"id": "google/gemma-3-1b-pt", "name": "gemma3-1b", "family": "gemma3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-3-4b-pt", "name": "gemma3-4b", "family": "gemma3",
     "dtype": "bfloat16", "n_gpus": 1, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-3-12b-pt", "name": "gemma3-12b", "family": "gemma3",
     "dtype": "bfloat16", "n_gpus": 2, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "gated"]},
    {"id": "google/gemma-3-27b-pt", "name": "gemma3-27b", "family": "gemma3",
     "dtype": "bfloat16", "n_gpus": 3, "attn": "eager", "trust_remote_code": True,
     "purpose": ["scaling", "generation", "scale-anchor", "gated"]},

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
