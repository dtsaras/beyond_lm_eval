"""Tiny CPU models and texts shared by the topology / depth-profile tests (family D)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_TESTS = Path(__file__).resolve().parents[2]
_SRC = _TESTS.parent / "src"
for _p in (str(_SRC), str(_TESTS), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import invariance as inv  # noqa: E402,F401  (re-exported for the tests)
from dynamics_fixtures import TEXTS as _DYN_TEXTS  # noqa: E402
from dynamics_fixtures import tiny_gpt2, tiny_neox, tokenizer  # noqa: E402,F401

TEXTS = list(_DYN_TEXTS) + [
    "A prime number has exactly two distinct positive divisors.",
    "Rivers carry sediment downstream and deposit it in deltas near the sea.",
    "The committee will vote on the proposal at its meeting next Thursday.",
    "Bees communicate the location of flowers through a waggle dance.",
    "The library closed early because of the storm warning issued that morning.",
    "Glaciers carve deep valleys as they advance and retreat over millennia.",
    "He tuned the old radio until a faint voice emerged from the static.",
    "The bridge was repainted after inspectors found rust on its cables.",
]


def tiny_llama(n_layers: int = 3, seed: int = 0, vocab: int = 1000):
    """Random Llama (post_attention_layernorm + model.norm) on the tiny gpt2 vocabulary."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval()


def dataset(n: int = len(TEXTS)):
    return [{"text": t} for t in TEXTS[:n]]
