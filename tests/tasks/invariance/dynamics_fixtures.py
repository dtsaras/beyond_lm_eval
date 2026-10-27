"""Tiny CPU models, texts and import paths shared by the dynamics-task invariance tests."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TESTS = Path(__file__).resolve().parents[2]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import invariance as inv  # noqa: E402,F401  (re-exported for the tests)

TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Volcanoes form where tectonic plates diverge or where one plate subducts.",
    "The orchestra tuned their instruments while the audience took their seats.",
]


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TINY_GPT2)


def tiny_gpt2():
    """hf-internal-testing/tiny-random-gpt2 in fp32, eager attention, eval mode."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(TINY_GPT2, attn_implementation="eager",
                                             torch_dtype=torch.float32)
    return m.eval()


def tiny_neox(n_layers: int = 3, seed: int = 0, parallel: bool = True):
    """Random GPT-NeoX (Pythia layout) on the tiny gpt2 vocabulary."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=1000, hidden_size=32, num_attention_heads=4,
                        intermediate_size=64, num_hidden_layers=n_layers,
                        max_position_embeddings=512, use_parallel_residual=parallel,
                        attn_implementation="eager")
    return GPTNeoXForCausalLM(cfg).eval()


def dataset(n: int = len(TEXTS)):
    return [{"text": t} for t in TEXTS[:n]]


def token_batches(tok, n: int = 3):
    return [tok(t, return_tensors="pt")["input_ids"] for t in TEXTS[:n]]
