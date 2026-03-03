"""
Shared fixtures for BLME tests.

Provides tiny real HuggingFace models (2 layers, dim=32) across 10 architectures
so that every task is tested against each structural variant:

- GPT2:       transformer.h         (classic GPT)
- Llama:      model.layers           (modern decoder, GQA)
- BERT:       bert.encoder.layer     (encoder, non-causal)
- Mistral:    model.layers           (sliding-window attention, GQA)
- Phi3:       model.layers           (different FFN)
- Gemma:      model.layers           (RMSNorm variant)
- Qwen2:      model.layers           (rotary variant)
- GPTNeoX:    gpt_neox.layers        (Pythia-style)
- Qwen3:      model.layers           (thinking tokens)
- Qwen3Next:  model.layers           (Qwen 3.5 architecture)
"""

import pytest
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    LlamaConfig, LlamaForCausalLM,
    BertConfig, BertLMHeadModel,
)

# ── Conditional imports for newer architectures ─────────────────────────
# These are guarded so tests skip gracefully on older transformers versions.

_OPTIONAL_ARCHS = {}

try:
    from transformers import MistralConfig, MistralForCausalLM
    _OPTIONAL_ARCHS["mistral"] = (MistralConfig, MistralForCausalLM)
except ImportError:
    pass

try:
    from transformers import Phi3Config, Phi3ForCausalLM
    _OPTIONAL_ARCHS["phi3"] = (Phi3Config, Phi3ForCausalLM)
except ImportError:
    pass

try:
    from transformers import GemmaConfig, GemmaForCausalLM
    _OPTIONAL_ARCHS["gemma"] = (GemmaConfig, GemmaForCausalLM)
except ImportError:
    pass

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    _OPTIONAL_ARCHS["qwen2"] = (Qwen2Config, Qwen2ForCausalLM)
except ImportError:
    pass

try:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    _OPTIONAL_ARCHS["gptneox"] = (GPTNeoXConfig, GPTNeoXForCausalLM)
except ImportError:
    pass

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    _OPTIONAL_ARCHS["qwen3"] = (Qwen3Config, Qwen3ForCausalLM)
except ImportError:
    pass

try:
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    _OPTIONAL_ARCHS["qwen3next"] = (Qwen3NextConfig, Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    _OPTIONAL_ARCHS["qwen3_5"] = (Qwen3_5TextConfig, Qwen3_5ForCausalLM)
except ImportError:
    pass


# ── Tiny config builders ────────────────────────────────────────────────

def _build_gpt2():
    config = GPT2Config(
        vocab_size=1000, n_positions=128, n_embd=32, n_layer=2, n_head=2,
    )
    return GPT2LMHeadModel(config), config.vocab_size

def _build_llama():
    config = LlamaConfig(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config), config.vocab_size

def _build_bert():
    config = BertConfig(
        vocab_size=1000, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=128, is_decoder=True,
    )
    return BertLMHeadModel(config), config.vocab_size

def _build_mistral():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["mistral"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_phi3():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["phi3"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=0, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_gemma():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["gemma"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, head_dim=16, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_qwen2():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["qwen2"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_gptneox():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["gptneox"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=128, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_qwen3():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["qwen3"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_qwen3next():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["qwen3next"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, head_dim=16, use_cache=False, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size

def _build_qwen3_5():
    cfg_cls, model_cls = _OPTIONAL_ARCHS["qwen3_5"]
    config = cfg_cls(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, use_cache=False, attn_implementation="eager",
    )
    return model_cls(config), config.vocab_size


_BUILDERS = {
    "gpt2": _build_gpt2,
    "llama": _build_llama,
    "bert": _build_bert,
    "mistral": _build_mistral,
    "phi3": _build_phi3,
    "gemma": _build_gemma,
    "qwen2": _build_qwen2,
    "gptneox": _build_gptneox,
    "qwen3": _build_qwen3,
    "qwen3next": _build_qwen3next,
    "qwen3_5": _build_qwen3_5,
}


# ── Determine which architectures to test ───────────────────────────────

def _available_archs():
    """Return list of architecture names available for testing."""
    always = ["gpt2", "llama", "bert"]
    optional = [k for k in _OPTIONAL_ARCHS if k in _BUILDERS]
    return always + sorted(optional)


# ── Dummy tokenizer ─────────────────────────────────────────────────────

class DummyTokenizer:
    """Minimal tokenizer that produces valid tensor shapes for any model."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        is_list = isinstance(text, (list, tuple))
        batch_size = len(text) if is_list else 1
        length = 8

        ids = torch.randint(0, self.vocab_size, (batch_size, length))
        mask = torch.ones_like(ids)

        if return_tensors == "pt":
            class BatchDict(dict):
                def to(self, dev): return self
                def __getattr__(self, name):
                    try:
                        return self[name]
                    except KeyError:
                        raise AttributeError(name)
            return BatchDict({"input_ids": ids, "attention_mask": mask})
        return {"input_ids": ids.tolist(), "attention_mask": mask.tolist()}

    def encode(self, text, return_tensors=None, **kwargs):
        ids = torch.randint(0, self.vocab_size, (1, 8))
        if return_tensors == "pt":
            return ids
        return ids[0].tolist()

    def decode(self, *args, **kwargs):
        return "dummy text"


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(params=_available_archs())
def mock_model_tokenizer(request):
    """
    Returns (model, tokenizer) parametrized over all available architectures.

    Models are tiny (2 layers, dim=32) and require no downloads.
    """
    arch = request.param
    builder = _BUILDERS[arch]

    model, vocab_size = builder()
    model.eval()

    return model, DummyTokenizer(vocab_size)


@pytest.fixture
def mock_model(mock_model_tokenizer):
    return mock_model_tokenizer[0]


@pytest.fixture
def mock_tokenizer(mock_model_tokenizer):
    return mock_model_tokenizer[1]
