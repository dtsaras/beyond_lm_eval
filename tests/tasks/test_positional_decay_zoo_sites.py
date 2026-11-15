"""geometry_positional_decay: the write-site lookup covers every model type of
the study zoo that transformers can build here, Qwen3.5's linear-attention
blocks included, and an unknown attention attribute gives a NaN headline with
a reason naming it."""
import math
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask  # noqa: E402

# model_type of every study model (configs/models.yaml, cached configs), except
# gemma4_text / gemma4_unified_text, which transformers 5.2 cannot build.
ZOO_TYPES = ["gpt2", "gpt_neox", "llama", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text",
             "phi", "phi3", "olmo", "olmo2", "olmo3", "mistral", "smollm3", "qwen3_5_text"]


def _tiny(model_type, n_layers=4):
    """A random-init 4-block model of ``model_type`` (eager attention)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    if model_type == "gpt2":
        kw = dict(vocab_size=1000, n_embd=64, n_layer=n_layers, n_head=4, n_positions=512)
    else:
        kw = dict(vocab_size=1000, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=512, pad_token_id=0)
    if model_type in ("gpt_neox", "phi"):
        kw.pop("num_key_value_heads")
    if model_type in ("gemma", "gemma2", "gemma3_text", "qwen3_5_text"):
        kw["head_dim"] = 16
    if model_type == "qwen3_5_text":
        kw["layer_types"] = ["linear_attention"] * (n_layers - 1) + ["full_attention"]
    cfg = AutoConfig.for_model(model_type, **kw)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()


def _run(model):
    return PositionalAttentionDecayTask(config={"num_samples": 3}).evaluate(
        model, fd.tokenizer(), fd.dataset(3))


@pytest.mark.parametrize("model_type", ZOO_TYPES)
def test_every_zoo_model_type_has_a_finite_headline(model_type):
    try:
        model = _tiny(model_type)
    except (ValueError, KeyError) as e:  # pragma: no cover - transformers version
        pytest.skip(f"{model_type} cannot be built: {e}")
    r = _run(model)
    assert "error" not in r, r
    assert math.isfinite(r["positional_decay_nosink_write_weighted"])
    assert r["_meta_positional_decay_nan_reason"] is None, r["_meta_positional_decay_nan_reason"]
    assert r["_meta_write_weights_complete"] is True
    assert r["_meta_n_blocks_without_softmax_attention"] == (3 if model_type == "qwen3_5_text" else 0)


def test_unknown_attention_attribute_gives_nan_with_reason():
    """A block whose attention hangs off an attribute the lookup does not know
    used to give a silent NaN."""
    model = fd.tiny_llama()
    for block in model.model.layers:
        block.mixer = block.self_attn
        del block.self_attn
        block.forward = (lambda b: (lambda *a, **k: type(b).forward(_Alias(b), *a, **k)))(block)
    r = _run(model)
    assert math.isnan(r["positional_decay_nosink_write_weighted"])
    reason = r["_meta_positional_decay_nan_reason"]
    assert "block 0 (LlamaDecoderLayer) has none of the attention attributes" in reason, reason
    assert "self_attn" in reason


class _Alias:
    """The block seen by its own forward under its old attribute name."""

    def __init__(self, block):
        object.__setattr__(self, "_b", block)

    def __getattr__(self, name):
        return getattr(self._b, "mixer" if name == "self_attn" else name)
