"""get_final_norm finds the final norm of every architecture the study runs.

phi-2 (in the study) keeps it at ``model.final_layernorm``, which the lookup
missed: logit-lens decodes, the unembedding and zigzag's final-norm hook all
ran without it.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.common import get_final_norm  # noqa: E402


def _tiny(cls, cfg_cls, **kw):
    torch.manual_seed(0)
    return getattr(transformers, cls)(getattr(transformers, cfg_cls)(**kw)).eval()


@pytest.mark.parametrize("cls,cfg_cls,attr", [
    ("PhiForCausalLM", "PhiConfig", "model.final_layernorm"),
    ("Phi3ForCausalLM", "Phi3Config", "model.norm"),
    ("LlamaForCausalLM", "LlamaConfig", "model.norm"),
    ("GPT2LMHeadModel", "GPT2Config", "transformer.ln_f"),
    ("GPTNeoXForCausalLM", "GPTNeoXConfig", "gpt_neox.final_layer_norm"),
])
def test_final_norm_is_found(cls, cfg_cls, attr):
    small = dict(vocab_size=64, num_hidden_layers=1, num_attention_heads=2)
    if cfg_cls == "GPT2Config":
        small = dict(vocab_size=64, n_layer=1, n_head=2, n_embd=16)
    else:
        small.update(hidden_size=16, intermediate_size=32, pad_token_id=0)
    model = _tiny(cls, cfg_cls, **small)
    expected = model
    for part in attr.split("."):
        expected = getattr(expected, part)
    assert get_final_norm(model) is expected
