"""Regression tests for repe_steering_effectiveness fixes (one test per fix)."""
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")

from blme.tasks.repe.steering import SteeringEffectivenessTask  # noqa: E402


def _gpt2_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")


def _tiny_llama(vocab, seed=0, hidden=32, layers=3):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval()


class _ReportedHugeParameter:
    """Counts 3e9 elements in ``model.parameters()``; float32, so nothing promotes it."""
    dtype = torch.float32
    grad = None
    requires_grad = False

    def numel(self):
        return 3_000_000_000

    def is_floating_point(self):
        return True


def _report_three_billion_parameters(model):
    cls = type(model)

    class _Huge(cls):
        def parameters(self, recurse=True):
            yield from cls.parameters(self, recurse)
            yield _ReportedHugeParameter()

    model.__class__ = _Huge
    return model


CFG = {"num_samples": 2, "n_random_draws": 4}
DIRECTIONAL = ("best_directional_effect", "mean_directional_effect",
               "directional_consistency", "steering_sign_reversal")


def test_fp32_compute_at_every_size():
    """A bf16 model that reports more than 2e9 parameters is still steered and
    scored in fp32: its output equals the fp32 copy's. The size gate scored it
    in bf16 (directional effects off by bf16 arithmetic)."""
    tok = _gpt2_tokenizer()
    bf16 = _tiny_llama(len(tok)).to(torch.bfloat16)
    ref = SteeringEffectivenessTask(dict(CFG)).evaluate(copy.deepcopy(bf16).float(), tok, None)

    big = _report_three_billion_parameters(copy.deepcopy(bf16))
    assert sum(p.numel() for p in big.parameters()) > 2_000_000_000
    out = SteeringEffectivenessTask(dict(CFG)).evaluate(big, tok, None)

    for k in DIRECTIONAL:
        assert out[k] == ref[k], (k, out[k], ref[k])
    assert out["_meta_compute_dtype"] == "float32"
    assert out["_meta_promoted_to_fp32"] is True
    # The shared model is handed back in bf16.
    assert all(p.dtype == torch.bfloat16 for p in big.parameters()
               if isinstance(p, torch.nn.Parameter))
