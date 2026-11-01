"""MoE expert weights that the weight-spectrum tasks cannot measure are counted.

transformers 5 packs a block's experts into 3-D parameters of a plain module,
which neither WeightWatcher's layer types nor FARMS's ``isinstance(m,
nn.Linear)`` covers; on an MoE model the headlines then cover the attention
weights only, so the skip must show in ``_meta_``.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.tasks.geometry import farms as FA  # noqa: E402
from blme.tasks.geometry import ww_randomized as WW  # noqa: E402


def _tiny_mixtral():
    cfg = transformers.MixtralConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, num_local_experts=4,
                                     num_experts_per_tok=2)
    torch.manual_seed(0)
    return transformers.MixtralForCausalLM(cfg).eval()


def _packed(model):
    return sum(1 for n, p in model.named_parameters() if "expert" in n and p.dim() >= 3)


def test_ww_counts_packed_experts():
    model = _tiny_mixtral()
    if _packed(model) == 0:
        pytest.skip("this transformers version stores experts as nn.Linear")
    _, _, counts = WW._ww_matrices(model)
    assert counts["packed_expert_tensors"] == _packed(model)


def test_farms_counts_packed_experts():
    model = _tiny_mixtral()
    if _packed(model) == 0:
        pytest.skip("this transformers version stores experts as nn.Linear")
    _, _, counts = FA._farms_matrices(model)
    assert counts["packed_expert_tensors"] == _packed(model)
