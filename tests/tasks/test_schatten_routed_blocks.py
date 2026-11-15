"""geometry_schatten: a block with routed experts gets a NaN MLP value with a
reason; the down-projection fallback used to pick its shared_expert_gate,
whose input is the residual stream."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry.schatten import SchattenNormTask  # noqa: E402


def _tiny_qwen2_moe():
    """Blocks 0 and 2 dense (mlp_only_layers), 1 and 3 routed with a shared
    expert and its shared_expert_gate, as Qwen1.5-MoE and Qwen3-Next."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(
        "qwen2_moe", vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=0, mlp_only_layers=[0, 2], num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=64)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()


def test_routed_blocks_are_nan_with_reason_and_dense_blocks_kept():
    r = SchattenNormTask(config={"num_samples": 4, "documents": "dataset"}).evaluate(
        _tiny_qwen2_moe(), fd.tokenizer(), fd.dataset(4))
    per_layer = r["_meta_schatten_1_mlp_per_layer"]
    assert math.isnan(per_layer[1]) and math.isnan(per_layer[3]), per_layer
    assert math.isfinite(per_layer[0]) and math.isfinite(per_layer[2]), per_layer
    assert r["_meta_schatten_mlp_n_routed_blocks"] == 2
    assert "blocks [1, 3] hold routed experts" in r["_meta_schatten_mlp_routed_reason"]
    assert r["_meta_schatten_mlp_layers_found"] == 2
    # The headline averages the dense blocks only (here the per-text means
    # of two blocks, so their per-layer means average to it exactly).
    assert r["schatten_1"] == pytest.approx(np.mean([per_layer[0], per_layer[2]]), rel=1e-12)


def test_dense_model_reports_no_routed_block():
    r = SchattenNormTask(config={"num_samples": 2, "documents": "dataset"}).evaluate(
        fd.tiny_llama(), fd.tokenizer(), fd.dataset(2))
    assert r["_meta_schatten_mlp_n_routed_blocks"] == 0
    assert r["_meta_schatten_mlp_routed_reason"] is None
    assert all(math.isfinite(v) for v in r["_meta_schatten_1_mlp_per_layer"])
