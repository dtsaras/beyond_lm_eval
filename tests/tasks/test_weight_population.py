"""geometry_spectral and geometry_weight_norms measure the language model's linear
maps. A convolution kernel (Qwen3.5's depthwise linear-attention conv1d) and a
multimodal checkpoint's vision tower (Gemma 3) are not scanned. Each fixture is a
known positive: it carries the module the rule excludes, and the result must equal
the result on the same model with that module removed."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch


def _qwen35():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=16,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"])
    model = Qwen3_5ForCausalLM(cfg).eval()
    conv = model.model.layers[0].linear_attn.conv1d
    assert isinstance(conv, torch.nn.Conv1d) and conv.weight.shape[1:] == (1, 4)
    return model


def _without_convs(model):
    stripped = copy.deepcopy(model)
    for layer in stripped.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.conv1d = torch.nn.Identity()
    return stripped


def _gemma3_pair():
    """A tiny Gemma3ForConditionalGeneration (what AutoModelForCausalLM loads for
    gemma-3-4b/12b/27b) and a Gemma3ForCausalLM holding the same text weights."""
    from transformers import (Gemma3Config, Gemma3ForCausalLM,
                              Gemma3ForConditionalGeneration, Gemma3TextConfig,
                              SiglipVisionConfig)
    torch.manual_seed(0)
    tc = Gemma3TextConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=32)
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                            num_attention_heads=2, image_size=28, patch_size=14)
    mm = Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=tc.to_dict(), vision_config=vc.to_dict(), mm_tokens_per_image=4)).eval()
    text = Gemma3ForCausalLM(mm.config.text_config).eval()
    text.model.load_state_dict(mm.model.language_model.state_dict())
    text.lm_head.load_state_dict(mm.lm_head.state_dict())
    n_vision = sum(isinstance(m, torch.nn.Linear) for m in mm.model.vision_tower.modules())
    assert n_vision > 0
    return mm, text, n_vision


def _headline(r):
    return {k: v for k, v in r.items() if not k.startswith("_meta_")}


def _assert_same(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(np.asarray(a[k], dtype=float),
                                      np.asarray(b[k], dtype=float), err_msg=k)


def _weight_norms(model):
    from blme.tasks.geometry.weight_norms import WeightNormProfileTask
    return WeightNormProfileTask({}).evaluate(model, None, None)


def _spectral(model):
    from blme.tasks.geometry.spectral import WeightSpectralTask
    return WeightSpectralTask({}).evaluate(model, None, None)


def test_weight_norms_skips_the_depthwise_conv():
    model = _qwen35()
    r, ref = _weight_norms(model), _weight_norms(_without_convs(model))
    _assert_same(_headline(r), _headline(ref))
    assert r["_meta_n_conv_kernels_skipped"] == 1
    assert ref["_meta_n_conv_kernels_skipped"] == 0
    # Layer 0 from first principles: the mean stable rank of its Linear weights.
    srs = []
    for m in model.model.layers[0].modules():
        if isinstance(m, torch.nn.Linear):
            s = torch.linalg.svdvals(m.weight.detach().double())
            srs.append(float((s ** 2).sum() / s[0] ** 2))
    assert r["stable_rank_per_layer"][0] == pytest.approx(np.mean(srs), rel=1e-12)


def test_spectral_skips_the_depthwise_conv():
    model = _qwen35()
    r, ref = _spectral(model), _spectral(_without_convs(model))
    _assert_same(_headline(r), _headline(ref))
    assert r["_meta_n_modules_below_min_evals"] == 1
    assert r["_meta_n_modules_scanned"] == ref["_meta_n_modules_scanned"]


def test_spectral_population_matches_weightwatcher():
    """Live WeightWatcher analyses the same Linear layers (it also fits the
    embedding, which geometry_spectral does not scan)."""
    ww = pytest.importorskip("weightwatcher")
    model = _qwen35()
    df = ww.WeightWatcher(model=model).analyze()
    n_ww = int((df["layer_type"].astype(str).str.lower() != "embedding").sum())
    assert n_ww == _spectral(model)["_meta_n_modules_scanned"]


def test_spectral_ignores_the_vision_tower():
    mm, text, n_vision = _gemma3_pair()
    r, ref = _spectral(mm), _spectral(text)
    _assert_same(_headline(r), _headline(ref))
    assert r["_meta_n_modules_outside_lm"] == n_vision
    assert ref["_meta_n_modules_outside_lm"] == 0
