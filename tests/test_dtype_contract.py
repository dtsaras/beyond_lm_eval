"""The loader's dtype contract and the precision provenance in the envelope (0.2.7).

On transformers 5.10, ``from_pretrained(path, dtype=torch.float32)`` on a
Qwen3.5 checkpoint returns bfloat16 parameters: the config loader consumes the
``dtype`` keyword into the composite config, the auto class swaps in the text
sub-config (``dtype: bfloat16`` in config.json) and the model loads "auto".
``load_model_and_tokenizer`` used to pass the request through and never check
the result. The contract checks every floating parameter and buffer after the
load, loads again from a config with the dtype pinned on every sub-config, and
casts what is still narrower; a loader that honours the request is untouched.

``test_tiny_qwen3_5_float32_request_is_honoured`` fails on 0.2.6 under
transformers 5.10 (bfloat16 parameters); under 5.2, which honours the request,
it checks that nothing is reloaded. The tests with a loader that ignores the
request fail on 0.2.6 on every stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.models import dtype_contract as dc  # noqa: E402
from blme.models import wrapper  # noqa: E402

TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
TEXTS = [{"text": t} for t in (
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
)]


def _save_tokenizer(path):
    transformers.AutoTokenizer.from_pretrained(TINY_GPT2).save_pretrained(path)


def _tiny_llama_dir(path, dtype=torch.float32):
    """A tiny random Llama checkpoint (RoPE: inv_freq is a float32 buffer) with a tokenizer."""
    cfg = transformers.LlamaConfig(vocab_size=1000, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=128)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).to(dtype).save_pretrained(path)
    _save_tokenizer(path)
    return Path(path)


def _tiny_gpt2_dir(path, dtype=torch.float32):
    """A tiny random GPT-2 checkpoint (tied embeddings) with a tokenizer."""
    cfg = transformers.GPT2Config(vocab_size=1000, n_embd=32, n_layer=2, n_head=4, n_positions=128)
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(cfg).to(dtype).save_pretrained(path)
    _save_tokenizer(path)
    return Path(path)


def _tiny_qwen3_5_dir(path):
    """A tiny random Qwen3.5 checkpoint laid out as the released ones: the multimodal
    class, bfloat16 weights with the linear-attention ``A_log`` and ``norm.weight`` in
    float32, and a ``dtype`` only on the text sub-config of config.json."""
    if not hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
        pytest.skip("this transformers has no Qwen3.5")
    from safetensors.torch import load_file, save_file
    text = dict(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                layer_types=["linear_attention", "full_attention"], linear_conv_kernel_dim=4,
                linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                linear_num_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                  num_position_embeddings=64, deepstack_visual_indexes=[])
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True)
    transformers.Qwen3_5ForConditionalGeneration(cfg).save_pretrained(path)
    path = Path(path)
    weights = path / "model.safetensors"

    def keep32(k):
        return k.endswith("linear_attn.A_log") or k.endswith("linear_attn.norm.weight")

    # Cloned: load_file maps the file, which save_file then rewrites.
    sd = {k: ((v.float() if keep32(k) else v.to(torch.bfloat16)) if v.is_floating_point() else v).clone()
          for k, v in load_file(str(weights)).items()}
    save_file(sd, str(weights), metadata={"format": "pt"})
    c = json.loads((path / "config.json").read_text())
    c.pop("dtype", None)
    c.pop("torch_dtype", None)
    c["text_config"]["dtype"] = "bfloat16"
    c["vision_config"].pop("dtype", None)
    (path / "config.json").write_text(json.dumps(c, indent=2))
    _save_tokenizer(path)
    fp32 = {k.replace("model.language_model.", "model."): v for k, v in sd.items() if keep32(k)}
    assert fp32 and all(v.dtype == torch.float32 for v in fp32.values())
    return path, fp32


def _param_dtypes(model):
    return {str(p.dtype) for p in model.parameters() if p.is_floating_point()}


# ── the affected class ──────────────────────────────────────────────────────

def test_tiny_qwen3_5_float32_request_is_honoured(tmp_path):
    path, fp32_in_file = _tiny_qwen3_5_dir(tmp_path)
    raw = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    loader_ignores_dtype = _param_dtypes(raw) != {"torch.float32"}
    del raw

    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=float32", device="cpu")
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert _param_dtypes(model) == {"torch.float32"}
    assert {str(b.dtype) for b in model.buffers() if b.is_floating_point()} <= {"torch.float32"}
    # The checkpoint's float32 tensors arrive bit for bit (a cast of a bfloat16
    # load would have rounded them).
    params = dict(model.named_parameters())
    for name, value in fp32_in_file.items():
        assert torch.equal(params[name].detach(), value), name
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["requested_dtype"] == "float32"
    assert rec["native_dtype"] == "bfloat16" and rec["native_dtype_source"] == "config"
    assert rec["loaded_param_dtypes"] == ["float32"]
    assert rec["cast_applied"] is False
    if loader_ignores_dtype:            # transformers 5.10
        assert rec["reloaded"] is True and rec["dtype_contract"] == "honoured after reload"
        assert rec["load_violation"]["dtypes"] == ["bfloat16"]
    else:                               # transformers 5.2
        assert rec["reloaded"] is False and rec["dtype_contract"] == "honoured"


def test_tiny_qwen3_5_bfloat16_request_is_untouched(tmp_path):
    path, _ = _tiny_qwen3_5_dir(tmp_path)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=bfloat16", device="cpu")
    assert _param_dtypes(model) == {"torch.bfloat16"}
    # RoPE buffers stay float32, as transformers builds them.
    assert {str(b.dtype) for n, b in model.named_buffers() if "inv_freq" in n} == {"torch.float32"}
    assert getattr(model, dc.PRECISION_ATTR)["dtype_contract"] == "honoured"


# ── a loader that honours the request is untouched ──────────────────────────

@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("make", [_tiny_gpt2_dir, _tiny_llama_dir], ids=["gpt2", "llama"])
def test_honouring_loader_is_loaded_once_and_untouched(tmp_path, monkeypatch, dtype, make):
    path = make(tmp_path)
    calls = []
    real = wrapper._load_model
    monkeypatch.setattr(wrapper, "_load_model", lambda name, kw: calls.append(dict(kw)) or real(name, kw))
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype={dtype}", device="cpu")
    assert len(calls) == 1 and "config" not in calls[0]
    direct = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=getattr(torch, dtype))
    got = dict(model.named_parameters()) | dict(model.named_buffers())
    want = dict(direct.named_parameters()) | dict(direct.named_buffers())
    assert got.keys() == want.keys()
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert torch.equal(got[name], want[name]), name
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["dtype_contract"] == "honoured" and not rec["reloaded"] and not rec["cast_applied"]


# ── a loader that ignores the request ───────────────────────────────────────

def _ignoring_loader(monkeypatch, loaded_dtype, calls, quantized=False):
    """Replace _load_model by one that always returns ``loaded_dtype``, config or not."""
    def fake(name, kw):
        calls.append(dict(kw))
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=loaded_dtype)
        if quantized:
            model.hf_quantizer = object()
        return model
    monkeypatch.setattr(wrapper, "_load_model", fake)


def test_ignored_float32_request_is_reloaded_then_cast(tmp_path, monkeypatch):
    path = _tiny_llama_dir(tmp_path)
    calls = []
    _ignoring_loader(monkeypatch, torch.bfloat16, calls)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=float32", device="cpu")
    # One reload, from a config with the dtype pinned on it.
    assert len(calls) == 2 and "config" not in calls[0]
    assert calls[1]["config"].dtype == torch.float32 and calls[1]["dtype"] == torch.float32
    assert _param_dtypes(model) == {"torch.float32"}
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["reloaded"] and rec["cast_applied"]
    assert rec["dtype_contract"] == "reloaded, then cast"
    assert rec["load_violation"]["dtypes"] == ["bfloat16"]
    assert rec["loaded_param_dtypes"] == ["float32"]
    assert any("cast to float32" in c for c in rec["casts"])


def test_cast_keeps_wider_buffers_and_tied_weights(tmp_path, monkeypatch):
    path = _tiny_gpt2_dir(tmp_path)
    calls = []
    _ignoring_loader(monkeypatch, torch.float16, calls)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=bfloat16", device="cpu")
    assert _param_dtypes(model) == {"torch.bfloat16"}
    assert model.lm_head.weight is model.transformer.wte.weight
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["dtype_contract"] == "reloaded, then cast"

    llama = _tiny_llama_dir(tmp_path / "llama")
    _ignoring_loader(monkeypatch, torch.float16, calls)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={llama},dtype=bfloat16", device="cpu")
    assert _param_dtypes(model) == {"torch.bfloat16"}
    assert {str(b.dtype) for n, b in model.named_buffers() if "inv_freq" in n} == {"torch.float32"}


def test_wider_load_is_reloaded_but_never_narrowed(tmp_path, monkeypatch):
    path = _tiny_llama_dir(tmp_path)
    calls = []
    _ignoring_loader(monkeypatch, torch.float32, calls)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=bfloat16", device="cpu")
    assert len(calls) == 2
    assert _param_dtypes(model) == {"torch.float32"}
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["load_violation"]["wider_bulk"] is True and rec["cast_applied"] is False
    assert rec["dtype_contract"] == "reloaded, then wider parameters kept"
    assert any("never narrows" in w for w in rec["warnings"])


def test_quantized_model_is_recorded_not_cast(tmp_path, monkeypatch):
    path = _tiny_llama_dir(tmp_path)
    calls = []
    _ignoring_loader(monkeypatch, torch.bfloat16, calls, quantized=True)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=float32", device="cpu")
    assert len(calls) == 1 and _param_dtypes(model) == {"torch.bfloat16"}
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["dtype_contract"] == "not checked: quantized model"
    assert rec["load_violation"]["dtypes"] == ["bfloat16"] and not rec["cast_applied"]


def test_auto_dtype_is_not_checked(tmp_path, monkeypatch):
    path = _tiny_llama_dir(tmp_path, dtype=torch.float16)
    calls = []
    real = wrapper._load_model
    monkeypatch.setattr(wrapper, "_load_model", lambda name, kw: calls.append(dict(kw)) or real(name, kw))
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path}", device="cpu")
    assert len(calls) == 1 and _param_dtypes(model) == {"torch.float16"}
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["requested_dtype"] == "auto" and rec["native_dtype"] == "float16"
    assert rec["dtype_contract"].startswith("not checked")


# ── the rule itself ─────────────────────────────────────────────────────────

class _Toy(torch.nn.Module):
    def __init__(self, dtypes, buffer_dtype=torch.float32):
        super().__init__()
        self.ws = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(4, dtype=d)) for d in dtypes])
        self.register_buffer("inv_freq", torch.zeros(2, dtype=buffer_dtype), persistent=False)
        self.register_buffer("ids", torch.zeros(3, dtype=torch.long))
        self.q = torch.nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False)


@pytest.mark.parametrize("dtypes, buf, requested, violated", [
    ([torch.float32], torch.float32, torch.float32, False),
    ([torch.bfloat16], torch.float32, torch.bfloat16, False),      # RoPE buffer kept wider
    ([torch.float16], torch.float32, torch.float16, False),
    ([torch.bfloat16], torch.float32, torch.float32, True),        # the Qwen3.5 case
    ([torch.float32], torch.bfloat16, torch.float32, True),        # a narrower buffer
    ([torch.float16], torch.float32, torch.bfloat16, True),        # the other 16-bit format
    ([torch.float32], torch.float32, torch.bfloat16, True),        # wider bulk
    ([torch.bfloat16, torch.bfloat16, torch.float32], torch.float32, torch.bfloat16, False),
    ([torch.float64], torch.float64, torch.float32, False),
])
def test_find_violations(dtypes, buf, requested, violated):
    assert (dc.find_violations(_Toy(dtypes, buf), requested) is not None) is violated


def test_integer_and_8bit_tensors_are_never_checked_or_cast():
    toy = _Toy([torch.bfloat16])
    if hasattr(torch, "float8_e4m3fn"):
        toy.f8 = torch.nn.Parameter(torch.zeros(4).to(torch.float8_e4m3fn), requires_grad=False)
    assert dc.cast_violations(toy, torch.float32) == {"parameters": 1, "buffers": 0, "rotary_kept": 0}
    assert toy.q.dtype == torch.int8 and toy.ids.dtype == torch.long
    if hasattr(torch, "float8_e4m3fn"):
        assert toy.f8.dtype == torch.float8_e4m3fn


def test_rotary_buffers_are_never_cast():
    """A blanket model.to(dtype) also casts RoPE inv_freq, which transformers keeps in
    float32 and which moves predictions when rounded. The contract casts parameters
    and other buffers only."""
    toy = _Toy([torch.bfloat16], buffer_dtype=torch.bfloat16)        # inv_freq arrived narrowed
    toy.register_buffer("scale", torch.ones(2, dtype=torch.bfloat16), persistent=False)
    n = dc.cast_violations(toy, torch.float32)
    assert n == {"parameters": 1, "buffers": 1, "rotary_kept": 1}
    assert toy.inv_freq.dtype == torch.bfloat16 and toy.scale.dtype == torch.float32


def test_contract_cast_keeps_inv_freq_through_the_loader(tmp_path, monkeypatch):
    """A loader that returns the whole model cast to bfloat16 (inv_freq included) for a
    float32 request: parameters come back to float32, inv_freq is left as loaded, and
    the record says so. A loader that honours the request keeps inv_freq in float32."""
    path = _tiny_llama_dir(tmp_path)
    calls = []

    def whole_model_cast(name, kw):
        calls.append(dict(kw))
        return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(torch.bfloat16)

    monkeypatch.setattr(wrapper, "_load_model", whole_model_cast)
    model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=float32", device="cpu")
    assert len(calls) == 2 and _param_dtypes(model) == {"torch.float32"}
    assert {str(b.dtype) for n, b in model.named_buffers() if "inv_freq" in n} == {"torch.bfloat16"}
    rec = getattr(model, dc.PRECISION_ATTR)
    assert rec["dtype_contract"] == "reloaded, then cast"
    assert any("rotary buffer" in w for w in rec["warnings"])

    monkeypatch.undo()
    for dtype in ("bfloat16", "float16", "float32"):
        model, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype={dtype}", device="cpu")
        assert {str(b.dtype) for n, b in model.named_buffers() if "inv_freq" in n} == {"torch.float32"}


def test_native_dtype_hints():
    assert dc.native_dtype_hints("bfloat16", "float16")
    assert dc.native_dtype_hints("float16", "float32") and dc.native_dtype_hints("bfloat16", "float32")
    for running, native in [("float32", "float16"), ("bfloat16", "bfloat16"), ("float16", "float16"),
                            ("float16", "bfloat16"), ("bfloat16", None), (None, "float32")]:
        assert dc.native_dtype_hints(running, native) == [], (running, native)


# ── provenance in the envelope ──────────────────────────────────────────────

def test_envelope_records_precision_outside_the_results(tmp_path):
    from blme.core import evaluate
    path = _tiny_gpt2_dir(tmp_path)
    env = evaluate(model_args=f"pretrained={path},dtype=bfloat16", tasks=["geometry_svd"],
                   device="cpu", dataset=TEXTS, task_configs={"geometry_svd": {"num_samples": 4}})
    prec = env["config"]["precision"]
    assert prec["requested_dtype"] == "bfloat16"
    assert prec["native_dtype"] == "float32" and prec["native_dtype_source"] == "config"
    assert prec["loaded_param_dtypes"] == ["bfloat16"] and prec["effective_param_dtypes"] == ["bfloat16"]
    assert prec["effective_dtype"] == "bfloat16"
    assert prec["dtype_contract"] == "honoured" and prec["cast_applied"] is False
    assert any("float32-native" in w for w in prec["warnings"])
    assert not {"precision", "requested_dtype", "native_dtype", "dtype_contract"} & set(env["results"]["geometry_svd"])


def test_eager_fp16_to_bf16_cast_is_recorded(tmp_path):
    from blme.core import evaluate
    path = _tiny_llama_dir(tmp_path, dtype=torch.float16)
    env = evaluate(model_args=f"pretrained={path}", tasks=["interpretability_attention_entropy"],
                   device="cpu", dataset=TEXTS,
                   task_configs={"interpretability_attention_entropy": {"num_samples": 4}})
    prec = env["config"]["precision"]
    assert prec["requested_dtype"] == "auto" and prec["native_dtype"] == "float16"
    assert prec["loaded_param_dtypes"] == ["float16"]
    assert prec["effective_param_dtypes"] == ["bfloat16"] and prec["cast_applied"] is True
    assert any(c.startswith("eager attention") for c in prec["casts"])
    assert any("float16-native" in w for w in prec["warnings"])


def test_a_model_loaded_elsewhere_still_gets_a_record(monkeypatch):
    from blme import core
    cfg = transformers.GPT2Config(vocab_size=1000, n_embd=32, n_layer=2, n_head=4, n_positions=128)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    tok = transformers.AutoTokenizer.from_pretrained(TINY_GPT2)
    monkeypatch.setattr(core, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    env = core.evaluate(model_args="pretrained=unused,dtype=float32", tasks=["geometry_svd"], device="cpu",
                        dataset=TEXTS, task_configs={"geometry_svd": {"num_samples": 4}})
    prec = env["config"]["precision"]
    assert prec["requested_dtype"] == "float32" and prec["effective_param_dtypes"] == ["float32"]
    assert prec["dtype_contract"].startswith("not recorded")
