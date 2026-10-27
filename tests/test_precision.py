"""fp32_compute: promote low-precision parameters AND buffers, restore exactly."""
import contextlib

import pytest
import torch

from blme.tasks._precision import (
    LOW_PRECISION, effective_dtype_meta, floating_dtype_summary, fp32_compute,
)

transformers = pytest.importorskip("transformers")


def _tiny_llama(dtype):
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    # `.to(dtype)` casts the floating buffers too (RoPE inv_freq), which is
    # what a user-cast checkpoint looks like.
    return transformers.LlamaForCausalLM(cfg).eval().to(dtype)


def _dtypes(model):
    return ([(n, p.dtype) for n, p in model.named_parameters()],
            [(n, b.dtype) for n, b in model.named_buffers()])


def _snapshot(model):
    return ({n: p.detach().clone() for n, p in model.named_parameters()},
            {n: b.detach().clone() for n, b in model.named_buffers()})


@contextlib.contextmanager
def _legacy_promote(model):
    """The pattern positional_decay and steering use today."""
    saved = [(p, p.dtype) for p in model.parameters()]
    model.float()
    try:
        yield
    finally:
        for p, dt in saved:
            p.data = p.data.to(dt)


@pytest.mark.parametrize("promote", [
    fp32_compute,
    pytest.param(_legacy_promote, marks=pytest.mark.xfail(
        strict=True, reason="restores parameters only; bf16 buffers stay fp32")),
])
def test_restores_every_parameter_and_buffer_dtype(promote):
    model = _tiny_llama(torch.bfloat16)
    before = _dtypes(model)
    assert any(dt == torch.bfloat16 for _, dt in before[1]), "fixture needs a bf16 buffer"
    with promote(model):
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert all(b.dtype == torch.float32 for b in model.buffers() if b.is_floating_point())
    assert _dtypes(model) == before


def test_legacy_pattern_narrows_float64():
    """`model.float()` rounds a float64 tensor to float32 for good."""
    model = _tiny_llama(torch.bfloat16)
    extra = torch.nn.Parameter(torch.tensor([1.0 + 2.0 ** -40], dtype=torch.float64))
    model.register_parameter("extra64", extra)
    with _legacy_promote(model):
        pass
    assert model.extra64.dtype == torch.float64
    assert float(model.extra64.detach()) != 1.0 + 2.0 ** -40   # the value is lost
    model2 = _tiny_llama(torch.bfloat16)
    model2.register_parameter("extra64", torch.nn.Parameter(
        torch.tensor([1.0 + 2.0 ** -40], dtype=torch.float64)))
    with fp32_compute(model2) as info:
        assert model2.extra64.dtype == torch.float64
    assert float(model2.extra64.detach()) == 1.0 + 2.0 ** -40
    assert info["compute_dtype"] == "float32+float64"


def test_values_round_trip_bit_exactly_and_forward_matches_fp32():
    model = _tiny_llama(torch.bfloat16)
    params, bufs = _snapshot(model)
    ids = torch.tensor([[1, 5, 9, 3, 7, 2]])
    ref = _tiny_llama(torch.bfloat16).float()          # same weights, fp32 copy
    with torch.no_grad():
        want = ref(ids).logits
        with fp32_compute(model):
            got = model(ids).logits
        native = model(ids).logits
    assert got.dtype == torch.float32 and torch.equal(got, want)
    assert native.dtype == torch.bfloat16
    for n, p in model.named_parameters():
        assert torch.equal(p, params[n]) and p.dtype == params[n].dtype
    for n, b in model.named_buffers():
        assert torch.equal(b, bufs[n]) and b.dtype == bufs[n].dtype


def test_restores_when_the_block_raises():
    model = _tiny_llama(torch.float16)
    before = _dtypes(model)
    params, _ = _snapshot(model)
    with pytest.raises(RuntimeError, match="boom"):
        with fp32_compute(model):
            assert next(model.parameters()).dtype == torch.float32
            raise RuntimeError("boom")
    assert _dtypes(model) == before
    for n, p in model.named_parameters():
        assert torch.equal(p, params[n])


def test_identity_tied_weights_and_shared_buffers_survive():
    model = _tiny_llama(torch.bfloat16)
    emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    assert emb is head
    shared = torch.ones(3, dtype=torch.bfloat16)
    model.model.layers[0].register_buffer("shared", shared)
    model.model.layers[1].register_buffer("shared", shared)
    with fp32_compute(model) as info:
        assert model.get_input_embeddings().weight is emb
        assert model.get_output_embeddings().weight is emb
        assert emb.dtype == torch.float32
        assert model.model.layers[0].shared is model.model.layers[1].shared
        assert model.model.layers[0].shared.dtype == torch.float32
    assert emb.dtype == torch.bfloat16 and model.get_output_embeddings().weight is emb
    assert model.model.layers[0].shared is model.model.layers[1].shared
    assert model.model.layers[0].shared.dtype == torch.bfloat16
    n_params = sum(1 for _ in model.parameters())     # tied weight counted once
    assert info["n_params_promoted"] == n_params
    # inv_freq / original_inv_freq on the rotary module + the one shared buffer.
    assert info["n_buffers_promoted"] == sum(
        1 for b in {id(b): b for b in model.buffers()}.values()
        if b.dtype in LOW_PRECISION)


def test_meta_and_no_op_cases():
    model = _tiny_llama(torch.bfloat16)
    with fp32_compute(model) as info:
        pass
    assert info["original_dtype"] == "bfloat16" and info["compute_dtype"] == "float32"
    assert info["promoted"] is True and info["n_params_promoted"] > 0
    meta = effective_dtype_meta(info)
    assert meta == {"_meta_original_dtype": "bfloat16", "_meta_compute_dtype": "float32",
                    "_meta_promoted_to_fp32": True,
                    "_meta_n_params_promoted": info["n_params_promoted"],
                    "_meta_n_buffers_promoted": info["n_buffers_promoted"]}

    fp32 = _tiny_llama(torch.float32)
    with fp32_compute(fp32) as info32:
        pass
    assert info32["promoted"] is False and info32["compute_dtype"] == "float32"
    assert info32["n_params_promoted"] == info32["n_buffers_promoted"] == 0

    with fp32_compute(model, enabled=False) as off:
        assert next(model.parameters()).dtype == torch.bfloat16
    assert off["promoted"] is False and off["compute_dtype"] == "bfloat16"
    assert effective_dtype_meta(model)["_meta_compute_dtype"] == "bfloat16"
    assert floating_dtype_summary(torch.nn.Linear(2, 2).double()) == "float64"


def test_gradients_follow_their_parameter():
    model = _tiny_llama(torch.bfloat16)
    ids = torch.tensor([[1, 2, 3, 4]])
    with fp32_compute(model):
        model(ids, labels=ids).loss.backward()
        assert all(p.grad.dtype == torch.float32 for p in model.parameters()
                   if p.grad is not None)
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads and all(p.grad.dtype == torch.bfloat16 == p.dtype for p in grads)
    # A gradient that existed on entry is promoted and restored too.
    with fp32_compute(model):
        model(ids, labels=ids).loss.backward()
    assert all(p.grad.dtype == torch.bfloat16 for p in grads)


def test_no_size_gate():
    """The promotion does not depend on the parameter count."""
    import inspect
    src = inspect.getsource(fp32_compute)
    assert "numel" not in src and "2_000_000_000" not in src
