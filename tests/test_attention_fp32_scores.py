"""Eager attention with float32 scores, ``attn_score_dtype=float32`` (0.2.7, opt-in).

transformers' eager attention computes ``q k^T * scaling`` in the model's
dtype, so in bfloat16 a large score (1e3-1e5 from large q/k biases) is rounded
to a spacing of 16-1024 before the float32 softmax. The toy models below carry
an exactly representable bias of 300 on four dimensions of q and k in every
head, with those weights zeroed: every score gains 4 x 300^2 = 360,000, which
the softmax ignores in exact arithmetic, while bfloat16's spacing there (2048)
erases the token-dependent part. The variant must match float32. Without the
option, eager attention must stay transformers' own, bit for bit.
"""
from __future__ import annotations

import math
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.models import attention as attn  # noqa: E402
from blme.models import dtype_contract as dc  # noqa: E402
from blme.models import wrapper  # noqa: E402

E, H = 64, 4
HD = E // H
TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"


def _model(arch, bias=300.0, gain=10.0, vocab=200):
    """A random 2-layer model; q/k weights scaled by ``gain`` (sharper attention) and,
    per head, four q and k dimensions with zero weights and bias ``bias`` (on the
    non-rotary dimensions for GPT-NeoX, whose RoPE covers a quarter of each head)."""
    torch.manual_seed(0)
    if arch == "gpt2":
        m = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=vocab, n_embd=E, n_layer=2, n_head=H, n_positions=256))
        with torch.no_grad():
            for block in m.transformer.h:
                w, b = block.attn.c_attn.weight, block.attn.c_attn.bias
                w[:, :2 * E] *= gain
                for h in range(H):
                    for base in (0, E):                       # the q block, then the k block
                        cols = slice(base + h * HD, base + h * HD + 4)
                        w[:, cols] = 0.0
                        b[cols] = bias
    else:
        m = transformers.GPTNeoXForCausalLM(transformers.GPTNeoXConfig(
            vocab_size=vocab, hidden_size=E, num_attention_heads=H, intermediate_size=128,
            num_hidden_layers=2, max_position_embeddings=256, rotary_pct=0.25))
        with torch.no_grad():
            for layer in m.gpt_neox.layers:
                w, b = layer.attention.query_key_value.weight, layer.attention.query_key_value.bias
                for h in range(H):
                    for part in (0, 1):                       # q, k in the head's [q | k | v] rows
                        start = h * 3 * HD + part * HD
                        w[start:start + HD] *= gain
                        rows = slice(start + HD // 2, start + HD // 2 + 4)
                        w[rows] = 0.0
                        b[rows] = bias
    return m.eval()


IDS = torch.randint(0, 200, (2, 96), generator=torch.Generator().manual_seed(1))


def _run(model, impl):
    model.set_attn_implementation(impl)
    with torch.no_grad():
        out = model(IDS, output_attentions=True)
    lg = out.logits.float()
    nll = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), IDS[:, 1:].reshape(-1))
    return float(nll), lg, [a.float() for a in out.attentions]


@pytest.mark.parametrize("arch", ["gpt2", "neox"])
def test_float32_scores_recover_float32_where_bf16_eager_does_not(arch):
    attn.register()
    n32, l32, a32 = _run(_model(arch).float(), "eager")
    nb, lb, ab = _run(_model(arch).to(torch.bfloat16), "eager")
    nf, lf, af = _run(_model(arch).to(torch.bfloat16), attn.FP32_SCORES)
    err = lambda xs, ys: max(float((x - y).abs().max()) for x, y in zip(xs, ys))  # noqa: E731
    # bf16 eager: the token-dependent scores are rounded away.
    assert err(ab, a32) > 0.5 and float((lb - l32).abs().max()) > 0.2 and abs(nb - n32) > 2e-3
    # float32 scores: at bf16's floor for this model (the same as with no bias at all).
    assert err(af, a32) < 0.05 and float((lf - l32).abs().max()) < 0.03 and abs(nf - n32) < 5e-4
    _, lb0, ab0 = _run(_model(arch, bias=0.0).to(torch.bfloat16), "eager")
    _, l320, a320 = _run(_model(arch, bias=0.0).float(), "eager")
    assert err(af, a32) < 3 * err(ab0, a320) + 1e-3


@pytest.mark.parametrize("arch", ["gpt2", "neox"])
def test_float32_model_is_bit_identical_to_eager(arch):
    attn.register()
    _, le, ae = _run(_model(arch).float(), "eager")
    _, lf, af = _run(_model(arch).float(), attn.FP32_SCORES)
    assert torch.equal(le, lf)
    assert all(torch.equal(x, y) for x, y in zip(ae, af))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_registering_the_variant_leaves_eager_unchanged(dtype):
    """Plain eager after register() is transformers' eager, bit for bit (default off)."""
    before = _run(_model("neox").to(dtype), "eager")
    attn.register()
    after = _run(_model("neox").to(dtype), "eager")
    assert torch.equal(before[1], after[1])
    assert all(torch.equal(x, y) for x, y in zip(before[2], after[2]))


def test_double_backward_and_attention_weights():
    attn.register()
    m = _model("neox").to(torch.bfloat16)
    m.set_attn_implementation(attn.FP32_SCORES)
    p = m.gpt_neox.layers[0].attention.query_key_value.weight
    loss = m(IDS).logits.float().logsumexp(-1).mean()
    (g,) = torch.autograd.grad(loss, p, create_graph=True)
    (hv,) = torch.autograd.grad((g.float() ** 2).sum(), p)
    assert torch.isfinite(hv).all() and hv.abs().sum() > 0
    with torch.no_grad():
        out = m(IDS, output_attentions=True)
    assert len(out.attentions) == 2 and out.attentions[0].dtype == torch.bfloat16
    rows = out.attentions[0].float().sum(-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=2e-2)


# ── unsupported attention fails loudly ─────────────────────────────────────

def _fake_module(name, eager=None):
    mod = types.ModuleType(name)
    if eager is not None:
        mod.eager_attention_forward = eager
    sys.modules[name] = mod
    cls = type("FakeAttention", (torch.nn.Module,), {"__module__": name})
    return cls()


def test_attention_without_an_eager_function_raises():
    m = _fake_module("_blme_test_no_eager")
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(NotImplementedError, match="eager_attention_forward"):
        attn.eager_fp32_scores_forward(m, q, q, q, None)


def test_eager_whose_first_matmul_is_not_on_the_query_raises():
    def scaled_first(module, query, key, value, attention_mask, **kw):
        w = torch.matmul(query * 0.5, key.transpose(2, 3))
        return torch.matmul(w.softmax(-1), value), w
    m = _fake_module("_blme_test_scaled_first", scaled_first)
    q = torch.randn(1, 2, 3, 4)
    with pytest.raises(RuntimeError, match="not on the query"):
        attn.eager_fp32_scores_forward(m, q, q, q, None)


def test_score_dtype_values():
    assert attn.parse_score_dtype(None) == "model" and attn.parse_score_dtype("auto") == "model"
    assert attn.parse_score_dtype("float32") == "float32" and attn.parse_score_dtype("FP32") == "float32"
    with pytest.raises(ValueError, match="attn_score_dtype"):
        attn.parse_score_dtype("bfloat16")


# ── through the loader and the envelope ──────────────────────────────────────

def _tiny_dir(path):
    _model("neox", bias=0.0, vocab=1000).save_pretrained(path)        # the tokenizer's vocabulary
    transformers.AutoTokenizer.from_pretrained(TINY_GPT2).save_pretrained(path)
    return path


def test_loader_switches_requested_eager_only(tmp_path):
    path = _tiny_dir(tmp_path)
    m, _ = wrapper.load_model_and_tokenizer(
        f"pretrained={path},dtype=bfloat16,attn_implementation=eager,attn_score_dtype=float32", device="cpu")
    assert m.config._attn_implementation == attn.FP32_SCORES
    assert getattr(m, dc.PRECISION_ATTR)["attn_score_dtype"] == "float32"
    m, _ = wrapper.load_model_and_tokenizer(
        f"pretrained={path},dtype=bfloat16,attn_implementation=sdpa,attn_score_dtype=float32", device="cpu")
    assert m.config._attn_implementation == "sdpa"            # SDPA does not round the scores
    m, _ = wrapper.load_model_and_tokenizer(f"pretrained={path},dtype=bfloat16,attn_implementation=eager",
                                            device="cpu")
    assert m.config._attn_implementation == "eager"
    assert getattr(m, dc.PRECISION_ATTR)["attn_score_dtype"] == "model"


@pytest.mark.parametrize("option, expected", [("", "eager"), (",attn_score_dtype=float32", attn.FP32_SCORES)])
def test_forced_eager_run_uses_the_variant_only_when_asked(tmp_path, option, expected):
    from blme.core import evaluate
    path = _tiny_dir(tmp_path)
    docs = [{"text": "The river carried silt past the mill and the town counted the boats."}] * 4
    env = evaluate(model_args=f"pretrained={path},dtype=bfloat16{option}",
                   tasks=["interpretability_attention_entropy"], device="cpu", dataset=docs,
                   task_configs={"interpretability_attention_entropy": {"num_samples": 4}})
    assert env["errors"] is None, env["errors"]
    assert env["config"]["attn_implementation"]["model"] == expected
    prec = env["config"]["precision"]
    assert prec["attn_implementation"] == expected
    assert prec["attn_score_dtype"] == ("float32" if option else "model")
    res = env["results"]["interpretability_attention_entropy"]
    assert all(math.isfinite(v) for k, v in res.items() if isinstance(v, float) and not k.startswith("_meta_"))
