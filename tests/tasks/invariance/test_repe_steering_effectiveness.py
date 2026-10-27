"""Direct mechanism tests for repe_steering_effectiveness.

Width: a Llama (RMSNorm) model zero-padded from d to k*d with every function
preserved (dead residual coordinates, gains divided by sqrt(k), eps by k).
Scale: the residual-scale gauge with eps rescaled (exact for power-of-two c).
"""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import invariance as inv  # noqa: E402
from blme.tasks.repe import steering  # noqa: E402
from blme.tasks.repe.steering import SteeringEffectivenessTask  # noqa: E402

CFG = {"num_samples": 2, "n_random_draws": 8}
TEXT = "The weather today is mild and the train arrived at noon."


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")


@pytest.fixture(scope="module")
def llama(tok):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval()


def zero_pad_llama_width(model, k=4):
    """Function-preserving width padding d -> k*d of a Llama-family model.

    Residual-reading projections get zero input columns, residual-writing ones
    zero output rows, embeddings/unembedding zero columns; RMSNorm over k*d sees
    mean(x^2)/k, so its eps is divided by k and its live gains by sqrt(k).
    """
    cfg = copy.deepcopy(model.config)
    d = cfg.hidden_size
    cfg.head_dim = getattr(cfg, "head_dim", None) or d // cfg.num_attention_heads
    cfg.hidden_size = k * d
    cfg.rms_norm_eps = cfg.rms_norm_eps / k
    new = type(model)(cfg).eval()
    src = model.state_dict()
    with torch.no_grad():
        for name, p in new.state_dict().items():
            s = src[name]
            p.zero_()
            if name.endswith("norm.weight"):
                p[:d] = s / math.sqrt(k)
            elif p.ndim == 2:
                p[:s.shape[0], :s.shape[1]] = s
            else:
                p[:s.shape[0]] = s
    return new


def _isotropic_controls(model, tokenizer, layers, num_layers, device, n_draws):
    """The control this task used before: norm-matched isotropic Gaussian
    directions in the full residual width."""
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(1234)
    return {l: [torch.randn(d, generator=g, dtype=torch.float64) for _ in range(n_draws)]
            for l in range(num_layers)}


def _run(model, tok, monkeypatch=None, isotropic=False):
    if isotropic:
        monkeypatch.setattr(steering, "_control_differences", _isotropic_controls)
    return SteeringEffectivenessTask(dict(CFG)).evaluate(model, tok, None)


def test_width_padding_preserves_the_function(llama, tok):
    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    pad = zero_pad_llama_width(llama)
    assert inv.max_logit_delta(llama, pad, ids) < 1e-5


INVARIANT_KEYS = ["steering_concept_over_random", "steering_beats_random_frac",
                  "directional_consistency", "directional_success_rate",
                  "steering_sign_reversal"]


def test_steering_keys_invariant_under_width_padding_and_residual_gauge(llama, tok):
    """Width: 4x function-preserving zero-padding (rtol 1e-4, float rounding of
    the padded reductions). Scale: residual-scale gauge c=2 with eps rescaled."""
    base = _run(llama, tok)
    pad = _run(zero_pad_llama_width(llama), tok)
    inv.assert_invariant(base, pad, INVARIANT_KEYS, rtol=1e-4)
    assert base["_meta_random_control"] == "text_difference"
    assert base["_meta_n_random_draws"] == CFG["n_random_draws"]

    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    gauged = inv.residual_scale_gauge(llama, 2.0, rescale_eps=True, verify_on=ids, atol=0.0)
    inv.assert_invariant(base, _run(gauged, tok), INVARIANT_KEYS, rtol=1e-5)


def test_isotropic_control_moves_under_width_padding(llama, tok, monkeypatch):
    """Known negative: dead coordinates dilute an isotropic draw, so the ratio
    against it grows with unused width."""
    base = _run(llama, tok, monkeypatch, isotropic=True)
    pad = _run(zero_pad_llama_width(llama), tok, monkeypatch, isotropic=True)
    inv.assert_moves(base, pad, "steering_concept_over_random", min_rel=0.2)
    assert pad["steering_concept_over_random"] > base["steering_concept_over_random"]
