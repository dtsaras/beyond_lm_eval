"""Parity: geometry_spectral_signature vs the RELEASED Spectral Signature code
(Ingrid-505/Spectral_Signature@d04eacd, ``utils.net_esd_estimator``).

The release is run on tiny models that exercise each of its population rules:
GPT-2's fused QKV split, the shape rule that also splits a GQA key/value
projection whose width is a third of the hidden size, packed MoE experts, and
the LM-head aspect-ratio skip. Module names and alphas must agree. On Qwen3.5
the release also reads the depthwise conv1d kernel as one-eigenvalue
"experts" (alpha = inf); BLME skips it, so there the comparison is over the
matrices both include. Skips when the clone is absent (fails under
BLME_REQUIRE_REFS=1).
"""
import importlib.util
import math
import os
import subprocess

import pytest
import torch

pytest.importorskip("pandas")
transformers = pytest.importorskip("transformers")

from blme.tasks.geometry import spectral_signature as S  # noqa: E402

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "Spectral_Signature")


@pytest.fixture(scope="module")
def ref():
    path = os.path.join(_REPO, "utils.py")
    if not os.path.isfile(path):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    spec = importlib.util.spec_from_file_location("spectral_signature_ref_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sources_pin_the_cloned_commit(ref):
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert S.SpectralSignatureTask.SOURCES["code"].endswith("@" + head)


def _gpt2():
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")


def _llama_kv_third():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=1000, hidden_size=96, intermediate_size=160,
                                   num_hidden_layers=2, num_attention_heads=6,
                                   num_key_value_heads=2)
    m = transformers.LlamaForCausalLM(cfg).eval()
    assert m.model.layers[0].self_attn.k_proj.weight.shape == (32, 96)   # kv width = d / 3
    return m


def _mixtral():
    torch.manual_seed(0)
    cfg = transformers.MixtralConfig(vocab_size=1000, hidden_size=64, intermediate_size=96,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, num_local_experts=3,
                                     num_experts_per_tok=2)
    m = transformers.MixtralForCausalLM(cfg).eval()
    packed = [p for p in m.parameters() if p.dim() == 3]
    assert packed, "expected packed 3-D expert weights"
    return m


def _qwen35():
    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=4,
        linear_num_value_heads=8, linear_key_head_dim=16, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"])
    return transformers.Qwen3_5ForCausalLM(cfg).eval()


def _release(ref, model):
    res = ref.net_esd_estimator(model, device="cpu")
    return dict(zip(res["longname"], res["alpha"]))


def _blme(model):
    names, mats, counts = S._signature_matrices(model)
    with torch.no_grad():
        return {n: S._hill_alpha_xmin_mid(w)[0] for n, w in zip(names, mats)}, counts


@pytest.mark.parametrize("build", [_gpt2, _llama_kv_third, _mixtral], ids=["gpt2", "llama_kv_third", "mixtral"])
def test_population_and_alphas_equal_the_release(ref, build):
    model = build()
    rel = _release(ref, model)
    ours, _ = _blme(model)
    assert list(ours) == list(rel)
    for name in rel:
        assert ours[name] == pytest.approx(rel[name], rel=1e-12, abs=0), name


def test_known_quirks_are_present(ref):
    """Known positives: the split and the skip happen in both."""
    ours, _ = _blme(_gpt2())
    assert "transformer.h.0.attn.c_attn_q" in ours and "lm_head" not in ours
    ours, _ = _blme(_llama_kv_third())
    assert "model.layers.0.self_attn.k_proj_k" in ours


def test_qwen35_conv_kernels_are_skipped(ref):
    model = _qwen35()
    rel = _release(ref, model)
    ours, counts = _blme(model)
    conv = [n for n in rel if ".conv1d.expert_" in n]
    assert conv and all(math.isinf(rel[n]) for n in conv)     # what the release does
    assert counts["conv_kernels_skipped"] == 1 and not any("conv1d" in n for n in ours)
    shared = [n for n in rel if n in ours]
    assert len(shared) == len(ours)
    for name in shared:
        assert ours[name] == pytest.approx(rel[name], rel=1e-12, abs=0), name


def test_matrix_alpha_invariant_to_zero_padding_and_isometry():
    torch.manual_seed(1)
    W = torch.randn(48, 80) / 9.0
    a = S._hill_alpha_xmin_mid(W)[0]
    padded = torch.zeros(96, 160)
    padded[:48, :80] = W
    assert S._hill_alpha_xmin_mid(padded)[0] == pytest.approx(a, rel=1e-5)
    Q, _ = torch.linalg.qr(torch.randn(120, 48))
    R, _ = torch.linalg.qr(torch.randn(80, 80))
    assert S._hill_alpha_xmin_mid(Q @ W @ R)[0] == pytest.approx(a, rel=1e-5)


def test_evaluate_reports_the_profile_summary():
    out = S.SpectralSignatureTask({}).evaluate(_llama_kv_third(), None, None)
    prof = torch.tensor(out["_meta_alpha_hill_profile"])
    assert out["alpha_hill_mean"] == pytest.approx(float(prof.double().mean()), rel=1e-12)
    assert out["_meta_n_matrices"] == len(out["_meta_alpha_hill_modules"]) == prof.numel()
