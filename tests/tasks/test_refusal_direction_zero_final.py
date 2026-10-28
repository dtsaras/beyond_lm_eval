"""repe_refusal_direction protocol checks (Arditi et al. 2024; Q4).

Base models (no chat template) get NaN headline values with a ``_meta_``
reason; instruct models are read at the post-instruction tokens of their chat
template, at resid_pre layers below 0.8 L. A selected direction whose class
means coincide is reported as 0.0, not replaced by another layer's norm.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")

from blme.tasks.repe.refusal import RefusalDirectionTask  # noqa: E402

CHAT = ("{% for m in messages %}<|user|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


class _Loose(RefusalDirectionTask):
    """The paper's rule with the KL/induce filters opened, so a random model
    keeps candidates; the layer prune (l < 0.8 L) stays."""
    KL_THRESHOLD, INDUCE_THRESHOLD = 1e9, -1e9


@pytest.fixture(scope="module")
def tiny():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval(), tok


def test_base_model_gets_nan_and_a_reason(mock_model, mock_tokenizer):
    res = RefusalDirectionTask({}).evaluate(mock_model, mock_tokenizer, None)
    if "error" in res:
        pytest.skip(f"task declined: {res['error']}")
    assert res["_meta_is_instruct"] is False
    assert res["_meta_instruct_detected_by"] == "tokenizer.chat_template"
    assert "chat template" in res["_meta_refusal_skipped_reason"]
    for k in RefusalDirectionTask.HEADLINE_KEYS:
        assert np.isnan(res[k]), k


def test_instruct_model_is_read_below_0_8L(tiny):
    model, tok = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    res = _Loose({}).evaluate(model, tok, None)
    assert res["_meta_is_instruct"] is True
    assert res["_meta_selected_layer"] < int(0.8 * 5)
    assert 0.0 <= res["selected_layer_frac"] < 0.8
    assert -res["_meta_n_post_instruction_tokens"] <= res["_meta_selected_position"] <= -1
    assert res["_meta_ablation_hooks_complete"] is True


def test_zero_selected_direction_is_reported_as_zero(tiny, monkeypatch):
    """Class-independent resid_pre states at every site: every candidate's mean
    difference is exactly zero, and the selected one reports 0.0."""
    from blme.tasks.repe import refusal as R
    model, _ = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    real = R._resid_pre_sites
    fixed = {}

    def same_for_every_prompt(model, encoded, n_pos, n_layers):
        if "x" not in fixed:
            fixed["x"] = next(real(model, encoded[:1], n_pos, n_layers))
        for _ in encoded:
            yield fixed["x"].clone()

    monkeypatch.setattr(R, "_resid_pre_sites", same_for_every_prompt)
    res = _Loose({}).evaluate(model, tok, None)
    assert res["direction_norm"] == 0.0


def test_instruct_model_is_scored_in_fp32(tiny):
    """A bf16 model's selection and scores equal its fp32 copy's."""
    import copy
    model, _ = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    ref = _Loose({}).evaluate(copy.deepcopy(bf16).float(), tok, None)
    out = _Loose({}).evaluate(bf16, tok, None)
    for k in ("direction_norm", "refusal_score_bypass", "refusal_score_induce",
              "_meta_kl_ablation", "_meta_selected_layer", "_meta_selected_position"):
        assert out[k] == ref[k], (k, out[k], ref[k])
    assert out["_meta_compute_dtype"] == "float32"
    assert all(p.dtype == torch.bfloat16 for p in bf16.parameters())
