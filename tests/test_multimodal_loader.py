"""A multimodal checkpoint outside the causal-LM mapping loads with its LM head (image-text-to-text class).

transformers 5.10 registers Gemma 3/4 multimodal configs under AutoModelForCausalLM but not Mistral 3, so the
loader fell back to AutoModel: a headless backbone with no logits, which made perplexity NaN and generation empty
on Ministral-3 (replication smoke run, 2026-11-21).
"""
import pytest
import torch

from blme.models.wrapper import _load_model

transformers = pytest.importorskip("transformers")


def _tiny_mistral3(tmp_path):
    try:
        from transformers import AutoModelForImageTextToText, Mistral3Config, MistralConfig, PixtralVisionConfig
    except ImportError:
        pytest.skip("this transformers has no Mistral3")
    text = MistralConfig(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    vision = PixtralVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 image_size=32, patch_size=16, head_dim=8)
    cfg = Mistral3Config(vision_config=vision, text_config=text)
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(cfg).eval().save_pretrained(tmp_path)
    return tmp_path


def test_mistral3_checkpoint_loads_with_its_lm_head(tmp_path):
    from transformers import AutoModelForCausalLM
    path = _tiny_mistral3(tmp_path)
    with pytest.raises(ValueError):          # the reason for the fallback: not a registered causal LM
        AutoModelForCausalLM.from_pretrained(path)
    model = _load_model(str(path), {"dtype": torch.float32})
    name = type(model).__name__        # the image-text-to-text class of Mistral 3, which carries the LM head
    assert name.startswith("Mistral3") and name.endswith("ConditionalGeneration"), name
    ids = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    assert out.logits.shape == (1, 4, 96) and len(out.hidden_states) == 3
    assert model.get_output_embeddings() is not None


def test_causal_lm_checkpoints_are_unchanged(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=50)).save_pretrained(tmp_path)
    assert type(_load_model(str(tmp_path), {})).__name__ == "GPT2LMHeadModel"
