"""Regression tests for repe_concept_separability fixes."""
import copy

import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")

from blme.tasks.repe.concept_separability import ConceptSeparabilityTask  # noqa: E402


def _tiny_llama(vocab):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        attn_implementation="eager")
    return transformers.LlamaForCausalLM(cfg).eval()


def test_concept_reading_computes_in_fp32():
    """A bf16 model is read in fp32 arithmetic: the PCA spectrum of the reading
    states equals the fp32 copy's, and the model is handed back in bf16."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 tokenizer not cached: {e}")
    bf16 = _tiny_llama(len(tok)).to(torch.bfloat16)
    ref = ConceptSeparabilityTask({}).evaluate(copy.deepcopy(bf16).float(), tok, None)
    out = ConceptSeparabilityTask({}).evaluate(bf16, tok, None)
    assert out["_meta_layer_pca_explained_ratio"] == ref["_meta_layer_pca_explained_ratio"]
    assert out["lat_test_acc"] == ref["lat_test_acc"]
    assert out["_meta_compute_dtype"] == "float32" and out["_meta_promoted_to_fp32"] is True
    assert all(p.dtype == torch.bfloat16 for p in bf16.parameters())
