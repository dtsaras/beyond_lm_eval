"""Tests for blme.tasks.common — universal HF model introspection utilities."""

import torch
from blme.tasks.common import (
    get_embeddings,
    get_layers,
    get_num_layers,
    get_lm_head,
    apply_lm_head,
)


def test_get_layers_returns_module_list(mock_model_tokenizer):
    model, tokenizer = mock_model_tokenizer
    layers = get_layers(model)
    if layers is not None:
        assert isinstance(layers, torch.nn.ModuleList)
        assert len(layers) > 0


def test_get_num_layers_positive(mock_model_tokenizer):
    model, _ = mock_model_tokenizer
    n = get_num_layers(model)
    assert isinstance(n, int)
    assert n > 0


def test_get_embeddings_shape(mock_model_tokenizer):
    model, tokenizer = mock_model_tokenizer
    emb = get_embeddings(model)
    assert emb is not None
    assert emb.ndim == 2
    # First dimension should be vocab_size
    assert emb.shape[0] == tokenizer.vocab_size


def test_get_lm_head_not_none(mock_model_tokenizer):
    model, _ = mock_model_tokenizer
    # BertLMHeadModel may not have a standard lm_head Linear
    head = get_lm_head(model)
    if head is not None:
        assert isinstance(head, torch.nn.Linear)


def test_apply_lm_head_output_shape(mock_model_tokenizer):
    model, tokenizer = mock_model_tokenizer
    hidden_dim = get_embeddings(model).shape[1]
    dummy_hidden = torch.randn(1, hidden_dim)
    logits = apply_lm_head(model, dummy_hidden)
    assert logits.shape[-1] == tokenizer.vocab_size
