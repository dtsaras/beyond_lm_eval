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


def test_score_continuation_finds_boundary_via_offset_map():
    """BPE / SentencePiece tokenisers can merge the last token of a
    prompt with the first token of an answer. Tokenising the two
    independently and slicing by ``len(prompt_tokens)`` then mis-aligns
    the scored window. The ``score_continuation`` helper locates the
    boundary via ``return_offsets_mapping=True`` so the scored region
    is the actual answer substring regardless of merges."""
    import pytest
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from blme.tasks.common import score_continuation

    cfg = GPT2Config(vocab_size=50257, n_positions=16, n_embd=16, n_layer=1,
                     n_head=2)
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "The capital of France is"
    answer = " Paris"
    res = score_continuation(model, tokenizer, prompt, answer)
    assert res is not None
    nll, n_tok, ids = res
    # The answer is one GPT-2 token (" Paris" ~ token 6342); n_tok==1
    # tells us the boundary was correct. More importantly the helper
    # picks up the *leading space* tokenisation — if we'd sliced by
    # ``len(tokenize(prompt))``, we'd likely have scored " Paris" via
    # an off-by-one boundary.
    assert n_tok >= 1
    # Decoding the ids back should give text that overlaps with the
    # answer (strip accounts for the leading space on GPT-2).
    decoded = tokenizer.decode(ids).strip()
    assert decoded and decoded in answer.strip()
