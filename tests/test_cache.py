"""
Tests for the ModelOutputCache.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from blme.cache import ModelOutputCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(n_layers=4, hidden_dim=32, vocab_size=100):
    """Create a mock model that returns proper hidden states + logits."""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.vocab_size = vocab_size

    param = MagicMock()
    param.device = torch.device("cpu")
    mock.parameters.return_value = iter([param])

    def forward_fn(**kwargs):
        result = MagicMock()

        # Hidden states: tuple of (1, T, D) — n_layers+1 (including embedding)
        if kwargs.get("output_hidden_states", False):
            T = 5  # mock sequence length
            result.hidden_states = tuple(
                torch.randn(1, T, hidden_dim) for _ in range(n_layers + 1)
            )
        else:
            result.hidden_states = None

        # Attentions: tuple of (1, H, T, T)
        if kwargs.get("output_attentions", False):
            T = 5
            H = 4  # mock heads
            result.attentions = tuple(
                torch.randn(1, H, T, T) for _ in range(n_layers)
            )
        else:
            result.attentions = None

        # Logits: (1, T, V)
        result.logits = torch.randn(1, 5, vocab_size)
        return result

    mock.side_effect = forward_fn
    return mock


def _make_mock_tokenizer():
    """Create a mock tokenizer."""
    mock = MagicMock()

    def tokenize(text, **kwargs):
        result = MagicMock()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        result.__getitem__ = lambda self, key: ids if key == "input_ids" else MagicMock()
        result.__contains__ = lambda self, key: key == "input_ids"
        result.keys.return_value = ["input_ids", "attention_mask"]
        result.to.return_value = result
        result["input_ids"] = ids
        return result

    mock.side_effect = tokenize
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelOutputCache:
    def test_populate_hidden_states(self):
        model = _make_mock_model(n_layers=4, hidden_dim=32)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "hello world"}] * 3

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=3)
        cache.populate(need_hidden=True)

        assert cache.is_populated

        # Should have 4 layers (excluding embedding)
        hs = cache.get_hidden_states(layer_idx="all")
        assert len(hs) == 4

        # Each layer should have shape (N_total_tokens, D)
        for li in range(4):
            assert hs[li].shape[1] == 32  # hidden_dim
            assert hs[li].shape[0] > 0     # some tokens

    def test_get_single_layer(self):
        model = _make_mock_model(n_layers=4, hidden_dim=32)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "test"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        cache.populate(need_hidden=True)

        # Last layer
        last = cache.get_hidden_states(layer_idx=-1)
        assert last.shape[1] == 32

        # First layer
        first = cache.get_hidden_states(layer_idx=0)
        assert first.shape[1] == 32

    def test_populate_attentions(self):
        model = _make_mock_model(n_layers=4)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "attention test"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        cache.populate(need_hidden=False, need_attentions=True)

        attns = cache.get_attentions()
        assert attns is not None
        assert len(attns) == 4  # 4 layers

    def test_logits_always_collected(self):
        model = _make_mock_model(n_layers=2, vocab_size=100)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "logit test"}] * 2

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=False)

        logits = cache.get_logits()
        assert logits is not None
        assert len(logits) == 2

    def test_lazy_population(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "lazy"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        assert not cache.is_populated

        # Calling get_hidden_states should auto-populate
        hs = cache.get_hidden_states(layer_idx=-1)
        assert cache.is_populated
        assert hs.shape[0] > 0

    def test_no_redundant_forward_passes(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "once"}] * 2

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=True)

        call_count_after_populate = model.call_count

        # Multiple get calls should NOT re-run forward pass
        cache.get_hidden_states(layer_idx="all")
        cache.get_hidden_states(layer_idx=-1)
        cache.get_hidden_states(layer_idx=0)
        cache.get_logits()

        assert model.call_count == call_count_after_populate

    def test_clear(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "clear test"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        cache.populate(need_hidden=True)
        assert cache.is_populated

        cache.clear()
        assert not cache.is_populated

    def test_prediction_stats(self):
        model = _make_mock_model(vocab_size=100)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "stats test"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        cache.populate(need_hidden=True)

        stats, _ = cache.get_prediction_stats()
        assert "logits" in stats
        assert "labels" in stats
        assert "token_counts" in stats

    def test_string_dataset(self):
        """Cache should handle plain string datasets."""
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        dataset = ["hello", "world"]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=True)
        assert cache.is_populated

    def test_none_dataset_uses_default(self):
        """When dataset is None, cache should use internal defaults."""
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()

        cache = ModelOutputCache(model, tokenizer, dataset=None, num_samples=3)
        cache.populate(need_hidden=True)
        assert cache.is_populated
