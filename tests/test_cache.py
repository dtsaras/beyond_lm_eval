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
        assert "hidden" in stats
        assert "token_counts" in stats

    def test_hidden_state_sample_slicing(self):
        model = _make_mock_model(n_layers=2, hidden_dim=8)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "sample one"}, {"text": "sample two"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=True)

        all_hidden = cache.get_hidden_states(layer_idx=-1)
        sliced_hidden = cache.get_hidden_states(layer_idx=-1, num_samples=1)

        assert all_hidden.shape[0] > sliced_hidden.shape[0]

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

    def test_hidden_states_per_sample(self):
        """get_hidden_states(per_sample=True) must return one tensor per
        sample so consumers that need sample-level statistics (CKA, RSA,
        LID, matrix entropy) don't operate on a cross-sample token pile.
        """
        model = _make_mock_model(n_layers=2, hidden_dim=8)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=3)
        cache.populate(need_hidden=True)

        per_sample = cache.get_hidden_states(layer_idx=-1, per_sample=True)
        assert isinstance(per_sample, list)
        assert len(per_sample) == 3

        # Each element is a per-sample (T_i, D) tensor.
        for chunk in per_sample:
            assert chunk.ndim == 2
            assert chunk.shape[1] == 8

        # Token counts per sample must sum to the flat tensor length.
        flat = cache.get_hidden_states(layer_idx=-1)
        assert sum(c.shape[0] for c in per_sample) == flat.shape[0]

    def test_prediction_stats_is_shifted_for_next_token(self):
        """``get_prediction_stats`` must align logits/labels/hidden for
        next-token prediction — the same convention as
        ``collect_prediction_stats``. Historic bug: cache returned
        unshifted tensors with different shapes, so any task that
        ``torch.cat`` them together got a 3-D array instead of
        ``(TotalTokens, ...)``, and computing ECE/NLL silently compared
        the distribution at position t to token at position t (identity).
        """
        model = _make_mock_model(n_layers=2, hidden_dim=8, vocab_size=50)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "a"}, {"text": "b"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=True)

        stats, _ = cache.get_prediction_stats()

        # Mock model uses T=5 (see _make_mock_model). Each sample must
        # contribute T-1 = 4 rows after the shift.
        assert len(stats["logits"]) == 2
        assert len(stats["labels"]) == 2
        assert len(stats["hidden"]) == 2

        for lg in stats["logits"]:
            # Shape convention for downstream consumers is flat (T-1, V)
            assert lg.ndim == 2
            assert lg.shape[0] == 4
            assert lg.shape[1] == 50
        for lb in stats["labels"]:
            assert lb.ndim == 1
            assert lb.shape[0] == 4
        for h in stats["hidden"]:
            assert h.ndim == 2
            assert h.shape[0] == 4
            assert h.shape[1] == 8

        # Concatenation should yield 2-D (TotalTokens, V) — this is what
        # consistency/calibration.py and geometry/perplexity.py expect.
        logits_cat = torch.cat(stats["logits"], dim=0)
        labels_cat = torch.cat(stats["labels"], dim=0)
        assert logits_cat.ndim == 2
        assert logits_cat.shape[0] == labels_cat.shape[0]

    def test_prediction_stats_sync_when_hidden_missing_for_a_sample(self):
        """If the cache somehow has hidden states for only some samples
        (e.g. a forward pass failed mid-sample), we must drop the
        matching logits/labels for the missing samples rather than
        silently keeping them and producing a length-mismatched dict."""
        model = _make_mock_model(n_layers=2, hidden_dim=8, vocab_size=50)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": f"sample {i}"} for i in range(3)]
        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=3)
        cache.populate(need_hidden=True)

        # Simulate a broken middle-sample hidden entry by re-splitting
        # with a bogus sample length for sample 1.
        original_lengths = list(cache._sample_lengths)
        # Simulate: keep the stored flat tensor but claim sample 1 has
        # only 1 token (less than T = 5), so the shape check fails.
        cache._sample_lengths = [original_lengths[0], 1, original_lengths[2]]

        stats, _ = cache.get_prediction_stats()

        # The task must honour alignment even under the bad-length
        # simulation — the three lists must have the same length.
        assert len(stats["logits"]) == len(stats["labels"])
        if "hidden" in stats:
            assert len(stats["hidden"]) == len(stats["logits"])

    def test_prediction_stats_list_lengths_stay_in_sync(self):
        """``stats["logits"]``, ``stats["labels"]`` and ``stats["hidden"]``
        must have identical lengths: downstream tasks zip them and
        silent length drift produces misaligned tensors. Historic bug:
        when a sample's hidden state failed a shape check it was dropped
        but its logits/labels were still appended."""
        model = _make_mock_model(n_layers=2, hidden_dim=8, vocab_size=50)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": f"sample {i}"} for i in range(4)]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=4)
        cache.populate(need_hidden=True)

        stats, _ = cache.get_prediction_stats()
        assert len(stats["logits"]) == len(stats["labels"])
        if "hidden" in stats:
            assert len(stats["hidden"]) == len(stats["logits"]), (
                f"hidden list ({len(stats['hidden'])}) out of sync with "
                f"logits ({len(stats['logits'])}) / labels "
                f"({len(stats['labels'])})"
            )

    def test_prediction_stats_labels_are_next_tokens(self):
        """labels[i] must correspond to input_ids[i+1] — the target of the
        prediction at position i."""
        model = _make_mock_model(n_layers=2, hidden_dim=4, vocab_size=50)

        # Deterministic tokenizer so we can verify the shift: input_ids
        # are [10, 11, 12, 13, 14] for every sample.
        fixed_ids = torch.tensor([[10, 11, 12, 13, 14]])

        class FixedTok:
            pad_token_id = 0
            def __call__(self, text, **kw):
                class B(dict):
                    def to(self, dev): return self
                    def __getattr__(self, n):
                        try: return self[n]
                        except KeyError: raise AttributeError(n)
                return B({"input_ids": fixed_ids, "attention_mask": torch.ones_like(fixed_ids)})
        tokenizer = FixedTok()
        dataset = [{"text": "x"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=1)
        cache.populate(need_hidden=True)

        stats, _ = cache.get_prediction_stats()
        labels = stats["labels"][0].tolist()
        # Unshifted inputs were [10, 11, 12, 13, 14]; targets for
        # next-token prediction are [11, 12, 13, 14].
        assert labels == [11, 12, 13, 14]

    def test_hidden_states_per_sample_all_layers(self):
        """per_sample=True with layer_idx='all' returns {layer: List[Tensor]}."""
        model = _make_mock_model(n_layers=3, hidden_dim=8)
        tokenizer = _make_mock_tokenizer()
        dataset = [{"text": "a"}, {"text": "b"}]

        cache = ModelOutputCache(model, tokenizer, dataset, num_samples=2)
        cache.populate(need_hidden=True)

        per_sample = cache.get_hidden_states(layer_idx="all", per_sample=True)
        assert isinstance(per_sample, dict)
        assert set(per_sample.keys()) == {0, 1, 2}
        for layer_chunks in per_sample.values():
            assert isinstance(layer_chunks, list)
            assert len(layer_chunks) == 2
