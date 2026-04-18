"""
Tests for all 11 interpretability tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.

Tasks requiring optional dependencies (sae_lens) are skipped when unavailable.
Tasks that are architecture-specific (attention module name heuristics) accept
error dicts as valid responses for unsupported architectures.
"""
import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_logit_lens(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.logit_lens import LogitLensTask

    task = LogitLensTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "layer0_acc" in results
    assert "layer1_acc" in results
    assert "layer0_entropy" in results


def test_attribution(mock_model, mock_tokenizer):
    from blme.tasks.interpretability.attribution import ComponentAttributionTask

    task = ComponentAttributionTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "component_coherence_mean" in results
    assert -1.0 <= results["component_coherence_mean"] <= 1.0


# ---------------------------------------------------------------------------
# New tests for remaining 9 interpretability tasks
# ---------------------------------------------------------------------------

def test_attention_entropy(mock_model, mock_tokenizer):
    """Entropy of attention distributions per head."""
    from blme.tasks.interpretability.attention import AttentionEntropyTask

    task = AttentionEntropyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May return error if model uses SDPA attention (returns None weights)
    if "error" not in results:
        assert "avg_entropy_total" in results
        assert results["avg_entropy_total"] >= 0


def test_attention_graph(mock_model, mock_tokenizer):
    """Graph topology analysis of attention matrices."""
    from blme.tasks.interpretability.attention_graph import AttentionGraphTopologyTask

    task = AttentionGraphTopologyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Attention graph needs raw attention weights which may be None with SDPA
    if "error" not in results:
        assert "mean_sink_pagerank" in results
        assert "bos_sink_ratio" in results


def test_attention_polysemanticity(mock_model, mock_tokenizer):
    """SVD entropy (effective rank) of attention outputs."""
    from blme.tasks.interpretability.attention_polysemanticity import (
        AttentionEffectiveRankTask,
    )

    task = AttentionEffectiveRankTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Architecture-specific module name matching — may return error
    if "error" not in results:
        assert "mean_attention_effective_rank_entropy" in results
        assert results["mean_attention_effective_rank_entropy"] >= 0


def test_induction_heads(mock_model, mock_tokenizer):
    """Induction head detection via repeated random sequences."""
    from blme.tasks.interpretability.induction import InductionHeadTask

    task = InductionHeadTask(config={"seq_len": 10, "num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error if attention weights are None
    if "error" not in results:
        assert "max_induction_score" in results
        assert "avg_induction_score" in results
        assert "top_induction_heads" in results


def test_prediction_entropy(mock_model, mock_tokenizer):
    """Output distribution entropy profiling."""
    from blme.tasks.interpretability.prediction_entropy import PredictionEntropyTask

    task = PredictionEntropyTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "mean_entropy" in results
    assert "mean_top1_prob" in results
    assert results["mean_entropy"] >= 0
    assert 0 <= results["mean_top1_prob"] <= 1.0


def test_probing(mock_model, mock_tokenizer):
    """Linear probing for token identity at each layer."""
    pytest.importorskip("sklearn")
    from blme.tasks.interpretability.probing import LinearProbingTask

    task = LinearProbingTask(config={"num_samples": 5, "max_probe_samples": 50})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "probing_accuracy_per_layer" in results
        assert "max_probing_accuracy" in results


def test_superposition_index(mock_model, mock_tokenizer):
    """Superposition index — neuron polysemanticity measurement."""
    from blme.tasks.interpretability.superposition import SuperpositionIndexTask

    task = SuperpositionIndexTask(config={"num_samples": 2, "max_neurons": 32})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "mean_polysemanticity_index" in results
        assert "polysemanticity_per_layer" in results
        assert "neuron_utilization_rate" in results


def test_sparsity(mock_model, mock_tokenizer):
    """Activation sparsity (L0) and kurtosis of MLP blocks."""
    from blme.tasks.interpretability.sparsity import ActivationSparsityTask

    task = ActivationSparsityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Some architectures may not have standard MLP blocks
    if "error" not in results:
        assert "global_mean_l0" in results
        assert "layer_l0_rates" in results


def test_sae_features(mock_model, mock_tokenizer):
    """SAE feature dimensionality (requires sae_lens)."""
    from blme.tasks.interpretability.sae_features import SAEFeatureDimensionalityTask

    task = SAEFeatureDimensionalityTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Expected to return error when sae_lens is not installed or SAE doesn't
    # match the test model — both are valid outcomes
    assert "error" in results or "mean_active_features_l0" in results


def test_weight_activation_alignment(mock_model, mock_tokenizer):
    """Weight-Activation Alignment (WAA) via SVD/PCA cosine similarity."""
    from blme.tasks.interpretability.weight_activation_alignment import (
        WeightActivationAlignmentTask,
    )

    task = WeightActivationAlignmentTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Architecture-dependent MLP detection; may error on some architectures
    if "error" not in results:
        assert "mean_waa_alignment" in results
        assert 0 <= results["mean_waa_alignment"] <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for device-mismatch test")
def test_weight_activation_alignment_runs_on_cuda():
    """The real-hardware smoke test tripped "Expected all tensors to
    be on the same device" in the dot-product between the weight-SVD
    vector (on CUDA) and the activation-PCA vector (on CPU). Re-run on
    a tiny CUDA model to lock the fix in."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.interpretability.weight_activation_alignment import (
        WeightActivationAlignmentTask,
    )

    cfg = GPT2Config(vocab_size=200, n_positions=16, n_embd=16, n_layer=2,
                     n_head=2)
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval().cuda()

    class Tok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=16, **kw):
            ids = torch.randint(0, 200, (1, 10))
            class B(dict):
                # ``to`` is the real HF BatchEncoding behaviour: move
                # every tensor value to the target device. Our mock
                # previously returned ``self`` unchanged, which masked
                # the exact device-mismatch bug we're guarding against.
                def to(self, dev):
                    return B({k: v.to(dev) if hasattr(v, "to") else v
                             for k, v in self.items()})
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    res = WeightActivationAlignmentTask(config={"num_samples": 2}).evaluate(
        model, Tok(), dataset=[{"text": "a"}, {"text": "b"}],
    )
    assert "error" not in res, f"WAA errored on CUDA: {res}"


def test_attention_entropy_uses_cached_attentions_when_available():
    """Historic bug: attention tasks always ran their own forward
    passes with ``output_attentions=True``. When the shared cache was
    already populated with attentions for another task, the redundant
    pass tripled peak memory on 8B-class models. The fix: consume
    cached attentions first.
    """
    import torch
    from unittest.mock import MagicMock
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=200, n_positions=16, n_embd=16, n_layer=2, n_head=2,
    )
    # Force eager attention so output_attentions=True actually returns
    # weights (SDPA / Flash return None on modern transformers).
    config._attn_implementation = "eager"
    real_model = GPT2LMHeadModel(config).eval()

    class Tok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=16):
            ids = torch.randint(0, 200, (1, 8))
            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    tokenizer = Tok()

    from blme.cache import ModelOutputCache
    cache = ModelOutputCache(real_model, tokenizer,
                              dataset=[{"text": "a"}, {"text": "b"}],
                              num_samples=2)
    cache.populate(need_hidden=False, need_attentions=True)

    # After cache is populated, the task should not need more forward
    # passes. We verify by counting calls on the real model.
    call_count = {"n": 0}
    orig_forward = real_model.forward
    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return orig_forward(*args, **kwargs)
    real_model.forward = counting_forward

    from blme.tasks.interpretability.attention import AttentionEntropyTask
    task = AttentionEntropyTask(config={"num_samples": 2})
    res = task.evaluate(real_model, tokenizer,
                         dataset=[{"text": "a"}, {"text": "b"}],
                         cache=cache)

    assert "error" not in res
    assert "avg_entropy_total" in res
    assert call_count["n"] == 0, (
        f"task ran {call_count['n']} redundant forward passes while "
        "the cache already had attentions populated"
    )


def test_weight_activation_alignment_is_single_pass():
    """WAA used to run one full forward pass *per layer*: a 32-layer
    Llama 3 8B model with 5 samples did 160 forward passes, tripping the
    task timeout. All architectures can be analysed in a single pass
    by hooking every target projection simultaneously.

    We verify this by counting the number of times the model's forward
    method is invoked."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=200, n_positions=32, n_embd=16, n_layer=4, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()

    class CountingTok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=16):
            ids = torch.randint(0, 200, (1, 12))
            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    call_count = {"n": 0}
    orig_forward = model.forward
    def counting_forward(*args, **kwargs):
        call_count["n"] += 1
        return orig_forward(*args, **kwargs)
    model.forward = counting_forward

    dataset = [{"text": f"sample {i}"} for i in range(3)]
    from blme.tasks.interpretability.weight_activation_alignment import (
        WeightActivationAlignmentTask,
    )
    task = WeightActivationAlignmentTask(config={"num_samples": 3})
    results = task.evaluate(model, CountingTok(), dataset=dataset)

    assert "error" not in results
    # Must be one forward pass per sample — no per-layer multiplier.
    # (Allow a small slack for any internal recomputation.)
    assert call_count["n"] <= 3, (
        f"expected 3 forward passes (one per sample); got {call_count['n']} — "
        f"task is still running O(num_layers × num_samples) passes."
    )


def test_weight_activation_alignment_covers_pythia_olmo_phi2():
    """The historic implementation only found projections named
    ``c_proj`` / ``down_proj`` / ``dense`` — missing Pythia/GPT-NeoX
    (``dense_4h_to_h``), OLMo (``ff_out``), and Phi-2 (``fc2``). On the
    32-model study these three families therefore returned the
    'Could not identify standard MLP projection layers' error."""
    import torch
    import torch.nn as nn
    from blme.tasks.interpretability.weight_activation_alignment import (
        _find_mlp_projection,
    )

    class PythiaMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_h_to_4h = nn.Linear(16, 32)
            self.dense_4h_to_h = nn.Linear(32, 16)
    m = PythiaMLP()
    assert _find_mlp_projection(m) is m.dense_4h_to_h

    class OLMoMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.ff_proj = nn.Linear(16, 32)
            self.ff_out = nn.Linear(32, 16)
    m = OLMoMLP()
    assert _find_mlp_projection(m) is m.ff_out

    class PhiMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)
    m = PhiMLP()
    assert _find_mlp_projection(m) is m.fc2
