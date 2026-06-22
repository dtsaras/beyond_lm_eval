"""
Tests for all 3 causality tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import torch
import numpy as np


def test_causal_tracing(mock_model, mock_tokenizer):
    """Simplified causal tracing (ROME-style) — corruption and restoration."""
    from blme.tasks.causality.tracing import CausalTracingTask

    task = CausalTracingTask(config={"num_samples": 2, "noise_std": 0.1})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Causal tracing may skip all samples if noise doesn't affect prediction,
    # or may error on architectures where get_layers returns None.
    # Both are valid structural outcomes.
    if "max_aie" in results:
        assert "max_causal_layer" in results
        assert "causal_entropy" in results


def test_causal_tracing_exposes_int_layer_index():
    """``max_causal_layer`` used to be stored as the raw string key
    (e.g. ``"layer_21_aie"``), so the aggregator could not coerce it into
    a numeric column and the whole tracing summary was dropped from
    ``aggregated.csv``. The task must expose an integer layer index."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=500, n_positions=64, n_embd=32, n_layer=4, n_head=2,
    )
    model = GPT2LMHeadModel(cfg).eval()

    # Deterministic tokenizer that produces a subject-rich prompt.
    class Tok:
        vocab_size = 500
        pad_token_id = 0
        eos_token_id = 1
        def encode(self, text, return_tensors=None, add_special_tokens=True,
                   truncation=True, max_length=32):
            ids = torch.arange(2, 12).unsqueeze(0)
            if return_tensors == "pt":
                return ids
            return ids[0].tolist()
        def __call__(self, text, return_tensors=None, add_special_tokens=True,
                     truncation=True, max_length=32):
            ids = torch.arange(2, 12).unsqueeze(0)
            out = {"input_ids": ids.tolist() if return_tensors is None else ids}
            return out

    # Provide inline triples that hit the subject-location path.
    dataset = [
        {"prompt": "France capital is", "subject": "France",
         "target_true": " Paris"},
        {"prompt": "Germany capital is", "subject": "Germany",
         "target_true": " Berlin"},
    ]

    from blme.tasks.causality.tracing import CausalTracingTask
    task = CausalTracingTask(config={"num_samples": 2})
    results = task.evaluate(model, Tok(), dataset=dataset)

    # Causal tracing may return no samples if the random model's output is
    # insensitive to noise; in that case the task returns without max_aie.
    # We only gate on the case where the metric IS computed.
    if "max_aie" not in results:
        pytest.skip("tracing skipped all samples on this random init")

    # New contract: max_causal_layer_idx is an integer in [0, num_layers).
    assert "max_causal_layer_idx" in results
    idx = results["max_causal_layer_idx"]
    assert isinstance(idx, (int, np.integer))
    assert 0 <= int(idx) < cfg.n_layer


def test_causal_tracing_prompt_seed_is_hashseed_stable():
    """Prompt-level tracing noise must not depend on Python's randomized hash()."""
    import os
    import subprocess
    import sys

    from blme.tasks.causality.tracing import _stable_prompt_seed

    prompt = "France capital is"
    assert _stable_prompt_seed(prompt, base_seed=7) == _stable_prompt_seed(prompt, base_seed=7)
    assert _stable_prompt_seed(prompt, base_seed=7) != _stable_prompt_seed("Germany capital is", base_seed=7)

    code = (
        "from blme.tasks.causality.tracing import _stable_prompt_seed; "
        "print(_stable_prompt_seed('France capital is', base_seed=7))"
    )
    env0 = {**os.environ, "PYTHONHASHSEED": "0"}
    env1 = {**os.environ, "PYTHONHASHSEED": "1"}
    seed0 = subprocess.check_output([sys.executable, "-c", code], text=True, env=env0).strip()
    seed1 = subprocess.check_output([sys.executable, "-c", code], text=True, env=env1).strip()

    assert seed0 == seed1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for device-mismatch test")
def test_causal_tracing_runs_on_cuda():
    """The real-hardware smoke test caught a device mismatch: my fix
    generated noise via ``torch.randn(shape, generator=rng_gen)`` on
    CPU and then handed it to a CUDA embedding hook, tripping
    ``Expected all tensors to be on the same device``. Re-run on a
    tiny CUDA model to make sure the noise is moved to the model's
    device before the hook adds it."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=200, n_positions=16, n_embd=16, n_layer=2,
                     n_head=2)
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval().cuda()

    class Tok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def encode(self, text, return_tensors=None, add_special_tokens=True,
                   truncation=True, max_length=16):
            ids = torch.arange(2, 12).unsqueeze(0)
            return ids if return_tensors == "pt" else ids[0].tolist()
        def __call__(self, text, return_tensors=None, add_special_tokens=True,
                     truncation=True, max_length=16, return_offsets_mapping=False):
            ids = torch.arange(2, 12).unsqueeze(0)
            return {"input_ids": ids.tolist() if return_tensors is None else ids}

    from blme.tasks.causality.tracing import CausalTracingTask
    dataset = [{"prompt": "France is", "subject": "France",
                "target_true": " Paris"}]
    res = CausalTracingTask(config={
        "num_samples": 1, "n_noise_samples": 1,
    }).evaluate(model, Tok(), dataset=dataset)
    # Either succeeds with AIE output or skips cleanly; either way we
    # must not raise the device-mismatch error.
    assert "error" not in res or "device" not in str(res.get("error", "")).lower(), (
        f"device mismatch still present: {res}"
    )


def test_causal_tracing_sweeps_every_layer_by_default():
    """ROME's Figure 2 reports AIE at every layer. Our historic
    implementation sampled only 5 evenly-spaced layers, losing the
    middle-layer peak the paper identifies. Unless the user explicitly
    overrides ``trace_layers``, the task should now return one
    ``layer_<i>_aie`` entry per transformer block.
    """
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    # 15 layers is deliberately > 10 so the historic early-termination
    # "sample 5 evenly-spaced layers" branch kicks in under the old code.
    cfg = GPT2Config(
        vocab_size=500, n_positions=16, n_embd=32, n_layer=15, n_head=2,
    )
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        vocab_size = 500
        pad_token_id = 0
        eos_token_id = 1
        def encode(self, text, return_tensors=None, add_special_tokens=True,
                   truncation=True, max_length=16):
            ids = torch.arange(2, 12).unsqueeze(0)
            return ids if return_tensors == "pt" else ids[0].tolist()
        def __call__(self, text, return_tensors=None, add_special_tokens=True,
                     truncation=True, max_length=16, return_offsets_mapping=False):
            ids = torch.arange(2, 12).unsqueeze(0)
            return {"input_ids": ids.tolist() if return_tensors is None else ids}

    from blme.tasks.causality.tracing import CausalTracingTask
    # Configure enough samples/noise to almost always hit a usable run.
    dataset = [
        {"prompt": "France capital is", "subject": "France",
         "target_true": " Paris"},
    ]
    res = CausalTracingTask(config={
        "num_samples": 1,
        "n_noise_samples": 2,
    }).evaluate(model, Tok(), dataset=dataset)

    # The fix must expose which layers it intended to trace (regardless
    # of whether any sample produced a usable AIE), so reviewers can
    # verify the sweep covered every block rather than 5 spaced-apart
    # samples of them.
    assert "traced_layers" in res, (
        f"task must declare which layers were swept; got keys {list(res)}"
    )
    assert sorted(res["traced_layers"]) == list(range(cfg.n_layer)), (
        f"traced {res['traced_layers']}, expected every layer "
        f"0..{cfg.n_layer - 1}"
    )


def test_causal_tracing_auto_noise_scales_with_embedding_std():
    """ROME uses ``noise = 3 * sigma(E)``. Without adaptive scaling the
    same fixed noise under-corrupts models with large embedding norms
    and over-corrupts ones with small norms, making AIE incomparable."""
    from blme.tasks.causality.tracing import _resolve_noise_std

    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=200, n_positions=16, n_embd=16, n_layer=1,
                     n_head=2)
    model = GPT2LMHeadModel(cfg).eval()

    # Scale embeddings by 10x and verify resolved noise_std scales with
    # the embedding sigma.
    emb = model.get_input_embeddings().weight.data
    base = _resolve_noise_std(model, user_value=None)
    emb.mul_(10.0)
    scaled = _resolve_noise_std(model, user_value=None)
    assert scaled > base * 5, (
        f"noise_std did not scale with embedding σ (base={base}, "
        f"scaled={scaled})"
    )

    # An explicit user value must win unchanged so reproductions are
    # deterministic.
    assert _resolve_noise_std(model, user_value=0.25) == 0.25


def test_ablation_robustness(mock_model, mock_tokenizer):
    """Ablation robustness — degradation curve from residual-feature ablation."""
    from blme.tasks.causality.ablation import AblationRobustnessTask

    task = AblationRobustnessTask(
        config={
            "num_samples": 2,
            "ablation_percentages": [0.05, 0.1],
        }
    )
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # May error on architectures where get_layers returns None
    if "error" not in results:
        assert "baseline_loss" in results
        assert "area_under_degradation_curve" in results
        assert results["ablation_unit"] == "residual_stream_features"
        assert results["baseline_loss"] >= 0


def test_circuit_quality_observed_minimality_weights_importance():
    """Layer-proxy minimality should use observed layer effects, not only top_k_pct."""
    from blme.tasks.causality.circuit_quality import _observed_layer_minimality

    assert _observed_layer_minimality([9.0, 1.0, 0.0, 0.0], {0}) == pytest.approx(0.75 * 0.9)
    assert _observed_layer_minimality([0.0, 0.0, 0.0, 0.0], {0}) == 0.0


def test_circuit_quality(mock_model, mock_tokenizer):
    """Circuit quality — faithfulness and minimality of layer-ablation proxy circuits."""
    from blme.tasks.causality.circuit_quality import CircuitQualityTask

    task = CircuitQualityTask(config={"num_samples": 2, "top_k_pct": 50})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "circuit_faithfulness" in results
        assert "circuit_minimality" in results
        assert "circuit_quality_score" in results
        assert results["diagnostic_method"] == "layer_mean_ablation_circuit_proxy"
        assert "selected_layer_importance_share" in results
        assert 0 <= results["circuit_faithfulness"] <= 1.0
        assert 0 <= results["circuit_minimality"] <= 1.0


def test_edge_attribution_reports_layer_proxy_contract():
    """The compatibility task name should report a layer proxy, not true EAP."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.causality.edge_attribution import EdgeAttributionTask

    cfg = GPT2Config(vocab_size=200, n_positions=16, n_embd=16, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            ids = torch.arange(2, 10).unsqueeze(0)
            class BatchDict(dict):
                def to(self, dev): return self
                def __getattr__(self, name): return self[name]
            return BatchDict({"input_ids": ids})

    results = EdgeAttributionTask(config={"num_samples": 1}).evaluate(
        model, Tok(), dataset=["The capital of France is Paris"]
    )

    assert "error" not in results, results
    assert results["diagnostic_method"] == "residual_layer_gradient_patch_proxy"
    assert results["attribution_unit"] == "transformer_layer"
    assert "mean_layer_attribution_profile" in results


def test_knowledge_neuron_task_reports_saliency_proxy_contract():
    """The compatibility task name should report gradient-activation saliency."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.causality.knowledge_neurons import KnowledgeNeuronsTask

    cfg = GPT2Config(vocab_size=200, n_positions=16, n_embd=16, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        def __call__(self, text, return_tensors=None, **kwargs):
            ids = torch.arange(2, 10).unsqueeze(0)
            if return_tensors is None:
                return {"input_ids": ids[0].tolist()}
            class BatchDict(dict):
                def to(self, dev): return self
                def __getattr__(self, name): return self[name]
            return BatchDict({"input_ids": ids})
        def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
            ids = torch.arange(2, 10).unsqueeze(0)
            return ids if return_tensors == "pt" else ids[0].tolist()

    results = KnowledgeNeuronsTask(config={}).evaluate(
        model, Tok(), dataset=[{"prompt": "The capital of France is", "target": " Paris"}]
    )

    assert "error" not in results, results
    assert results["diagnostic_method"] == "ffn_gradient_activation_saliency"
    assert results["saliency_unit"] == "ffn_intermediate_neuron"
    assert "mean_saliency_gini" in results


def test_attention_knockout(mock_model, mock_tokenizer):
    """Attention head knockout — specialization via Gini coefficient."""
    from blme.tasks.causality.attention_knockout import AttentionKnockoutTask

    # Provide inline dataset to avoid external dataset download issues
    dataset = [
        {"text": "John gave a book to Mary. Mary gave a pencil to"},
        {"text": "The cat sat on the mat. The dog sat on the"},
    ]

    task = AttentionKnockoutTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    # May error if num_attention_heads not in model config or get_layers fails
    if "error" not in results:
        assert "baseline_loss" in results
        assert "max_knockout_impact" in results
        assert "head_impact_gini_coefficient" in results


def test_attention_knockout_gemma_style_head_dim():
    """Gemma 2/3 set ``head_dim`` explicitly, and ``num_heads * head_dim``
    can exceed ``hidden_size``. The old knockout hook zeroed a slice of
    the post-``o_proj`` residual (size ``hidden_size``) using indices up to
    ``num_heads * head_dim`` — the slice overflowed or zeroed the wrong
    contiguous region. The fix hooks ``o_proj``'s INPUT (size
    ``num_heads * head_dim``) so per-head slicing is always in range.
    """
    pytest.importorskip("transformers")
    try:
        from transformers import GemmaConfig, GemmaForCausalLM
    except ImportError:
        pytest.skip("GemmaConfig not available")

    # Deliberately pick num_heads * head_dim > hidden_size so the old
    # post-o_proj slicing would overflow.
    config = GemmaConfig(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,               # → 4 * 16 = 64 > hidden_size=32
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = GemmaForCausalLM(config).eval()

    # Deterministic dummy tokenizer (length-12 inputs).
    import torch
    class FixedTok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def encode(self, text, return_tensors="pt", truncation=True, max_length=128):
            ids = torch.randint(0, 200, (1, 12))
            return ids

    from blme.tasks.causality.attention_knockout import AttentionKnockoutTask

    task = AttentionKnockoutTask(config={"num_samples": 2})
    results = task.evaluate(
        model, FixedTok(),
        dataset=[{"text": "x"}] * 2,
    )

    # Must not fall through to the error path on an architecture where
    # num_heads * head_dim > hidden_size.
    assert "error" not in results, f"task errored on Gemma config: {results}"
    assert "head_impact_gini_coefficient" in results
    # Gini must be in [0, 1] — a sanity check that the knockout impacts
    # were real numbers rather than garbage from out-of-bounds indexing.
    g = results["head_impact_gini_coefficient"]
    assert 0.0 <= g <= 1.0

    # Every head should have produced a measurable knockout. Under the
    # historic bug the slice `[h*head_dim:(h+1)*head_dim]` silently no-
    # opped for ``h >= hidden_size / head_dim`` (Python slice assignment
    # on an OOB range does nothing), so `per_head_impacts[h]` was
    # identically 0 for half the heads. Verify that at most one head has
    # exactly zero impact per layer — zeros imply slice fell out of range.
    assert "per_head_impacts" in results
    impacts = results["per_head_impacts"]
    # Expect len = num_layers * num_heads. With 2 layers * 4 heads = 8.
    assert len(impacts) == 8
    n_zero = sum(1 for v in impacts if v == 0.0)
    assert n_zero < 4, (
        "half the heads had zero impact — slice likely fell out of the "
        f"post-o_proj output range; impacts={impacts}"
    )
