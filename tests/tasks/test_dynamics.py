"""
Tests for all 3 dynamics tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_interpolation(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 2, "steps": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "convexity_gap" in results
    assert "interp_entropy_0.0" in results
    assert "interp_entropy_0.5" in results


def test_interpolation_default_steps_include_true_midpoint(mock_model, mock_tokenizer):
    """The default ``steps=10`` must still measure alpha=0.5.

    ``np.linspace(0, 1, 10)`` skips 0.5, and the previous implementation
    silently used a zero fallback for convexity_gap.
    """
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 1})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=["left", "right"])

    assert "interp_entropy_0.5" in results
    expected_gap = results["interp_entropy_0.5"] - (
        results["interp_entropy_0.0"] + results["interp_entropy_1.0"]
    ) / 2
    assert results["convexity_gap"] == pytest.approx(expected_gap)


def test_stability(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.stability import NeighborhoodStabilityTask

    # Default should be an informative perturbation stability estimate, not
    # a self-comparison that is identically 1.0.
    task = NeighborhoodStabilityTask(config={"k": 5, "num_samples": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "stability_mean" in results
    assert results["stability_mode"] == "embedding_noise"
    assert "reference_required" not in results


# ---------------------------------------------------------------------------
# New test
# ---------------------------------------------------------------------------

def test_chain_of_embedding(mock_model, mock_tokenizer):
    """Chain-of-Embedding (CoE) — magnitude and angle changes across
    layers of a single forward pass (Wang et al., ICLR 2025)."""
    from blme.tasks.dynamics.coe import ChainOfEmbeddingTask

    task = ChainOfEmbeddingTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "mean_magnitude_change" in results
        assert "mean_angle_change" in results
        assert results["mean_magnitude_change"] >= 0
        assert results["mean_angle_change"] >= 0


def test_gradient_flow_uses_cross_entropy_loss():
    """Gradient-flow analysis should backprop a representative loss
    (shifted next-token cross-entropy), not a single argmax-logit.
    Backpropagating the argmax ties the measurement to the model's own
    greedy prediction — different target tokens → incommensurable norms
    across models — and is not what Pascanu et al. 2013 describes.
    """
    pytest.importorskip("transformers")
    import torch
    import torch.nn.functional as F
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=200, n_positions=16, n_embd=16, n_layer=3, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()

    class Tok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors=None, truncation=True,
                     max_length=16):
            ids = torch.arange(2, 12).unsqueeze(0)
            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    from blme.tasks.dynamics.gradient_flow import GradientFlowTask
    task = GradientFlowTask(config={"num_samples": 2})
    out = task.evaluate(
        model, Tok(),
        dataset=[{"text": "x"}, {"text": "y"}],
    )

    assert "error" not in out
    # The per-layer norms must be finite non-negative reals.
    assert all(
        isinstance(v, (int, float)) and v >= 0 and np.isfinite(v)
        for v in out["gradient_norm_per_layer"]
    )
    # The new contract exposes which loss was used so downstream scripts
    # can verify the fix.
    assert out.get("loss") == "cross_entropy", (
        f"expected cross-entropy loss; got {out.get('loss')}"
    )
    # Entropy must be finite (slope computation used to explode when
    # ``np.log`` was applied to an identically-zero layer norm).
    assert np.isfinite(out["gradient_flow_entropy"])


def test_chain_of_embedding_emits_paper_scores():
    """Wang et al. 2025 Eq. 3 normalises every per-layer magnitude and
    angle by the end-to-end distance/angle between ``h_0`` (embedding)
    and ``h_L`` (final block). Eq. 5 defines the output-free score
    ``CoE-R`` and Eq. 7 the complex-plane score ``CoE-C``. These are
    the paper's headline metrics and must be exposed."""
    import numpy as np
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.dynamics.coe import ChainOfEmbeddingTask

    cfg = GPT2Config(
        vocab_size=200, n_positions=16, n_embd=16, n_layer=3, n_head=2,
    )
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        vocab_size = 200
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=16, **kw):
            ids = torch.randint(0, 200, (1, 10))
            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        def encode(self, text, return_tensors=None, **kw):
            ids = torch.randint(0, 200, (1, 10))
            return ids if return_tensors == "pt" else ids[0].tolist()

    res = ChainOfEmbeddingTask(config={"num_samples": 2}).evaluate(
        model, Tok(), dataset=[{"text": f"s{i}"} for i in range(2)],
    )
    assert "error" not in res
    # Paper Eq. 5 + Eq. 7 — both must be emitted.
    assert "coe_r" in res, "missing CoE-R (paper Eq. 5)"
    assert "coe_c" in res, "missing CoE-C (paper Eq. 7)"
    assert np.isfinite(res["coe_r"])
    assert np.isfinite(res["coe_c"])

    # Normalised magnitude/angle — also reported per Eq. 3.
    assert "mean_normalized_magnitude" in res
    assert "mean_normalized_angle" in res
    assert np.isfinite(res["mean_normalized_magnitude"])
    assert np.isfinite(res["mean_normalized_angle"])


def test_chain_of_embedding_coe_c_matches_reference_normalized():
    """CoE-C matches the reference score.py (Alsace08/Chain-of-Embedding):
    NORMALIZED magnitude as radius AND NORMALIZED angle as phase.

    Reference compute_CoE_C: x=Mag̃·cos(Ãng), y=Mag̃·sin(Ãng),
    CoE-C = sqrt(mean(x)^2 + mean(y)^2). (Corrected 2026-06-22 from the
    earlier raw-magnitude/raw-angle implementation.)
    """
    from blme.tasks.dynamics.coe import _coe_from_chain

    chain = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ]

    result = _coe_from_chain(chain)
    rn = result["normalized_magnitudes"]
    sn = result["normalized_angles"]
    x = np.mean([rn[i] * np.cos(sn[i]) for i in range(len(rn))])
    y = np.mean([rn[i] * np.sin(sn[i]) for i in range(len(rn))])
    expected = float(np.sqrt(x ** 2 + y ** 2))

    assert result["coe_c"] == pytest.approx(expected, abs=1e-6)
    # The reference CoE-C is the fully-normalized form; the alias matches.
    assert result["normalized_coe_c"] == pytest.approx(result["coe_c"], abs=1e-9)


def test_chain_of_embedding_is_layerwise(mock_model, mock_tokenizer):
    """Wang et al. define CoE as the layer-wise chain
    ``h^{(0)}, h^{(1)}, …, h^{(L)}`` at a fixed token. The historic
    implementation instead walked ``hidden_states[-1]`` across
    generation steps — a token-to-token trajectory on the final layer,
    which is categorically a different object. A correct implementation
    produces one chain **per sample** of length ``num_layers`` —
    independent of ``generation_steps``."""
    from blme.tasks.dynamics.coe import ChainOfEmbeddingTask

    task = ChainOfEmbeddingTask(config={"num_samples": 2})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "error" not in results
    assert "per_sample_mean_magnitude" in results
    # Each sample should contribute one chain summary.
    assert len(results["per_sample_mean_magnitude"]) == 2
    # The task must expose the axis it walked so downstream consumers
    # can verify the fix.
    assert results.get("axis") == "layers"
    # And for the mock model (2 layers), each chain has exactly
    # num_layers steps between hidden states.
    assert results.get("chain_length_per_sample") == 2


def test_generation_diversity_seed_and_trims_special_tokens():
    from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask

    class Batch(dict):
        def to(self, device):
            return self

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    class Tok:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, return_tensors="pt"):
            return Batch({"input_ids": torch.tensor([[10, 11]])})

    class Out:
        pass

    class FakeModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.generator_was_passed = False

        def parameters(self):
            return iter([self.param])

        def generate(self, **kwargs):
            gen = kwargs.get("generator")
            self.generator_was_passed = gen is not None and gen.initial_seed() == 123
            out = Out()
            out.sequences = torch.tensor([
                [10, 11, 5, 1, 0, 0, 0],
                [10, 11, 6, 1, 0, 0, 0],
            ])
            out.scores = [torch.zeros(2, 8) for _ in range(5)]
            return out

    model = FakeModel()
    result = GenerationDiversityTask(config={
        "num_prompts": 1,
        "n_samples_per_prompt": 2,
        "max_new_tokens": 5,
        "seed": 123,
    }).evaluate(model, Tok(), dataset=["prompt"])

    assert model.generator_was_passed
    assert result["completion_filter"] == "trim_at_eos_drop_special"
    assert result["distinct_n_scope"] == "per_completion_mean"
    assert result["mean_distinct_1"] == pytest.approx(1.0)


def test_generation_diversity_entropy_masks_post_eos_positions():
    from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask

    class Batch(dict):
        def to(self, device):
            return self

    class Tok:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, return_tensors="pt"):
            return Batch({"input_ids": torch.tensor([[10, 11]])})

    class Out:
        pass

    class FakeModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def generate(self, **kwargs):
            out = Out()
            out.sequences = torch.tensor([
                [10, 11, 5, 6, 1, 0, 0],
                [10, 11, 7, 8, 9, 10, 11],
            ])
            # High entropy after EOS for row 0; low entropy tail for row 1.
            scores = []
            for step in range(6):
                row0 = torch.full((2, 8), 100.0 if step >= 3 else 0.0)
                row1 = torch.full((2, 8), 0.0 if step < 4 else 100.0)
                scores.append(torch.stack([row0[0], row1[1]], dim=0))
            out.scores = scores
            return out

    result = GenerationDiversityTask(config={
        "num_prompts": 1,
        "n_samples_per_prompt": 2,
        "max_new_tokens": 6,
    }).evaluate(FakeModel(), Tok(), dataset=["prompt"])

    assert result["entropy_collapse_delta"] == pytest.approx(0.0, abs=1e-6)
