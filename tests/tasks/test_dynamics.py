"""
Tests for all 3 dynamics tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import numpy as np
import torch


def test_interpolation_does_not_hang_on_unusable_samples():
    """Campaign-3 BUG: `while count < num_pairs` only advanced after a fully
    usable pair, so rows lacking text/input_ids spun the loop forever. Such a
    dataset must terminate with an error dict, not hang."""
    import torch
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    class _M:
        def parameters(self):
            return iter([torch.zeros(1)])

    # rows match neither str / 'text' / 'input_ids' -> every pair is unusable
    dataset = [{"tokens": [1, 2]}, {"tokens": [3, 4]}, {"tokens": [5, 6]}]
    r = LatentInterpolationTask(config={"num_pairs": 5}).evaluate(_M(), None, dataset)
    assert isinstance(r, dict) and "error" in r


def test_generation_diversity_seed_does_not_crash_generation():
    """Campaign-3 BUG: a seed config created a torch.Generator and passed it
    as generate(generator=...), which transformers rejects with ValueError;
    the broad except swallowed it per-prompt so the task returned
    {"error": "No completions generated"}. With a seed set the task must
    still produce real diversity metrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from blme.tasks.dynamics.generation_diversity import GenerationDiversityTask

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    task = GenerationDiversityTask(config={
        "seed": 123, "num_prompts": 2, "n_samples_per_prompt": 3,
        "max_new_tokens": 8,
    })
    r = task.evaluate(model, tok, dataset=None)
    assert "error" not in r, f"seeded run returned error dict: {r}"
    assert "mean_distinct_1" in r and r["seed"] == 123


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_interpolation(mock_model, mock_tokenizer):
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 2, "steps": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "convexity_gap" in results
    assert "_meta_interp_entropy_0.0" in results
    assert "_meta_interp_entropy_0.5" in results


def test_interpolation_default_steps_include_true_midpoint(mock_model, mock_tokenizer):
    """The default ``steps=10`` must still measure alpha=0.5.

    ``np.linspace(0, 1, 10)`` skips 0.5, and the previous implementation
    silently used a zero fallback for convexity_gap.
    """
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 1})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=["left", "right"])

    assert "_meta_interp_entropy_0.5" in results
    expected_gap = results[f"_meta_interp_entropy_0.5"] - (
        results[f"_meta_interp_entropy_0.0"] + results[f"_meta_interp_entropy_1.0"]
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
    assert np.isfinite(out["_meta_gradient_flow_entropy"])


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
    assert "_meta_mean_normalized_magnitude" in res
    assert "mean_normalized_angle" in res
    assert np.isfinite(res["_meta_mean_normalized_magnitude"])
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
    # `_meta_`-prefixed: with the full h_0..h_L chain this equals n_layers
    # EXACTLY, so unprefixed it was a perfect depth proxy in the feature matrix.
    assert results.get("_meta_chain_length_per_sample") == 2
    assert "chain_length_per_sample" not in results


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

    # Corrected contract (Campaign-3): transformers' generate() has no
    # per-call generator kwarg — passing it raised ValueError and the task
    # silently returned an error dict. Determinism now comes from
    # torch.manual_seed(seed), so generate() must NOT receive a generator.
    assert not model.generator_was_passed
    assert result["seed"] == 123
    assert result["completion_filter"] == "trim_at_eos_drop_special"
    # Both scopes are emitted now: `distinct_n` is Li et al.'s collection-level
    # ratio (ten duplicate completions -> 0.100000) and `mean_distinct_n` is
    # the per-completion mean (-> 1.000000). The old string named only the
    # second while the first is the paper's definition.
    assert result["distinct_n_scope"] == (
        "corpus (distinct_n) + per_completion_mean (mean_distinct_n)")
    assert result["self_bleu_scope"] == "all_pairs_over_collection"
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

    assert result["sampling_entropy_collapse_delta"] == pytest.approx(0.0, abs=1e-6)


def test_chain_of_embedding_zmag_not_one_pins_normalization():
    """Regression (campaign-4 B3): the original fixture chain had
    Z_Mag = ||h_L - h_0|| exactly 1, so dropping the Eq. 3 magnitude
    normalization was invisible to every pinned test. This chain has
    Z_Mag = sqrt(13); the reference values are computed in-test with a
    verbatim transcription of the official Alsace08/Chain-of-Embedding
    score.py (compute_CoE_Mag/Ang/R/C @ eef2d8f)."""
    from blme.tasks.dynamics.coe import _coe_from_chain

    chain = [
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        torch.tensor([3.0, 3.0], dtype=torch.float64),
    ]
    h = [c.numpy() for c in chain]
    L = len(h) - 1

    # --- verbatim-official reference (numpy float64) ---
    def mag(a, b):
        return float(np.linalg.norm(b - a))

    def ang(a, b):
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.arccos(np.clip(cos, -1.0, 1.0)))

    z_mag = mag(h[0], h[-1])
    z_ang = ang(h[0], h[-1])
    assert abs(z_mag - 1.0) > 0.5, "scenario must have Z_Mag far from 1"

    mags_n = [mag(h[i], h[i + 1]) / z_mag for i in range(L)]
    angs_n = [ang(h[i], h[i + 1]) / z_ang for i in range(L)]
    ref_mag_norm_mean = float(np.mean(mags_n))
    ref_coe_r = float(np.mean([m - a for m, a in zip(mags_n, angs_n)]))
    xs = [m * np.cos(a) for m, a in zip(mags_n, angs_n)]
    ys = [m * np.sin(a) for m, a in zip(mags_n, angs_n)]
    ref_coe_c = float(np.sqrt(np.mean(xs) ** 2 + np.mean(ys) ** 2))

    got = _coe_from_chain(chain)
    assert got["mag_norm_mean"] == pytest.approx(ref_mag_norm_mean, abs=1e-9)
    assert got["coe_r"] == pytest.approx(ref_coe_r, abs=1e-9)
    assert got["coe_c"] == pytest.approx(ref_coe_c, abs=1e-9)
    # Removing the magnitude normalization must NOT go unnoticed here:
    raw_mag_mean = float(np.mean([mag(h[i], h[i + 1]) for i in range(L)]))
    assert abs(raw_mag_mean - ref_mag_norm_mean) > 0.1


def test_sharpness_evaluate_end_to_end_deterministic(mock_model, mock_tokenizer):
    """Drive the REAL LossSharpnessTask.evaluate() (no test previously did):
    finite outputs and bitwise determinism across two runs."""
    from blme.tasks.dynamics.sharpness import LossSharpnessTask

    cfg = {"n_hutchinson": 2, "n_power_iter": 3, "num_samples": 2,
           "param_scope": "last_layer", "seed": 0}
    r1 = LossSharpnessTask(config=dict(cfg)).evaluate(
        mock_model, mock_tokenizer, dataset=None)
    if "error" in r1:
        pytest.skip(f"unsupported architecture: {r1['error']}")
    for key in ("baseline_loss", "hutchinson_trace_estimate",
                "top_eigenvalue_estimate", "sam_sharpness",
                "sam_perturbed_loss"):
        assert key in r1
        assert np.isfinite(r1[key]), key
    # SAM identity holds inside the result dict itself.
    assert r1["sam_perturbed_loss"] - r1["baseline_loss"] == pytest.approx(
        r1["sam_sharpness"], abs=1e-9)

    r2 = LossSharpnessTask(config=dict(cfg)).evaluate(
        mock_model, mock_tokenizer, dataset=None)
    for key in ("baseline_loss", "hutchinson_trace_estimate",
                "top_eigenvalue_estimate", "sam_sharpness"):
        assert r1[key] == pytest.approx(r2[key], abs=0.0), (
            f"{key} not deterministic")


def test_interpolation_endpoints_lerp_equals_slerp(mock_model, mock_tokenizer):
    """Analytic invariant pinning the entropy kernel: at alpha = 0 and 1 the
    norm-corrected lerp and slerp trajectories are the SAME point (the raw
    endpoint), so the two entropy curves must coincide there exactly. A
    wrong softmax axis or entropy kernel would break this equality."""
    from blme.tasks.dynamics.trajectories import LatentInterpolationTask

    task = LatentInterpolationTask(config={"num_pairs": 2, "steps": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in results:
        pytest.skip(f"unsupported architecture: {results['error']}")

    for a in ("0.0", "1.0"):
        li, sl = (results[f"_meta_interp_entropy_{a}"],
                  results[f"_meta_slerp_entropy_{a}"])
        assert li == pytest.approx(sl, rel=1e-6), (
            f"lerp/slerp entropies differ at endpoint alpha={a}")
        assert np.isfinite(li) and li >= 0.0


def test_stability_reference_model_self_reference_is_perfect(
        mock_model, mock_tokenizer, tmp_path):
    """Regression (campaign-4 B3): the reference_model branch had zero test
    coverage. Self-reference (the model compared to a saved copy of itself)
    has the analytic answer: Jaccard exactly 1.0 with std 0.0."""
    from blme.tasks.dynamics.stability import NeighborhoodStabilityTask

    save_dir = tmp_path / "ref_model"
    try:
        mock_model.save_pretrained(save_dir)
    except Exception as exc:
        pytest.skip(f"save_pretrained unsupported: {exc}")

    task = NeighborhoodStabilityTask(config={
        "reference_model_path": str(save_dir), "k": 10, "num_samples": 100,
    })
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in results:
        pytest.skip(f"reference load unsupported here: {results['error']}")

    assert results["stability_mode"] == "reference_model"
    assert results["stability_mean"] == 1.0
    assert results["stability_std"] == 0.0
