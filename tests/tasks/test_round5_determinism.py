"""Regression tests pinning determinism for RNG-using tasks.

Previously these tasks used the global Python / numpy / torch RNG
without seeds, so rerunning the study produced different feature values
across invocations. Paper-grade numbers should be deterministic.
"""
import inspect
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))


def test_causality_ablation_seeds_its_randperm():
    """causality/ablation.py must seed its ``torch.randperm`` so the
    same set of feature indices is ablated across reruns."""
    from blme.tasks.causality import ablation
    src = inspect.getsource(ablation)
    # Unseeded call is the bug.
    # (Allow the seeded form which uses a Generator.)
    assert "torch.randperm(dim)[" not in src, (
        "ablation.py still calls torch.randperm(dim)[...] without a seed"
    )
    # Must use an explicit Generator.
    assert "torch.Generator" in src


def test_causality_edge_attribution_seeds_shuffle():
    """edge_attribution.py must seed its corruption shuffle."""
    from blme.tasks.causality import edge_attribution
    src = inspect.getsource(edge_attribution)
    assert "torch.randperm(input_ids.shape[1], device=device)" not in src, (
        "edge_attribution.py still uses unseeded torch.randperm for the "
        "corruption shuffle"
    )
    assert "torch.Generator" in src


def test_dynamics_trajectories_uses_seeded_sampler():
    """trajectories.py must use a seeded ``random.Random`` for pair
    sampling, not the global ``random`` module."""
    from blme.tasks.dynamics import trajectories
    src = inspect.getsource(trajectories)
    assert "random.sample(samples, 2)" not in src, (
        "trajectories.py still uses global random.sample for pair selection"
    )
    assert "random.Random" in src or "_random.Random" in src


def test_attention_polysemanticity_uses_seeded_sampler():
    """attention_polysemanticity.py must seed the random.sample call
    that picks 4 modules."""
    from blme.tasks.interpretability import attention_polysemanticity
    src = inspect.getsource(attention_polysemanticity)
    assert "random.sample(target_modules, 4)" not in src, (
        "attention_polysemanticity.py still uses global random.sample "
        "for module selection"
    )
    assert "random.Random" in src


def test_bias_task_uses_offset_mapping_helper():
    """bias.py must expose the offset-mapping-based word locator."""
    from blme.tasks.consistency import bias
    assert hasattr(bias, "_find_word_token_position")
    src = inspect.getsource(bias)
    assert "return_offsets_mapping=True" in src
    # The fallback-to-end sentinel should still be gone from the main path.
    assert "input_ids.index(first_word_tok)" not in src


def test_repe_steering_hook_casts_vector_to_hidden_dtype():
    """The steering hook must cast the task-vector addition to the
    hidden state dtype — without this, fp32 vec added to bf16 hidden
    silently upcasts the residual to fp32, which changes every
    downstream layer's output and biases KL against bf16 models."""
    from blme.tasks import representation_engineering
    src = inspect.getsource(representation_engineering)
    # Raw ``out_t[:, -1, :] += alpha * vec`` without a dtype cast is the bug.
    # Fix line must have ``.to(out_t.dtype)``.
    assert "(alpha * vec).to(out_t.dtype)" in src, (
        "repe_steering_effectiveness still adds fp32 task-vector to "
        "bf16/fp16 hidden state without dtype cast"
    )


def test_sae_features_target_layer_parsed_from_sae_id():
    """sae_features.py must parse the layer number from sae_id
    (``blocks.8.hook_resid_pre`` → 8) instead of using ``num_layers//2``
    — the SAE is trained on a specific layer's hidden state and
    applying it to a different layer produces meaningless L0 counts."""
    from blme.tasks.interpretability import sae_features
    src = inspect.getsource(sae_features)
    # The old hardcoded mid-layer assignment without fallback must be
    # gated behind the sae_id parse.
    assert 'blocks\\.(\\d+)\\.' in src or 'blocks\\.(\\d+)' in src, (
        "sae_features.py should parse the SAE's trained layer from sae_id"
    )


def test_persistence_landscape_uses_trapezoid_with_fallback():
    """persistence_landscape.py must use np.trapezoid (NumPy 2.0) with
    fallback to np.trapz — the deprecated np.trapz emits warnings on
    NumPy 2.x."""
    from blme.tasks.topology import persistence_landscape
    src = inspect.getsource(persistence_landscape)
    assert 'getattr(np, "trapezoid", np.trapz)' in src, (
        "persistence_landscape.py should prefer np.trapezoid with "
        "np.trapz fallback"
    )


def test_prediction_alignment_uses_output_projection_not_input_embedding():
    """geometry_prediction_alignment must project hidden states onto
    lm_head.weight (the model's actual next-token projection), not
    the input embedding table. Input embeddings coincide with lm_head
    only for TIED-head architectures (GPT-2, Pythia, etc.); on untied
    heads (Gemma 3/4) they differ and measuring input-embedding
    alignment no longer reflects prediction."""
    from blme.tasks.geometry import consistency
    src = inspect.getsource(consistency)
    assert "_get_output_projection_weight" in src
    assert "get_lm_head" in src
    # The old unconditional ``get_embeddings(model)`` fallback must be
    # gated behind the lm_head path.
    assert "embeddings = _get_output_projection_weight(model)" in src
