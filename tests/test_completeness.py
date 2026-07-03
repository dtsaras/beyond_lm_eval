"""
Meta-test: verifies that ALL expected tasks are registered and instantiable.
Catches silent registration failures (missing imports, decorator issues, etc.).
"""
import pytest
from blme.core import _register_all_tasks
from blme.registry import list_tasks, get_task

# Ensure all tasks are registered before tests run
_register_all_tasks()


# The complete set of expected task names across all 7 categories
EXPECTED_TASKS = [
    # --- Geometry (32) ---
    "geometry_svd",
    "geometry_isoscore",
    "geometry_categories",
    "geometry_cka",
    "geometry_collapse",
    "geometry_contextualization",
    "geometry_correlation_dimension",
    "geometry_hsic",
    "geometry_hubness",
    "geometry_intrinsic_dim",
    "geometry_lid",
    "geometry_lipschitz",
    "geometry_mahalanobis",
    "geometry_matrix_entropy",
    "geometry_neural_collapse",
    "geometry_prediction_alignment",
    "geometry_perplexity",
    "geometry_positional_decay",
    "geometry_rsa",
    "geometry_spectral",
    "geometry_representation_sensitivity",
    "geometry_unembedding",
    "geometry_weight_norms",
    "geometry_tokenizer_efficiency",
    "geometry_schatten",
    "geometry_trajectory_curvature",
    "geometry_mp_bulk_deviation",
    # Campaign-2 additions (2026-06):
    "geometry_vendi_score",
    "geometry_phd_dimension",
    "geometry_cknna",
    "geometry_magnitude",
    "geometry_procrustes_linearity",
    # --- Interpretability (17) ---
    "interpretability_attention_effective_rank",
    "interpretability_attention_entropy",
    "interpretability_attention_graph",
    "interpretability_attention_rank",
    "interpretability_activation_sinks",
    "interpretability_attribution",
    "interpretability_head_roles",
    "interpretability_induction_heads",
    "interpretability_logit_lens",
    "interpretability_prediction_entropy",
    "interpretability_probing",
    "interpretability_sae_features",
    "interpretability_sparsity",
    "interpretability_superposition",
    "interpretability_waa",
    "interpretability_activation_kurtosis",
    "interpretability_attention_rollout",
    # --- Consistency (12) ---
    "consistency_bias_weat",
    "consistency_calibration",
    "consistency_contamination",
    "consistency_contrastive",
    "consistency_format_robustness",
    "consistency_knowledge_capacity",
    "consistency_logical",
    "consistency_membership_inference",
    "consistency_paraphrase",
    "consistency_position_sensitivity",
    "consistency_self_consistency",
    "consistency_icl_slope",
    # --- Dynamics (6) ---
    "dynamics_coe",
    "dynamics_generation_diversity",
    "dynamics_interpolation",
    "dynamics_gradient_flow",
    "dynamics_sharpness",
    "dynamics_stability",
    # --- Causality (6) ---
    "causality_ablation",
    "causality_attention_knockout",
    "causality_circuit_quality",
    "causality_edge_attribution",
    "causality_knowledge_neurons",
    "causality_tracing",
    # --- Topology (5) ---
    "topology_betti_curve",
    "topology_homology",
    "topology_persistence_entropy",
    "topology_persistence_landscape",
    "topology_zigzag_persistence",
    # --- Representation Engineering (4) ---
    "repe_concept_separability",
    "repe_refusal_direction",
    "repe_steering_effectiveness",
    "repe_task_vectors",
]


def test_all_tasks_registered():
    """Every expected task name must be present in the registry."""
    registered = set(list_tasks())

    missing = [t for t in EXPECTED_TASKS if t not in registered]
    assert not missing, f"Tasks missing from registry: {missing}"


def test_no_unexpected_tasks():
    """Flag any tasks in the registry not in our expected list.

    This is a soft check — new tasks being added is fine, but it
    ensures developers consciously add them to EXPECTED_TASKS.
    """
    registered = set(list_tasks())
    expected = set(EXPECTED_TASKS)
    extra = registered - expected

    if extra:
        pytest.skip(
            f"New tasks found not yet in EXPECTED_TASKS: {extra}. "
            "Add them to the list in test_completeness.py."
        )


@pytest.mark.parametrize("task_name", EXPECTED_TASKS)
def test_task_instantiable(task_name):
    """Every registered task must be instantiable with default config."""
    task_cls = get_task(task_name)
    assert task_cls is not None, f"Task '{task_name}' not found in registry"

    # Should instantiate without errors (default config)
    instance = task_cls(config={})
    assert hasattr(instance, "evaluate"), f"Task '{task_name}' has no evaluate method"


def test_task_count():
    """Registry must expose the current 82-task catalog."""
    registered = list_tasks()
    assert len(registered) == 82, (
        f"Expected exactly 82 registered tasks, got {len(registered)}"
    )
    assert len(registered) >= len(EXPECTED_TASKS), (
        f"Expected at least {len(EXPECTED_TASKS)} tasks, "
        f"got {len(registered)}"
    )


def test_getting_started_docs_task_count():
    """Docs drift guard: getting_started must mention the current task total."""
    from pathlib import Path
    text = Path("docs/getting_started.md").read_text(encoding="utf-8")
    assert "82 diagnostic tasks" in text
