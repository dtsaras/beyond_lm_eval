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
    # --- Geometry (19) ---
    "geometry_svd",
    "geometry_categories",
    "geometry_cka",
    "geometry_collapse",
    "geometry_consistency",
    "geometry_correlation_dimension",
    "geometry_hubness",
    "geometry_information_fisher",
    "geometry_intrinsic_dim",
    "geometry_lid",
    "geometry_lipschitz",
    "geometry_mahalanobis",
    "geometry_matrix_entropy",
    "geometry_mutual_info",
    "geometry_perplexity",
    "geometry_positional_decay",
    "geometry_rsa",
    "geometry_spectral",
    "geometry_unembedding",
    # --- Interpretability (12) ---
    "interpretability_attention_entropy",
    "interpretability_attention_graph",
    "interpretability_attention_polysemanticity",
    "interpretability_attribution",
    "interpretability_induction_heads",
    "interpretability_logit_lens",
    "interpretability_prediction_entropy",
    "interpretability_probing",
    "interpretability_sae_features",
    "interpretability_sparsity",
    "interpretability_superposition",
    "interpretability_waa",
    # --- Consistency (6) ---
    "consistency_calibration",
    "consistency_contamination",
    "consistency_contrastive",
    "consistency_knowledge_capacity",
    "consistency_logical",
    "consistency_paraphrase",
    # --- Dynamics (3) ---
    "dynamics_coe",
    "dynamics_interpolation",
    "dynamics_stability",
    # --- Causality (4) ---
    "causality_ablation",
    "causality_attention_knockout",
    "causality_circuit_quality",
    "causality_tracing",
    # --- Topology (3) ---
    "topology_betti_curve",
    "topology_homology",
    "topology_persistence_entropy",
    # --- Representation Engineering (3) ---
    "repe_concept_separability",
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
    """Minimum expected task count as a guardrail against accidental deletion."""
    registered = list_tasks()
    assert len(registered) >= len(EXPECTED_TASKS), (
        f"Expected at least {len(EXPECTED_TASKS)} tasks, "
        f"got {len(registered)}"
    )
