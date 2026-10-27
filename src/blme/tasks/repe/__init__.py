"""Representation-engineering tasks (one module per task).

Importing the package registers ``repe_task_vectors``,
``repe_concept_separability``, ``repe_steering_effectiveness`` and
``repe_refusal_direction``.
"""
from ._common import HAS_SKLEARN, find_final_norm
from .task_vectors import TaskVectorGeometryTask
from .concept_separability import ConceptSeparabilityTask, _auc_summary
from .steering import SteeringEffectivenessTask, _N_RAND_DRAWS_DEFAULT
from .refusal import RefusalDirectionTask, _HARMFUL_PROMPTS, _HARMLESS_PROMPTS

__all__ = [
    "HAS_SKLEARN", "find_final_norm", "TaskVectorGeometryTask",
    "ConceptSeparabilityTask", "SteeringEffectivenessTask", "RefusalDirectionTask",
    "_auc_summary", "_N_RAND_DRAWS_DEFAULT", "_HARMFUL_PROMPTS", "_HARMLESS_PROMPTS",
]
