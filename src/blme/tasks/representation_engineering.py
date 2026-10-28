"""Compatibility shim: the representation-engineering tasks live in
``blme.tasks.repe`` (one module per task). Importing this module registers them."""
from .repe import (  # noqa: F401
    HAS_SKLEARN, ConceptSeparabilityTask, RefusalDirectionTask,
    SteeringEffectivenessTask, TaskVectorGeometryTask, _HARMFUL_PROMPTS,
    _HARMLESS_PROMPTS, _N_RAND_DRAWS_DEFAULT, _auc_summary,
)
from .repe._common import find_final_norm as _find_final_norm  # noqa: F401
