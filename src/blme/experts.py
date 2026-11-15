"""Mixture-of-experts kernel selection.

transformers dispatches the routed experts of an MoE block (Qwen3-Next,
Qwen1.5-MoE, ...) to a grouped-matmul kernel by default. Some torch builds ship
that kernel for CUDA only, and the forward pass then raises
``NotImplementedError: Could not run 'aten::_grouped_mm' ... 'CPU' backend``.
``ensure_runnable_experts`` selects the eager experts in exactly that case.
"""

import functools
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("blme")

FALLBACK_ATTR = "_blme_experts_fallback"


@functools.lru_cache(maxsize=1)
def grouped_mm_runs_on_cpu() -> bool:
    """Whether the torch grouped-matmul kernel transformers would call has a CPU
    implementation. True when torch has no such kernel at all: transformers then
    uses its own loop."""
    kernel = getattr(torch.nn.functional, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
    if kernel is None:
        return True
    try:
        kernel(torch.zeros(2, 2), torch.zeros(1, 2, 2), offs=torch.tensor([2], dtype=torch.int32))
    except NotImplementedError:
        return False
    except Exception:  # an argument-convention change is not a missing kernel
        return True
    return True


def _has_cpu_parameter(model) -> bool:
    return any(p.device.type == "cpu" for p in model.parameters())


def ensure_runnable_experts(model) -> Optional[Dict[str, Any]]:
    """Switch ``model`` to the eager experts when its grouped-matmul experts
    cannot run where its weights live; a no-op for every other model.

    Returns the record of the switch (also kept on the model as
    ``_blme_experts_fallback``), or None when nothing was changed.
    """
    config = getattr(model, "config", None)
    if getattr(config, "_experts_implementation", None) != "grouped_mm":
        return getattr(model, FALLBACK_ATTR, None) if isinstance(model, torch.nn.Module) else None
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "set_experts_implementation"):
        return None
    if grouped_mm_runs_on_cpu() or not _has_cpu_parameter(model):
        return None
    model.set_experts_implementation("eager")
    record = {
        "from": "grouped_mm",
        "to": model.config._experts_implementation,
        "reason": f"torch {torch.__version__} has no CPU grouped_mm kernel and the model holds CPU weights",
    }
    setattr(model, FALLBACK_ATTR, record)
    logger.warning(
        "MoE experts: grouped_mm -> eager (%s). The two compute the same function.", record["reason"])
    return record
