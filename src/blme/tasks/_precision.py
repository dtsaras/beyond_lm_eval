"""Compute in float32 on a low-precision model, then hand the model back as found.

Tasks that measure small differences or derivatives lose their signal to
bf16/fp16 arithmetic: keeping the low-precision weights but doing the
arithmetic in fp32 reproduces an fp32 run, while the bf16 run does not. The
pattern several tasks use for this is

    saved = [(p, p.dtype) for p in model.parameters()]
    model.float()
    ...
    for p, dt in saved:
        p.data = p.data.to(dt)

which has three defects: ``model.float()`` also promotes every floating
BUFFER and the loop restores parameters only, so a bf16/fp16 buffer (e.g.
RoPE ``inv_freq`` on a model cast with ``.to(torch.bfloat16)``) stays fp32 for
every later task on the shared model; ``model.float()`` NARROWS float64
tensors, losing their precision for good; and it is gated on a parameter count,
which makes the compute dtype a function of model size.

:func:`fp32_compute` promotes only fp16/bf16 parameters and buffers, never
narrows, restores every one of them (on success and on exception), and reports
what it did.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterator, List, Tuple

import torch

logger = logging.getLogger("blme")

LOW_PRECISION = (torch.float16, torch.bfloat16)


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def floating_dtype_summary(model) -> str:
    """The floating dtype(s) of ``model``'s parameters, e.g. ``"bfloat16"``.

    A model whose floating parameters disagree gives the sorted names joined
    by ``+`` (``"bfloat16+float32"``); a model with none gives ``"none"``.
    """
    seen = sorted({_dtype_name(p.dtype) for p in model.parameters()
                   if p.is_floating_point()})
    return "+".join(seen) if seen else "none"


def _buffer_slots(model) -> List[Tuple[torch.nn.Module, str, torch.Tensor]]:
    """Every (module, name, tensor) buffer slot, shared buffers included."""
    slots = []
    for mod in model.modules():
        for name, buf in mod._buffers.items():
            if buf is not None:
                slots.append((mod, name, buf))
    return slots


@contextlib.contextmanager
def fp32_compute(model, enabled: bool = True) -> Iterator[Dict[str, Any]]:
    """Run the block with every fp16/bf16 parameter and buffer of ``model`` in fp32.

    Only low-precision floating tensors are promoted; float32/float64 and
    non-floating tensors are left alone. Parameters keep their identity (their
    ``.data`` is swapped, so tied weights stay tied); a buffer shared between
    modules stays shared. On exit -- also when the block raises -- every
    promoted tensor is cast back to its original dtype (exact: fp16/bf16 embed
    in fp32). There is no size gate.

    Yields a dict, filled in before the block runs:
      ``original_dtype``     floating parameter dtype(s) on entry
      ``compute_dtype``      floating parameter dtype(s) inside the block
      ``promoted``           whether anything was promoted
      ``n_params_promoted``  parameter tensors promoted
      ``n_buffers_promoted`` buffer tensors promoted (distinct tensors)

    ``enabled=False`` promotes nothing and reports the model as it is.
    """
    info: Dict[str, Any] = {
        "original_dtype": floating_dtype_summary(model),
        "compute_dtype": None,
        "promoted": False,
        "n_params_promoted": 0,
        "n_buffers_promoted": 0,
    }
    saved_params: List[Tuple[torch.nn.Parameter, torch.dtype]] = []
    saved_buffers: List[Tuple[torch.nn.Module, str, torch.dtype, int]] = []
    promoted_buffers: Dict[int, torch.Tensor] = {}
    try:
        if enabled:
            with torch.no_grad():
                for p in model.parameters():
                    if p.dtype in LOW_PRECISION:
                        saved_params.append((p, p.dtype))
                        p.data = p.data.float()
                        if p.grad is not None:
                            p.grad = p.grad.float()
                for mod, name, buf in _buffer_slots(model):
                    if buf.dtype in LOW_PRECISION:
                        key = id(buf)
                        if key not in promoted_buffers:
                            promoted_buffers[key] = buf.float()
                        saved_buffers.append((mod, name, buf.dtype, key))
                        mod._buffers[name] = promoted_buffers[key]
        info["promoted"] = bool(saved_params or saved_buffers)
        info["n_params_promoted"] = len(saved_params)
        info["n_buffers_promoted"] = len(promoted_buffers)
        info["compute_dtype"] = floating_dtype_summary(model)
        if info["promoted"]:
            logger.info(f"  fp32_compute: {info['original_dtype']} -> "
                        f"{info['compute_dtype']} ({len(saved_params)} params, "
                        f"{len(promoted_buffers)} buffers)")
        yield info
    finally:
        with torch.no_grad():
            for p, dt in saved_params:
                p.data = p.data.to(dt)
                # A gradient (kept or created in the block) follows its parameter.
                if p.grad is not None:
                    p.grad = p.grad.to(dt)
            restored: Dict[int, torch.Tensor] = {}
            for mod, name, dt, key in saved_buffers:
                cur = mod._buffers.get(name)
                if cur is None:
                    continue
                if key not in restored:
                    restored[key] = cur.to(dt)
                mod._buffers[name] = restored[key]


def effective_dtype_meta(source, prefix: str = "_meta_") -> Dict[str, Any]:
    """``_meta_`` keys describing the dtype a task actually computed in.

    ``source`` is either the dict yielded by :func:`fp32_compute` or a model
    (for a task that computes in the model's native dtype).
    """
    if isinstance(source, dict):
        info = source
    else:
        d = floating_dtype_summary(source)
        info = {"original_dtype": d, "compute_dtype": d, "promoted": False,
                "n_params_promoted": 0, "n_buffers_promoted": 0}
    return {
        f"{prefix}original_dtype": info["original_dtype"],
        f"{prefix}compute_dtype": info["compute_dtype"],
        f"{prefix}promoted_to_fp32": bool(info["promoted"]),
        f"{prefix}n_params_promoted": int(info["n_params_promoted"]),
        f"{prefix}n_buffers_promoted": int(info["n_buffers_promoted"]),
    }
