"""The loader's dtype contract, and the precision provenance of a loaded model.

When ``dtype`` is requested explicitly, ``from_pretrained`` can still return
another one. On transformers 5.10, ``AutoModelForCausalLM.from_pretrained(path,
dtype=torch.float32)`` on Qwen3.5 returns bfloat16 parameters:
``PreTrainedConfig.from_dict`` consumes the ``dtype`` keyword into the
top-level (composite) config and drops it from the keywords it hands back; the
auto class then swaps in the text sub-config, whose own ``dtype`` is bfloat16
in ``config.json``, and the model's ``from_pretrained``, called without a
``dtype``, defaults to ``"auto"`` and reads that. transformers 5.2 keeps the
keyword and honours it. The causal-LM classes built on a text sub-config this
way are emu3, llama4, mllama, qwen3_5 and qwen3_5_moe.

The contract (``load_model_and_tokenizer``):

* Only floating tensors of 16 bits or more are checked: integer, boolean and
  8-bit float tensors (quantized storage, token ids, masks) never are, and a
  quantized model is recorded but neither reloaded nor cast.
* A tensor honours the request when its dtype is the requested one or wider.
  transformers' own mixed precision only ever widens: ``_keep_in_fp32_modules``
  (``_strict``) parameters, and buffers built in float32 whatever the default
  dtype, such as RoPE ``inv_freq`` and ``original_inv_freq``. A float16 load
  keeps those float32, as does a bfloat16 one.
* It is violated by a tensor in a narrower dtype or in the other 16-bit format,
  or, for a 16-bit request, by float32 parameters holding most of the
  parameter elements outside transformers' keep-in-fp32 plan.
* On a violation the model is loaded again from a config whose ``dtype``, and
  every sub-config's, is the requested one (the config object is passed in, so
  the keyword is never consumed). On Qwen3.5 and transformers 5.10 this load
  equals transformers 5.2's own float32 load bit for bit. Tensors still
  narrower or in the other 16-bit format are then cast in place; nothing is
  ever narrowed from a wider dtype the reload kept.

A model whose loader honours the request is never reloaded or cast.
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger("blme")

# The record load_model_and_tokenizer leaves on the model (copied into the
# results envelope's config.precision by core.evaluate).
PRECISION_ATTR = "_blme_precision"

HALF_DTYPES = (torch.float16, torch.bfloat16)
# Mantissa-and-range ordering for the contract: the two 16-bit formats are
# different, not narrower than each other.
_WIDTH = {torch.float16: 16, torch.bfloat16: 16, torch.float32: 32, torch.float64: 64}
_SAFETENSORS_DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
_EXAMPLES = 5


def dtype_name(dtype) -> Optional[str]:
    """``"float32"`` for ``torch.float32`` (and for its string forms); None for None."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16", "half": "float16",
                   "float": "float32"}
        s = dtype.lower().replace("torch.", "")
        return aliases.get(s, s)
    return str(dtype).replace("torch.", "")


def _checked(t: torch.Tensor) -> bool:
    """Floating tensors of 16 bits or more (8-bit floats are quantized storage)."""
    try:
        return bool(t.is_floating_point()) and t.element_size() >= 2
    except Exception:  # noqa: BLE001 - foreign tensor subclasses
        return False


def _named_tensors(model) -> Iterable[Tuple[str, str, torch.Tensor]]:
    """``(kind, name, tensor)`` for every checked parameter and buffer."""
    try:
        params = list(model.named_parameters())
        buffers = list(model.named_buffers())
    except Exception:  # noqa: BLE001 - not an nn.Module (a test double)
        return []
    out = [("parameter", n, p) for n, p in params if isinstance(p, torch.Tensor) and _checked(p)]
    out += [("buffer", n, b) for n, b in buffers if isinstance(b, torch.Tensor) and _checked(b)]
    return out


def floating_dtypes(model, kind: str = "parameter") -> List[str]:
    """Sorted names of the floating dtypes of the model's parameters (or buffers)."""
    return sorted({dtype_name(t.dtype) for k, _, t in _named_tensors(model) if k == kind})


def dominant_param_dtype(model) -> Optional[torch.dtype]:
    """The floating dtype holding the most parameter elements."""
    count: Dict[torch.dtype, int] = {}
    for kind, _, t in _named_tensors(model):
        if kind == "parameter":
            count[t.dtype] = count.get(t.dtype, 0) + t.numel()
    return max(count, key=count.get) if count else None


def is_quantized(model) -> bool:
    cfg = getattr(model, "config", None)
    return bool(getattr(model, "hf_quantizer", None) is not None
                or getattr(model, "is_quantized", False) is True
                or (cfg is not None and getattr(cfg, "quantization_config", None) is not None))


def _keep_fp32_matcher(model, dtype: torch.dtype):
    """A regex for transformers' keep-in-fp32 plan at ``dtype`` (as its loader
    matches it: the plan's globs, searched in each name), or None."""
    try:
        plan = model._get_dtype_plan(dtype)
    except Exception:  # noqa: BLE001 - no plan on this class / version
        return None
    if not plan:
        return None
    return re.compile("|".join(f"(?:{str(g).replace('*', '.*')})" for g in plan))


def _violates(t_dtype: torch.dtype, requested: torch.dtype) -> bool:
    """Narrower than requested, or the other 16-bit format."""
    if t_dtype == requested:
        return False
    w_t, w_r = _WIDTH.get(t_dtype), _WIDTH.get(requested)
    if w_t is None or w_r is None:
        return False
    return w_t <= w_r


def find_violations(model, requested: torch.dtype) -> Optional[Dict[str, Any]]:
    """What in ``model`` breaks the contract for ``requested``, or None.

    ``{"tensors": n, "dtypes": [...], "examples": [...], "wider_bulk": bool}``:
    ``tensors`` counts tensors narrower than requested or in the other 16-bit
    format; ``wider_bulk`` is set when, for a 16-bit request, wider parameters
    outside the keep-in-fp32 plan hold most of the parameter elements (the
    request was ignored the other way).
    """
    bad = [(kind, name, t.dtype) for kind, name, t in _named_tensors(model)
           if _violates(t.dtype, requested)]
    wider_bulk = False
    if requested in HALF_DTYPES:
        keep = _keep_fp32_matcher(model, requested)
        total = wide = 0
        for kind, name, t in _named_tensors(model):
            if kind != "parameter":
                continue
            total += t.numel()
            if _WIDTH.get(t.dtype, 0) > _WIDTH[requested] and not (keep is not None and keep.search(name)):
                wide += t.numel()
        wider_bulk = total > 0 and 2 * wide > total
    if not bad and not wider_bulk:
        return None
    return {
        "tensors": len(bad),
        "dtypes": sorted({dtype_name(d) for _, _, d in bad}),
        "examples": [f"{kind} {name}: {dtype_name(d)}" for kind, name, d in bad[:_EXAMPLES]],
        "wider_bulk": bool(wider_bulk),
    }


def is_rotary_buffer(module, key: str) -> bool:
    """A rotary-embedding buffer (``inv_freq``, ``original_inv_freq``, cos/sin
    caches): transformers builds these in float32 whatever the model's dtype,
    and casting them changes predictions (pythia-70m: top-1 changed at 32% of
    positions with inv_freq in bfloat16)."""
    return "inv_freq" in key or "rotary" in type(module).__name__.lower()


def cast_violations(model, requested: torch.dtype) -> Dict[str, int]:
    """Cast every floating parameter, and every buffer transformers would have
    built in the requested dtype, that is narrower than ``requested`` or in the
    other 16-bit format, to ``requested``, in place. Wider tensors are kept, and
    rotary buffers (``is_rotary_buffer``) are never cast: ``rotary_kept`` counts
    any that are in a violating dtype, left as loaded.

    A parameter keeps its object (``.data`` is replaced), so tied weights stay
    tied; a buffer shared by several modules is replaced by one new tensor.
    """
    n_param = n_buf = n_rotary = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            if _checked(p) and _violates(p.dtype, requested):
                p.data = p.data.to(requested)
                n_param += 1
        new: Dict[int, torch.Tensor] = {}
        kept = set()
        for module in model.modules():
            for key, b in list(module._buffers.items()):
                if b is None or not _checked(b) or not _violates(b.dtype, requested):
                    continue
                if is_rotary_buffer(module, key):
                    kept.add(id(b))
                    continue
                if id(b) not in new:
                    new[id(b)] = b.to(requested)
                    n_buf += 1
                module._buffers[key] = new[id(b)]
        n_rotary = len(kept)
    return {"parameters": n_param, "buffers": n_buf, "rotary_kept": n_rotary}


def pin_config_dtype(config, dtype: torch.dtype, _seen=None) -> None:
    """Set ``dtype`` on ``config`` and, recursively, on every sub-config."""
    _seen = set() if _seen is None else _seen
    if config is None or id(config) in _seen:
        return
    _seen.add(id(config))
    config.dtype = dtype
    for key in getattr(config, "sub_configs", None) or {}:
        sub = getattr(config, key, None)
        if sub is not None and hasattr(sub, "to_dict"):
            pin_config_dtype(sub, dtype, _seen)


def config_with_dtype(model_name: str, model_kwargs: Dict[str, Any], dtype: torch.dtype):
    """The checkpoint's config with ``dtype`` pinned on it and every sub-config."""
    from transformers import AutoConfig
    cfg_kwargs = {k: model_kwargs[k] for k in ("revision", "trust_remote_code", "token") if k in model_kwargs}
    config = AutoConfig.from_pretrained(model_name, **cfg_kwargs)
    pin_config_dtype(config, dtype)
    return config


# ── the checkpoint's own dtype ─────────────────────────────────────────────

def _safetensors_dtype(path: str) -> Optional[str]:
    """The floating dtype holding the most elements in a safetensors header."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    count: Dict[str, int] = {}
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        name = _SAFETENSORS_DTYPES.get(entry.get("dtype"))
        if name is None:
            continue
        numel = 1
        for s in entry.get("shape", []):
            numel *= int(s)
        count[name] = count.get(name, 0) + numel
    return max(count, key=count.get) if count else None


def _checkpoint_file(model_name: str, model_kwargs: Dict[str, Any]) -> Optional[str]:
    """A local safetensors file of the checkpoint (the first shard), or None."""
    def _local(name):
        if os.path.isdir(model_name):
            p = os.path.join(model_name, name)
            return p if os.path.isfile(p) else None
        from transformers.utils import cached_file
        return cached_file(model_name, name, revision=model_kwargs.get("revision"),
                           _raise_exceptions_for_missing_entries=False,
                           _raise_exceptions_for_connection_errors=False)

    single = _local("model.safetensors")
    if single:
        return single
    index = _local("model.safetensors.index.json")
    if not index:
        return None
    with open(index) as fh:
        shards = sorted(set(json.load(fh).get("weight_map", {}).values()))
    return _local(shards[0]) if shards else None


def native_dtype(model_name: str, model_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(dtype, source)``: the checkpoint's dtype from its config (``dtype`` /
    ``torch_dtype``, or the text sub-config's), else the dtype holding most
    elements in its safetensors header; ``(None, None)`` when neither is found.
    Read from the files, because ``from_pretrained`` overwrites ``config.dtype``
    with the dtype it loaded."""
    model_kwargs = model_kwargs or {}
    try:
        from transformers import AutoConfig
        cfg_kwargs = {k: model_kwargs[k] for k in ("revision", "trust_remote_code", "token") if k in model_kwargs}
        cfg = AutoConfig.from_pretrained(model_name, **cfg_kwargs)
        dt = getattr(cfg, "dtype", None)
        if dt is None:
            try:
                dt = getattr(cfg.get_text_config(), "dtype", None)
            except Exception:  # noqa: BLE001 - configs without a text config
                dt = None
        if dt is not None:
            return dtype_name(dt), "config"
    except Exception as e:  # noqa: BLE001 - provenance never breaks a load
        logger.debug(f"native dtype: config unreadable ({e})")
    try:
        path = _checkpoint_file(model_name, model_kwargs)
        if path:
            dt = _safetensors_dtype(path)
            if dt is not None:
                return dt, "checkpoint"
    except Exception as e:  # noqa: BLE001
        logger.debug(f"native dtype: checkpoint header unreadable ({e})")
    return None, None


def native_dtype_hints(running: Optional[str], native: Optional[str]) -> List[str]:
    """Warnings for arithmetic in a dtype that may lose precision against the
    checkpoint's own. Only a hint: a bfloat16-native checkpoint can still
    degrade in bfloat16, which only ``blme precision-check`` measures."""
    if running is None or native is None:
        return []
    if running == "bfloat16" and native == "float16":
        return ["running bfloat16 on a float16-native checkpoint: bfloat16 keeps 3 fewer mantissa "
                "bits than the weights were trained in. Measure the loss with `blme precision-check` "
                "and run float32 if bfloat16 - float32 next-token NLL exceeds 0.02 nats."]
    if running in ("float16", "bfloat16") and native == "float32":
        return [f"running {running} on a float32-native checkpoint. Measure the loss with "
                f"`blme precision-check --dtype {running}` and run float32 if {running} - float32 "
                f"next-token NLL exceeds 0.02 nats."]
    return []


def precision_provenance(model, model_args: str, eager_cast: bool = False) -> Optional[Dict[str, Any]]:
    """The envelope's ``config.precision``: what was requested, what the
    checkpoint is, what was loaded, any cast BLME applied, and what the tasks
    ran with. Starts from the record ``load_model_and_tokenizer`` left on the
    model; a model loaded some other way gets the fields it can still show."""
    if model is None:
        return None
    rec = dict(getattr(model, PRECISION_ATTR, None) or {})
    if not rec:
        from .wrapper import parse_model_args
        rec = {"requested_dtype": dtype_name(parse_model_args(model_args).get("dtype", "auto")),
               "native_dtype": None, "native_dtype_source": None,
               "loaded_param_dtypes": floating_dtypes(model, "parameter"),
               "loaded_buffer_dtypes": floating_dtypes(model, "buffer"),
               "dtype_contract": "not recorded (model not loaded by blme)",
               "load_violation": None, "reloaded": False, "cast_applied": False, "casts": [],
               "warnings": []}
    rec["casts"] = list(rec.get("casts", []))
    rec["warnings"] = list(rec.get("warnings", []))
    if eager_cast:
        rec["casts"].append("eager attention: float16 parameters -> bfloat16 (buffers kept)")
        rec["cast_applied"] = True
    rec.setdefault("attn_score_dtype", "model")
    rec["attn_implementation"] = getattr(getattr(model, "config", None), "_attn_implementation", None)
    rec["effective_param_dtypes"] = floating_dtypes(model, "parameter")
    running = dtype_name(dominant_param_dtype(model))
    rec["effective_dtype"] = running
    for w in native_dtype_hints(running, rec.get("native_dtype")):
        if w not in rec["warnings"]:
            logger.warning(f"{type(model).__name__}: {w}")
            rec["warnings"].append(w)
    return rec
