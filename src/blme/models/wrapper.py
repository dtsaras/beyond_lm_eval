"""
HuggingFace model loader for BLME.

Supports loading any HuggingFace model with configurable dtype, device mapping,
quantization, attention implementation, and automatic tokenizer setup.

Usage from CLI:
    blme evaluate --model-args pretrained=meta-llama/Llama-2-7b,dtype=bfloat16,device_map=auto

Usage from Python:
    model, tokenizer = load_model_and_tokenizer("pretrained=gpt2,dtype=float16")
"""

import gc
import logging
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer

from ..experts import ensure_runnable_experts
from . import dtype_contract as _dc

logger = logging.getLogger("blme")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_model_args(model_args_str: str) -> Dict[str, str]:
    """
    Parse a comma-separated key=value string into a dictionary.

    Example:
        "pretrained=gpt2,dtype=float16,trust_remote_code=true"
        -> {"pretrained": "gpt2", "dtype": "float16", "trust_remote_code": "true"}
    """
    args: Dict[str, str] = {}
    if not model_args_str:
        return args
    for pair in model_args_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            args[k.strip()] = v.strip()
    return args


def _parse_bool(value: str) -> bool:
    """Parse string booleans like 'true', 'false', '1', '0'."""
    return value.lower() in ("true", "1", "yes")


def _resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    """Map string dtype to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    result = dtype_map.get(dtype_str.lower())
    if result is None:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. Supported: {list(dtype_map.keys())}"
        )
    return result


def _parse_max_memory(value: str) -> Dict[Any, str]:
    """
    Parse a max_memory string into a HuggingFace device_map dict.

    Multiple devices must be separated by ';' (e.g. '0:20GiB;cpu:40GiB'),
    because parse_model_args splits the whole model_args string on ',' first —
    a comma-separated max_memory would be silently truncated to its first
    device. Commas are still accepted here for the single-token case.
    """
    import re
    mem = {}
    for part in re.split(r"[;,]", value):
        part = part.strip()
        if not part:
            continue
        dev, size = part.split(":")
        try:
            dev_key = int(dev)
        except ValueError:
            dev_key = dev  # 'cpu' stays as string
        mem[dev_key] = size
    return mem


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_args_str: str,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace model and tokenizer from a model_args string.

    Supported model_args keys:
        pretrained          - HF model ID or local path (required)
        dtype               - float16 | bfloat16 | float32 | auto (default: auto)
        device_map          - auto | balanced | sequential | specific device
        trust_remote_code   - true | false (default: false)
        attn_implementation - eager | sdpa | flash_attention_2
        revision            - model revision / branch
        load_in_8bit        - true | false (bitsandbytes 8-bit)
        load_in_4bit        - true | false (bitsandbytes 4-bit)
        max_memory           - per-device memory limit; separate devices with
                               ';' e.g. '0:20GiB;cpu:40GiB' (commas are consumed
                               by the model_args parser)

    Args:
        model_args_str: Comma-separated key=value string.
        device: Target device when device_map is not used.

    Returns:
        (model, tokenizer) tuple.
    """
    args = parse_model_args(model_args_str)

    if "pretrained" not in args:
        if model_args_str:
            # A bare model name (e.g. "meta-llama/Llama-3-8B") or a typo has no
            # '=', so parse_model_args drops it and we would silently load gpt2 —
            # evaluating the wrong model. Fail loudly instead.
            raise ValueError(
                f"model_args={model_args_str!r} does not specify 'pretrained='. "
                "Pass 'pretrained=<hf-id-or-path>' (a bare model name is ignored)."
            )
        model_name = "gpt2"  # explicit no-args default (demos/tests)
    else:
        model_name = args["pretrained"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Tokenizer -----------------------------------------------------------
    tokenizer_kwargs = {}
    if _parse_bool(args.get("trust_remote_code", "false")):
        tokenizer_kwargs["trust_remote_code"] = True
    if "revision" in args:
        tokenizer_kwargs["revision"] = args["revision"]

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Ensure pad_token is set (critical for batched inputs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = tokenizer.bos_token
            tokenizer.pad_token_id = tokenizer.bos_token_id

    # --- Model kwargs --------------------------------------------------------
    model_kwargs: Dict[str, Any] = {}

    # dtype  (transformers v5 renamed torch_dtype= -> dtype=; the old name
    # prints a deprecation warning on every load)
    dtype_str = args.get("dtype", "auto")
    dtype = _resolve_dtype(dtype_str)
    if dtype == "auto":
        model_kwargs["dtype"] = "auto"
    elif dtype is not None:
        model_kwargs["dtype"] = dtype

    # trust_remote_code
    if _parse_bool(args.get("trust_remote_code", "false")):
        model_kwargs["trust_remote_code"] = True

    # revision
    if "revision" in args:
        model_kwargs["revision"] = args["revision"]

    # attention implementation
    if "attn_implementation" in args:
        model_kwargs["attn_implementation"] = args["attn_implementation"]

    # device_map (takes priority over --device for placement)
    use_device_map = "device_map" in args
    if use_device_map:
        model_kwargs["device_map"] = args["device_map"]

    # max_memory
    if "max_memory" in args:
        model_kwargs["max_memory"] = _parse_max_memory(args["max_memory"])

    # quantization (bitsandbytes). Passing load_in_8bit=/load_in_4bit= directly
    # to from_pretrained was removed in transformers v5 (raises TypeError); they
    # must be wrapped in a BitsAndBytesConfig passed as quantization_config=.
    quant_8bit = _parse_bool(args.get("load_in_8bit", "false"))
    quant_4bit = _parse_bool(args.get("load_in_4bit", "false"))
    if quant_8bit or quant_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quant_8bit,
            load_in_4bit=quant_4bit and not quant_8bit,  # 8-bit takes priority
        )
        use_device_map = True  # quantized models require device_map
        model_kwargs.setdefault("device_map", "auto")

    # --- Load model ----------------------------------------------------------
    logger.info(f"Loading model: {model_name} (kwargs: {model_kwargs})")

    model, record = _load_with_dtype_contract(model_name, model_kwargs, dtype, dtype_str,
                                              quantized=bool(quant_8bit or quant_4bit))

    # Move to device if device_map was not used
    if not use_device_map:
        model = model.to(device)

    model.eval()

    # MoE experts on a kernel this torch cannot run on CPU fall back to eager.
    ensure_runnable_experts(model)

    try:
        first_param = next(model.parameters())
        logger.info(
            f"Model loaded: {model.__class__.__name__}, "
            f"dtype={first_param.dtype}, "
            f"device={first_param.device}"
        )
    except StopIteration:
        logger.info(f"Model loaded: {model.__class__.__name__}")

    record["loaded_param_dtypes"] = _dc.floating_dtypes(model, "parameter")
    record["loaded_buffer_dtypes"] = _dc.floating_dtypes(model, "buffer")
    loaded = _dc.dtype_name(_dc.dominant_param_dtype(model))
    for w in _dc.native_dtype_hints(loaded, record.get("native_dtype")):
        logger.warning(f"{model.__class__.__name__}: {w}")
        record["warnings"].append(w)
    try:
        setattr(model, _dc.PRECISION_ATTR, record)
    except Exception:  # noqa: BLE001 - a model that refuses attributes keeps no record
        pass
    return model, tokenizer


def _load_with_dtype_contract(model_name: str, model_kwargs: Dict[str, Any], dtype, dtype_str: str,
                              quantized: bool):
    """Load the model under the dtype contract (``models/dtype_contract.py``):
    ``(model, record)``.

    With an explicit ``dtype``, a model whose floating parameters or buffers
    are narrower than requested (or in the other 16-bit format), or mostly
    wider for a 16-bit request, is loaded again from a config with the dtype
    pinned on it and every sub-config; what is then still narrower is cast.
    A model whose loader honoured the request is returned as loaded.
    """
    model = _load_model(model_name, model_kwargs)
    native, native_src = _dc.native_dtype(model_name, model_kwargs)
    record: Dict[str, Any] = {
        "requested_dtype": _dc.dtype_name(dtype_str),
        "native_dtype": native,
        "native_dtype_source": native_src,
        "dtype_contract": None,
        "load_violation": None,
        "reloaded": False,
        "cast_applied": False,
        "casts": [],
        "warnings": [],
    }
    if not isinstance(dtype, torch.dtype):
        record["dtype_contract"] = "not checked: no explicit dtype requested"
        return model, record
    found = _dc.find_violations(model, dtype)
    if quantized or _dc.is_quantized(model):
        record["dtype_contract"] = "not checked: quantized model"
        record["load_violation"] = found
        return model, record
    if found is None:
        record["dtype_contract"] = "honoured"
        return model, record

    cls = model.__class__.__name__
    want = _dc.dtype_name(dtype)
    what = (f"{found['tensors']} tensor(s) in {', '.join(found['dtypes'])}" if found["tensors"]
            else "most parameters in a wider dtype")
    example = f" (e.g. {'; '.join(found['examples'])})" if found["examples"] else ""
    logger.warning(f"{cls}: dtype={want} was requested but the loader returned {what}{example}. "
                   f"Loading again with dtype={want} pinned on the config and every sub-config.")
    record["load_violation"] = found
    try:
        config = _dc.config_with_dtype(model_name, model_kwargs, dtype)
    except Exception as e:  # noqa: BLE001 - no readable config: cast what was loaded
        logger.warning(f"{cls}: could not read the config to load again ({e}); casting instead.")
        config = None
    if config is not None:
        model = None        # the only reference: free it before the second load
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = _load_model(model_name, {**model_kwargs, "config": config})
        record["reloaded"] = True
        cls = model.__class__.__name__
        found = _dc.find_violations(model, dtype)
        if found is None:
            record["dtype_contract"] = "honoured after reload"
            logger.warning(f"{cls}: dtype={want} honoured after the reload.")
            return model, record
    steps = []
    if found["tensors"]:
        n = _dc.cast_violations(model, dtype)
        msg = (f"dtype contract: {n['parameters']} parameter(s) and {n['buffers']} buffer(s) in "
               f"{', '.join(found['dtypes'])} cast to {want}")
        logger.warning(f"{cls}: {msg}; wider tensors kept.")
        record["casts"].append(msg)
        record["cast_applied"] = True
        steps.append("cast")
    if found["wider_bulk"]:
        w = (f"dtype={want} was requested but most parameters are in a wider dtype as loaded; "
             f"kept (BLME never narrows a tensor the loader widened).")
        logger.warning(f"{cls}: {w}")
        record["warnings"].append(w)
        steps.append("wider parameters kept")
    record["dtype_contract"] = ("reloaded, then " if record["reloaded"] else "") + ", ".join(steps)
    return model, record


def _load_image_text_to_text(model_name: str, model_kwargs: Dict[str, Any]):
    """The image-text-to-text model for a checkpoint registered there, else None."""
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    except ImportError:
        return None
    config_kwargs = {k: model_kwargs[k] for k in ("revision", "trust_remote_code", "token") if k in model_kwargs}
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)
    if getattr(config, "model_type", None) not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return None
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    logger.info("Loaded as AutoModelForImageTextToText (multimodal checkpoint; the text path is evaluated)")
    return model


def _load_model(model_name: str, model_kwargs: Dict[str, Any]):
    """
    Try loading as CausalLM first (GPT2, LLaMA, etc.); then, for a multimodal checkpoint the causal-LM
    mapping does not cover, its image-text-to-text class; then AutoModel (BERT, encoders).
    """
    from transformers import AutoModelForCausalLM, AutoModel

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("Loaded as AutoModelForCausalLM")
        return model
    except (ValueError, OSError) as e:
        logger.info(f"CausalLM failed ({e}), trying AutoModel...")

    # A multimodal checkpoint whose text model transformers does not register as a causal LM (Mistral 3 /
    # Ministral 3 in transformers 5.10, unlike Gemma 3/4) loads through its image-text-to-text class, which
    # carries the LM head; AutoModel would return the headless backbone (no logits, no generate).
    model = _load_image_text_to_text(model_name, model_kwargs)
    if model is not None:
        return model

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    logger.info("Loaded as AutoModel (encoder-style)")
    return model
