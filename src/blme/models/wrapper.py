"""
HuggingFace model loader for BLME.

Supports loading any HuggingFace model with configurable dtype, device mapping,
quantization, attention implementation, and automatic tokenizer setup.

Usage from CLI:
    blme evaluate --model-args pretrained=meta-llama/Llama-2-7b,dtype=bfloat16,device_map=auto

Usage from Python:
    model, tokenizer = load_model_and_tokenizer("pretrained=gpt2,dtype=float16")
"""

import logging
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer

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
    Parse max_memory string like '0:20GiB,1:20GiB,cpu:40GiB'.
    Returns dict for HuggingFace device_map.
    """
    mem = {}
    for part in value.split(","):
        dev, size = part.strip().split(":")
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
        max_memory           - per-device memory limit, e.g. '0:20GiB,cpu:40GiB'

    Args:
        model_args_str: Comma-separated key=value string.
        device: Target device when device_map is not used.

    Returns:
        (model, tokenizer) tuple.
    """
    args = parse_model_args(model_args_str)

    model_name = args.get("pretrained", "gpt2")
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

    # dtype
    dtype_str = args.get("dtype", "auto")
    dtype = _resolve_dtype(dtype_str)
    if dtype == "auto":
        model_kwargs["torch_dtype"] = "auto"
    elif dtype is not None:
        model_kwargs["torch_dtype"] = dtype

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

    # quantization (bitsandbytes)
    if _parse_bool(args.get("load_in_8bit", "false")):
        model_kwargs["load_in_8bit"] = True
        use_device_map = True  # quantized models require device_map
        model_kwargs.setdefault("device_map", "auto")
    elif _parse_bool(args.get("load_in_4bit", "false")):
        model_kwargs["load_in_4bit"] = True
        use_device_map = True
        model_kwargs.setdefault("device_map", "auto")

    # --- Load model ----------------------------------------------------------
    logger.info(f"Loading model: {model_name} (kwargs: {model_kwargs})")

    model = _load_model(model_name, model_kwargs)

    # Move to device if device_map was not used
    if not use_device_map:
        model = model.to(device)

    model.eval()

    try:
        first_param = next(model.parameters())
        logger.info(
            f"Model loaded: {model.__class__.__name__}, "
            f"dtype={first_param.dtype}, "
            f"device={first_param.device}"
        )
    except StopIteration:
        logger.info(f"Model loaded: {model.__class__.__name__}")

    return model, tokenizer


def _load_model(model_name: str, model_kwargs: Dict[str, Any]):
    """
    Try loading as CausalLM first (GPT2, LLaMA, etc.),
    fall back to AutoModel (BERT, encoders).
    """
    from transformers import AutoModelForCausalLM, AutoModel

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("Loaded as AutoModelForCausalLM")
        return model
    except (ValueError, OSError) as e:
        logger.info(f"CausalLM failed ({e}), trying AutoModel...")

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    logger.info("Loaded as AutoModel (encoder-style)")
    return model
