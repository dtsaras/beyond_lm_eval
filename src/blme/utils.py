"""Shared utilities for BLME."""

import logging
import sys


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("blme")


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isolated_torch_rng(model=None):
    """A block that may reseed torch's global RNG (``generate()`` takes no
    generator) without the reset reaching the code that runs after it: the
    CPU state and that of every CUDA device holding ``model`` are restored."""
    import torch

    devices = set()
    for p in (model.parameters() if model is not None else ()):
        if p.device.type == "cuda":
            devices.add(p.device.index if p.device.index is not None
                        else torch.cuda.current_device())
    return torch.random.fork_rng(devices=sorted(devices))
