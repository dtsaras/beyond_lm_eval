"""Fixed thread counts for bit-reproducible comparisons.

A multi-threaded BLAS reduction's order depends on the thread count it gets:
gpt2 logits move by 7.6e-5 between 1 and 2 torch threads, and on a loaded host
the effective count can change between two calls. Exact comparisons therefore
run at a fixed count (torch, plus numpy's BLAS via threadpoolctl when present).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Optional

import torch


@contextmanager
def limit_threads(n: Optional[int]):
    """Temporarily cap torch and BLAS threads at ``n`` (None: leave them)."""
    if n is None:
        yield
        return
    old = torch.get_num_threads()
    with ExitStack() as stack:
        try:
            from threadpoolctl import threadpool_limits
            stack.enter_context(threadpool_limits(limits=n))
        except ImportError:
            pass
        torch.set_num_threads(n)
        try:
            yield
        finally:
            torch.set_num_threads(old)
