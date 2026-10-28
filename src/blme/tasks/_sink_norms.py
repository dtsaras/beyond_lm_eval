"""Sink keep masks for tasks that run their own forward passes.

Cache-fed tasks hold every layer of every document and call
:func:`~blme.tasks._tokens.middle_depth_layers` and
:func:`~blme.tasks._tokens.sink_keep_masks` directly. A task that runs its own
forward pass sees one document at a time, and the massive-row threshold is a
median over the whole corpus, so no mask can be built until the last document
has been seen. :class:`SinkNormCollector` keeps what the mask needs and nothing
more: the row norms at the middle depths (``sink_keep_masks`` reads nothing
else), plus one middle layer for the share diagnostics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from ._tokens import middle_depth_layers, sink_keep_masks


class SinkNormCollector:
    """Accumulate per-document block outputs; build the sink masks once.

    ``add`` takes one document's block outputs in depth order (HF
    ``hidden_states[1:]``: ``(1, T, D)`` or ``(T, D)`` tensors). The middle
    depths are chosen from the number of entries, as
    ``middle_depth_layers`` does for a cache-fed task, and each row's float64
    norm is stored as a ``(T, 1)`` column: its norm is the row norm, so
    ``sink_keep_masks`` on the columns returns the masks it returns on the
    full rows (tests/tasks/invariance/test_sink_adoption.py). ``mid_docs``
    keeps the full rows of depth ``L // 2`` -- the layer the cache-fed tasks
    report their shares on.
    """

    def __init__(self) -> None:
        self._n_depths = None
        self.keys: List[int] = []
        self.mid: int = -1
        self._norms: Dict[int, List[torch.Tensor]] = {}
        self.mid_docs: List[torch.Tensor] = []

    def add(self, block_outputs: Sequence[torch.Tensor]) -> None:
        hs = [h[0] if h.ndim == 3 else h for h in block_outputs]
        if self._n_depths is None:
            self._n_depths = len(hs)
            self.keys = middle_depth_layers(dict.fromkeys(range(len(hs))))
            self.mid = len(hs) // 2
        elif len(hs) != self._n_depths:
            raise ValueError(f"document has {len(hs)} depths, the first had {self._n_depths}")
        for k in self.keys:
            n = torch.linalg.vector_norm(hs[k].detach().to("cpu", torch.float64), dim=1)
            self._norms.setdefault(k, []).append(n.unsqueeze(1))
        self.mid_docs.append(hs[self.mid].detach().cpu())

    @property
    def n_docs(self) -> int:
        return len(self.mid_docs)

    def masks(self) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """``sink_keep_masks`` over the middle depths of every document added."""
        if not self.n_docs:
            return [], {"n_massive_dropped": 0, "n_rows_pos_gt0": 0, "n_depths": 0}
        return sink_keep_masks([self._norms[k] for k in self.keys])
