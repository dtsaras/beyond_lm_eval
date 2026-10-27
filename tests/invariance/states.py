"""Per-document hidden states in the layout the shared cache serves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


@dataclass
class States:
    """Hidden states of one model on N documents.

    blocks[l][i]   (T_i, d) output of block l on document i, l = 0..L-1.
                   blocks[L-1] is the RAW deepest block output (before the
                   final norm), which the cache serves on replace_final_norm.
    final_norm[i]  HF's last hidden_states entry, final_norm(blocks[L-1][i]).
                   None: the deepest block output is served in its place.
    embedding[i]   hidden_states[0], the embedding output. Optional.
    """

    blocks: Dict[int, List[Array]]
    final_norm: Optional[List[Array]] = None
    embedding: Optional[List[Array]] = None

    def __post_init__(self):
        keys = sorted(self.blocks)
        if keys != list(range(len(keys))) or not keys:
            raise ValueError(f"block keys must be 0..L-1, got {keys}")
        lengths = self.lengths
        width = self.d
        for name, docs in self._named_lists():
            if len(docs) != len(lengths):
                raise ValueError(f"{name}: {len(docs)} documents, expected {len(lengths)}")
            for i, x in enumerate(docs):
                if x.ndim != 2 or x.shape[0] != lengths[i] or x.shape[1] != width:
                    raise ValueError(
                        f"{name}[{i}] has shape {tuple(x.shape)}, expected ({lengths[i]}, {width})")

    def _named_lists(self):
        for k in sorted(self.blocks):
            yield f"blocks[{k}]", self.blocks[k]
        if self.final_norm is not None:
            yield "final_norm", self.final_norm
        if self.embedding is not None:
            yield "embedding", self.embedding

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @property
    def n_docs(self) -> int:
        return len(self.blocks[0])

    @property
    def lengths(self) -> List[int]:
        return [int(x.shape[0]) for x in self.blocks[0]]

    @property
    def d(self) -> int:
        return int(self.blocks[0][0].shape[1])

    def map(self, fn: Callable[[Array], Array]) -> "States":
        """Apply ``fn`` to every array (blocks, final norm, embedding)."""
        return States(
            blocks={k: [fn(x) for x in v] for k, v in self.blocks.items()},
            final_norm=None if self.final_norm is None else [fn(x) for x in self.final_norm],
            embedding=None if self.embedding is None else [fn(x) for x in self.embedding],
        )

    def select_docs(self, idx: Sequence[int]) -> "States":
        idx = [int(i) for i in idx]
        pick = (lambda docs: [docs[i] for i in idx])
        return States(
            blocks={k: pick(v) for k, v in self.blocks.items()},
            final_norm=None if self.final_norm is None else pick(self.final_norm),
            embedding=None if self.embedding is None else pick(self.embedding),
        )

    def with_blocks(self, blocks: Dict[int, List[Array]]) -> "States":
        return States(blocks=blocks, final_norm=self.final_norm, embedding=self.embedding)


def as_states(obj, final_norm=None, embedding=None) -> States:
    """Accept a ``States`` or a ``{layer: [doc arrays]}`` dict."""
    if isinstance(obj, States):
        if final_norm is not None or embedding is not None:
            return States(blocks=obj.blocks,
                          final_norm=final_norm if final_norm is not None else obj.final_norm,
                          embedding=embedding if embedding is not None else obj.embedding)
        return obj
    if isinstance(obj, dict):
        return States(blocks={int(k): list(v) for k, v in obj.items()},
                      final_norm=final_norm, embedding=embedding)
    raise TypeError(f"expected States or dict of per-document lists, got {type(obj).__name__}")


def states_from_model(model, tokenizer, texts: Sequence[str],
                      threads: Optional[int] = 1) -> States:
    """Collect ``States`` with the library's own ``ModelOutputCache``.

    Same tokenisation, truncation, embedding capture and deepest-block hook as
    a production run, so injected copies are directly comparable. Runs with
    ``threads`` threads (default 1) so repeated collections are
    bit-reproducible (see ``_threads``).
    """
    from blme.cache import ModelOutputCache

    from ._threads import limit_threads

    cache = ModelOutputCache(model, tokenizer, dataset=[{"text": t} for t in texts],
                             num_samples=len(texts))
    with limit_threads(threads):
        cache.populate(need_hidden=True, need_attentions=False, need_logits=False)
    if not cache._hidden_states:
        raise RuntimeError("cache collected no hidden states")
    if cache._last_block_raw is None:
        raise RuntimeError("the deepest-block hook did not fire; layout would be ambiguous")

    def split(t):
        return [c.numpy() for c in cache._split_by_samples(t, None)]

    L = len(cache._hidden_states)
    blocks = {li: split(cache._hidden_states[li]) for li in range(L - 1)}
    blocks[L - 1] = split(cache._last_block_raw)
    return States(
        blocks=blocks,
        final_norm=split(cache._hidden_states[L - 1]),
        embedding=None if cache._embedding_states is None else split(cache._embedding_states),
    )
