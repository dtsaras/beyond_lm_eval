"""One point per sequence, plus a sink-masked token cloud for ``_meta_``.

The sequence-level estimators (intrinsic_dim, lid, vendi_score, magnitude,
correlation_dimension) take one point per document: the hidden state of its
last content token (``select_rows(policy="last")`` on the whole document, no
token cap). Pooling every token of a document into the cloud makes a token's
nearest neighbour another token of the same document 28-83% of the time,
which the text-LLM precedents avoid by reducing each sequence to one point
(Cheng et al. 2024; Yin et al. 2024). The point count N is pinned per task.

The token-cloud value each task used to report is kept as ``_meta_``. Those
clouds follow the sink policy: position 0 and every massive row (norm > 10x
the median at any middle depth, Sun et al. 2024) are dropped, with one mask
per document shared by every layer.

:func:`sequence_clouds` reads both from the shared cache when it holds at
least N documents and otherwise runs one forward pass per document, keeping
only the last row of each layer for the N documents and the token rows of the
first ``token_docs`` documents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .._tokens import (DocList, hidden_states_for_task, massive_token_share,
                       middle_depth_layers, position0_share, select_rows,
                       sink_keep_masks, sink_meta, strip_trailing_special)

logger = logging.getLogger("blme")

# Q1: one point per sequence, the same N for every model.
DEFAULT_N_SEQUENCES = 1000
# Documents whose token rows make the ``_meta_`` token cloud (the study's
# previous 200-document cloud).
DEFAULT_TOKEN_DOCS = 200


# TOKEN_POLICY -> the select_rows readout of one point per sequence.
_READOUTS = {"last_per_sequence": "last"}
# The policy of the _meta_ token clouds (non-LLM-origin point clouds).
TOKEN_CLOUD_POLICY = "exclude_sinks"


def readout_for(token_policy: str) -> str:
    """The per-sequence readout a task's ``TOKEN_POLICY`` names."""
    if token_policy not in _READOUTS:
        raise ValueError(f"sequence_clouds supports TOKEN_POLICY in {sorted(_READOUTS)}, "
                         f"got {token_policy!r}")
    return _READOUTS[token_policy]


@dataclass
class SequenceClouds:
    """What :func:`sequence_clouds` returns.

    ``last[k]``: ``(N, D)`` last-content-token rows of layer ``k`` (one per
    document, in document order). ``tokens[k]``: per-document token rows of
    the first ``token_docs`` documents for the requested token layers.
    ``keep``: the sink keep masks for those documents. ``mid``: the middle
    layer's token rows of the same documents (for the share diagnostics).
    """

    last: Dict[Any, torch.Tensor]
    tokens: Dict[Any, DocList] = field(default_factory=dict)
    keep: Optional[List[torch.Tensor]] = None
    sink_info: Dict[str, Any] = field(default_factory=dict)
    mid: Optional[DocList] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(next(iter(self.last.values())).shape[0]) if self.last else 0

    def token_rows(self, key, policy: str = TOKEN_CLOUD_POLICY):
        """``select_rows`` on the token cloud of layer ``key`` under ``policy``."""
        docs = self.tokens[key]
        if policy in ("exclude_sinks", "mean_excl_sinks"):
            return select_rows(docs, policy, keep=self.keep)
        return select_rows(docs, policy)

    def meta(self) -> Dict[str, Any]:
        """``_meta_`` keys: point count, route, sink mask and share diagnostics."""
        out = {
            "_meta_n_points": self.n_points,
            "_meta_n_points_requested": int(self.info.get("n_requested", 0)),
            "_meta_points_source": self.info.get("source"),
            "_meta_points_readout": "last_content_token",
            "_meta_token_cloud_n_docs": int(len(self.keep or [])),
        }
        if self.keep is not None:
            out.update(sink_meta(self.sink_info))
        if self.mid is not None and len(self.mid):
            out["_meta_position0_share"] = position0_share(self.mid)
            out["_meta_massive_token_share"] = massive_token_share(self.mid)
        return out


def _norm_columns(docs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Per-row float64 L2 norms as ``(T_i, 1)`` columns.

    ``sink_keep_masks`` only reads row norms, and the norm of a one-entry row
    is the entry itself, so passing these columns gives the masks the full
    rows would give while the forward route keeps no middle-depth rows.
    """
    return [torch.linalg.vector_norm(c.detach().to("cpu", torch.float64), dim=1).reshape(-1, 1)
            for c in docs]


def _masks_from_norms(norms_by_depth: Sequence[Sequence[torch.Tensor]]):
    if not norms_by_depth:
        return None, {}
    return sink_keep_masks(norms_by_depth)


def _store(t: torch.Tensor) -> torch.Tensor:
    """CPU copy of one row block in the dtype the shared cache stores.

    The cache stores ``.float()``; half types are widened the same way so the
    two routes return identical rows, and float32/float64 are kept.
    """
    t = t.detach().to("cpu")
    if not t.is_floating_point() or t.dtype in (torch.float16, torch.bfloat16):
        t = t.float()
    return t


def _resolve_key(keys: List[Any], layer: Union[int, str]) -> Any:
    if layer == "all":
        raise ValueError("_resolve_key needs one layer")
    li = int(layer)
    actual = li if li >= 0 else keys[-1] + 1 + li
    return max(keys[0], min(actual, keys[-1]))


def sequence_clouds(
    model,
    tokenizer,
    dataset,
    cache=None,
    *,
    n: int,
    layer: Union[int, str] = "all",
    replace_final_norm: bool,
    use_cache: bool = True,
    token_docs: int = DEFAULT_TOKEN_DOCS,
    readout: str = "last",
) -> SequenceClouds:
    """Last-token rows of ``n`` documents, plus the ``_meta_`` token cloud.

    ``layer`` is ``"all"`` (every block output, keys 0..L-1, the deepest the
    raw block output when ``replace_final_norm``) or one cache layer index
    (``-1`` = HF's last entry, post-norm unless ``replace_final_norm``). The
    token cloud covers the first ``min(token_docs, n)`` documents at the same
    layer(s), and its sink masks come from the middle depths of those
    documents. The shared cache is used only when it holds at least ``n``
    documents (``hidden_states_for_task(min_cached_docs=n)``), so the cache
    cap cannot shrink N.
    """
    if readout != "last":
        raise ValueError(f"unsupported readout {readout!r}; only 'last' is implemented")
    n = int(n)
    m = max(0, min(int(token_docs), n))
    states, info = hidden_states_for_task(
        model, tokenizer, dataset, cache, num_samples=n,
        replace_final_norm=replace_final_norm, layer_idx="all", use_cache=use_cache,
        per_sample=True, min_cached_docs=n, fallback="none")
    if states:
        out = _from_states(states, layer, m, dict(info, n_requested=n))
    elif dataset is None:
        raise ValueError("sequence_clouds: the cache cannot supply the documents "
                         "and no dataset was given")
    else:
        out = _from_forward(model, tokenizer, dataset, n=n, layer=layer, m=m,
                            replace_final_norm=replace_final_norm,
                            info=dict(info, n_requested=n))
    if out.n_points < n:
        logger.warning(f"sequence_clouds: {out.n_points} of the {n} requested sequences; "
                       f"the dataset is shorter than the task's N")
    return out


def _from_states(states, layer, m, info) -> SequenceClouds:
    keys = sorted(states)
    want = keys if layer == "all" else [_resolve_key(keys, layer)]
    last = {k: select_rows(states[k], "last")[0] for k in want}
    mid_key = keys[len(keys) // 2]
    tokens, keep, sink_info, mid = {}, None, {}, None
    if m > 0:
        tokens = {k: DocList(list(states[k])[:m]) for k in want}
        mid = DocList(list(states[mid_key])[:m])
        depth = middle_depth_layers(states)
        keep, sink_info = _masks_from_norms(
            [_norm_columns(list(states[k])[:m]) for k in depth])
    info = dict(info, last_is_block=bool(info.get("last_is_block", False)), layer_keys=want)
    return SequenceClouds(last=last, tokens=tokens, keep=keep, sink_info=sink_info,
                          mid=mid, info=info)


def _from_forward(model, tokenizer, dataset, *, n, layer, m, replace_final_norm,
                  info) -> SequenceClouds:
    from ...cache import CACHE_TOKENIZER_MAX_LENGTH
    from ..common import get_layers

    device = next(model.parameters()).device
    hook_out: Dict[str, torch.Tensor] = {}
    hook = None
    if replace_final_norm:
        blocks = get_layers(model)
        if blocks is not None and len(blocks) > 0:
            def _capture(module, inputs, output):
                hook_out["h"] = (output[0] if isinstance(output, tuple) else output).detach()
            hook = blocks[-1].register_forward_hook(_capture)

    last: Dict[Any, List[torch.Tensor]] = {}
    tok: Dict[Any, List[torch.Tensor]] = {}
    mid_rows: List[torch.Tensor] = []
    depth_norms: Dict[Any, List[torch.Tensor]] = {}
    keys = want = depth = None
    mid_key = None
    n_done = n_swapped = 0
    try:
        with torch.no_grad():
            for sample in dataset:
                if n_done >= n:
                    break
                text = sample.get("text", "") if isinstance(sample, dict) else str(sample)
                enc = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=CACHE_TOKENIZER_MAX_LENGTH).to(device)
                enc, _ = strip_trailing_special(enc, tokenizer)
                if enc["input_ids"].shape[1] < 1:
                    continue
                hook_out.pop("h", None)
                # post-final-norm: SUBSTITUTED when replace_final_norm (the hooked
                # raw block output replaces HF's ln_f entry, as the cache does);
                # otherwise hs[-1] is DELIBERATELY the post-norm output state.
                hs = list(model(**enc, output_hidden_states=True).hidden_states[1:])
                if replace_final_norm and "h" in hook_out and \
                        tuple(hook_out["h"].shape) == tuple(hs[-1].shape):
                    hs[-1] = hook_out["h"]
                    n_swapped += 1
                if keys is None:
                    keys = list(range(len(hs)))
                    want = keys if layer == "all" else [_resolve_key(keys, layer)]
                    mid_key = keys[len(keys) // 2]
                    depth = middle_depth_layers({k: None for k in keys})
                # post-final-norm: the entries read below follow the declaration
                # above (cache layout: keys 0..L-1 are the block outputs).
                for k in want:
                    last.setdefault(k, []).append(_store(hs[k][0, -1:]))
                if n_done < m:
                    for k in want:
                        tok.setdefault(k, []).append(_store(hs[k][0]))
                    mid_rows.append(_store(hs[mid_key][0]))
                    for k in depth:
                        depth_norms.setdefault(k, []).extend(_norm_columns([_store(hs[k][0])]))
                n_done += 1
    finally:
        if hook is not None:
            hook.remove()

    if not last:
        return SequenceClouds(last={}, info=dict(info, source="self_collected",
                                                 fallback="sequence_forward"))
    keep, sink_info = _masks_from_norms([depth_norms[k] for k in depth]) if m > 0 else (None, {})
    info = dict(info, source="self_collected", fallback="sequence_forward",
                last_is_block=bool(replace_final_norm and n_done > 0 and n_swapped == n_done),
                layer_keys=want)
    return SequenceClouds(
        last={k: torch.cat(v, dim=0) for k, v in last.items()},
        tokens={k: DocList(v) for k, v in tok.items()},
        keep=keep, sink_info=sink_info,
        mid=DocList(mid_rows) if mid_rows else None, info=info)
