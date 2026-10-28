"""Token-row selection that knows which document and position each row is.

The collectors (``geometry.utils.collect_hidden_states`` and
``ModelOutputCache.get_hidden_states``) return token rows flattened across
documents by default. Policies that depend on the document or on the token
position (drop position 0, take the last token, pool per document) need the
per-document split and the position of every row. This module supplies:

* :func:`select_rows` -- one policy applied to a list of per-document
  ``(T_i, D)`` tensors, with row metadata (document index, position).
* :func:`massive_token_share` / :func:`position0_share` -- the share of the
  centred sum of squares held by massive non-initial rows / by position 0.
* :func:`hidden_states_for_task` -- the "use the shared cache if it can supply
  this task, else collect" gate, written once with explicit parameters.

Positions are read from ``positions=`` or from a :class:`DocList`'s
``.positions``; without either, rows of document ``i`` are positions
``0..T_i-1``. ``collect_hidden_states(per_sample=True)`` returns
:class:`DocList` objects, so a subsampled single-layer collection still knows
which row was position 0.

Cache gate variants in the tasks (call sites of ``get_hidden_states`` /
``collect_hidden_states``). "populated" is ``cache is not None and
cache.is_populated and use_cache``; "None-fallback" means a populated cache
that returns no hidden states (attention-only population) leads to
self-collection.

=====================================  =======  ===========================================  =============================  =====================================================
task (file)                            layer    cache accepted when                          unusable cache                 fallback route
=====================================  =======  ===========================================  =============================  =====================================================
geometry_collapse                      all      populated                                    ``is None`` -> fallback        collect, rfn=True
geometry_rsa                           all      populated                                    ``is None`` -> fallback        collect, rfn=True
geometry_hsic (mutual_info)            all      populated                                    ``is None`` -> fallback        collect, rfn=True
geometry_cka                           all      populated                                    ``is None`` -> fallback        collect, rfn=True
geometry_cknna                         all      populated                                    ``is None`` -> fallback        collect, rfn=False
geometry_mp_bulk_deviation (rmt_bulk)  all      populated                                    ``is None`` -> fallback        collect, rfn=True
interpretability_activation_kurtosis   all      populated                                    ``is None`` -> fallback        collect, rfn=True
geometry_lipschitz                     all+emb  populated                                    ``is None`` -> fallback        collect, rfn=True, include_embedding
geometry_intrinsic_dim (layerwise)     all      populated                                    NOT handled (``None.keys()``)  collect, rfn=True
geometry_lid (single layer)            -1       populated                                    ``is None`` -> fallback        collect, rfn=False, every token
geometry_lid (layerwise)               all      populated                                    falsy -> single-layer mode     none (drops to the single-layer branch)
geometry_svd (layerwise)               all      populated                                    falsy -> single-layer mode     none (drops to the single-layer branch)
geometry_svd (single layer)            -1       populated and cached docs >= N               ``is None`` -> fallback        collect, rfn=False, every token
geometry_isoscore                      -1       populated and cached docs >= N               ``is None`` -> fallback        collect, rfn=False, every token
geometry_vendi_score                   all      populated and cached docs >= N               falsy -> fallback              private ModelOutputCache, rfn=True
geometry_magnitude                     cfg      populated and cached docs == N (exact)       ``is None`` -> fallback        collect max(N, 50) docs, rfn=False, every token
geometry_trajectory_curvature          all/doc  populated                                    ``is None`` -> fallback        private ModelOutputCache, per_sample, rfn=True
geometry_procrustes_linearity          all+emb  populated                                    ``is None`` -> fallback        private ModelOutputCache, rfn=True, include_embedding
geometry_schatten                      all/doc  populated and max_length == cache window     falsy -> fallback              task's own forward (max_length)
geometry_matrix_entropy                all/doc  populated (no window check)                  falsy -> fallback              task's own forward (max_length)
geometry_phd_dimension                 cfg      ``getattr(is_populated)`` (no window check)  ``is None`` -> fallback        task's own forward (max_length, pooling)
topology_zigzag_persistence            all/doc  ``getattr(is_populated)``                    falsy -> fallback              task's own forward (window 512)
=====================================  =======  ===========================================  =============================  =====================================================

"all/doc" = ``layer_idx="all", per_sample=True``; "cfg" = a configured single
layer; "every token" = ``max_tokens_per_doc=None``; rfn = ``replace_final_norm``.
The flag ``last_is_block`` is read from ``cache._last_block_substituted``
immediately after the cache call (a later ``replace_final_norm=False`` call
resets it) or from ``collect_hidden_states(status=...)``. The routes differ in
dtype: the cache (shared or private) stores float32, ``collect_hidden_states``
keeps the model's float dtype (bf16 rows on a bf16 model). The prediction-stats
gates (perplexity, calibration, prediction_alignment) read logits, not hidden
states, and are out of scope here.

:func:`hidden_states_for_task` reproduces the canonical row (populated cache,
None-or-empty fallback, collect) by default; ``min_cached_docs`` /
``exact_cached_docs`` give the population-gated rows, ``fallback=
"private_cache"`` the private-cache rows and ``fallback="none"`` the
layerwise-drop rows. The task-private forward passes are not reproduced: they
differ in window, pooling or hooking and stay in their tasks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger("blme")

POLICIES = ("all", "exclude_pos0", "exclude_sinks", "last", "mean", "mean_excl_pos0",
            "mean_excl_sinks")
_POOLED = ("mean", "mean_excl_pos0", "mean_excl_sinks")
_EXCL_POS0 = ("exclude_pos0", "mean_excl_pos0")
# Position 0 plus massive-activation rows (Sun et al. 2024), given as a
# per-document keep mask from :func:`sink_keep_masks`.
_EXCL_SINKS = ("exclude_sinks", "mean_excl_sinks")
MASSIVE_FACTOR = 10.0
MIDDLE_DEPTHS = (0.1, 0.9)


class DocList(list):
    """A list of per-document ``(T_i, D)`` tensors carrying row provenance.

    ``positions[i]`` is an int64 tensor of length ``T_i`` with the original
    token position of each row of document ``i``; ``doc_ids[i]`` is that
    document's index in the input dataset. Either may be None (unknown).
    """

    def __init__(self, chunks=(), positions=None, doc_ids=None):
        super().__init__(chunks)
        self.positions = positions
        self.doc_ids = doc_ids


@dataclass
class RowMeta:
    """Provenance of the rows returned by :func:`select_rows`.

    ``doc`` and ``position`` have one entry per returned row. ``position`` is
    -1 for a pooled row (``mean`` / ``mean_excl_pos0``).
    """

    policy: str
    doc: torch.Tensor
    position: torch.Tensor
    n_docs: int
    n_docs_used: int
    n_docs_empty: int
    n_docs_dropped_pos0_only: int

    @property
    def n_rows(self) -> int:
        return int(self.doc.numel())

    def as_meta(self, prefix: str = "_meta_rows_") -> Dict[str, Any]:
        return {
            f"{prefix}policy": self.policy,
            f"{prefix}n_rows": self.n_rows,
            f"{prefix}n_docs": self.n_docs,
            f"{prefix}n_docs_used": self.n_docs_used,
            f"{prefix}n_docs_empty": self.n_docs_empty,
            f"{prefix}n_docs_dropped_pos0_only": self.n_docs_dropped_pos0_only,
        }


def _as_2d(x, i: int) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    if t.ndim != 2:
        raise ValueError(f"document {i}: expected a (T, D) tensor, got shape {tuple(t.shape)}")
    return t


def _resolve_positions(chunks: Sequence[torch.Tensor], positions) -> List[torch.Tensor]:
    if positions is None:
        return [torch.arange(c.shape[0], dtype=torch.int64) for c in chunks]
    if len(positions) != len(chunks):
        raise ValueError(f"{len(positions)} position vectors for {len(chunks)} documents")
    out = []
    for i, (c, p) in enumerate(zip(chunks, positions)):
        p = torch.as_tensor(p, dtype=torch.int64).reshape(-1).cpu()
        if p.numel() != c.shape[0]:
            raise ValueError(f"document {i}: {p.numel()} positions for {c.shape[0]} rows")
        out.append(p)
    return out


def _prepare(per_sample, positions=None, doc_ids=None):
    """Validated chunks, positions and document ids for a per-document list."""
    if isinstance(per_sample, torch.Tensor):
        raise TypeError("select_rows takes a list of per-document (T_i, D) tensors, "
                        "not a flat tensor; use per_sample=True on the collector")
    chunks = [_as_2d(c, i) for i, c in enumerate(per_sample)]
    if positions is None:
        positions = getattr(per_sample, "positions", None)
    if doc_ids is None:
        doc_ids = getattr(per_sample, "doc_ids", None)
    pos = _resolve_positions(chunks, positions)
    if doc_ids is None:
        ids = list(range(len(chunks)))
    else:
        ids = [int(d) for d in doc_ids]
        if len(ids) != len(chunks):
            raise ValueError(f"{len(ids)} doc ids for {len(chunks)} documents")
    dims = {int(c.shape[1]) for c in chunks}
    if len(dims) > 1:
        raise ValueError(f"documents disagree on the feature dimension: {sorted(dims)}")
    return chunks, pos, ids


def _empty_rows(chunks, dtype) -> torch.Tensor:
    d = int(chunks[0].shape[1]) if chunks else 0
    return torch.empty((0, d), dtype=dtype)


def _pool_float64(rows: torch.Tensor) -> torch.Tensor:
    """Per-document mean, accumulated in float64 (returned as float64, on CPU)."""
    x64 = rows.detach().to("cpu", torch.float64).numpy()
    return torch.from_numpy(np.ascontiguousarray(x64.mean(axis=0, dtype=np.float64)))


def select_rows(
    per_sample: Sequence[torch.Tensor],
    policy: str,
    positions: Optional[Sequence[torch.Tensor]] = None,
    doc_ids: Optional[Sequence[int]] = None,
    keep: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, RowMeta]:
    """Apply one token-selection ``policy`` to per-document hidden states.

    Args:
        per_sample: list of ``(T_i, D)`` tensors, one per document.
        policy: one of

            * ``"all"`` -- every row.
            * ``"exclude_pos0"`` -- every row whose position is not 0.
            * ``"last"`` -- one row per document: the row with the largest
              position (the last token of an unsubsampled document).
            * ``"mean"`` -- one row per document: the mean of its rows.
            * ``"mean_excl_pos0"`` -- one row per document: the mean of its
              rows at position > 0.
            * ``"exclude_sinks"`` / ``"mean_excl_sinks"`` -- as the two above,
              but dropping every row that ``keep`` marks False as well:
              position 0 plus the massive-activation rows found by
              :func:`sink_keep_masks`. ``keep`` is required, and the same
              masks should be passed for every layer so that all layers
              keep the same rows.
        positions: optional per-document int64 positions (default: the
            ``.positions`` of a :class:`DocList`, else ``0..T_i-1``).
        doc_ids: optional document ids to report (default: the ``.doc_ids``
            of a :class:`DocList`, else the list index).
        keep: per-document boolean masks (True = keep), used by the
            ``*_sinks`` policies.

    Returns:
        ``(rows, meta)``. Unpooled policies return rows in the input dtype in
        document order (rows of one document keep their order); pooled
        policies return float64 rows on CPU. A document with no row left under
        the policy contributes nothing and is counted in ``meta``
        (``n_docs_empty`` for ``T_i == 0``, ``n_docs_dropped_pos0_only`` for a
        document whose only rows sit at position 0 under ``exclude_pos0`` /
        ``mean_excl_pos0``). Rows are never merged across documents.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown token policy {policy!r}; expected one of {POLICIES}")
    chunks, pos, ids = _prepare(per_sample, positions, doc_ids)
    if policy in _EXCL_SINKS:
        if keep is None:
            raise ValueError(f"policy {policy!r} needs keep masks from sink_keep_masks()")
        if len(keep) != len(chunks):
            raise ValueError(f"{len(keep)} keep masks for {len(chunks)} documents")

    rows: List[torch.Tensor] = []
    doc_col: List[int] = []
    pos_col: List[int] = []
    n_empty = n_pos0_only = n_used = 0
    for i, (c, p, did) in enumerate(zip(chunks, pos, ids)):
        if c.shape[0] == 0:
            n_empty += 1
            continue
        if policy in _EXCL_POS0:
            mask = p != 0
        elif policy in _EXCL_SINKS:
            mask = torch.as_tensor(keep[i], dtype=torch.bool).reshape(-1).cpu()
            if mask.numel() != c.shape[0]:
                raise ValueError(f"document {i}: keep mask has {mask.numel()} entries "
                                 f"for {c.shape[0]} rows")
            mask = mask & (p != 0)
        else:
            mask = None
        if mask is not None and not bool(mask.any()):
            n_pos0_only += 1
            continue

        if policy == "all":
            rows.append(c)
            doc_col.extend([did] * c.shape[0])
            pos_col.extend(p.tolist())
        elif policy in ("exclude_pos0", "exclude_sinks"):
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            rows.append(c[idx.to(c.device)])
            doc_col.extend([did] * idx.numel())
            pos_col.extend(p[idx].tolist())
        elif policy == "last":
            j = int(torch.argmax(p))
            rows.append(c[j:j + 1])
            doc_col.append(did)
            pos_col.append(int(p[j]))
        else:
            sub = c if mask is None else c[torch.nonzero(mask, as_tuple=False).reshape(-1).to(c.device)]
            rows.append(_pool_float64(sub).reshape(1, -1))
            doc_col.append(did)
            pos_col.append(-1)
        n_used += 1

    if policy in _POOLED:
        out = torch.cat(rows, dim=0) if rows else _empty_rows(chunks, torch.float64)
    else:
        dtype = chunks[0].dtype if chunks else torch.float32
        out = torch.cat(rows, dim=0) if rows else _empty_rows(chunks, dtype)
    meta = RowMeta(
        policy=policy,
        doc=torch.tensor(doc_col, dtype=torch.int64),
        position=torch.tensor(pos_col, dtype=torch.int64),
        n_docs=len(chunks),
        n_docs_used=n_used,
        n_docs_empty=n_empty,
        n_docs_dropped_pos0_only=n_pos0_only,
    )
    return out, meta


def _stack_float64(per_sample, positions=None):
    chunks, pos, _ = _prepare(per_sample, positions)
    if not chunks or sum(int(c.shape[0]) for c in chunks) == 0:
        return None, None
    X = np.concatenate([c.detach().to("cpu", torch.float64).numpy() for c in chunks], axis=0)
    P = torch.cat(pos).numpy()
    return X, P


def _centred_row_ss(X: np.ndarray) -> np.ndarray:
    C = X - X.mean(axis=0, dtype=np.float64)
    return np.einsum("ij,ij->i", C, C)


def massive_token_share(
    per_sample: Sequence[torch.Tensor],
    factor: float = 10.0,
    positions: Optional[Sequence[torch.Tensor]] = None,
) -> float:
    """Share of the centred sum of squares held by massive non-initial rows.

    A row is massive when its position is > 0 and its L2 norm exceeds
    ``factor`` times the median norm of the rows at position > 0. The cloud is
    every row of every document, centred on its float64 column mean; the share
    is ``sum_massive ||x_i - mu||^2 / sum_all ||x_i - mu||^2``. NaN when there
    is no row at position > 0 or the total is zero.
    """
    X, P = _stack_float64(per_sample, positions)
    if X is None:
        return float("nan")
    rest = P > 0
    if not rest.any():
        return float("nan")
    norms = np.linalg.norm(X, axis=1)
    med = float(np.median(norms[rest]))
    big = rest & (norms > float(factor) * med)
    ss = _centred_row_ss(X)
    tot = float(ss.sum())
    if not (tot > 0 and math.isfinite(tot)):
        return float("nan")
    return float(ss[big].sum() / tot)


def position0_share(
    per_sample: Sequence[torch.Tensor],
    positions: Optional[Sequence[torch.Tensor]] = None,
) -> float:
    """Share of the centred sum of squares held by the rows at position 0.

    Same cloud and centring as :func:`massive_token_share`. NaN when the cloud
    is empty or its total is zero.
    """
    X, P = _stack_float64(per_sample, positions)
    if X is None:
        return float("nan")
    ss = _centred_row_ss(X)
    tot = float(ss.sum())
    if not (tot > 0 and math.isfinite(tot)):
        return float("nan")
    return float(ss[P == 0].sum() / tot)


# ── Sink masks ───────────────────────────────────────────────────────────

def middle_depth_layers(states_by_layer: Dict[Any, Sequence[torch.Tensor]],
                        lo: float = MIDDLE_DEPTHS[0], hi: float = MIDDLE_DEPTHS[1]) -> List[Any]:
    """Keys of the layers whose relative depth lies in ``[lo, hi]``.

    Depth is the key's rank among the sorted keys over the last rank, so it
    does not depend on whether the dict starts at the embedding or at block 0.
    With fewer than three layers every key is returned: there is no middle to
    choose from, and the caller should record which depths it used.
    """
    keys = sorted(states_by_layer)
    if len(keys) < 3:
        return keys
    n = len(keys) - 1
    mid = [k for r, k in enumerate(keys) if lo <= r / n <= hi]
    return mid or keys


def sink_keep_masks(
    per_sample_by_depth: Sequence[Sequence[torch.Tensor]],
    factor: float = MASSIVE_FACTOR,
    positions: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """One keep mask per document: drop position 0 and every massive row.

    A row (token) is massive when, at ANY of the given depths, its L2 norm
    exceeds ``factor`` times the median norm of the rows at position > 0 at
    that depth (pooled over documents). Massive activations sit on a few
    tokens and act as attention-sink biases (Sun et al. 2024); which tokens
    carry them differs by model family (BOS, the first delimiter, position 1,
    frequent words), so the rule is set by magnitude, not by position.

    ``per_sample_by_depth`` holds the same documents at each depth, with
    matching row counts; pass the middle depths (:func:`middle_depth_layers`)
    -- the final post-norm layer has no massive rows. The SAME masks are then
    used at every layer, so cross-layer metrics compare the same rows.

    Returns ``(masks, info)``; ``info`` has the number of rows dropped as
    massive (beyond position 0) and the depths used.
    """
    if not per_sample_by_depth:
        raise ValueError("sink_keep_masks needs at least one depth")
    first, pos, _ = _prepare(per_sample_by_depth[0], positions)
    masks = [p != 0 for p in pos]
    for depth_docs in per_sample_by_depth:
        chunks, _, _ = _prepare(depth_docs, positions)
        if [c.shape[0] for c in chunks] != [c.shape[0] for c in first]:
            raise ValueError("every depth must hold the same documents and rows")
        norms = [torch.linalg.vector_norm(c.detach().to("cpu", torch.float64), dim=1)
                 for c in chunks]
        rest = torch.cat([n[p != 0] for n, p in zip(norms, pos)])
        if rest.numel() == 0:
            continue
        thr = float(factor) * float(rest.median())
        masks = [m & ~((n > thr) & (p != 0)) for m, n, p in zip(masks, norms, pos)]
    n_rest = int(sum(int((p != 0).sum()) for p in pos))
    n_kept = int(sum(int(m.sum()) for m in masks))
    info = {"n_massive_dropped": n_rest - n_kept, "n_rows_pos_gt0": n_rest,
            "n_depths": len(per_sample_by_depth), "factor": float(factor)}
    return masks, info


def sink_meta(info: Dict[str, Any], prefix: str = "_meta_sink_") -> Dict[str, Any]:
    """``_meta_`` keys recording what :func:`sink_keep_masks` dropped."""
    n = info.get("n_rows_pos_gt0") or 0
    return {f"{prefix}n_massive_dropped": int(info.get("n_massive_dropped", 0)),
            f"{prefix}frac_massive_dropped": (float(info["n_massive_dropped"]) / n) if n else float("nan"),
            f"{prefix}n_depths": int(info.get("n_depths", 0)),
            f"{prefix}factor": float(info.get("factor", MASSIVE_FACTOR))}


def strip_trailing_special(encoding: Dict[str, torch.Tensor], tokenizer) -> Tuple[Dict[str, torch.Tensor], int]:
    """Drop special tokens a tokenizer APPENDS (EOS, SEP) from a batch-1 encoding.

    Some tokenizers append EOS to every sequence (OLMo-1B), which turns every
    last-token readout into a read of that special token. In a causal LM no
    earlier position attends to a trailing token, so removing it leaves every
    other row bit-identical. A leading BOS is never touched, and a sequence is
    never emptied. Returns the encoding and the number of tokens removed.
    """
    ids = encoding["input_ids"]
    # Only the tokens a tokenizer appends: EOS / SEP. UNK, PAD and the like
    # can end real text and are left alone.
    special = {t for t in (getattr(tokenizer, "eos_token_id", None),
                           getattr(tokenizer, "sep_token_id", None)) if t is not None}
    if ids.ndim != 2 or ids.shape[0] != 1 or not special:
        return encoding, 0
    row = ids[0].tolist()
    end = len(row)
    while end > 1 and row[end - 1] in special:
        end -= 1
    n = len(row) - end
    if n == 0:
        return encoding, 0
    out = {k: (v[:, :end] if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == len(row) else v)
           for k, v in encoding.items()}
    return out, n


# ── The cache-or-collect gate ────────────────────────────────────────────


def _is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, (dict, list, tuple)):
        return len(x) == 0
    return False


def hidden_states_for_task(
    model,
    tokenizer,
    dataset,
    cache=None,
    *,
    num_samples: int,
    replace_final_norm: bool,
    layer_idx: Union[int, str] = "all",
    use_cache: bool = True,
    include_embedding: bool = False,
    per_sample: bool = False,
    max_tokens_per_doc: Optional[int] = None,
    min_cached_docs: Optional[int] = None,
    exact_cached_docs: bool = False,
    max_length: Optional[int] = None,
    fallback: str = "collect",
    use_disk_offload: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Hidden states from the shared cache when it can supply them, else collected.

    The shared cache is used only when it is populated, ``use_cache`` is set,
    it holds enough documents (``min_cached_docs``; exactly that many with
    ``exact_cached_docs``), its tokenisation window equals ``max_length`` (when
    given), no per-document token cap is requested, and it actually returns
    hidden states. Otherwise the states come from ``fallback``:

    * ``"collect"`` -- ``collect_hidden_states`` (window ``max_length`` or the
      cache window), with ``max_tokens_per_doc`` applying to a single layer.
    * ``"private_cache"`` -- a fresh ``ModelOutputCache`` over ``dataset``.
    * ``"none"`` -- return ``None`` and let the caller decide.

    ``replace_final_norm`` has no default: every hidden-state read declares
    what it does with HF's post-norm last entry.

    Returns:
        ``(states, info)`` where ``states`` has the shape the collectors
        return for ``layer_idx`` / ``per_sample`` and ``info`` holds
        ``source`` (``"shared_cache"``, ``"self_collected"`` or None),
        ``fallback``, ``cache_rejected`` (why the cache was not used, or None),
        ``cached_docs`` and ``last_is_block`` (whether the deepest entry is the
        hooked raw block output -- derived from what happened, not requested).
    """
    from ..cache import CACHE_TOKENIZER_MAX_LENGTH, ModelOutputCache

    if fallback not in ("collect", "private_cache", "none"):
        raise ValueError(f"unknown fallback {fallback!r}")
    window = CACHE_TOKENIZER_MAX_LENGTH if max_length is None else int(max_length)
    cached_docs = len(getattr(cache, "_sample_lengths", None) or []) if cache is not None else 0
    info: Dict[str, Any] = {"source": None, "fallback": None, "cache_rejected": None,
                            "cached_docs": int(cached_docs), "last_is_block": False}

    reason = None
    if not use_cache:
        reason = "disabled"
    elif cache is None:
        reason = "no_cache"
    elif not getattr(cache, "is_populated", False):
        reason = "not_populated"
    elif min_cached_docs is not None and exact_cached_docs and cached_docs != int(min_cached_docs):
        reason = "doc_count_mismatch"
    elif min_cached_docs is not None and not exact_cached_docs and cached_docs < int(min_cached_docs):
        reason = "too_few_docs"
    elif window != CACHE_TOKENIZER_MAX_LENGTH:
        reason = "window"
    elif max_tokens_per_doc is not None and layer_idx != "all":
        reason = "token_cap"

    if reason is None:
        states = cache.get_hidden_states(
            layer_idx=layer_idx, num_samples=num_samples, per_sample=per_sample,
            replace_final_norm=replace_final_norm, include_embedding=include_embedding)
        # Read the flag now: the next get_hidden_states call resets it.
        last_is_block = bool(getattr(cache, "_last_block_substituted", False))
        if not _is_empty(states):
            info.update(source="shared_cache", last_is_block=last_is_block)
            return states, info
        reason = "no_hidden_states"
    info["cache_rejected"] = reason
    if reason not in ("disabled", "no_cache", "not_populated"):
        logger.info(f"  shared cache not used ({reason}); fallback={fallback}")

    if fallback == "none":
        return None, info
    if dataset is None:
        raise ValueError("hidden_states_for_task: the cache cannot supply these states "
                         "and no dataset was given to collect them from")
    if fallback == "private_cache":
        if window != CACHE_TOKENIZER_MAX_LENGTH:
            raise ValueError(f"a private ModelOutputCache tokenises at "
                             f"{CACHE_TOKENIZER_MAX_LENGTH}, not max_length={window}")
        local = ModelOutputCache(model, tokenizer, dataset=dataset, num_samples=num_samples)
        local.populate(need_hidden=True, need_logits=False)
        states = local.get_hidden_states(
            layer_idx=layer_idx, num_samples=num_samples, per_sample=per_sample,
            replace_final_norm=replace_final_norm, include_embedding=include_embedding)
        info.update(source="self_collected", fallback="private_cache",
                    last_is_block=bool(getattr(local, "_last_block_substituted", False)))
        return states, info

    from .geometry.utils import collect_hidden_states
    st: Dict[str, Any] = {}
    states = collect_hidden_states(
        model, tokenizer, dataset, num_samples=num_samples, layer_idx=layer_idx,
        use_disk_offload=use_disk_offload, replace_final_norm=replace_final_norm,
        status=st, max_tokens_per_doc=max_tokens_per_doc,
        include_embedding=include_embedding, per_sample=per_sample, max_length=window)
    info.update(source="self_collected", fallback="collect",
                last_is_block=bool(st.get("last_is_block", False)))
    return states, info
