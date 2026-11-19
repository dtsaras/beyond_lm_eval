"""Fixed-length word windows: the sequences behind the one-point-per-sequence clouds.

Cheng et al. (2024, id-llm-abstraction@4a170a2, ``corpora/*_sane_ds.txt`` and
``scripts/extract_final_representations.py``) represent a sequence by the
last-token hidden state of 20 consecutive whitespace words cut from a longer
text. Every model reads the same words. A whole passage instead lets its
length and its closing " ." organise the cloud.

The window of a passage depends only on the passage text and ``seed``
(sha256). It does not depend on N, on the other passages, or on the run
seed, so it is part of the item.
"""
from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

N_WORDS = 20
WINDOW_SEED = 0
READOUTS = ("window", "passage")


def text_of(sample: Any) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, Mapping):
        return str(sample.get("text", ""))
    return str(sample)


def word_window(text: str, n_words: int = N_WORDS, seed: int = WINDOW_SEED) -> Optional[str]:
    """``n_words`` consecutive whitespace words of ``text`` at a hashed offset, or None."""
    words = text.split()
    if n_words < 1 or len(words) < n_words:
        return None
    digest = hashlib.sha256(f"{int(seed)}\x00{text}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % (len(words) - n_words + 1)
    return " ".join(words[offset:offset + n_words])


def is_punct_final(window: str) -> bool:
    """True when the window's last word has no letter or digit (".", ",", ")" ...)."""
    last = window.rsplit(" ", 1)[-1]
    return not any(ch.isalnum() for ch in last)


@dataclass
class WindowSet:
    texts: List[str]
    n_words: int
    seed: int
    n_scanned: int = 0
    n_short: int = 0
    n_punct_dropped: int = 0
    punct_final: List[bool] = field(default_factory=list)

    @property
    def digest(self) -> str:
        h = hashlib.sha256(f"{self.n_words}:{self.seed}\n".encode("utf-8"))
        h.update("\n".join(self.texts).encode("utf-8"))
        return h.hexdigest()[:16]

    def meta(self) -> Dict[str, Any]:
        n = len(self.texts)
        return {
            "_meta_window_words": self.n_words,
            "_meta_window_seed": self.seed,
            "_meta_window_set_hash": self.digest,
            "_meta_window_punct_final_share": (sum(self.punct_final) / n) if n else float("nan"),
            "_meta_window_punct_final_dropped": self.n_punct_dropped,
            "_meta_window_short_passages_skipped": self.n_short,
        }


def window_set(dataset: Optional[Iterable[Any]], n: int, *, n_words: int = N_WORDS,
               seed: int = WINDOW_SEED, drop_punct_final: bool = False) -> WindowSet:
    """One window from each of the first ``n`` passages (in dataset order) long enough to hold one."""
    if dataset is None:
        raise ValueError("word windows need the passages: pass a dataset")
    ws = WindowSet(texts=[], n_words=int(n_words), seed=int(seed))
    for sample in dataset:
        if len(ws.texts) >= int(n):
            break
        ws.n_scanned += 1
        w = word_window(text_of(sample).strip(), ws.n_words, ws.seed)
        if w is None:
            ws.n_short += 1
            continue
        final = is_punct_final(w)
        if final and drop_punct_final:
            ws.n_punct_dropped += 1
            continue
        ws.texts.append(w)
        ws.punct_final.append(final)
    return ws


def window_options(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The readout settings a task's config names (defaults: 20-word windows, seed 0)."""
    config = config or {}
    readout = str(config.get("points_readout", "window"))
    if readout not in READOUTS:
        raise ValueError(f"points_readout must be one of {READOUTS}, got {readout!r}")
    massive = str(config.get("massive_points", "keep"))
    if massive not in MASSIVE_POINT_MODES:
        raise ValueError(f"massive_points must be one of {MASSIVE_POINT_MODES}, got {massive!r}")
    return {"readout": readout,
            "n_words": int(config.get("window_words", N_WORDS)),
            "seed": int(config.get("window_seed", WINDOW_SEED)),
            "drop_punct_final": bool(config.get("drop_punct_final_windows", False)),
            "massive_points": massive}


# Windows whose last token carries a massive activation: "keep" (default; the
# references keep every point) or "drop" (the token-cloud sink rule applied to
# the one-point-per-sequence cloud). Counted either way.
MASSIVE_POINT_MODES = ("keep", "drop")


def massive_window_mask(rows: Mapping[int, torch.Tensor]) -> Tuple[torch.Tensor, float]:
    """Windows whose last-token norm exceeds MASSIVE_FACTOR times the median
    last-token norm at any middle depth: the token-cloud sink rule
    (:func:`blme.tasks._tokens.sink_keep_masks`) on the one-point cloud.

    Returns ``(flag, max_ratio)``: a bool ``(N,)`` mask and the largest
    norm/median ratio seen at those depths. One massive point (OLMo-1B: one
    window ending in "." at 54x the median) sets a cloud's extremal statistics,
    e.g. the largest H0 persistence.
    """
    from ._tokens import MASSIVE_FACTOR, middle_depth_layers
    keys = middle_depth_layers(dict(rows)) if rows else []
    n = int(next(iter(rows.values())).shape[0]) if rows else 0
    flag = torch.zeros(n, dtype=torch.bool)
    max_ratio = float("nan")
    for k in keys:
        nrm = torch.linalg.vector_norm(rows[k].detach().to("cpu", torch.float64), dim=1)
        med = float(nrm.median()) if n else 0.0
        if not med > 0:
            continue
        ratio = nrm / med
        flag |= ratio > MASSIVE_FACTOR
        r = float(ratio.max())
        max_ratio = r if max_ratio != max_ratio else max(max_ratio, r)
    return flag, max_ratio


def _apply_massive_rule(rows, info, mode):
    """Count massive windows; with ``mode == "drop"`` remove them at every depth."""
    flag, max_ratio = massive_window_mask(rows)
    n_massive = int(flag.sum())
    meta = {"_meta_window_massive_points": mode, "_meta_window_n_massive_points": n_massive,
            "_meta_window_massive_points_max_ratio": max_ratio}
    info = dict(info, massive_meta=meta)
    if mode != "drop" or not n_massive:
        return rows, info
    keep = ~flag
    rows = {k: v[keep] for k, v in rows.items()}
    for key in ("n_tokens", "positions"):
        if isinstance(info.get(key), list) and len(info[key]) == len(keep):
            info[key] = [x for x, kp in zip(info[key], keep.tolist()) if kp]
    return rows, info


def _store(t: torch.Tensor) -> torch.Tensor:
    """CPU row in the dtype the shared cache stores (half types widened to float32)."""
    t = t.detach().to("cpu")
    if not t.is_floating_point() or t.dtype in (torch.float16, torch.bfloat16):
        t = t.float()
    return t


def window_last_rows(model, tokenizer, texts: List[str], *,
                     select: Callable[[int], Iterable[int]],
                     replace_final_norm: bool) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """``{key: (N, D)}`` last-token rows of each window, one forward per window.

    Keys index ``hidden_states[1:]`` (block outputs, 0..L-1); ``select(L)``
    picks them. With ``replace_final_norm`` the deepest entry is the hooked
    raw output of the last block, as on the shared cache. Each window is
    tokenised alone with the tokenizer's defaults (a BOS where it adds one),
    and appended EOS/SEP tokens are stripped, so no padding enters the rows.
    """
    from ._tokens import strip_trailing_special
    from .common import get_layers

    device = next(model.parameters()).device
    sink: Dict[str, torch.Tensor] = {}
    hook = None
    if replace_final_norm:
        blocks = get_layers(model)
        if blocks is not None and len(blocks) > 0:
            def _capture(module, inputs, output):
                sink["h"] = (output[0] if isinstance(output, tuple) else output).detach()
            hook = blocks[-1].register_forward_hook(_capture)
    rows: Dict[int, List[torch.Tensor]] = {}
    n_tokens: List[int] = []
    keys: Optional[List[int]] = None
    n_sub = 0
    try:
        with torch.no_grad():
            for text in texts:
                enc = dict(tokenizer(text, return_tensors="pt"))
                enc, _ = strip_trailing_special(enc, tokenizer)
                T = int(enc["input_ids"].shape[1])
                if T < 1:
                    continue
                sink.pop("h", None)
                # post-final-norm: SUBSTITUTED when replace_final_norm (the hooked
                # raw block output replaces HF's final-norm entry, as the cache
                # does); otherwise hs[-1] is DELIBERATELY the post-norm state.
                hs = list(model(**{k: v.to(device) for k, v in enc.items()},
                                output_hidden_states=True).hidden_states[1:])
                if replace_final_norm and "h" in sink and \
                        tuple(sink["h"].shape) == tuple(hs[-1].shape):
                    hs[-1] = sink["h"]
                    n_sub += 1
                if keys is None:
                    keys = sorted({int(k) for k in select(len(hs)) if 0 <= int(k) < len(hs)})
                # post-final-norm: the entries read below follow the declaration above.
                for k in keys:
                    rows.setdefault(k, []).append(_store(hs[k][0, T - 1]))
                n_tokens.append(T)
    finally:
        if hook is not None:
            hook.remove()
    out = {k: torch.stack(v) for k, v in rows.items()}
    info = {
        "n_tokens": n_tokens,
        "positions": [t - 1 for t in n_tokens],
        "last_is_block": bool(replace_final_norm and n_tokens and n_sub == len(n_tokens)),
    }
    return out, info


def token_count_meta(n_tokens: List[int]) -> Dict[str, float]:
    """Mean and SD of the tokens per window: the tokenizer-dependent length of the same text."""
    if not n_tokens:
        return {"_meta_window_tokens_mean": float("nan"), "_meta_window_tokens_sd": float("nan")}
    return {"_meta_window_tokens_mean": float(statistics.fmean(n_tokens)),
            "_meta_window_tokens_sd": float(statistics.pstdev(n_tokens))}


def window_rows(model, tokenizer, dataset, cache, *, n: int, opts: Mapping[str, Any],
                replace_final_norm: bool, use_cache: bool = True):
    """``(rows, window_set, info)`` for the first ``n`` windows of ``dataset``.

    ``rows`` is ``{block index: (N, D)}`` over every block output. A cache with
    a ``window_rows`` method computes them once per run and serves every task;
    otherwise this runs the window forwards itself. Both give the same rows.
    """
    mode = opts.get("massive_points", "keep")
    if use_cache and cache is not None and callable(getattr(cache, "window_rows", None)):
        rows, ws, info = cache.window_rows(dataset, n=n, opts=opts, replace_final_norm=replace_final_norm)
    else:
        ws = window_set(dataset, n, n_words=opts["n_words"], seed=opts["seed"],
                        drop_punct_final=opts["drop_punct_final"])
        rows, info = window_last_rows(model, tokenizer, ws.texts, select=range,
                                      replace_final_norm=replace_final_norm)
        info = dict(info, source="window_forward")
    rows, info = _apply_massive_rule(rows, info, mode)
    return rows, ws, info
