"""One point per sequence for the Vietoris-Rips / kNN topology tasks.

The LLM precedents for barcode summaries read the LAST token of each prompt
at a layer (Fay et al. 2025, arXiv:2505.20435 Sec. 2.2; Gardinazzi et al.
2025, arXiv:2410.11042 Sec. 3), with thousands of prompts. The shared cache
supplies the block outputs when it holds at least ``num_samples`` documents;
otherwise a private forward keeps only the last-token rows of the requested
blocks, so memory is O(N * layers * d) rather than every token of every layer.
Both routes tokenise like the cache (window 512, appended EOS/SEP stripped)
and read the raw last block in place of HF's ``ln_f`` entry.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch

from ..common import get_layers

logger = logging.getLogger("blme")

LayerSpec = Union[Iterable[int], Callable[[int], Iterable[int]]]


def rows_policy(token_policy: str) -> str:
    """The ``select_rows`` policy a task's ``TOKEN_POLICY`` declares."""
    return "last" if token_policy == "last_per_sequence" else token_policy


def role_layers(n_layers: int) -> List[int]:
    """Block indices of the first, middle and last layer (deduplicated)."""
    return sorted({0, n_layers // 2, n_layers - 1})


def role_names(indices: List[int]) -> Dict[int, str]:
    names = {3: ("first", "mid", "last"), 2: ("first", "last"), 1: ("first",)}
    return dict(zip(indices, names.get(len(indices), ())))


def fallback_dataset(num_samples: int, label: str) -> List[Dict[str, str]]:
    """``dataset=None`` for a direct call: LAMBADA test texts, else bundled
    sentences when ``datasets`` is missing or offline (a run through
    ``blme.core.evaluate`` always passes its corpus)."""
    try:
        from datasets import load_dataset
        dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        return [{"text": dset[i]["text"]} for i in range(min(num_samples, len(dset)))]
    except Exception as e:
        logger.info(f"Warning: could not load lambada_openai ({type(e).__name__}: missing "
                    "`datasets`, offline, or download error); falling back to default examples.")
        return [{"text": f"{label} sample {i} for topological analysis of language models."}
                for i in range(num_samples)]


def _wanted(layers: LayerSpec, n_layers: int) -> List[int]:
    want = layers(n_layers) if callable(layers) else layers
    return sorted({int(l) for l in want if 0 <= int(l) < n_layers})


def _text(sample) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        return sample.get("text", "")
    return str(sample)


def last_token_points(model, tokenizer, dataset, cache=None, *, num_samples: int,
                      layers: LayerSpec, policy: str = "last",
                      use_cache: bool = True) -> Tuple[Dict[int, np.ndarray], Dict]:
    """``{block index: (N, d) float64}`` last-token clouds, and provenance.

    ``policy`` is passed to :func:`blme.tasks._tokens.select_rows` on the cache
    route; the private route implements ``"last"`` only.
    """
    from .._tokens import hidden_states_for_task, select_rows

    states, ginfo = hidden_states_for_task(
        model, tokenizer, dataset, cache, num_samples=int(num_samples),
        replace_final_norm=True, layer_idx="all", use_cache=use_cache, per_sample=True,
        min_cached_docs=int(num_samples), fallback="none")
    if states is not None:
        keys = sorted(states)
        want = _wanted(layers, len(keys))
        pts, positions, n_docs = {}, [], 0
        for l in want:
            rows, meta = select_rows(states[keys[l]], policy)
            pts[l] = rows.detach().to("cpu", torch.float64).numpy()
            positions, n_docs = meta.position.tolist(), meta.n_docs_used
        return pts, {"source": "shared_cache", "last_is_block": bool(ginfo["last_is_block"]),
                     "n_layers": len(keys), "positions": positions, "n_docs": int(n_docs),
                     "cache_rejected": None}
    if policy != "last":
        raise ValueError(f"the private last-token route implements policy 'last', not {policy!r}")
    return _private_last_rows(model, tokenizer, dataset, int(num_samples), layers,
                              ginfo.get("cache_rejected"))


def _private_last_rows(model, tokenizer, dataset, num_samples, layers, rejected):
    from ...cache import CACHE_TOKENIZER_MAX_LENGTH
    from .._tokens import strip_trailing_special

    blocks = get_layers(model)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("topology: could not locate the transformer blocks")
    n_layers = len(blocks)
    want = _wanted(layers, n_layers)
    device = next(model.parameters()).device
    sink = {}
    hook = blocks[-1].register_forward_hook(
        lambda mod, inp, out: sink.__setitem__(
            "h", (out[0] if isinstance(out, tuple) else out).detach()))
    rows = {l: [] for l in want}
    positions = []
    n_fwd = n_sub = 0
    try:
        with torch.no_grad():
            for i, sample in enumerate(dataset or []):
                if i >= num_samples:
                    break
                enc = tokenizer(_text(sample), return_tensors="pt", truncation=True,
                                max_length=CACHE_TOKENIZER_MAX_LENGTH)
                enc, _ = strip_trailing_special(enc, tokenizer)
                T = int(enc["input_ids"].shape[1])
                if T < 1:
                    continue
                sink.pop("h", None)
                out = model(**{k: v.to(device) for k, v in enc.items()},
                            output_hidden_states=True)
                hs = list(out.hidden_states)
                n_fwd += 1
                # post-final-norm: SUBSTITUTED -- the deepest entry is the raw
                # last block output, as on the cache route.
                h_raw = sink.get("h")
                if h_raw is not None and tuple(h_raw.shape) == tuple(hs[-1].shape):
                    hs[-1] = h_raw
                    n_sub += 1
                for l in want:
                    rows[l].append(hs[l + 1][0, T - 1].detach().to("cpu", torch.float64))
                positions.append(T - 1)
    finally:
        hook.remove()
    pts = {l: (torch.stack(v).numpy() if v else np.empty((0, 0))) for l, v in rows.items()}
    return pts, {"source": "self_collected", "last_is_block": bool(n_fwd > 0 and n_sub == n_fwd),
                 "n_layers": n_layers, "positions": positions, "n_docs": len(positions),
                 "cache_rejected": rejected}


def point_meta(info: Dict, cloud: np.ndarray) -> Dict[str, object]:
    """Provenance and sink-share ``_meta_`` keys for one task's clouds.

    The shares are computed on ``cloud`` (the middle layer's last-token cloud)
    with the shared helpers; a last-token cloud carries position 0 only for
    one-token documents.
    """
    from .._tokens import massive_token_share, position0_share
    docs = [torch.from_numpy(np.ascontiguousarray(cloud[i:i + 1])) for i in range(len(cloud))]
    pos = [torch.tensor([int(p)]) for p in info["positions"][:len(docs)]]
    out = {
        "_meta_n_points": int(len(cloud)),
        "_meta_points_source": info["source"],
        "_meta_points_cache_rejected": info["cache_rejected"],
        "_meta_rows_policy": "last",
        "_meta_last_entry_is_block_output": bool(info["last_is_block"]),
    }
    if docs:
        out["_meta_massive_token_share"] = massive_token_share(docs, positions=pos)
        out["_meta_position0_share"] = position0_share(docs, positions=pos)
    return out


def rips_diagrams(X: np.ndarray, maxdim: int = 1):
    """ripser on a float64 pairwise-distance matrix (``distance_matrix=True``).

    ripser's own point-cloud path computes distances from float32 coordinates,
    which loses fine structure that sits far from the centroid (two clusters a
    1e6 apart: 3% error on the within-cluster bars); float64 ``pdist`` does
    not. An out-of-range matrix is shifted by a power of two (exact) before
    ripser's float32 cast and the diagram is shifted back.

    Returns ``(dgms, scale)``; ``scale`` is the power of two divided out.
    """
    import math
    import warnings

    from ripser import ripser
    from scipy.spatial.distance import pdist, squareform

    X = np.asarray(X, dtype=np.float64)
    D = squareform(pdist(X - X.mean(axis=0, keepdims=True)))
    e = 0
    dmax = float(D.max()) if D.size else 0.0
    if dmax > 0 and np.isfinite(dmax) and not (1e-6 < dmax < 1e6):
        e = int(max(-126, min(127, round(math.log2(dmax)))))
        D = np.ldexp(D, -e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgms = ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"]
    return ([np.ldexp(d, e) for d in dgms] if e else list(dgms)), float(np.ldexp(1.0, e))
