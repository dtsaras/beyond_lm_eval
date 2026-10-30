"""Compare task results before and after a transform."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence

import numpy as np


def get_key(result: Dict[str, Any], key: str):
    """``a.b.c`` looks up nested dicts; key names may themselves contain dots
    (``alpha_0.5``), so the longest prefix present at each level wins."""
    if key in result:
        return result[key]
    parts = key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in result and isinstance(result[head], dict):
            return get_key(result[head], ".".join(parts[i:]))
    raise KeyError(key)


def rel_change(a, b) -> float:
    """|b - a| / |a| (elementwise max for sequences); 0 if equal, inf if a == 0 != b."""
    A = np.asarray(a, dtype=np.float64).ravel()
    B = np.asarray(b, dtype=np.float64).ravel()
    if A.shape != B.shape:
        return math.inf
    both_nan = np.isnan(A) & np.isnan(B)
    diff = np.where(both_nan, 0.0, np.abs(B - A))
    if np.isnan(diff).any():
        return math.inf
    scale = np.abs(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(diff == 0, 0.0, diff / scale)
    return float(np.max(r)) if r.size else 0.0


def assert_invariant(base: Dict[str, Any], other: Dict[str, Any], keys: Iterable[str],
                     rtol: float = 0.0) -> Dict[str, float]:
    """Assert each key moved by at most ``rtol`` (relative). rtol=0 is bit equality."""
    out, bad = {}, []
    for k in keys:
        r = rel_change(get_key(base, k), get_key(other, k))
        out[k] = r
        if not r <= rtol:
            bad.append(f"{k}: {get_key(base, k)!r} -> {get_key(other, k)!r} (rel {r:.3e})")
    if bad:
        raise AssertionError(f"not invariant at rtol={rtol:g}:\n  " + "\n  ".join(bad))
    return out


def assert_moves(base: Dict[str, Any], other: Dict[str, Any], key: str,
                 min_rel: float) -> float:
    """Assert ``key`` moved by at least ``min_rel`` (relative); return the change."""
    r = rel_change(get_key(base, key), get_key(other, key))
    if not r >= min_rel:
        raise AssertionError(
            f"{key} did not move: {get_key(base, key)!r} -> {get_key(other, key)!r} "
            f"(rel {r:.3e} < {min_rel:g})")
    return r


def uniform_layer_mean(profile: Sequence[float]) -> float:
    return float(np.mean(np.asarray(profile, dtype=np.float64)))
