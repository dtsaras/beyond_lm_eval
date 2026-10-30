"""
Persistence landscapes — Bubenik 2015 (JMLR, arXiv:1207.6437);
Bubenik & Dlotko 2017 toolbox reference (arXiv:1501.00179).

A persistence landscape is a functional summary of a persistence diagram
that is richer than scalar statistics (lifespans, entropy) and amenable
to statistical operations (mean, variance, hypothesis tests).

For each birth-death pair (b, d) in a persistence diagram, define the
*tent function* f(t) = max(0, min(t - b, d - t)). This is a triangle
peaked at the midpoint (b+d)/2 with height (d-b)/2.

The k-th persistence landscape L_k(t) is the k-th largest tent function
value at each t: sort all tent function values at t in decreasing order,
then L_k = the k-th value (1-indexed). L_1 captures the most prominent
topological feature, L_2 the second, etc.

Reported metrics (per layer prefix ``layer_{l}_{h0|h1}_``):
  - **landscape_integrals**: list of integrals of L_k over t, k = 1..K
  - **landscape_max_1**: max of the first landscape L_1 over t
  - **landscape_norm_1**: L2 norm of L_1
  - **mean_landscape_integral**: mean integral across k = 1..K

These complement the scalar persistence entropy and Betti curve tasks.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers
from ._points import (last_token_points, point_meta, rips_diagrams, role_layers, role_meta,
                      role_names, rows_policy)

logger = logging.getLogger("blme")

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


def _compute_landscape(dgm: np.ndarray, n_landscapes: int = 5,
                       n_points: int = 200) -> np.ndarray:
    """Compute the first `n_landscapes` persistence landscapes from a
    persistence diagram.

    Args:
        dgm: (N, 2) array of (birth, death) pairs (finite only).
        n_landscapes: number of landscape functions to compute.
        n_points: resolution of the discretised t-axis.

    Returns:
        (n_landscapes, n_points) array. L[k, i] = L_{k+1}(t_i).
    """
    # Filter out infinite-death and degenerate pairs.
    finite = dgm[np.isfinite(dgm[:, 1])] if dgm.size else dgm
    finite = finite[finite[:, 1] > finite[:, 0]]
    if finite.size == 0:
        return np.zeros((n_landscapes, n_points), dtype=np.float64)

    births = finite[:, 0]
    deaths = finite[:, 1]
    t_min = float(births.min())
    t_max = float(deaths.max())
    if t_max <= t_min:
        return np.zeros((n_landscapes, n_points), dtype=np.float64)

    ts = np.linspace(t_min, t_max, n_points)
    # Evaluate all tent functions at all t-points: (N_pairs, n_points)
    # f_i(t) = max(0, min(t - b_i, d_i - t))
    left = ts[np.newaxis, :] - births[:, np.newaxis]   # (N, n_points)
    right = deaths[:, np.newaxis] - ts[np.newaxis, :]   # (N, n_points)
    tents = np.maximum(0.0, np.minimum(left, right))     # (N, n_points)

    # At each t-point, sort tent values descending and take the top K.
    sorted_vals = np.sort(tents, axis=0)[::-1]  # (N, n_points) descending
    K = min(n_landscapes, sorted_vals.shape[0])
    landscapes = np.zeros((n_landscapes, n_points), dtype=np.float64)
    landscapes[:K, :] = sorted_vals[:K, :]

    return landscapes


# Above this many bars the O(n^2) pairwise-intersection candidate set is not
# built in full (9e6 doubles, ~72 MB, at the cap); see _landscape_stats.
_PAIRWISE_CAND_CAP = 3000


def _topk_tents(b, dth, ts, k):
    """Top-``k`` tent values at every point of ``ts``, evaluated in chunks so
    peak memory is ``O(n_bars * chunk)``; each column is independent, so the
    result equals the dense evaluation."""
    n = ts.size
    out = np.zeros((k, n), dtype=np.float64)
    chunk = max(1, int(4_000_000 // max(1, b.size)))
    for s0 in range(0, n, chunk):
        t = ts[s0:s0 + chunk]
        tents = np.maximum(0.0, np.minimum(t[None, :] - b[:, None],
                                           dth[:, None] - t[None, :]))
        kk = min(k, tents.shape[0])
        part = np.partition(tents, -kk, axis=0)[-kk:]
        part = np.sort(part, axis=0)[::-1]
        out[:kk, s0:s0 + chunk] = part
    return out


def _landscape_stats(landscapes: np.ndarray, dt: float,
                     dgm: np.ndarray = None) -> Dict[str, float]:
    """Summary statistics from a (K, n_points) landscape array.

    With the diagram given, maxima, integrals and L2 norms are evaluated
    exactly on the landscape's critical points (births, deaths, apexes and
    pairwise tent crossings), where every lambda_k is piecewise linear; the
    grid values are the fallback and are labelled by the status keys.
    """
    # np.trapezoid (NumPy 2.0) replaces np.trapz. Looked up lazily: the eager
    # getattr(np, "trapezoid", np.trapz) form evaluates np.trapz, gone in 2.x.
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    K = landscapes.shape[0]
    integrals = []
    maxes = []
    norms = []
    for k in range(K):
        Lk = landscapes[k]
        integrals.append(float(_trapz(Lk, dx=dt)))
        maxes.append(float(Lk.max()))

        norms.append(float(np.sqrt(_trapz(Lk ** 2, dx=dt))))
    # lambda_k's maximum is attained at a breakpoint (a birth, death, apex or
    # crossing of two tents), not at the k-th largest half-persistence for
    # k >= 2: {[1,6],[2,8],[3,4]} has maxima (3.0, 2.0, 0.5).
    n_land = landscapes.shape[0]
    exact_maxes = list(maxes)
    _max_status = "grid"
    _quad_status = "grid_trapezoid"
    _n_bars = 0
    if dgm is not None and len(dgm):
        _d = np.asarray(dgm, dtype=np.float64)
        _d = _d[np.isfinite(_d).all(axis=1)]
        _d = _d[_d[:, 1] > _d[:, 0]]
        if len(_d):
            _n_bars = int(len(_d))
            b, dth = _d[:, 0], _d[:, 1]
            cands = [b, dth, (b + dth) / 2.0]
            # Pairwise tent crossings: the rising edge of one tent meets the
            # falling edge of another at (b_i + d_j)/2. When every birth is
            # IDENTICAL (ripser's H0) these coincide with the per-bar apexes,
            # so the set is complete at any size.
            _all_same_birth = bool(np.all(b == b[0])) if b.size else False
            if len(_d) <= _PAIRWISE_CAND_CAP:
                bb = b[:, None] + dth[None, :]
                cands.append((bb / 2.0).ravel())
                _exact_ok = True
            else:
                # Above the cap: a lower bound L on lambda_K's maximum (from
                # the cheap candidates) means only bars with half-persistence
                # >= L can be among the K tallest tents at the argmax, so their
                # pairwise crossings suffice for the maxima.
                _half = (dth - b) / 2.0
                _ts0 = np.unique(np.concatenate(cands))
                _t0 = np.maximum(0.0, np.minimum(_ts0[None, :] - b[:, None],
                                                 dth[:, None] - _ts0[None, :]))
                _srt0 = np.sort(_t0, axis=0)[::-1]
                _L = float(_srt0[min(n_land, _srt0.shape[0]) - 1].max()) \
                    if _srt0.size else 0.0
                _sel = np.flatnonzero(_half >= _L)
                # At least K bars always survive.
                if _sel.size < n_land:
                    _sel = np.argsort(_half)[::-1][:n_land]
                if _sel.size * _sel.size <= _PAIRWISE_CAND_CAP ** 2:
                    _bb = b[_sel][:, None] + dth[_sel][None, :]
                    cands.append((_bb / 2.0).ravel())
                    _exact_ok = True
                    _n_sufficient_set = int(_sel.size)
                else:
                    _exact_ok = _all_same_birth
                    _n_sufficient_set = int(_sel.size)
            ts_c = np.unique(np.concatenate(cands))
            srt = _topk_tents(b, dth, ts_c, n_land)
            # Every candidate and grid point is a genuine evaluation, so the
            # max over their union is never worse than either alone.
            _cand_maxes = [
                float(srt[k].max()) if k < srt.shape[0] else 0.0
                for k in range(n_land)
            ]
            exact_maxes = [max(_c, _g) for _c, _g in zip(_cand_maxes, maxes)]
            # Maxima and quadrature have different sufficiency conditions: the
            # restricted set above is exact for the argmax only, while the
            # integrals need every breakpoint at every t.
            _full_pairwise = bool(len(_d) <= _PAIRWISE_CAND_CAP or _all_same_birth)
            _max_status = "exact" if _exact_ok else "lower_bound"
            _quad_status = ("exact" if _full_pairwise
                            else ("refined_approximation" if _exact_ok
                                  else "grid_trapezoid"))

            # Between consecutive critical points every lambda_k is linear:
            #   integral of a segment       = (b-a) * (y_a + y_b) / 2
            #   integral of its square      = (b-a) * (y_a^2 + y_a*y_b + y_b^2) / 3
            if not _exact_ok:
                # Incomplete candidate set: integrate on its union with the
                # uniform grid, a refinement of both.
                _grid = np.linspace(float(ts_c.min()), float(ts_c.max()),
                                    landscapes.shape[1])
                ts_c = np.unique(np.concatenate([ts_c, _grid]))
                srt = _topk_tents(b, dth, ts_c, n_land)
                _cand_maxes = [
                    float(srt[k].max()) if k < srt.shape[0] else 0.0
                    for k in range(n_land)
                ]
                exact_maxes = [max(_c, _g) for _c, _g in zip(_cand_maxes, maxes)]
                _quad_status = "refined_lower_bound"
            _dt_seg = np.diff(ts_c)
            if _dt_seg.size and _quad_status != "grid_trapezoid":
                _ex_int, _ex_norm = [], []
                for k in range(n_land):
                    yk = srt[k] if k < srt.shape[0] else np.zeros_like(ts_c)
                    ya, yb = yk[:-1], yk[1:]
                    _ex_int.append(float(np.sum(_dt_seg * (ya + yb) / 2.0)))
                    # The ordinates and widths are factored out before squaring
                    # (exact homogeneity), so tiny scales do not flush to zero.
                    _ym = float(np.max(np.abs(yk))) if yk.size else 0.0
                    _dm = float(np.max(_dt_seg)) if _dt_seg.size else 0.0
                    if _ym > 0 and _dm > 0:
                        _yan, _ybn = ya / _ym, yb / _ym
                        _val = np.sum((_dt_seg / _dm)
                                      * (_yan * _yan + _yan * _ybn + _ybn * _ybn) / 3.0)
                        _ex_norm.append(float(_ym * np.sqrt(_dm) * np.sqrt(max(_val, 0.0))))
                    else:
                        _ex_norm.append(0.0)
                integrals_grid, norms_grid = integrals, norms
                integrals, norms = _ex_int, _ex_norm

    return {
        "landscape_integrals": integrals,
        "landscape_maxes": exact_maxes,
        "landscape_maxes_grid": maxes,
        "landscape_norms": norms,
        # How exact the numbers above are (always emitted).
        "_meta_landscape_max_status": _max_status,
        "_meta_landscape_quadrature": _quad_status,
        "_meta_landscape_n_bars": _n_bars,
        "mean_landscape_integral": float(np.mean(integrals)),
    }


_ROLES = ("first", "mid", "last")
_TWINS = ("mean_landscape_integral_mpd2", "landscape_max_1_mpd", "landscape_norm_1_mpd15")
_HEADLINES = tuple(f"{r}_layer_h1_{k}" for r in _ROLES
                   for k in ("mean_landscape_integral", "landscape_max_1") + _TWINS)
_SIZE_INVARIANT = {
    f"{r}_layer_h1_{k}": (("embed_isometric", "zero_pad", "rescale"),
                          "tests/tasks/invariance/test_topology_persistence_landscape.py::"
                          "test_mpd_twins_are_width_and_scale_free")
    for r in _ROLES for k in _TWINS}


def _subsamples(n: int, size: int, k: int, seed: int) -> List[np.ndarray]:
    """``k`` seeded draws of ``size`` rows without replacement; one draw of
    every row when the cloud is no larger than ``size``."""
    if n <= size or k <= 1:
        return [np.arange(n)]
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n, size=size, replace=False)) for _ in range(k)]


def _mean_value(values):
    if values and isinstance(values[0], list):
        return [float(x) for x in np.mean(np.asarray(values, dtype=np.float64), axis=0)]
    return float(np.mean(values)) if values else float("nan")


def _layer_landscape(X: np.ndarray, n_landscapes: int, n_points: int) -> Dict[str, dict]:
    """Landscape summaries of one cloud, per homology dimension, in the
    cloud's own units."""
    # ripser reads float64 pairwise distances, divided by a power of two when
    # out of float32's range; the degree-1/2/1.5 restores undo the shift.
    dgms, _resc = rips_diagrams(X)
    dgms = [d / _resc for d in dgms]
    out = {}
    for dim_idx, dim_label in enumerate(["h0", "h1"]):
        if dim_idx >= len(dgms):
            continue
        dgm = dgms[dim_idx]
        landscapes = _compute_landscape(dgm, n_landscapes, n_points)
        finite = dgm[np.isfinite(dgm[:, 1])]
        finite = finite[finite[:, 1] > finite[:, 0]] if finite.size else finite
        if finite.size > 0:
            dt = (finite[:, 1].max() - finite[:, 0].min()) / max(1, n_points - 1)
        else:
            dt = 1.0
        stats = _landscape_stats(landscapes, dt, dgm=finite)
        # A max is degree 1 in the metric, an integral degree 2, an L2 norm 3/2.
        _m1, _m2, _m32 = _resc, _resc ** 2, _resc ** 1.5
        vals = {
            "mean_landscape_integral": stats["mean_landscape_integral"] * _m2,
            "landscape_max_1": (stats["landscape_maxes"][0] * _m1
                                if stats["landscape_maxes"] else 0.0),
            "landscape_norm_1": (stats["landscape_norms"][0] * _m32
                                 if stats["landscape_norms"] else 0.0),
            "landscape_integrals": [v * _m2 for v in stats["landscape_integrals"]],
        }
        # Twins divided by the cloud's mean pairwise distance to the matching
        # degree: scale- and width-free (test_topology_persistence_landscape.py).
        mpd = float(pdist(np.asarray(X, dtype=np.float64)).mean()) if len(X) > 1 else 0.0
        if mpd > 0:
            vals["mean_landscape_integral_mpd2"] = vals["mean_landscape_integral"] / mpd ** 2
            vals["landscape_max_1_mpd"] = vals["landscape_max_1"] / mpd
            vals["landscape_norm_1_mpd15"] = vals["landscape_norm_1"] / mpd ** 1.5
        status = {k: stats[f"_meta_{k}"] for k in
                  ("landscape_max_status", "landscape_quadrature", "landscape_n_bars")}
        out[dim_label] = {"vals": vals, "status": status, "prescale": float(_resc)}
    return out


@register_task("topology_persistence_landscape")
class PersistenceLandscapeTask(DiagnosticTask):
    """Persistence landscapes (Bubenik 2015) at early / mid / late layers."""

    TOKEN_POLICY = "last_per_sequence"
    SOURCES = {
        "paper": "arXiv:1207.6437",
        "code": "scikit-tda/persim 0.3.8 (installed package; landscape parity tests)",
        "llm_precedent": ("arXiv:2505.20435 (Fay et al. 2025: VR barcodes of last-token "
                          "clouds, summaries averaged over subsamples)"),
    }
    ADAPTATION_NOTE = (
        "Persistence landscapes (Bubenik 2015) of the Vietoris-Rips barcodes of one point per "
        "sequence, its last token, at the first, middle and last block (the LLM precedent's "
        "point cloud, Fay et al. 2025), with N = 1000 sequences. Following Bubenik's statistical "
        "use, each summary is the mean over 16 subsamples of 500 points: sum_{k<=5} of the "
        "landscape integrals (Eq. 3.4, reported divided by K), the sup of lambda_1 (Prop. 2.5) "
        "and the L2 norm of lambda_1. For H0 every bar is born at 0, so these are closed forms "
        "of the largest H0 deaths and are _meta_; the H1 summaries are the features. Raw values "
        "carry the representation's units (degree 2, 1 and 1.5), so each is also divided by the "
        "mean pairwise distance to that power, which removes a rescale and a width change "
        "exactly; the twins still grow with N, so models are compared at the same N."
    )
    HEADLINE_KEYS = _HEADLINES
    SIZE_INVARIANT = _SIZE_INVARIANT

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistence Landscape Analysis...")

        if not HAS_RIPSER:
            return {"error": "ripser not installed. Install with: pip install ripser"}

        # Q1: one point per sequence (its last token), N = 1000 for every model.
        num_samples = int(self.config.get("num_samples", 1000))
        n_landscapes = self.config.get("n_landscapes", 5)
        n_points = self.config.get("n_points", 200)
        use_cache = self.config.get("use_cache", True)
        n_sub = int(self.config.get("n_subsamples", 16))
        sub_size = int(self.config.get("subsample_size", 500))
        seed = int(self.config.get("seed", 42))

        if dataset is None:
            dataset = [{"text": f"Sample {i} for topological landscape analysis."}
                       for i in range(num_samples)]
        if get_layers(model) is None:
            return {"error": "Could not detect layers"}
        pts, pinfo = last_token_points(
            model, tokenizer, dataset, cache, num_samples=num_samples, layers=role_layers,
            policy=rows_policy(self.TOKEN_POLICY), use_cache=use_cache)
        target_layers = sorted(pts)
        layer_reps: Dict[int, List[np.ndarray]] = {l: list(pts[l]) for l in target_layers}
        _last_is_block = bool(pinfo["last_is_block"])

        results: Dict[str, object] = {}
        # Depth-role names (first/mid/last); absolute-index names alias
        # different depths across models and are _meta_.
        roles = role_names(target_layers)
        results.update(role_meta(roles, pinfo["n_layers"]))

        # Bubenik's statistical use: the landscape is a random variable whose
        # mean is estimated over samples. Each layer's cloud is subsampled (the
        # same documents at every layer) and the summaries are averaged; the
        # integrals are linear, so their mean is the mean landscape's integral.
        subsets = _subsamples(min(len(v) for v in layer_reps.values()), sub_size, n_sub, seed)
        results["_meta_landscape_n_subsamples"] = int(len(subsets))
        results["_meta_landscape_subsample_size"] = int(len(subsets[0])) if subsets else 0

        for l_idx in target_layers:
            if len(layer_reps[l_idx]) < 3:
                continue
            X = np.stack(layer_reps[l_idx])
            per = [_layer_landscape(X[idx], n_landscapes, n_points) for idx in subsets]
            full = _layer_landscape(X, n_landscapes, n_points)
            for dim_label in ("h0", "h1"):
                if dim_label not in full:
                    continue
                prefix = f"layer_{l_idx}_{dim_label}"
                rows = [p[dim_label] for p in per if dim_label in p]
                vals = {k: _mean_value([r["vals"][k] for r in rows]) for k in full[dim_label]["vals"]}
                # H0 bars are all born at 0, so its landscape summaries are closed
                # forms of the largest deaths (integral of lambda_k = d_(k)^2 / 4,
                # max = d_(1) / 2): topology_homology's H0 lifetimes again. _meta_.
                pfx = "_meta_" if dim_label == "h0" else ""
                for k, v in vals.items():
                    results[f"_meta_{prefix}_{k}"] = v
                    if l_idx in roles:
                        results[f"{pfx}{roles[l_idx]}_layer_{dim_label}_{k}"] = v
                # The single landscape of the whole cloud, for continuity.
                for k, v in full[dim_label]["vals"].items():
                    results[f"_meta_{prefix}_full_{k}"] = v
                # Status keys carry the _meta_ prefix at the START of the name
                # (they were layer_<l>_<dim>_meta_*, which a reader testing
                # startswith("_meta_") took for features).
                results[f"_meta_{prefix}_last_entry_is_block_output"] = bool(
                    _last_is_block) if l_idx == target_layers[-1] else True
                results[f"_meta_{prefix}_prescale"] = float(full[dim_label]["prescale"])
                for _sk in ("landscape_max_status", "landscape_quadrature",
                            "landscape_n_bars"):
                    results[f"_meta_{prefix}_{_sk}"] = full[dim_label]["status"][_sk]
                results[f"_meta_{prefix}_n_inexact_subsamples"] = int(sum(
                    r["status"]["landscape_quadrature"] != "exact" for r in rows))

        if not results:
            return {"error": "No layers produced landscape features"}
        results.update(point_meta(pinfo, pts[target_layers[len(target_layers) // 2]]))
        return results
