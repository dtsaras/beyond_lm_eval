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

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_layers

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
    # Transpose to (n_points, N), sort along axis=1 descending, take :K.
    sorted_vals = np.sort(tents, axis=0)[::-1]  # (N, n_points) descending
    K = min(n_landscapes, sorted_vals.shape[0])
    landscapes = np.zeros((n_landscapes, n_points), dtype=np.float64)
    landscapes[:K, :] = sorted_vals[:K, :]

    return landscapes


# Cap on the O(n^2) pairwise-intersection candidate set. Above it the
# critical-point set is incomplete and the "exact" segment sums are NOT
# substituted (see _landscape_stats).
#
# Raised 400 -> 3000 (2026-09-04). 400 put ordinary diagrams on the
# approximate path for no real saving: a 401-bar diagram needs 160,801
# candidates, which is ~1.3 MB and milliseconds, and on the approximate path
# lambda_4/lambda_5 maxima came out at 2.349163/2.242039 against persim's
# 2.352079/2.252138 (-0.124% / -0.448%) (adversarial review, 2026-09-04). At
# 3000 the array is 9e6 doubles (~72 MB), which is the point where the memory
# genuinely starts to matter.
_PAIRWISE_CAND_CAP = 3000


def _topk_tents(b, dth, ts, k):
    """Top-``k`` tent values at every point of ``ts``, evaluated in CHUNKS.

    The dense form ``tents = maximum(0, minimum(ts - b, d - ts))`` materialises
    an ``(n_bars, n_points)`` array, which is the real scaling limit here: at
    3,000 bars the candidate set has up to 9e6 points, so the array is 2.2e10
    doubles -- about 180 GB. The cap that was supposed to bound COST was
    therefore bounding nothing useful and would have OOMed long before it fired
    (found while verifying the round-5 landscape restriction, 2026-09-05).

    Chunking over points makes peak memory ``O(n_bars * chunk)`` and leaves the
    result bit-identical, since each column is independent.
    """
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
    """Summary statistics from a (K, n_points) landscape array."""
    # np.trapezoid (NumPy 2.0) replaces np.trapz, which was REMOVED in 2.0.
    # Must be a lazy lookup: the eager `getattr(np, "trapezoid", np.trapz)`
    # form raises AttributeError on NumPy 2.x (np.trapz evaluated as the
    # default arg). Numerically identical (same algorithm, renamed).
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
    # EXACT maxima, not grid maxima.
    #
    # Every tent peaks at (b+d)/2 with height (d-b)/2. An evenly spaced grid
    # need not contain that midpoint, so the grid max is systematically LOW --
    # and for ripser's H0 bars, which all start at 0, it is low by exactly
    # 1/(n_points-1): measured 0.5025% on EVERY real gpt2 H0 layer (3.572845
    # against 3.590890, 10.596878 against 10.650397, 14.002644 against
    # 14.073364) at n_points = 200 (adversarial review, 2026-09-03).
    #
    # L_1's exact maximum is the largest half-persistence; L_k's is the k-th
    # largest, since at that tent's own midpoint no other tent can exceed it in
    # rank order. This is a closed form and needs no grid at all.
    # EXACT maxima via CRITICAL POINTS, not the even grid.
    #
    # L_k is piecewise linear, so its maximum is attained at a breakpoint: a
    # tent's birth, death or midpoint, or an intersection of two tents.
    # Evaluating there is exact; an even grid is not, and for ripser's H0 bars
    # -- which all start at 0 -- it is low by exactly 1/(n_points-1), measured
    # at 0.5025% on EVERY real gpt2 H0 layer at n_points = 200 (adversarial
    # review, 2026-09-03).
    #
    # NOTE the k >= 2 subtlety: L_1's max IS the largest half-persistence, but
    # L_k's is NOT the k-th largest. At the [1,6] tent's midpoint the [2,8] tent
    # is only 1.5, so for {[1,6],[2,8],[3,4]} the exact maxima are
    # (3.0, 2.0, 0.5), not (3.0, 2.5, 0.5). Hence the candidate-point sweep
    # rather than a sort.
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
            # Pairwise tent intersections: the rising edge of one meets the
            # falling edge of another at (b_i + d_j)/2.
            #
            # The n^2 candidate set is capped for cost. Above the cap the set is
            # INCOMPLETE, so the segment sums below are no longer exact and must
            # not be substituted: the error jumps from 1.3e-11 at 400 bars to
            # 4.9e-4 at 401. `_exact_ok` records whether the set is complete.
            #
            # One exception makes the cap harmless in the common case: when all
            # births are equal (ripser's H0, where every bar starts at 0) the
            # pairwise midpoints (b_i + d_j)/2 coincide with the per-bar
            # midpoints already in the set, so nothing is lost at any size.
            # EXACT equality, not np.allclose. The claim being made is that
            # every pairwise midpoint (b_i + d_j)/2 already lies in the
            # candidate set, and that is an algebraic identity that holds only
            # when the births are IDENTICAL. allclose's 1e-5 relative /1e-8
            # absolute window certified a diagram of {[1,6],[2,8]} plus 399
            # copies of [100,100.001] translated by +1e9 as "same birth", and
            # the resulting integrals were not exact at all (adversarial
            # review, 2026-09-04).
            _all_same_birth = bool(np.all(b == b[0])) if b.size else False
            if len(_d) <= _PAIRWISE_CAND_CAP:
                bb = b[:, None] + dth[None, :]
                cands.append((bb / 2.0).ravel())
                _exact_ok = True
            else:
                # ABOVE THE CAP, RESTRICT SOUNDLY -- with a proof, not a
                # heuristic.
                #
                # The previous rule kept the pairwise set of the most PERSISTENT
                # bars. That is not sound: a shorter bar can enter the top K
                # where the longer ones have disjoint support, and on a valid
                # 3,001-bar diagram it reported lambda_5 = 1.5 against the
                # authors' Persistence Landscape Toolbox at 2.0, a 25% error
                # (independent Opus review, 2026-09-05).
                #
                # The sound argument: let t* be where lambda_K attains its
                # maximum. At t* the K tallest tents each have VALUE >=
                # lambda_K(t*), and a tent's value never exceeds its
                # half-persistence, so every one of them has half-persistence
                # >= lambda_K(t*). Therefore any LOWER BOUND L on lambda_K(t*)
                # gives a provably sufficient set S = {bars with
                # half-persistence >= L}: no bar outside S can be among the K
                # tallest at t*, so no crossing involving one can be a
                # breakpoint of lambda_K there.
                #
                # L comes free from the cheap candidate set already built
                # (births, deaths, apexes), which lower-bounds every lambda_k
                # because each is a genuine evaluation. Iterating tightens S.
                _half = (dth - b) / 2.0
                _ts0 = np.unique(np.concatenate(cands))
                _t0 = np.maximum(0.0, np.minimum(_ts0[None, :] - b[:, None],
                                                 dth[:, None] - _ts0[None, :]))
                _srt0 = np.sort(_t0, axis=0)[::-1]
                _L = float(_srt0[min(n_land, _srt0.shape[0]) - 1].max()) \
                    if _srt0.size else 0.0
                _sel = np.flatnonzero(_half >= _L)
                # A tent can only be in the top K if at least K tents reach L,
                # so K bars always survive; keep at least that many.
                if _sel.size < n_land:
                    _sel = np.argsort(_half)[::-1][:n_land]
                if _sel.size * _sel.size <= _PAIRWISE_CAND_CAP ** 2:
                    _bb = b[_sel][:, None] + dth[_sel][None, :]
                    cands.append((_bb / 2.0).ravel())
                    # EXACT: S provably contains every bar that can define a
                    # top-K breakpoint at the argmax.
                    _exact_ok = True
                    _n_sufficient_set = int(_sel.size)
                else:
                    _exact_ok = _all_same_birth
                    _n_sufficient_set = int(_sel.size)
            ts_c = np.unique(np.concatenate(cands))
            srt = _topk_tents(b, dth, ts_c, n_land)
            # Every candidate AND every grid point is a genuine evaluation of
            # lambda_k, so each is a lower bound on its true maximum; the max
            # over their union is therefore never worse than either alone. On
            # the complete-candidate path this is exact. On the incomplete path
            # it stays a lower bound rather than silently replacing the grid
            # answer with a possibly-worse one, which is what the unconditional
            # substitution did (adversarial review, 2026-09-04).
            _cand_maxes = [
                float(srt[k].max()) if k < srt.shape[0] else 0.0
                for k in range(n_land)
            ]
            exact_maxes = [max(_c, _g) for _c, _g in zip(_cand_maxes, maxes)]
            # MAXIMA and QUADRATURE have DIFFERENT sufficiency conditions.
            #
            # The sufficient-set proof is about the ARGMAX of lambda_K: only
            # bars whose half-persistence reaches a lower bound on
            # lambda_K(t*) can be among the K tallest THERE. Quadrature is a
            # different claim -- it needs every breakpoint at every t -- and
            # propagating one `_exact_ok` to both labelled the integrals
            # "exact" when they were not. On the 3,001-bar diagram the omitted
            # crossing at t = 4.0 costs lambda_1's integral +2.4618% and
            # lambda_2's -3.8285%, confirmed against the authors' compiled
            # Persistence Landscape Toolbox, persim 0.3.8 and a full-pairwise
            # reference (independent Opus review, 2026-09-05). The errors have
            # BOTH signs in one call, so it is not even a bound.
            #
            # The maxima keep their (verified exact) status; the integrals are
            # labelled for what they are.
            # `_all_same_birth` is ALSO a complete-candidate condition, not
            # just a maxima one: when every bar starts at the same b, each
            # pairwise crossing (b_i + d_j)/2 already equals a per-bar midpoint
            # (b + d_j)/2, so the candidate set contains every breakpoint and
            # the quadrature is exact. Omitting it made the label CONSERVATIVE
            # rather than wrong -- an all-same-birth 3,510-bar H0 diagram is
            # exact (rel 0.0) but was reported as `refined_approximation`
            # (independent Opus review, 2026-09-05). ripser's H0 is exactly
            # this case, so it mattered for the common path.
            _full_pairwise = bool(len(_d) <= _PAIRWISE_CAND_CAP or _all_same_birth)
            _max_status = "exact" if _exact_ok else "lower_bound"
            _quad_status = ("exact" if _full_pairwise
                            else ("refined_approximation" if _exact_ok
                                  else "grid_trapezoid")) 

            # EXACT INTEGRALS AND L2 NORMS, on the same critical points.
            #
            # The uniform grid does not merely lose precision -- it ERASES any
            # bar narrower than one grid step. On {[0,1000], [100,100.1]} the
            # exact integrals are [250000.0, 0.0025] and the grid returned
            # [249993.687, 0.0, 0.0, 0.0, 0.0]: lambda_2 reported as identically
            # zero when it is not. Raising n_points shrinks the window but never
            # closes it.
            #
            # Between consecutive critical points every tent is linear, so the
            # k-th largest of them is linear too, and both quantities have exact
            # closed forms there:
            #   integral of a linear segment  = (b-a) * (y_a + y_b) / 2
            #   integral of its SQUARE        = (b-a) * (y_a^2 + y_a*y_b + y_b^2) / 3
            # The square is quadratic, so a trapezoid would NOT be exact for the
            # norm; the three-term form is (it is Simpson's rule for a quadratic).
            if not _exact_ok:
                # Above the cap the candidate set is incomplete, but it is
                # still a far better quadrature grid than a fixed 200-point
                # uniform one -- and taking the UNION of the two is a strict
                # refinement of BOTH, so it can only improve on either. The
                # previous fallback threw the candidates away and integrated
                # the uniform grid alone, which erases any bar narrower than a
                # grid step: 2,999 copies of [100, 100.001] were invisible
                # (adversarial review round 2, 2026-09-04). Recompute the
                # tent values on the union before integrating.
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
                    # SCALE-EXACT: the integrand squares the ordinates, so
                    # for D = {[0,3],[1,5]} scaled by 1e-109 the sum flushed to
                    # zero and landscape_norm_1 read 0 against the exact
                    # homogeneous 8.316649967e-164 (adversarial review round 3,
                    # 2026-09-04). The L2 norm is homogeneous of degree 1 in
                    # the ordinates and degree 1/2 in the segment widths, so
                    # factoring both out is exact.
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
        # APPROXIMATION STATUS, always emitted. Above the pairwise cap the
        # critical-point set is incomplete, the maxima are lower bounds and the
        # integrals/norms fall back to grid quadrature. Silently returning
        # 200-point quadrature with no status field was itself the defect.
        "_meta_landscape_max_status": _max_status,
        "_meta_landscape_quadrature": _quad_status,
        "_meta_landscape_n_bars": _n_bars,
        "mean_landscape_integral": float(np.mean(integrals)),
    }


@register_task("topology_persistence_landscape")
class PersistenceLandscapeTask(DiagnosticTask):
    """Persistence landscapes (Bubenik 2015) at early / mid / late layers."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Persistence Landscape Analysis...")

        if not HAS_RIPSER:
            return {"error": "ripser not installed. Install with: pip install ripser"}

        num_samples = self.config.get("num_samples", 20)
        n_landscapes = self.config.get("n_landscapes", 5)
        n_points = self.config.get("n_points", 200)

        if dataset is None:
            dataset = [
                {"text": f"Sample {i} for topological landscape analysis."}
                for i in range(num_samples)
            ]

        device = next(model.parameters()).device
        layers = get_layers(model)
        if layers is None:
            return {"error": "Could not detect layers"}
        n_layers = len(layers)
        # Dedupe so models with <=2 layers don't collect (and process) the
        # same layer twice — same fix as topology_homology.
        target_layers = sorted({0, n_layers // 2, n_layers - 1})

        # Collect mean-pooled per-sentence representations at target layers.
        layer_reps: Dict[int, List[np.ndarray]] = {l: [] for l in target_layers}

        # RECOVER THE DEEPEST BLOCK'S RAW OUTPUT with a forward hook.
        #
        # HF's last `hidden_states` entry is `ln_f(b_{L-1})`, not a block
        # output, so the deepest layer's landscape was computed on a LayerNorm
        # result: its statistics differ from the true last block's by
        # -83.6% / -93.3% / -97.5% (independent Opus review, 2026-09-04). This
        # is the systemic defect already fixed in six other tasks; this one was
        # missed because it reads `out.hidden_states` directly rather than
        # going through `collect_hidden_states`.
        _sink = {}
        _hook = layers[-1].register_forward_hook(
            lambda mod, inp, outp: _sink.__setitem__(
                "h", (outp[0] if isinstance(outp, tuple) else outp).detach()))
        _n_fwd = _n_sub = 0
        try:
            with torch.no_grad():
                for i, s in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = s["text"] if isinstance(s, dict) else str(s)
                    enc = tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=128).to(device)
                    _sink.pop("h", None)
                    out = model(**enc, output_hidden_states=True)
                    # post-final-norm: SUBSTITUTED -- the deepest sampled layer must be a
                    # block output like every other entry in this profile.
                    _hs = list(out.hidden_states)
                    _n_fwd += 1
                    _h_raw = _sink.get("h")
                    if (_h_raw is not None
                            and tuple(_h_raw.shape) == tuple(_hs[-1].shape)):
                        _hs[-1] = _h_raw
                        _n_sub += 1
                    for l_idx in target_layers:
                        h = _hs[l_idx + 1][0]  # (T, D)
                        layer_reps[l_idx].append(
                            # The trailing .float() DISCARDED the precision the .double() was added to
                            # gain: promote, average, then narrow back is self-defeating. Removing only
                            # this cast moved geometry_correlation_dimension from 7.486064047571674 to
                            # 15.295160200026809 on native Pythia-70m hidden states (+104%), matching an
                            # independently reconstructed Grassberger-Procaccia value, and restored a
                            # lossless translation from 0.0 to 2.3540184486398976 (codex round 15,
                            # 2026-09-09). Swept as a class: 7 sites carried this exact pattern.
                            h.double().mean(dim=0).cpu().numpy())
        finally:
            _hook.remove()
        _last_is_block = bool(_n_fwd > 0 and _n_sub == _n_fwd)

        results: Dict[str, object] = {}

        for l_idx in target_layers:
            if len(layer_reps[l_idx]) < 3:
                continue
            X = np.stack(layer_reps[l_idx])
            # GLOBAL PRE-RESCALE before ripser. Persistent homology is
            # homogeneous of degree 1 in the metric, so dividing by the largest
            # magnitude only shifts exponents -- but ripser consumes float32
            # and at cloud scale 1e-25, with all 160 coordinates still nonzero,
            # 19 of the H0 bars collapsed to zero length and every statistic
            # read 0.0 (independent Opus review, 2026-09-04). The scale is
            # multiplied back into the length-1-homogeneous outputs below.
            # CENTRE FIRST, then measure the scale.
            #
            # `_xmax` is the max of the POINT CLOUD, so a common offset sets the
            # rescale rather than the geometry does: translating by +1e6 makes
            # `_xmax` ~1e6, the guard fires, and `X / 1e6` shrinks the OFFSET
            # along with the structure, leaving `1 + P*1e-6` with barely a digit
            # of the geometry left. Rips persistence is translation invariant --
            # the raw ripser output on the translated cloud is unchanged (39 H0
            # deaths at 0.125, H1 [0.125, 1.25] on the reviewer's 40x8
            # square-perimeter fixture) -- yet the reported H0
            # integral/L2/max moved (gpt-5.6-sol round 9, 2026-09-07).
            #
            # Subtracting the centroid is a mathematical no-op for the diagram
            # and makes `_xmax` describe the data, so the rescale fires on
            # genuine extremes only. Third topology site to need this today,
            # after `homology` and `persistence_entropy`.
            #
            # The landscape ordinates stay homogeneous in the METRIC, which
            # centring does not touch, so the degree-1/2/1.5 restores below are
            # unaffected.
            X = np.asarray(X, dtype=np.float64)
            X = X - X.mean(axis=0, keepdims=True)
            _xmax = float(np.max(np.abs(X))) if X.size else 0.0
            _resc = 1.0
            if _xmax > 0 and np.isfinite(_xmax) and not (1e-6 < _xmax < 1e6):
                X = X / _xmax
                _resc = _xmax

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dgms = ripser(X, maxdim=1)["dgms"]

            for dim_idx, dim_label in enumerate(["h0", "h1"]):
                if dim_idx >= len(dgms):
                    continue
                dgm = dgms[dim_idx]
                landscapes = _compute_landscape(dgm, n_landscapes, n_points)
                # Compute dt for integration.
                finite = dgm[np.isfinite(dgm[:, 1])]
                finite = finite[finite[:, 1] > finite[:, 0]] if finite.size else finite
                if finite.size > 0:
                    dt = (finite[:, 1].max() - finite[:, 0].min()) / max(1, n_points - 1)
                else:
                    dt = 1.0
                stats = _landscape_stats(landscapes, dt, dgm=finite)
                prefix = f"layer_{l_idx}_{dim_label}"
                # UNDO THE PRE-RESCALE. A landscape ordinate is homogeneous of
                # degree 1 in the metric, so a max is degree 1, an integral
                # (ordinate x width) degree 2, and an L2 norm
                # (sqrt(ordinate^2 x width)) degree 3/2. Restoring the exponents
                # keeps every reported value in the ORIGINAL units.
                _m1, _m2, _m32 = _resc, _resc ** 2, _resc ** 1.5
                results[f"{prefix}_mean_landscape_integral"] = (
                    stats["mean_landscape_integral"] * _m2)
                results[f"{prefix}_landscape_max_1"] = (
                    stats["landscape_maxes"][0] * _m1 if stats["landscape_maxes"] else 0.0)
                results[f"{prefix}_landscape_norm_1"] = (
                    stats["landscape_norms"][0] * _m32 if stats["landscape_norms"] else 0.0)
                results[f"{prefix}_landscape_integrals"] = [
                    v * _m2 for v in stats["landscape_integrals"]]
                # FORWARD the approximation status. `_landscape_stats` computed
                # it, `evaluate` dropped it, so an end-to-end caller could not
                # tell an exact result from a bounded one (adversarial review
                # round 2, 2026-09-04).
                results[f"{prefix}_meta_last_entry_is_block_output"] = bool(
                    _last_is_block) if l_idx == target_layers[-1] else True
                results[f"{prefix}_meta_prescale"] = float(_resc)
                for _sk in ("_meta_landscape_max_status",
                            "_meta_landscape_quadrature",
                            "_meta_landscape_n_bars"):
                    results[f"{prefix}{_sk}"] = stats[_sk]

        if not results:
            return {"error": "No layers produced landscape features"}

        return results
