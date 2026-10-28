"""
Correlation Dimension (Hidden-State Geometry) Task
──────────────────────────────────────────────────────────────────────
Evaluates hidden-state point-cloud complexity by computing the
Grassberger-Procaccia correlation dimension on final-layer representations.

A standard "intrinsic dimension" assumes a locally smooth Euclidean manifold.
This task is explicitly a hidden-state GP estimator, not a log-probability
trajectory diagnostic.

References:
- Grassberger & Procaccia 1983 correlation integral estimator.


The 5th-95th-percentile radius window of ``_gp_correlation_dimension``
lies in the saturation regime and is compressive at higher dimensions; it is
kept under ``_meta_`` only.
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
import logging
logger = logging.getLogger("blme")


def _gp_correlation_dimension(distances, num_radii=30, rvals=None):
    """Grassberger-Procaccia (1983) correlation dimension from a 1-D array of
    pairwise distances (upper triangle, i<j).

    Builds the correlation integral C(r) = fraction of pairs with distance < r
    over a percentile-spaced (5th..95th) log radius grid, then returns the slope
    of log C(r) vs log r (the GP dimension estimate).

    Returns ``(slope, r_squared, status)`` where ``status`` is ``"ok"``,
    ``"degenerate"`` (non-positive / collapsed percentile range) or
    ``"insufficient"`` (<3 usable radii); ``slope``/``r_squared`` are ``None``
    unless ``"ok"``. ``rvals`` overrides the percentile grid (used by the
    reference-parity test to compare on an identical radius grid).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if rvals is None:
        r_min = np.percentile(distances, 5)
        r_max = np.percentile(distances, 95)
        if r_min <= 0 or r_max <= 0 or r_min >= r_max:
            return None, None, "degenerate"
        radii = np.logspace(np.log10(r_min), np.log10(r_max), num=num_radii)
    else:
        radii = np.asarray(rvals, dtype=np.float64)

    total_pairs = len(distances)
    C_r, valid_radii = [], []
    for r in radii:
        c = np.sum(distances < r) / total_pairs
        if c > 0:
            C_r.append(c)
            valid_radii.append(r)

    if len(valid_radii) < 3:
        return None, None, "insufficient"

    log_r = np.log(valid_radii)
    log_Cr = np.log(C_r)
    slope, intercept = np.polyfit(log_r, log_Cr, 1)
    predicted = slope * log_r + intercept
    ss_res = np.sum((log_Cr - predicted) ** 2)
    ss_tot = np.sum((log_Cr - np.mean(log_Cr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r_squared), "ok"


@register_task("geometry_correlation_dimension")
class CorrelationDimensionTask(DiagnosticTask):
    """
    Computes the Grassberger-Procaccia fractional correlation dimension
    on the final representation space to evaluate fractal complexity.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Hidden-State Correlation Dimension Analysis...")
        num_samples = self.config.get("num_samples", 100)
        max_length = self.config.get("max_length", 128)
        num_radii = self.config.get("num_radii", 30)
        # Use mean-pooled representations by default; set to "last" for
        # last-token only (original behaviour).
        pooling = self.config.get("pooling", "mean")
        # Cap total points for pairwise distance computation
        max_points = self.config.get("max_points", 2000)
        seed = int(self.config.get("seed", 42))

        if dataset is None:
             from ...cache import load_default_corpus
             dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}

        device = next(model.parameters()).device
        all_hidden_states = []

        with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

                out = model(**inputs, output_hidden_states=True)
                # Use all token positions to get more data points
                # post-final-norm: DELIBERATELY post-norm. This is a SINGLE-LAYER point
                # cloud of the model's output representation, so there is no profile for
                # a mixed convention to corrupt; `_meta_correlation_dimension_layer`
                # records which representation was measured.
                last_hidden = out.hidden_states[-1][0].detach().cpu()  # (T, D)
                if pooling == "mean":
                    # One mean-pooled vector per sample, kept in float64.
                    all_hidden_states.append(last_hidden.double().mean(dim=0))
                elif pooling == "all_tokens":
                    # Each token is a separate data point (many more points)
                    all_hidden_states.extend(list(last_hidden))
                else:  # "last"
                    all_hidden_states.append(last_hidden[-1])

        if len(all_hidden_states) < 20:
             return {"error": "Need at least 20 points for Grassberger-Procaccia correlation dimension."}

        H = torch.stack(all_hidden_states)  # (N, D)
        N = H.shape[0]

        # Subsample if too many points (pairwise distances are O(N^2))
        if N > max_points:
            rng = np.random.default_rng(seed)
            indices = rng.choice(N, max_points, replace=False)
            H = H[indices]
            N = max_points

        # 1. All pairwise L2 distances, promoted to float64 BEFORE cdist (fp16
        # states overflow ||x||^2) and with direct differences (the mm path
        # cancels for the small distances D2 depends on).
        H = H.double()
        dist_matrix = torch.cdist(H, H, p=2,
                                  compute_mode="donot_use_mm_for_euclid_dist")

        # Extract upper triangle distances (i < j)
        tri_indices = torch.triu_indices(N, N, offset=1)
        # float64 end to end (a float32 cast flushes small distances to zero).
        distances = dist_matrix[tri_indices[0], tri_indices[1]].double().cpu().numpy()

        # 2-3. Grassberger-Procaccia: correlation integral C(r) -> log-log
        # slope. C(r) counts coincident pairs (they set its floor c0); one
        # radius setting reaches every plateau call.
        _plateau_radii = int(self.config.get("plateau_num_radii", 60))
        _atom_frac = float(np.count_nonzero(distances <= 0.0) / max(1, distances.size))
        _fallback_reason = None
        # No estimand switch on the atom mass: the headline is the scaling
        # exponent, the coincident-pair fraction is reported beside it (an
        # atomic measure has D2 = 0 exactly, applied below when no fit exists;
        # test_correlation_dimension_atom_e2e.py).
        _atom_dominates = False
        # The continuous part (positive distances) is fitted separately.
        _cont = None
        _d_pos = distances[distances > 0]
        slope, r_squared, status = _gp_correlation_dimension(
            distances, num_radii=num_radii)
        # The percentile-window fit, kept under its own name before any
        # fallback overwrites slope / r_squared.
        _pct_slope, _pct_r2 = slope, r_squared
        if _atom_frac > 0.0 and _d_pos.size >= 10:
            _cs, _cr2, _cst, _cw = _gp_correlation_dimension_plateau(
                _d_pos, num_radii=_plateau_radii)
            if _cs is not None and np.isfinite(_cs):
                _cont = float(_cs)
                slope, r_squared, status = _cont, float("nan"), "continuous_part"
                _fallback_reason = "atoms_removed"
        if status in ("degenerate", "insufficient"):
            # The demoted percentile kernel must not veto the plateau kernel.
            _fallback_reason = status
        if status in ("degenerate", "insufficient"):
            _sr = _gp_correlation_dimension_plateau(
                distances, num_radii=_plateau_radii)
            if _sr[0] is not None and np.isfinite(_sr[0]):
                slope, r_squared = float(_sr[0]), float("nan")
            elif _atom_frac > 0.0 and _sr[2] in ("unidentified", "insufficient",
                                                 "degenerate"):
                # Atomic measure: C(r) has a constant floor, D2 = 0.
                slope, r_squared = 0.0, float("nan")
                _fallback_reason = "atomic"
            else:
                return {"error": "Failed to compute correlation integral across scales."}

        if pooling == "all_tokens":
            point_space = "token_hidden_states"
        elif pooling == "last":
            point_space = "last_token_hidden_states"
        else:
            point_space = "mean_pooled_hidden_states"

        # The scaling-region (plateau) fit on the FULL pair population:
        # radii come from the positive distances, but C(r) counts every pair.
        _cr_input = distances
        sr_slope, sr_r2, sr_status, sr_w = _gp_correlation_dimension_plateau(
            _cr_input, num_radii=_plateau_radii)
        _cr_n_pairs = int(np.asarray(_cr_input).size)

        # The headline is the scaling-region fit: the 5th-95th percentile
        # window sits in the saturation region of C(r).
        _primary = (float(sr_slope)
                    if (sr_slope is not None and sr_status == "ok")
                    else float("nan"))

        # Atomic measure with no fit -> D2 = 0 exactly
        # (test_correlation_dimension_atom_e2e.py).
        if (_atom_frac > 0.0
                and (sr_slope is None or sr_status != "ok"
                     or not np.isfinite(_primary))):
            _primary = 0.0
        return {
            "correlation_dimension": _primary,
            # Derived: the pair population of the headline C(r) (all pairs).
            "_meta_correlation_integral_n_pairs": _cr_n_pairs,
            "_meta_coincident_pair_fraction": float(_atom_frac),
            # _meta_ alias of the headline.
            "_meta_hidden_state_correlation_dimension": _primary,
            # The percentile-window fit, retained under its own honest name.
            "_meta_correlation_dimension_percentile_window": (
                float(_pct_slope) if _pct_slope is not None else float("nan")),
            "correlation_dimension_method": "hidden_state_grassberger_procaccia_scaling_region",
            "correlation_dimension_point_space": point_space,
            "_meta_num_points": N,
            # R^2 of the reported fit (a _meta_ duplicate of the scaling R^2).
            "_meta_fit_r_squared": (float(sr_r2) if (sr_r2 is not None
                                                     and np.isfinite(sr_r2))
                                    else float("nan")),
            "_meta_fit_r_squared_percentile_window": (
                float(_pct_r2) if _pct_r2 is not None else float("nan")),
            # The scaling-region value, NaN when its span is below the floor.
            "_meta_correlation_dimension_scaling_region": (
                float(sr_slope)
                if (sr_slope is not None and sr_status == "ok")
                else float("nan")),
            "_meta_correlation_dimension_scaling_region_raw": (
                float(sr_slope) if sr_slope is not None else float("nan")),
            "_meta_correlation_dimension_min_decades": float(_MIN_SCALING_DECADES),
            # Eckmann-Ruelle 2 log10(N), for interpretation only: an estimate
            # above it is under-resolved and biased LOW, not invalid.
            "_meta_correlation_dimension_eckmann_ruelle_slope_bound": float(
                2.0 * np.log10(max(N, 10))),
            "_meta_correlation_dimension_under_resolved": bool(
                sr_slope is not None and np.isfinite(sr_slope)
                and float(sr_slope) > 2.0 * np.log10(max(N, 10))),
            "correlation_dimension_scaling_r2": (
                float(sr_r2) if sr_r2 is not None else float("nan")),
            "_meta_correlation_dimension_scaling_status": sr_status,
            "_meta_correlation_dimension_zero_pair_frac": _atom_frac,
            "_meta_correlation_dimension_atom_dominates": bool(_atom_dominates),
            "_meta_correlation_dimension_fallback": _fallback_reason,
            # D2 of the continuous part (NaN without exact duplicates).
            "_meta_correlation_dimension_continuous": (
                float(_cont) if _cont is not None else float("nan")),
            "_meta_correlation_dimension_scaling_window": sr_w,
        }


# Minimum span of the fitted scaling region, in decades of r.
_MIN_SCALING_DECADES = 0.05


def _gp_correlation_dimension_plateau(distances, num_radii=60,
                                      min_window=5, min_pairs_frac=None):
    """Grassberger-Procaccia D2 from the SCALING REGION, not a fixed window.

    D2 is defined as ``lim_{r->0} d log C(r) / d log r``. Fitting one line over
    a fixed 5th-95th percentile window instead measures the SATURATION region,
    where C(r) flattens toward 1, and is severely biased low: on a uniform 6-D
    cube the percentile fit returns 3.371 (-43.8%) with an R^2 of 0.9745 --
    which is why R^2 is not a validity diagnostic here. A curved saturation
    region looks perfectly linear over a badly chosen grid (adversarial review,
    2026-09-03).

    This estimator instead computes local slopes of log C vs log r and searches
    for the widest window whose slope is stable, returning that plateau slope,
    or ``status="unidentified"`` when no plateau exists.

    Returns ``(slope, r_squared, status, n_window)``.
    """
    distances = np.asarray(distances, dtype=np.float64)
    d_pos = distances[distances > 0]
    if d_pos.size < 10:
        return None, None, "insufficient", 0

    # Radii from the smallest positive separation to the 5th percentile of the
    # positive distances: the small-radius regime, below saturation.
    r_lo = float(d_pos.min())
    r_hi = np.percentile(d_pos, 5)
    if not (0 < r_lo < r_hi):
        return None, None, "degenerate", 0
    radii = np.logspace(np.log10(r_lo), np.log10(r_hi), num=num_radii)

    total = float(distances.size)
    C = np.array([np.count_nonzero(distances < r) / total for r in radii])
    # A radius is usable once it holds 10 pairs (resolution, not a fixed mass).
    _floor = (min_pairs_frac if min_pairs_frac is not None
              else 10.0 / max(1.0, total))
    keep = C > _floor
    if keep.sum() < min_window + 1:
        return None, None, "insufficient", int(keep.sum())
    lr = np.log(radii[keep])
    lC = np.log(C[keep])

    # Local slopes between consecutive grid points.
    slopes = np.diff(lC) / np.diff(lr)
    if slopes.size < min_window:
        return None, None, "insufficient", int(slopes.size)

    # Widest window whose local slopes are mutually consistent.
    best = None
    for w in range(slopes.size, min_window - 1, -1):
        cands = []
        for i in range(0, slopes.size - w + 1):
            seg = slopes[i:i + w]
            # A dead-flat window is no data, not a perfect plateau.
            if float(np.abs(seg).max()) <= 1e-9:
                continue
            spread = float(seg.std() / max(abs(seg.mean()), 1e-12))
            cands.append((spread, i))
        if not cands:
            continue
        spread, i = min(cands)
        if spread < 0.15:          # a genuine plateau at this width
            best = (w, spread, i, w)
            break
        if best is None or spread < best[1]:
            best = (w, spread, i, w)
    # Resolution ceiling ~2 log10(N), reported alongside the estimate.
    n_points_est = int(round((1.0 + np.sqrt(1.0 + 8.0 * distances.size)) / 2.0))
    d2_ceiling = 2.0 * np.log10(max(n_points_est, 10))

    # Use the selected window; refuse when none is stable.
    if best is None:
        return None, None, "unidentified", 0
    w, spread, i, _ = best
    if spread > 0.35:
        # No stable scaling region: refuse rather than return a confident
        # wrong number.
        return None, None, "unidentified", int(w)

    # Least-squares slope over the plateau's own radius range.
    j0, j1 = i, i + w + 1
    coef = np.polyfit(lr[j0:j1], lC[j0:j1], 1)
    pred = np.polyval(coef, lr[j0:j1])
    ss_res = float(np.sum((lC[j0:j1] - pred) ** 2))
    ss_tot = float(np.sum((lC[j0:j1] - lC[j0:j1].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    slope = float(coef[0])
    # A non-positive slope means the window carries no scaling behaviour.
    if not np.isfinite(slope) or slope <= 0.0:
        return None, None, "unidentified", int(w)
    # Resolvability: the FULL usable radius grid must span >= 0.05 decades
    # (resolvable clouds 0.22-0.89, iid Gaussian in d=768 0.007-0.017).
    _decades = (abs(float(lr[-1] - lr[0])) / np.log(10.0)
                if lr.size > 1 else 0.0)
    status = "ok" if _decades >= _MIN_SCALING_DECADES else "at_resolution_ceiling"
    return slope, float(r2), status, int(w)
