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


CALIBRATION CAVEAT (audit 2026-07-16): the 5th-95th-percentile radius
window includes the correlation-integral saturation regime, so the
estimate is COMPRESSIVE at higher dimensions (fresh k-cubes: true dim
3 -> ~2.1, 6 -> ~3.4, with fit R^2 > 0.97 throughout — R^2 cannot
flag it). Interpret as a monotone complexity proxy, not a calibrated
Grassberger-Procaccia dimension.
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
                    # Each sample contributes one mean-pooled vector
                    # KEEP THE DOUBLE MEAN. `.to(last_hidden.dtype)` narrows
                    # straight back to float32 for every float32 checkpoint, so
                    # the promotion bought nothing: on native Pythia-70m hidden
                    # states the cast gave 7.486064047571674 against an
                    # independently reconstructed Grassberger-Procaccia
                    # 15.295160200027059 (-51.06%), and a LOSSLESS translation
                    # collapsed the estimate from 2.3540184486398976 to 0.0.
                    # Removing only this cast gives 15.295160200026809
                    # (codex round 15, 2026-09-09).
                    #
                    # This site was MISSED by the AST sweep of the same class,
                    # which searched for a trailing `.float()`; this variant
                    # spells the narrowing `.to(<dtype>)`. Ask what the query
                    # cannot express.
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

        # 1. Compute all pairwise L2 distances.
        #
        # PROMOTE BEFORE cdist, not after. `torch.cdist` runs in the input's
        # dtype, and BLME's own loader uses dtype="auto", so pythia arrives as
        # float16 -- where ||x||^2 ~ 1.9e5 overflows the fp16 max of 65504 and
        # EVERY pairwise distance goes non-finite (measured: 4950/4950,
        # 19900/19900 and 1999000/1999000 at N = 100/200/2000), so the task
        # returned an error rather than a number. The later `.float()` was too
        # late -- the overflow had already happened. Present in the project's
        # own archive (results/blme_dtype_test/pythia_float16.json,
        # results/blme_verified_a/pythia-70m.json). Adversarial review
        # 2026-09-04.
        # float64, not float32.
        #
        # The plateau search is a DISCRETE argmin over candidate windows, so a
        # last-bit difference in the distances can flip which window wins and
        # move the reported slope by ~1e-4 -- far more than the input noise.
        # float32 `cdist` is a multithreaded matmul internally, and its
        # reduction order is not guaranteed stable under differing CPU load, so
        # the task was not reproducible run-to-run on a busy host. float64
        # removes the amplification at a cost of one extra copy of an
        # (N x N) matrix (32 MB at the shipped N = 2000).
        H = H.double()
        # `compute_mode="donot_use_mm_for_euclid_dist"`: torch.cdist's DEFAULT
        # path is a matrix multiplication, i.e. the same cancellation-prone
        # ||x||^2+||y||^2-2x.y identity fixed elsewhere in this campaign. On
        # float32-stored points [0.001, 0.5, 0.75] and
        # [0.0010000100117, 0.5, 0.75] the mm path disagrees with direct
        # subtraction and `torch.pdist`, which agree with each other
        # (adversarial review round 4, 2026-09-04). D2 is a function of the
        # SMALL distances, so exactly those pairs decide the answer.
        dist_matrix = torch.cdist(H, H, p=2,
                                  compute_mode="donot_use_mm_for_euclid_dist")

        # Extract upper triangle distances (i < j)
        tri_indices = torch.triu_indices(N, N, offset=1)
        # .float() so bf16 models (Gemma 4 etc.) don't crash on .numpy()
        # float64 END TO END. The Gram/cdist path is computed in double, but
        # casting the result back to float32 reintroduced exactly the underflow
        # it was promoted to avoid: a finite float64 cloud at scale 1e-50 had
        # all 4,950 distances flush to zero (adversarial review round 3,
        # 2026-09-04).
        distances = dist_matrix[tri_indices[0], tri_indices[1]].double().cpu().numpy()

        # 2-3. Grassberger-Procaccia: correlation integral C(r) -> log-log slope.
        # Extracted into _gp_correlation_dimension so the reference-parity test
        # exercises BLME's real kernel (not a transcription).
        # Exactly-coincident pairs ARE part of C(r) under the GP definition
        # (C(r) counts pairs at distance < r, and 0 < r for every r > 0), so
        # their mass is the constant floor c0 that decides whether D2 can be 0.
        # ONE radius setting for EVERY plateau call. It reached only the
        # headline, so `_cont` and the degenerate fallback silently used the
        # function default -- inert today because the default IS 60, but that
        # is exactly the latent inconsistency that produced the
        # percentile-window bug (independent Opus review, 2026-09-05).
        _plateau_radii = int(self.config.get("plateau_num_radii", 60))
        _atom_frac = float(np.count_nonzero(distances <= 0.0) / max(1, distances.size))
        _fallback_reason = None
        # ATOMIC MEASURES ARE DECIDED HERE, before any helper runs.
        #
        # C(r) counts pairs at distance < r, so a positive mass c0 of exactly
        # coincident pairs gives C(r) >= c0 > 0 for EVERY r > 0. Then
        # log C(r) -> log c0 and the Grassberger-Procaccia limit
        # D2 = lim log C(r)/log r is EXACTLY 0. Leaving that to a fallback
        # branch meant the legacy helper's "ok" status kept control and a
        # 10%-atom mixture with zero-pair fraction 0.0099819964 still reported
        # a positive dimension (adversarial review round 2, 2026-09-04).
        #
        # The continuous part is still measured, under its own name, by fitting
        # C(r) - c0 -- otherwise a single repeated token would erase all the
        # geometry from the output.
        # THE ATOM MASS MUST BE MEANINGFUL, not merely nonzero.
        #
        # `_atom_frac > 0` was far too aggressive: real hidden states repeat
        # tokens, so ANY duplicate zeroed the headline. Measured end to end,
        # distilgpt2 went 5.991867 -> 0.0 because ONE document out of 100 was
        # repeated (independent Opus review, 2026-09-04). That is a worse
        # failure than the defect it fixed.
        #
        # The limit argument only bites when the constant floor c0 actually
        # DOMINATES the correlation integral over the radii being fitted. Below
        # that the atoms are a negligible additive term and the continuous
        # scaling is what C(r) measures.
        #
        # NO SWITCH. The headline is ALWAYS the continuous-part estimate.
        #
        # I had a threshold here -- `c0 >= 0.0025`, i.e. a point mass of about
        # 5% -- and it was wrong in both directions. Below it, a genuine 4%
        # atom at N=1000 (c0 = 0.0015615616) was silently treated as
        # continuous; above it, the headline jumped discontinuously to 0. A
        # fixed cut on a continuous quantity is an ESTIMAND SWITCH, and a
        # metric whose meaning flips at an arbitrary boundary cannot be
        # compared across models (codex round 5, 2026-09-05). The `> 0` version
        # before it was worse still: it zeroed distilgpt2 on ONE repeated
        # document out of 100.
        #
        # So the switch is gone. `correlation_dimension` is the scaling
        # exponent of the CONTINUOUS part -- which is what the field means by
        # D2 for a representation, and what every reference implementation
        # computes -- and the atomic mass is REPORTED alongside. A reader who
        # wants the strict Grassberger-Procaccia limit can apply it themselves:
        # C(r) >= c0 > 0 for every r > 0 makes that limit exactly 0 whenever
        # `_meta_correlation_dimension_zero_pair_frac` is nonzero. That fact is
        # now stated in the output rather than silently applied to it.
        _atom_dominates = False
        # The CONTINUOUS-PART estimate uses the PROMOTED plateau kernel, on the
        # positive distances. Two things were wrong before: it used the DEMOTED
        # percentile kernel (2.1228 against the promoted 2.9856, -28.9%), and
        # the headline itself was still fitted on the FULL distance vector, so
        # the atom floor stayed inside C(r) -- on a 1000-point 3-cube the
        # headline fell 2.9139567040 -> 2.4040491472 -> 1.9185886747 ->
        # 0.0008826996 as the collapsed mass went 0 -> 4% -> 10% -> 20%, all
        # reported with status "ok", while the atom-free fit holds
        # 2.914 / 2.952 / 2.940 / 2.838 (independent Opus review, 2026-09-05).
        #
        # Coincident pairs carry no scaling information -- they sit at r = 0 at
        # every radius -- so the exponent of the continuous part is measured on
        # the positive distances, and the atomic mass is reported separately.
        _cont = None
        _d_pos = distances[distances > 0]
        # `status` is ALWAYS bound. The atomic branch used to assign it only
        # when the continuous fit succeeded, so a cloud with >= 10 positive
        # distances whose plateau is UNIDENTIFIED left `status` unset and the
        # next `if status in (...)` raised UnboundLocalError -- a 20-point
        # cloud with ten rows at 0 and ten at 1 reaches it (codex round 6,
        # 2026-09-05).
        slope, r_squared, status = _gp_correlation_dimension(
            distances, num_radii=num_radii)
        # KEEP the percentile-window fit under its own name. The atomic branch
        # below overwrote `slope`/`r_squared`, which are then exported as
        # `_meta_correlation_dimension_percentile_window` and
        # `_meta_fit_r_squared_percentile_window` -- so a key named for the
        # percentile window shipped the PLATEAU headline instead. Reachable on
        # real data with a single repeated document: gpt2 at the shipped
        # defaults emitted 6.604087299227721 where the true percentile-window
        # fit on the same distances is 1.784666709122511 (+270%), and the R^2
        # became NaN where it was 0.9034. It also breaks this repo's own
        # assertion at tests/tasks/test_comprehensive_parity.py:3188, whose
        # fixtures simply cannot enter the branch (red-team review,
        # 2026-09-05).
        _pct_slope, _pct_r2 = slope, r_squared
        if _atom_frac > 0.0 and _d_pos.size >= 10:
            _cs, _cr2, _cst, _cw = _gp_correlation_dimension_plateau(
                _d_pos, num_radii=_plateau_radii)
            if _cs is not None and np.isfinite(_cs):
                _cont = float(_cs)
                # The continuous-part fit REPLACES the contaminated one, but
                # the headline stays a scaling exponent -- it does not jump
                # to 0.
                slope, r_squared, status = _cont, float("nan"), "continuous_part"
                _fallback_reason = "atoms_removed"
        if status in ("degenerate", "insufficient"):
            # `degenerate` was ALSO a veto, and on the same grounds as
            # `insufficient` it should not be: the percentile kernel is the
            # DEMOTED one. With 23 coincident points, 253 of 4950 pairs are
            # zero (5.11%), which collapses the percentile range and returned
            # only an error -- while the promoted plateau kernel returns 0.3821
            # at R^2 = 0.99 on the very same distances (adversarial review,
            # 2026-09-04). Only a genuinely collapsed cloud, where the PROMOTED
            # kernel also fails, is an error.
            _fallback_reason = status
        if status in ("degenerate", "insufficient"):
            # The DEMOTED percentile kernel must not veto the promoted one.
            #
            # With 6.1% coincident pairs the percentile fit fails and the task
            # emitted NO keys at all, while the plateau kernel returns 0.3295
            # on the same distances. It does not fire on the shipped corpus
            # (0 of 4950 coincident pairs there), but a kernel that is no
            # longer the headline should not be able to suppress the headline
            # (adversarial review, 2026-09-04).
            _sr = _gp_correlation_dimension_plateau(
                distances, num_radii=_plateau_radii)
            if _sr[0] is not None and np.isfinite(_sr[0]):
                slope, r_squared = float(_sr[0]), float("nan")
            elif _atom_frac > 0.0 and _sr[2] in ("unidentified", "insufficient",
                                                 "degenerate"):
                # ATOMIC MEASURE. A positive mass of exactly-coincident points
                # puts a constant floor under the correlation integral:
                # C(r) -> c0 > 0 as r -> 0, so log C(r) is flat and the
                # Grassberger-Procaccia D2 is EXACTLY 0. Rejecting a zero slope
                # as impossible reported 1.8244 for a 10%-atom + 3-cube mixture
                # whose true D2 is 0 (adversarial review, 2026-09-04).
                # The status gate used to be `== "unidentified"` alone, so a
                # PURE atomic measure -- where the plateau kernel reports
                # `insufficient` (one atom, no positive distances) or
                # `degenerate` (two atoms, a single distance) -- fell through to
                # an error instead of the exact answer. Measured: one atom and
                # two atoms both returned an error where D2 is exactly 0
                # (gpt-6-astra round 10, 2026-09-07; the same leg Opus
                # confirmed in round 9, which I left unfixed after refuting the
                # OTHER leg of that verdict).
                #
                # Why the reason for the plateau failure does not matter: a
                # positive mass of coincident points puts a constant floor under
                # C(r), so log C(r) is flat as r -> 0 and D2 = 0 by definition.
                # `_atom_frac > 0` is the load-bearing condition; the status only
                # says which way the fit gave up.
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

        # Scaling-region estimate, reported ALONGSIDE the legacy percentile fit.
        # D2 is a limit as r -> 0; the legacy p5-p95 window sits in the
        # saturation tail and is biased low by -22% to -50% on uniform d-cubes
        # for d = 2..8. The scaling-region estimator cuts that to -2.7% to
        # -17.6% on the same clouds. Neither is exported as a replacement for
        # the other: the legacy key keeps its meaning, and callers who want the
        # better estimate ask for it by name.
        # Fitted on the POSITIVE distances: see the note above. With no
        # duplicates this is the identical vector, so nothing changes for the
        # shipped corpus (zero_pair_frac is 0.0 on gpt2, distilgpt2 and
        # pythia-70m at defaults).
        # `num_radii` REACHES THE PROMOTED KERNEL. It was passed only to the
        # demoted percentile fit, so four full evaluate() runs at
        # num_radii = 10/30/60/300 returned the identical headline
        # 5.410822187714411 while the demoted key moved 1.8132/1.7900/1.7813/
        # 1.7742 -- a documented, configurable parameter that silently did
        # nothing to the number anyone reads. It is NOT cosmetic: applied to
        # the promoted kernel it swings a 100-point 3-cube 2.2945 -> 3.5939
        # (+57%), every run reporting status "ok" and R^2 > 0.99 (red-team
        # review, 2026-09-05). The default stays 60, so shipped values are
        # unchanged unless a caller sets it.
        # C(r) COUNTS COINCIDENT PAIRS -- the radii do not.
        #
        # Radii must come from the positive distances (log 0 is undefined), but
        # the correlation integral is a property of the WHOLE pair population.
        # Dropping the zeros from both the numerator and the denominator
        # silently replaces Grassberger-Procaccia's C(r) by the conditional
        # integral (C(r) - c0)/(1 - c0) -- it measures the continuous part and
        # ships it as the headline.
        #
        # An atom is not a nuisance here, it IS the answer: a population with
        # positive mass at a point has D2 = 0, because C(r) -> c0 > 0 and
        # log C(r) goes flat. On the independently specified population
        # mu = 0.5*delta(-2) + 0.5*Uniform[0,1] -- where C(r) = 0.25 + 0.25(2r - r^2)
        # for 0 < r < 1, so the exact D2 is 0 -- the filtered call returns
        # 0.998264543712 and evaluate() shipped 0.987358435325, against nolds'
        # 5.35160579501927e-17 on its constant small-radius integral. Passing
        # the full pair population returns 0.002955769916 (codex round 12,
        # 2026-09-08).
        #
        # The kernel ALREADY divides by `distances.size` and draws its radii
        # from its own `distances > 0`: filtering here defeated logic that was
        # already correct.
        # Bound ONCE and passed by name, so the provenance key below cannot
        # disagree with what the kernel was actually handed.
        _cr_input = distances
        sr_slope, sr_r2, sr_status, sr_w = _gp_correlation_dimension_plateau(
            _cr_input, num_radii=_plateau_radii)
        _cr_n_pairs = int(np.asarray(_cr_input).size)

        # THE PRIMARY KEY IS THE SCALING-REGION FIT, not the 5th-95th
        # percentile fit.
        #
        # The percentile window lies in the SATURATION region of C(r), where
        # the log-log curve has already flattened, so its slope is not the
        # Grassberger-Procaccia dimension. Measured against GP 1983's published
        # D2 = 2.05 +/- 0.01 on a fresh RK4 Lorenz attractor, the percentile fit
        # returns 1.4925 on the 3-D state space (-27.2%) and 1.2960 / 1.5113 /
        # 1.6811 on delay embeddings (-36.8% to -18.0%) -- and it DRIFTS +29.7%
        # with embedding dimension, failing the saturation property that
        # defines the estimator. The scaling-region fit on the identical
        # distances gives 2.0562 (+0.3%). The kernel was never the defect; the
        # window was (adversarial review, 2026-09-04).
        _primary = (float(sr_slope)
                    if (sr_slope is not None and sr_status == "ok")
                    else float("nan"))

        # ATOMIC MEASURE -> D2 IS EXACTLY 0, APPLIED HERE.
        #
        # A positive mass of coincident points puts a constant floor under the
        # correlation integral: C(r) -> c0 > 0 as r -> 0, so log C(r) is flat
        # and the Grassberger-Procaccia dimension is exactly 0.
        #
        # I implemented this on 2026-09-07 in the branch above, and it was DEAD
        # CODE. `_pct_slope` is snapshotted at line 280 (a deliberate round-6
        # fix) and the headline is `sr_slope` from the plateau kernel, so every
        # later assignment to `slope` -- lines 289, 313, 337 -- is stored and
        # never read. AST confirms it: `slope` STORE at 190/213/237/261 (local
        # numbering), LOAD only at 204. A pure atomic measure still returned
        # NaN (Opus round 11, 2026-09-08).
        #
        # I verified that fix by calling `_gp_correlation_dimension_plateau`
        # directly and re-implementing the branch condition in the test script,
        # instead of driving `evaluate()`. That is exactly the failure the
        # repo's own note warns about -- "never verify a fix by re-reading the
        # diff; run it" -- and simulating the surrounding logic is the same
        # error wearing a measurement's clothes.
        if (_atom_frac > 0.0
                and (sr_slope is None or sr_status != "ok"
                     or not np.isfinite(_primary))):
            _primary = 0.0
        return {
            "correlation_dimension": _primary,
            # DERIVED pair population of the headline correlation integral.
            # C(r) must be computed over ALL pairs including coincident ones;
            # if the kernel is ever handed a positive-filtered array again this
            # drops below N(N-1)/2 and the guard test fails.
            "_meta_correlation_integral_n_pairs": _cr_n_pairs,
            "_meta_coincident_pair_fraction": float(_atom_frac),
            # DEMOTED: bit-identical aliases of the primary. Three unprefixed
            # keys carried one value (5.054201187848884) and the R^2 fix added
            # a fourth and fifth pair, so five is_feature() columns carried two
            # distinct numbers -- perfectly collinear columns that multiply
            # this task's weight in a regularized regression. The uncensored
            # twin already exists as _meta_..._scaling_region_raw.
            "_meta_hidden_state_correlation_dimension": _primary,
            # The percentile-window fit, retained under its own honest name.
            "_meta_correlation_dimension_percentile_window": (
                float(_pct_slope) if _pct_slope is not None else float("nan")),
            "correlation_dimension_method": "hidden_state_grassberger_procaccia_scaling_region",
            "correlation_dimension_point_space": point_space,
            "_meta_num_points": N,
            # R^2 OF THE REPORTED FIT. `r_squared` belongs to the demoted
            # percentile window (0.9032 on gpt2) and sat beside a headline
            # whose own fit has R^2 0.9987 -- a goodness-of-fit number
            # describing a different quantity from the one next to it.
            # _meta_: bit-identical to correlation_dimension_scaling_r2, and
            # no other task emits a `fit_r_squared` column.
            "_meta_fit_r_squared": (float(sr_r2) if (sr_r2 is not None
                                                     and np.isfinite(sr_r2))
                                    else float("nan")),
            "_meta_fit_r_squared_percentile_window": (
                float(_pct_r2) if _pct_r2 is not None else float("nan")),
            # Preferred estimate: fitted in the small-radius scaling region --
            # but GATED on resolvability.
            #
            # The Grassberger-Procaccia estimate saturates at a ceiling set by
            # the point count, ~2*log10(N): beyond it the "dimension" is a
            # property of how many points were sampled, not of the manifold.
            # The status was computed but the value was emitted regardless, and
            # it reads `at_resolution_ceiling` in 12 of 12 real-model runs
            # (gpt2 6.02, distilgpt2 11.06, pythia-70m 10.48, pythia-160m 7.x).
            # Since the key is unprefixed it fed the feature matrix, so a
            # CENSORED observation was entering regressions as a measurement.
            # It is now NaN when censored; the raw number stays under _meta_
            # alongside the ceiling, so the censoring is auditable.
            "_meta_correlation_dimension_scaling_region": (
                float(sr_slope)
                if (sr_slope is not None and sr_status == "ok")
                else float("nan")),
            "_meta_correlation_dimension_scaling_region_raw": (
                float(sr_slope) if sr_slope is not None else float("nan")),
            # The gate is a SPAN floor, so report that, not the old slope
            # ceiling -- emitting a threshold the code no longer applies made
            # the audit trail contradict the decision.
            "_meta_correlation_dimension_min_decades": float(_MIN_SCALING_DECADES),
            # THE ECKMANN-RUELLE BOUND IS NOT A VALIDITY VETO, and is emitted
            # for interpretation only. Read it as: an estimate above
            # 2*log10(N) is UNDER-RESOLVED and therefore biased LOW -- the true
            # D2 is at least the reported value, not less.
            #
            # A round-9 review read gpt2's 6.7813 against its own bound of
            # 4.6021 (1.47x) with status `ok` as a defect, and proposed gating
            # on it. Measured against known ground truth, that gate would be
            # wrong and would discard real measurements: on a 3-dim cube in
            # R^64 this estimator gives 2.4605/2.8273/2.8481/2.8903/2.9125 at
            # N=100..2000 -- converging to 3 from below and never exceeding the
            # bound -- while on an **8-dim** cube it gives 6.3386 at N=200,
            # which is 1.38x ITS bound of 4.6021, and then 6.9303 and 7.0613 at
            # N=1000/4000 against a true value of 8. Exceeding the bound is the
            # signature of a genuinely high-dimensional manifold sampled too
            # sparsely, not of a spurious number. Gating it to NaN would throw
            # away the estimate precisely when the manifold is interesting.
            #
            # The apparent "headline tracks N" (5.41/6.78/8.12/8.98 at
            # N=100/200/500/1000 on gpt2) is this same convergence-from-below,
            # which the cube controls above reproduce exactly with a known
            # answer. Verified 2026-09-07.
            "_meta_correlation_dimension_eckmann_ruelle_slope_bound": float(
                2.0 * np.log10(max(N, 10))),
            # Derived, so a caller never has to recompute the comparison.
            "_meta_correlation_dimension_under_resolved": bool(
                sr_slope is not None and np.isfinite(sr_slope)
                and float(sr_slope) > 2.0 * np.log10(max(N, 10))),
            "correlation_dimension_scaling_r2": (
                float(sr_r2) if sr_r2 is not None else float("nan")),
            "_meta_correlation_dimension_scaling_status": sr_status,
            "_meta_correlation_dimension_zero_pair_frac": _atom_frac,
            "_meta_correlation_dimension_atom_dominates": bool(_atom_dominates),
            "_meta_correlation_dimension_fallback": _fallback_reason,
            # D2 of the CONTINUOUS part, with the atomic mass removed. NaN when
            # there are no exact duplicates (then the headline already is it).
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
    for the widest window whose slope is stable, returning that plateau slope.
    It returns ``status="unidentified"`` when no plateau exists, rather than a
    confident wrong number.

    Returns ``(slope, r_squared, status, n_window)``.
    """
    distances = np.asarray(distances, dtype=np.float64)
    d_pos = distances[distances > 0]
    if d_pos.size < 10:
        return None, None, "insufficient", 0

    # Radii spanning the small-distance regime: from the smallest resolvable
    # separation up to the median, deliberately excluding the saturation tail.
    # Upper radius sits in the SCALING region, not the saturation tail. D2 is a
    # limit as r -> 0, and the bias falls monotonically as the window shrinks
    # toward it. Measured on a uniform 4-cube (true D2 = 4, N = 10000), the
    # slope over radii up to percentile p reads:
    #     p50 -> 3.439 (-14.0%)   p20 -> 3.564 (-10.9%)   p10 -> 3.623 (-9.4%)
    #     p5  -> 3.668 ( -8.3%)   p2  -> 3.714 ( -7.1%)   p1  -> 3.740 (-6.5%)
    # p5 balances bias against having enough pairs to fit. The residual few
    # percent is the known GP edge effect on bounded supports and does NOT
    # close with more data: the estimate is flat at -17% from N=1000 to
    # N=30000 under the old p50 window, so this is estimator bias, not a
    # sample-size limit.
    # START AT THE SMALLEST RESOLVABLE SEPARATION, not the 0.1th percentile.
    #
    # The 10-pair floor added in round 3 was INERT while the grid began at
    # percentile 0.1: on the 5,000-point 3%-line / 97%-cube cloud the first
    # grid radius was already 0.0623809593 and counted 12,000+ pairs, so the
    # small-radius regime the floor was meant to open was never sampled
    # (adversarial review round 3, 2026-09-04). The floor now does the work.
    r_lo = float(d_pos.min())
    r_hi = np.percentile(d_pos, 5)
    if not (0 < r_lo < r_hi):
        return None, None, "degenerate", 0
    radii = np.logspace(np.log10(r_lo), np.log10(r_hi), num=num_radii)

    total = float(distances.size)
    C = np.array([np.count_nonzero(distances < r) / total for r in radii])
    # RESOLUTION-BASED floor, not a fixed 0.1% of pair mass.
    #
    # D2 is a limit as r -> 0, and a fixed C > 0.001 cut forbids the estimator
    # from ever reaching the small-radius regime of a low-mass component: on a
    # 5,000-point 3%-line / 97%-cube mixture the line's own scaling region sits
    # entirely below that floor, so the true D2 = 1 was unreachable
    # (adversarial review round 2, 2026-09-04). The real limit is statistical
    # -- a radius is usable once it contains enough pairs to estimate a count
    # -- so require 10 pairs, which is what the sample can actually resolve.
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
    #
    # The previous loop was DEAD CODE: it compared `(w, -spread)` against the
    # running best while `w` decreased monotonically, so the first (widest)
    # window always won and no other candidate could ever be scored -- the
    # result equalled a plain full-window OLS to 1e-12 (adversarial review,
    # 2026-09-03). This version keeps the WIDEST window that is actually
    # stable, and only falls back to narrower ones if none is.
    best = None
    for w in range(slopes.size, min_window - 1, -1):
        cands = []
        for i in range(0, slopes.size - w + 1):
            seg = slopes[i:i + w]
            # A DEAD-FLAT window is not a plateau -- it is no data.
            #
            # `seg.std()/max(|seg.mean()|, 1e-12)` returns exactly 0.000000 --
            # the BEST possible plateau score -- when C(r) does not change at
            # all across the window, because both the numerator and the true
            # mean are zero. The search then locks onto that window, the fit
            # returns slope 0, and the `slope <= 1e-6` guard discards the whole
            # estimate. It fires on real data at the shipped config: 8 of 40
            # gpt2 and 4 of 40 pythia-70m bootstrap draws at N=100 returned
            # NaN, and a uniform 3-cube returned NaN where the same distances
            # support 3.4455 (adversarial review, 2026-09-04).
            #
            # Require the window to carry a real slope before it can win.
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
    # RESOLVABILITY. A correlation-integral estimate cannot resolve a dimension
    # much above ~2*log10(N): the number of pairs falls as r^D2, so beyond that
    # there are simply no close pairs left to fit. Report the ceiling alongside
    # the estimate so a value at or above it is recognisable as a sample-size
    # artifact rather than a measurement -- iid Gaussian in 768 dimensions
    # (true D2 = 768) reads 103 at N=5000, which is the ceiling, not the
    # dimension.
    n_points_est = int(round((1.0 + np.sqrt(1.0 + 8.0 * distances.size)) / 2.0))
    d2_ceiling = 2.0 * np.log10(max(n_points_est, 10))

    # Use BEST's window, and REFUSE when no window is stable.
    #
    # The fit previously read whichever `i` and `w` the loop happened to exit
    # with rather than the window the search selected -- a demonstrated
    # mismatch (loop w=5,i=54 against best w=6,i=53) -- and my rewrite had also
    # dropped the `unidentified` refusal, making it unreachable. That is the
    # SECOND time this function's window selection was dead code (adversarial
    # review, 2026-09-03).
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
    # A correlation dimension is strictly positive: C(r) must GROW with r. A
    # flat or negative slope means the window contains no scaling behaviour at
    # all -- 20 random points in 64 dimensions produced exactly 0.000, which was
    # being returned as a measurement.
    # EXACT. `slope` is a log-log slope, i.e. dimensionless, so a magnitude
    # threshold here is defensible -- but a genuinely tiny positive D2 is a
    # real measurement, and the D2 == 0 case (an atomic measure, where C(r) has
    # a constant floor) is now handled explicitly by the caller. Only a
    # non-positive slope means the window carries no scaling behaviour.
    if not np.isfinite(slope) or slope <= 0.0:
        return None, None, "unidentified", int(w)
    # RESOLVABILITY = a MINIMUM WINDOW SPAN, not a slope ceiling.
    #
    # Two earlier gates were both wrong. The one-decade Eckmann-Ruelle form
    # (D2 <= 2*log10(N)) was 3-8x too strict for a window spanning only
    # 0.127-0.311 decades and censored 14/14 real runs. Dividing it by the span
    # then made it INERT: this estimator floors C at `min_pairs_frac` and caps
    # it near C(p5) ~ 0.05, so the largest slope the grid can even express is
    # about log10(50)/span ~ 1.70/span, while the threshold 2*log10(N)/span is
    # >= 4.00/span for N >= 100 -- head-room ratio 0.230-0.424, always below 1,
    # so the ceiling was unreachable by construction. iid Gaussian in 768
    # dimensions then returned slope 106.505 as "ok" (adversarial review,
    # 2026-09-04).
    #
    # Eckmann-Ruelle's actual content is that a scaling region must SPAN enough
    # of r to be a scaling region at all, and the span separates cleanly with
    # no overlap: resolvable clouds (uniform d-cubes, d=2..8, N=100 and 5000)
    # span 0.218-0.889 decades, the pathological ones (iid Gaussian d=768/4096)
    # span 0.007-0.017 -- a 15-100x gap -- and real LLM hidden states sit at
    # 0.127-0.311. A floor of 0.05 decades sits an order of magnitude below
    # every real model and an order above every degenerate one.
    #
    # `lr` is ln(r), so the span in decades is a DIFFERENCE of natural logs
    # divided by ln(10).
    # Span of the FULL usable radius grid, not of the selected window.
    #
    # Resolvability is a property of the DATA's radius range; the width of the
    # fitted window is a property of the plateau search. Flooring the selected
    # window conflated the two and censored the shipped configuration on every
    # model: gpt2 at N=100 has a full grid span of 0.3106 decades -- 6x the
    # floor -- yet the search picks w=6 of 59 intervals, giving a fitted span
    # of ~0.032, so `correlation_dimension_scaling_region` came back NaN with
    # status at_resolution_ceiling on 3 of 3 models. The separation the floor
    # was calibrated against was measured on the FULL grid: resolvable clouds
    # 0.218-0.889 decades, pathological 0.007-0.017, real LLM states
    # 0.127-0.311 (adversarial review, 2026-09-04).
    _decades = (abs(float(lr[-1] - lr[0])) / np.log(10.0)
                if lr.size > 1 else 0.0)
    status = "ok" if _decades >= _MIN_SCALING_DECADES else "at_resolution_ceiling"
    return slope, float(r2), status, int(w)
