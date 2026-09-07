"""Scale normalization for structural metrics.

Many structural quantities grow mechanically with architecture: a stable rank
is bounded by d_model, a "layer index of maximum norm" is bounded by depth, a
chain length counts layers. Left raw, those features correlate with parameter
count at |r| up to 0.85 and a regression handed them will rediscover scale and
report it as structure.

This module derives a normalized twin for each such metric and appends it to
the envelope under ``<key>_scalenorm``. RAW VALUES ARE NEVER MODIFIED OR
REMOVED — the normalized entry sits alongside, so any published number stays
reproducible and the normalization can be audited or redone differently.

Divisors come from the model's own architecture (d_model, n_layers), not from
the parameter count, so the correction is structural rather than a regression
on size.

Design notes
------------
* Only metrics with a documented mechanical dependence are normalized. A
  metric that merely happens to correlate with size (e.g. a loss, which is
  lower for better models) is NOT normalized: that correlation is signal, not
  an artifact, and dividing it out would destroy it. Those are listed in
  ``SIZE_LINKED_BY_MEANING`` for disclosure instead.
* Normalizers are declared as data so the paper's normalization table and the
  code cannot drift apart.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# (metric key) -> (divisor name, short rationale)
#
# divisor names resolve against the architecture dict built by arch_info():
#   d_model     hidden size
#   n_layers    number of transformer blocks
#   log_d_model natural log of hidden size (for entropies over d_model states)
#   sqrt_d      sqrt(hidden size) (for vector norms in d_model-dim space)
#   inv_sqrt_n_layers  1/sqrt(depth) -- dividing by it MULTIPLIES by sqrt(depth),
#               for per-layer quantities that shrink as depth grows

# ─────────────────────────────────────────────────────────────────────────────
# TEN REGISTRATIONS WERE REMOVED ON 2026-09-04 AFTER MEASUREMENT.
#
# Every one had a plausible-sounding mechanism. Measurement refuted them, and
# the failures fall into four repeating kinds. Read this before adding a twin.
#
# 1. THE DIVISOR MANUFACTURES THE DEPENDENCE (the `schatten` failure, again).
#    A stable rank is invariant to zero-padding the operator: padding a fixed
#    matrix from d = 64 to 4096 leaves stable rank at 15.846333 while sr/d
#    falls x64. sr is proportional to d only for iid-Gaussian spectra
#    (d^+0.993); for heavy-tailed ones it is d^+0.157 to d^+0.002, and real
#    matrices use 5-11% of the bound. Both `spectral.avg_stable_rank/d` and
#    `weight_norms.mean_stable_rank/d` FLIPPED THE SIGN AND KEPT THE MAGNITUDE
#    (+0.793 -> -0.575, +0.637 -> -0.537). Same for
#    `unembedding.eff_rank/d`: erank ~ d^+1.146, so /d leaves rho(d) = +0.643
#    and DOUBLES the vocabulary term.
#
# 2. THE MECHANISM IS SIMPLY FALSE. `lid_mean/log(d_model)` was registered
#    because "an ID estimate over a d-dimensional cloud carries a log(d) term".
#    It does not: LID is BIT-IDENTICAL (9.7467) under isometric embedding at
#    d = 64..4096. The twin converted a null vocabulary association into a
#    significant one (-0.182 -> -0.408). It fit 66 models with no mechanism --
#    exactly the reason the `vendi/n_layers` fit was rejected, and it should
#    have been applied here too.
#
# 3. THE KEY IS DEAD OR THE TWIN IS A NEAR-CONSTANT.
#    `spectral.{min,max}_stable_rank` are emitted nowhere.
#    `intrinsic_dim.intrinsic_dimension` is never emitted in the study config
#    (layerwise: true), and its justification was measured on a DIFFERENT key.
#    `unembedding.eff_rank_centered/d` spans 0.913-0.982, cv 0.023 -- the same
#    near-constant failure that removed two twins before it.
#
# 4. THE TWIN DUPLICATES A COLUMN THE TASK ALREADY EMITS.
#    `refusal_direction.best_layer/n_layers` is Pearson 0.9991 with the task's
#    own `best_layer_fraction`, but with the WRONG denominator (ceiling
#    (L-1)/L, and 11 of 20 models saturate at the deepest block), leaving
#    rho(L) = +0.505 against -0.032 for the in-task column.
#    `task_vectors.mean_vector_norm/sqrt_d` is bit-identical to the in-task
#    `mean_vector_norm_normalized`, and sqrt(d) is not the offending term
#    anyway -- a residual-scale reparameterization (c = 2 on embed_in and every
#    block output projection) multiplies every residual norm by 2.000-2.008
#    with argmax agreement 1.0000, and leaves the depth term untouched
#    (+0.697 -> +0.676).
#
# THE BAR, restated: a twin needs a MECHANISM THAT SURVIVES A DIRECT TEST
# (isometric embedding, zero-padding, or a function-preserving
# reparameterization), AND a measured reduction that does not simply move the
# association to another architecture variable. A good fit is not evidence.
# ─────────────────────────────────────────────────────────────────────────────

# ONE TWIN IN THIS CAMPAIGN PASSED THE BAR, and it is not in this registry.
#
# `geometry_isoscore.isoscore_effective_dims` = iso*(d-1)+1 inverts IsoScore's
# structural 1/(d-1) factor. It survives the direct mechanism test that removed
# all eleven registered normalizers: a d = 64 -> 4096 ZERO-PADDING ISOMETRY
# moves the raw score 0.108590176 -> 0.001670618 (a factor of 65) while leaving
# the twin at 7.841181091 -- EXACTLY constant, to 5.5e-14 in an independent
# reviewer's measurement. It also reduces the width association without moving
# it elsewhere: rho(d_model) goes -0.440 (p = 3.5e-4) -> -0.105 (p = 0.42).
#
# It is absent from `NORMALIZERS` for a mechanical reason, not a substantive
# one: `add_normalized` can only DIVIDE by a named architecture scalar, and
# this transform is AFFINE. The task emits it directly. So "the registry is
# empty" must NOT be read as "no normalized variant passed" -- exactly one did,
# and it is emitted by `geometry_isoscore`.
#
# Per the campaign's standard a normalized variant needs BOTH reviewers'
# sign-off. Opus 5 has signed off on this one (2026-09-04); gpt-5.6-sol's
# round-4 verdict is pending at the time of writing.
NORMALIZERS: Dict[str, tuple] = {
    # --- per-layer relative change shrinks with depth ---
    #
    # ||dh||/||h|| per block falls mechanically with depth: the residual stream
    # norm grows while each block's update does not, so a model is rated
    # "smoother" largely for being deeper. Raw r(relative_change_mean,
    # n_layers) = -0.697. Multiplying by sqrt(n_layers) flattens it to
    # r = -0.151 (p = 0.747) while family separation survives -- the pythia
    # family goes 2.176 / 2.241 / 2.153 at L = 6 / 12 / 24.
    #
    # Note what does NOT work: reporting the value at fixed normalized depths
    # (25/50/75%) leaves r = -0.510 / -0.656 / -0.724, so it does not fix the
    # confound (adversarial review, 2026-09-03, refuting its own earlier
    # suggestion).
    # REMOVED 2026-09-04 -- the last surviving twin, now also refuted.
    #
    # It fails the direct mechanism test. Re-decomposing the SAME function
    # x -> 4x from 3 blocks [x, 2x, 4x] into 5 blocks
    # [x, sqrt2 x, 2x, 2sqrt2 x, 4x] leaves the computed function identical
    # (final-state max difference 0.0) but moves the raw mean 1.0000000 ->
    # 0.4142136 and the sqrt(n_layers) twin 1.7320508 -> 0.9262097 (-46.5%).
    # A normalizer whose whole claim is "removes the depth artefact" must not
    # move under a pure re-decomposition of one fixed function.
    #
    # Its recorded numeric rationale is also STALE: the flat pythia series
    # 2.176 / 2.241 / 2.153 at L = 6/12/24 was measured before the
    # post-final-norm hook fix. The current values are 4.0280 / 3.1811 /
    # 2.2296, i.e. monotone in depth, and the current raw/twin depth exponents
    # are -0.9266 / -0.4266 -- the twin halves the exponent rather than
    # removing it (adversarial review, 2026-09-04).
    #
    # "geometry_lipschitz.relative_change_mean":
    #     ("inv_sqrt_n_layers", ...),

    # NOT normalized: `effective_depth_90` is ALREADY k90/L in (0, 1].
    # Dividing by n_layers again manufactures a STRONGER size proxy than the
    # raw value: over 10 models raw rho(n_layers) = -0.810 while the
    # `_scalenorm` twin is -0.916 (adversarial review, 2026-09-03).
    # "geometry_effective_depth.effective_depth_90": ("n_layers", ...),




}

# Metrics that correlate with scale for a SUBSTANTIVE reason. Not normalized:
# the dependence is the measurement, not an artifact of it. Recorded so the
# write-up can disclose them rather than have a reader discover them.
SIZE_LINKED_BY_MEANING = {
    "dynamics_sharpness.baseline_loss":
        "larger models genuinely reach lower loss; this is capability, not geometry",
    "geometry_perplexity.perplexity":
        "same — perplexity is a capability measure",
}

# Architecture descriptors and DV-side controls. These are legitimate envelope
# content but must never enter a structural feature matrix: they ARE size, or
# they are the dependent variable's neighbours.
EXCLUDE_FROM_FEATURES = (
    "_meta_", ".n_params", ".hidden_dim", ".vocab_size", ".welch_bound",
    ".n_layers", ".num_layers", ".d_model", ".n_heads",
)
CONTROL_TASK_PREFIXES = (
    "consistency_calibration.",
    "interpretability_prediction_entropy.",
)


def arch_info(model: Any) -> Dict[str, float]:
    """Pull d_model / n_layers off a HF model config, defensively.

    Config field names differ across architectures (n_embd, hidden_size,
    d_model; n_layer, num_hidden_layers, num_layers), and multimodal configs
    nest the text tower — so resolve through text_config when present.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    cfg = getattr(cfg, "text_config", cfg) or cfg

    def pick(*names):
        for n in names:
            v = getattr(cfg, n, None)
            if isinstance(v, int) and v > 0:
                return float(v)
        return None

    d = pick("hidden_size", "n_embd", "d_model", "model_dim")
    L = pick("num_hidden_layers", "n_layer", "num_layers", "n_layers")
    out: Dict[str, float] = {}
    if d:
        import math
        out["d_model"] = d
        out["log_d_model"] = math.log(d)
        out["sqrt_d"] = math.sqrt(d)
    if L:
        import math
        out["n_layers"] = L
        # `add_normalized` can only DIVIDE, so a twin that needs to MULTIPLY by
        # sqrt(n_layers) divides by 1/sqrt(n_layers) instead.
        out["inv_sqrt_n_layers"] = 1.0 / math.sqrt(L)
    return out


def add_normalized(results: Dict[str, Any], arch: Dict[str, float]
                   ) -> Dict[str, Dict[str, float]]:
    """Append ``<key>_scalenorm`` entries in place; return what was added.

    Raw values are left untouched. Returns {task: {new_key: value}} so the
    caller can record exactly what was derived.
    """
    if not arch:
        logger.info("normalization: no architecture info; skipping")
        return {}

    added: Dict[str, Dict[str, float]] = {}
    for full_key, (divisor_name, _why) in NORMALIZERS.items():
        task, _, metric = full_key.partition(".")
        res = results.get(task)
        if not isinstance(res, dict) or metric not in res:
            continue
        raw = res[metric]
        div = arch.get(divisor_name)
        if not isinstance(raw, (int, float)) or not div:
            continue
        res[f"{metric}_scalenorm"] = float(raw) / float(div)
        added.setdefault(task, {})[f"{metric}_scalenorm"] = res[f"{metric}_scalenorm"]

    if added:
        n = sum(len(v) for v in added.values())
        logger.info(f"normalization: added {n} scale-normalized twin(s)")
    return added


def is_feature(key: str) -> bool:
    """True if ``task.metric`` may enter a structural feature matrix."""
    if any(key.startswith(p) for p in CONTROL_TASK_PREFIXES):
        return False
    return not any(tok in key for tok in EXCLUDE_FROM_FEATURES)
