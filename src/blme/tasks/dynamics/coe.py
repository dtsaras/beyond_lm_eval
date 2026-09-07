"""Chain-of-Embedding (CoE) — Wang et al., ICLR 2025 (arXiv:2410.13640).

Given the chain of hidden states
``H = h^(0), h^(1), …, h^(L)`` at a fixed token (``h^(0)`` is the
embedding output, ``h^(L)`` is the final transformer block output),
CoE measures how the representation *changes* between adjacent layers:

  * Magnitude change   ``M(h_l, h_{l+1}) = ‖h_{l+1} − h_l‖₂``               (Eq. 2)
  * Angle change       ``A(h_l, h_{l+1}) = arccos(<h_{l+1}, h_l>/(‖.‖·‖.‖))`` (Eq. 2)

(The paper defines M and A together in its Eq. 2; its Eq. 1 is the
layer-composition ``f = f_head ∘ … ∘ f_emb``.)

The paper's headline "output-free" scores:

  * ``Mag̃ = (1/L) Σ M(h_l,h_{l+1}) / M(h_0, h_L)``                         (Eq. 3)
  * ``Ãng = (1/L) Σ A(h_l,h_{l+1}) / A(h_0, h_L)``                          (Eq. 3)
  * ``CoE-R = (1/L) Σ [M(h_l,h_{l+1})/M(h_0,h_L) − A(h_l,h_{l+1})/A(h_0,h_L)]`` (Eq. 5)
  * ``CoE-C = |(1/L) Σ C(h_l,h_{l+1})|`` where
              ``C = (M/Z_Mag) · exp(i · A/Z_Ang)`` — the NORMALIZED form
              of the official code (score.py compute_CoE_C). NOTE: the
              paper's printed Eq. 6/7 uses RAW ``M · exp(i·A)`` and its
              Algorithm 2 normalizes only the magnitude; the original
              work is internally inconsistent, and BLME follows the
              official implementation.

Note on token position (departure from the paper): Wang et al. run
``model.generate`` and mean-pool ``h^(l)`` across every generated
output token. BLME is comparing *base and instruct models without
generating* — we read the final token of the prompt instead. The two
give highly correlated but not identical signals; we flag
``token_position`` in the result so reviewers can see the choice.
"""

import logging

import numpy as np
import math
import torch
import torch.nn.functional as F

from ...registry import register_task
from ..geometry.utils import block_output_count
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Arccos of the clamped cosine similarity — Eq. 2."""
    # SCALE-EXACT normalisation, matching `F.normalize`'s form.
    #
    # `F.normalize` divides by `max(||x||, 1e-12)`, and `||x||` SQUARES before
    # summing: at 1e-30 the float32 norm underflows to exactly 0, the 1e-12
    # floor takes over, and the "unit" vector is nothing of the kind. CoE is
    # dimensionless -- every reported quantity is a ratio -- so a positive
    # rescale must cancel exactly, and instead a dimensionless anchor whose
    # exact CoE-R/CoE-C is 0.0398345638/0.5398345638 returned **NaN/NaN at 5 of
    # 7** finite rescalings (gpt-5.6-sol round 9, 2026-09-07).
    #
    # Factoring out each vector's largest magnitude first only shifts
    # exponents, so it is exact and reproduces `F.normalize` bit-for-bit on
    # ordinary data (asserted in the parity test) -- the same discipline
    # `cknna._l2_normalize` already uses.
    def _unit(v):
        # `F.normalize` FIRST -- it is what the released code calls, and on
        # ordinary data it must remain bit-for-bit authoritative. Only when its
        # norm has underflowed (squaring 1e-30 gives 1e-60, far below float32's
        # 1.2e-38) does the 1e-12 floor take over and return something that is
        # not a unit vector; that is the only case the scale-exact path handles.
        n = v.norm(p=2)
        if bool(torch.isfinite(n)) and float(n) > 0.0:
            return v / n
        m = v.abs().max()
        if not bool(torch.isfinite(m)) or float(m) == 0.0:
            return F.normalize(v, p=2, dim=-1)
        vs = v / m
        return vs / vs.norm(p=2).clamp_min(torch.finfo(vs.dtype).tiny)

    an = _unit(a)
    bn = _unit(b)
    c = float(torch.dot(an, bn).item())
    if c >= 1.0 - 1e-12:
        return 0.0
    if c <= -1.0 + 1e-12:
        return float(np.pi)
    # math.acos, not torch.acos(torch.tensor(c)): the latter builds a
    # float32 tensor under the default dtype, capping angle precision at
    # ~1e-7 and making it depend on torch.set_default_dtype (audit
    # 2026-07-16).
    return float(math.acos(max(-1.0, min(1.0, float(c)))))


def _coe_from_chain(chain: list[torch.Tensor]) -> dict:
    """Compute all Wang et al. 2025 CoE statistics for one sample.

    ``chain`` is ``[h_0, h_1, …, h_L]`` with ``L ≥ 1``. Returns a dict
    containing per-pair magnitudes + angles, the Eq. 3 normalised
    means, and the Eq. 5 / Eq. 7 scalar scores.
    """
    def _norm2(v):
        """`v.norm(p=2)`, with a double fallback ONLY where float32 fails.

        CoE is dimensionless -- every reported quantity is a ratio to Z_Mag or
        Z_Ang -- so a positive rescale must cancel exactly. It did not: a
        dimensionless anchor whose exact CoE-R/CoE-C is
        0.0398345638 / 0.5398345638 returned **NaN/NaN at 5 of 7** finite
        positive rescalings (gpt-5.6-sol round 9, 2026-09-07), because
        `.norm()` squares before summing and the chain is materialised in
        float32 at `coe.py`'s `h[0, pos].detach().float()`.

        float32 stays authoritative wherever it produces a finite, nonzero
        answer. This file's headline matches the released raw L+1 chain with an
        **exact 0.000e+00** difference, and promoting the arithmetic outright
        would move that -- the mistake made in `cknna._remove_outliers` the
        same day, where a strictly more accurate computation was a regression
        against a bit-exact reference. Double is consulted only where float32
        has already failed, which is the regime the reference cannot reach
        either.
        """
        n = float(v.norm(p=2).item())
        if not math.isfinite(n) or (n == 0.0 and bool(v.any())):
            n = float(v.double().norm(p=2).item())
        return n

    L = len(chain) - 1
    mags = []
    angs = []
    for i in range(L):
        mags.append(_norm2(chain[i + 1] - chain[i]))
        angs.append(_angle_between(chain[i], chain[i + 1]))

    # End-to-end normalisers Z_Mag, Z_Ang (Eq. 3).
    z_mag = _norm2(chain[-1] - chain[0])
    z_ang = _angle_between(chain[0], chain[-1])

    # Guard degenerate cases where the representation doesn't move at
    # all: Z_Mag → 0 makes Mag̃ undefined, Z_Ang → 0 makes Ãng and the
    # Eq. 7 complex phase undefined. Return NaN for those fields.
    # RELATIVE degeneracy test, not an absolute 1e-12.
    #
    # `Mag~ = m / Z_Mag` is a RATIO, so it is scale-free -- and an absolute
    # threshold inside a scale-free quantity is this campaign's single most
    # repeated numeric defect. At a chain scale of 1e-30 a perfectly healthy
    # `z_mag` of 3.33e-30 is below 1e-12, so every pair was declared
    # "representation doesn't move at all" and the whole CoE family returned
    # NaN. That, not the norms, is what produced NaN/NaN at 5 of 7 rescalings:
    # the norms had already been repaired above and the values were still NaN.
    #
    # Degenerate means "did not move RELATIVE TO ITS OWN SIZE", so the
    # threshold scales with the chain. At ordinary magnitudes this is the same
    # decision as before (real hidden-state chains have z_mag of order 1, tens
    # of orders above either threshold), so no shipped number moves.
    _chain_scale = max((float(c.abs().max()) for c in chain), default=0.0)
    _z_floor = 1e-12 * _chain_scale if _chain_scale > 0 else 1e-12
    if not math.isfinite(z_mag) or z_mag <= _z_floor:
        mag_norm_pair = [float("nan")] * L
    else:
        mag_norm_pair = [m / z_mag for m in mags]
    if z_ang <= 1e-12:
        ang_norm_pair = [float("nan")] * L
    else:
        ang_norm_pair = [a / z_ang for a in angs]

    mag_norm_mean = (
        float(np.mean(mag_norm_pair)) if np.all(np.isfinite(mag_norm_pair))
        else float("nan")
    )
    ang_norm_mean = (
        float(np.mean(ang_norm_pair)) if np.all(np.isfinite(ang_norm_pair))
        else float("nan")
    )

    # CoE-R (Eq. 5) — only meaningful when both normalisers are finite.
    if np.isfinite(mag_norm_mean) and np.isfinite(ang_norm_mean):
        coe_r = mag_norm_mean - ang_norm_mean
    else:
        coe_r = float("nan")

    # CoE-C (Eq. 6/7) — magnitude of the complex-plane centroid, using the
    # NORMALIZED magnitude as radius AND the NORMALIZED angle as phase,
    # exactly matching the reference score.py compute_CoE_C
    # (Alsace08/Chain-of-Embedding): x = Mag̃·cos(Ãng), y = Mag̃·sin(Ãng),
    # CoE-C = sqrt(mean(x)^2 + mean(y)^2).
    # (Fixed 2026-06-22: previously used RAW magnitude and RAW angle — which
    # IS the paper's printed Eq. 6/7 form — but does not match the official
    # score.py (normalized/normalized). The original work is 3-way
    # inconsistent (prose raw/raw; Algorithm 2 normalized-mag/raw-angle;
    # official code normalized/normalized); we follow the official code.)
    if np.all(np.isfinite(mag_norm_pair)) and np.all(np.isfinite(ang_norm_pair)):
        x = np.array([mn * np.cos(an) for mn, an in zip(mag_norm_pair, ang_norm_pair)])
        y = np.array([mn * np.sin(an) for mn, an in zip(mag_norm_pair, ang_norm_pair)])
        coe_c = float(np.sqrt(x.mean() ** 2 + y.mean() ** 2))
        # The reference CoE-C IS the fully-normalized form; keep the
        # ``normalized_coe_c`` key as a documented alias for compatibility.
        normalized_coe_c = coe_c
    else:
        coe_c = float("nan")
        normalized_coe_c = float("nan")

    return {
        "magnitudes": mags,
        "angles": angs,
        "z_mag": z_mag,
        "z_ang": z_ang,
        "normalized_magnitudes": mag_norm_pair,
        "normalized_angles": ang_norm_pair,
        "mag_norm_mean": mag_norm_mean,
        "ang_norm_mean": ang_norm_mean,
        "coe_r": coe_r,
        "coe_c": coe_c,
        "normalized_coe_c": normalized_coe_c,
    }


@register_task("dynamics_coe")
class ChainOfEmbeddingTask(DiagnosticTask):
    """Wang et al. 2025 Chain-of-Embedding scores (Eq. 3/5/7)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Chain-of-Embedding (Wang et al. 2025)...")
        num_samples = int(self.config.get("num_samples", 5))
        token_position = self.config.get("token_position", "last")

        device = next(model.parameters()).device

        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)

        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        sample_results = []
        chain_length = None
        _used_hook = [False]
        _blockout_results = []

        # Recover the deepest block's RAW output so the chain is the FULL
        # h_0 … h_L that Eq. 3's M(h_0, h_L) requires. HF's last entry is
        # ln_f(b_{L-1}); the previous fix trimmed it away, but that dropped the
        # last BLOCK from the chain entirely (blocks 0..L-2 only, i.e. L-1 of
        # L), and the unmeasured share 1/L is larger for shallower models.
        # Hooking layers[-1] keeps every block and admits no LayerNorm step.
        from ..common import get_layers as _gl
        _blocks = _gl(model)
        _sink = {}
        _hook = None
        if _blocks is not None and len(_blocks) > 0:
            def _cap_last(module, inputs, output):
                _sink["h"] = (output[0] if isinstance(output, tuple)
                              else output).detach()
            _hook = _blocks[-1].register_forward_hook(_cap_last)

        try:
          with torch.no_grad():
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
                if input_ids.shape[1] < 1:
                    continue

                out = model(input_ids, output_hidden_states=True)
                if not getattr(out, "hidden_states", None):
                    continue

                pos = -1 if token_position == "last" else int(token_position)
                # Full chain h_0, h_1, …, h_L at the chosen position.
                # post-final-norm=False, DELIBERATELY: THE RELEASED REFERENCE
                # BUILDS THE CHAIN FROM THE RAW HF TUPLE.
                #
                # `Alsace08/Chain-of-Embedding`'s `inference.py::print_hidden_states`
                # takes `generation_output.hidden_states` and loops
                # `for j in range(layer_num)` with `layer_num = len(hidden_states[1])`
                # -- all L+1 entries, INCLUDING the final `ln_f(...)`, with no
                # hook and no trim. This task substituted the deepest block's
                # raw pre-norm output instead, and on the same 100 WikiText
                # passages, through BLME's own `_coe_from_chain`, the headline
                # CHANGES SIGN on every model tested (independent Opus review,
                # 2026-09-05):
                #
                #            reference chain   substituted chain
                #   gpt2        +0.013202          -0.183024
                #   pythia-70m  -0.365734          +0.135602
                #   distilgpt2  +0.120925          -0.214348
                #
                # coe_c moves -56.3% / +222.1% / -57.1%.
                #
                # This is the THIRD CLAUSE of the post-final-norm rule: a
                # per-layer profile must be homogeneous, a single-layer estimand
                # may want post-norm, AND THE RELEASED REFERENCE WINS. It is the
                # third task to need it (with `matrix_entropy` and
                # `contextualization`) and the largest of the three. The paper's
                # Eq. 3 defines the chain over "the hidden states of the layers"
                # and a raw block output is a defensible reading -- but it is
                # not the reading the released code implements, and faithfulness
                # is measured against the code.
                #
                # The hook is still installed, because `_meta_coe_r_blockout`
                # below reports what the substituted chain would have given: the
                # divergence is disclosed rather than hidden behind a choice.
                _hs = list(out.hidden_states)
                _h = _sink.get("h")
                if _h is not None and tuple(_h.shape) == tuple(_hs[-1].shape):
                    _used_hook[0] = True
                    chain = [h[0, pos].detach().float().cpu() for h in _hs]
                    _hs_blockout = list(_hs)
                    _hs_blockout[-1] = _h
                    _chain_blockout = [
                        h[0, pos].detach().float().cpu() for h in _hs_blockout]
                else:
                    # No usable hook: fall back to the trim, which at least
                    # keeps the LayerNorm step out of the path length.
                    #
                    # No hook: the chain is the raw tuple either way, so this
                    # path now differs from the one above only in that
                    # `_meta_coe_r_blockout` cannot be reported.
                    _used_hook[0] = False
                    chain = [h[0, pos].detach().float().cpu() for h in _hs]
                    _chain_blockout = None
                if len(chain) < 2:
                    continue
                chain_length = len(chain) - 1
                sample_results.append(_coe_from_chain(chain))
                if _chain_blockout is not None and len(_chain_blockout) >= 2:
                    _blockout_results.append(_coe_from_chain(_chain_blockout))

        finally:
            if _hook is not None:
                _hook.remove()

        if not sample_results:
            return {"error": "Failed to extract layer trajectory"}

        # Aggregate across samples. For Eq. 5 / Eq. 7 the paper reports
        # mean (and later uses this for ROC-AUC) — we report mean + std.
        def _mean(key):
            vs = [r[key] for r in sample_results if np.isfinite(r[key])]
            return float(np.mean(vs)) if vs else float("nan")
        def _std(key):
            vs = [r[key] for r in sample_results if np.isfinite(r[key])]
            return float(np.std(vs)) if vs else float("nan")

        flat_mags = [m for r in sample_results for m in r["magnitudes"]]
        flat_angs = [a for r in sample_results for a in r["angles"]]

        return {
            # Paper Eq. 3 — normalised per-layer averages.
            #
            # `mean_normalized_magnitude` is `_meta_`: THREE COLUMNS, TWO
            # DEGREES OF FREEDOM. `coe_r` is exactly
            # `mean_normalized_magnitude - mean_normalized_angle` -- measured
            # to <= 8.3e-17 on every model tested, i.e. an algebraic identity,
            # not a correlation (independent Opus review, 2026-09-05; verified
            # here on gpt2 2.8e-17, distilgpt2 8.3e-17, pythia-70m 2.1e-17).
            # The same key is also pearson 0.999961 with `coe_c`, so it is the
            # one column implicated in BOTH redundancies and therefore the one
            # to demote: `coe_r`, `coe_c` and `mean_normalized_angle` span the
            # same space with no linear dependency between them.
            "_meta_mean_normalized_magnitude": _mean("mag_norm_mean"),
            "mean_normalized_angle": _mean("ang_norm_mean"),

            # Paper Eq. 5 + Eq. 7 — the headline output-free scores.
            "coe_r": _mean("coe_r"),
            "coe_r_std": _std("coe_r"),
            "coe_c": _mean("coe_c"),
            "coe_c_std": _std("coe_c"),
            # `_meta_`: a DOCUMENTED ALIAS -- `normalized_coe_c = coe_c`
            # literally, three lines apart in `_coe_for_sample`, because the
            # reference CoE-C already IS the fully-normalized form. Emitted
            # unprefixed it put ONE degree of freedom into the feature matrix
            # under two names, and a sweep over 5 models confirms the pair is
            # BIT-IDENTICAL (as is the `_std` pair) -- so a sparse selection
            # was splitting one coefficient across two copies of the same
            # number (duplicate-column sweep, 2026-09-05). The keys stay for
            # compatibility; they are not features.
            "_meta_normalized_coe_c": _mean("normalized_coe_c"),
            "_meta_normalized_coe_c_std": _std("normalized_coe_c"),

            # Raw (unnormalised) magnitudes/angles for completeness.
            "mean_magnitude_change": float(np.mean(flat_mags)),
            "std_magnitude_change": float(np.std(flat_mags)),
            "mean_angle_change": float(np.mean(flat_angs)),
            "std_angle_change": float(np.std(flat_angs)),

            # Per-sample summaries (useful for downstream scripts).
            "per_sample_mean_magnitude": [
                float(np.mean(r["magnitudes"])) for r in sample_results
            ],
            "per_sample_mean_angle": [
                float(np.mean(r["angles"])) for r in sample_results
            ],
            "per_sample_coe_r": [r["coe_r"] for r in sample_results],
            "per_sample_coe_c": [r["coe_c"] for r in sample_results],
            "_meta_per_sample_normalized_coe_c": [
                r["normalized_coe_c"] for r in sample_results
            ],

            # Metadata so reviewers can interpret the numbers.
            "axis": "layers",
            # _meta_: with the full h_0..h_L chain this equals n_layers
            # EXACTLY, so unprefixed it would be a perfect depth proxy in the
            # feature matrix.
            "_meta_chain_length_per_sample": int(chain_length or 0),
            # FALSE, and deliberately so. The headline chain is the RAW
            # HF tuple, matching `Alsace08/Chain-of-Embedding`'s
            # `inference.py`, so its deepest entry is `ln_f(b_{L-1})`.
            # This flag previously reported whether the HOOK FIRED --
            # True on 3/3 models while the headline ended at ln_f -- so a
            # flag whose entire job is to say which representation
            # produced the number was saying the opposite (independent
            # Opus review, 2026-09-06). `_meta_coe_r_blockout` below is
            # where the hooked value lives.
            "_meta_last_entry_is_block_output": False,
            "_meta_blockout_hook_fired": bool(_used_hook[0]),
            # DISCLOSURE, not an alternative headline. What `coe_r` would
            # be if the deepest entry were the raw block output instead of
            # `ln_f(...)`. The released code uses the raw tuple, so the
            # headline follows it; the substituted chain flips the SIGN on
            # every model tested, which a reader of either number needs to
            # know (independent Opus review, 2026-09-05).
            "_meta_coe_r_blockout": (
                float(np.mean([r["coe_r"] for r in _blockout_results]))
                if _blockout_results else float("nan")),
            "_meta_coe_c_blockout": (
                float(np.mean([r["coe_c"] for r in _blockout_results]))
                if _blockout_results else float("nan")),
            "token_position": token_position,
            "note": (
                "Token position is the last-token hidden state of the "
                "prompt, not a mean over generated output tokens as in "
                "the paper — used to keep the metric output-free for "
                "base models that cannot be safely generated with."
            ),
        }
