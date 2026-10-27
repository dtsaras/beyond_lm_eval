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
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Arccos of the clamped cosine similarity — Eq. 2."""
    # Scale-exact unit vectors: the plain norm wherever it is finite and
    # nonzero, the max-factored form only where it under/overflows, so a
    # positive rescale of the chain cancels (test_coverage_gap_parity.py).
    def _unit(v):
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
    # math.acos on a Python float: torch.acos would follow the default dtype.
    return float(math.acos(max(-1.0, min(1.0, float(c)))))


def _coe_from_chain(chain: list[torch.Tensor], storage_eps: float = None) -> dict:
    """Compute all Wang et al. 2025 CoE statistics for one sample.

    ``chain`` is ``[h_0, h_1, …, h_L]`` with ``L ≥ 1``. Returns a dict
    containing per-pair magnitudes + angles, the Eq. 3 normalised
    means, and the Eq. 5 / Eq. 7 scalar scores.

    Arithmetic of the released score.py: every ratio is divided by Z_Mag /
    Z_Ang whatever their size, and a value is NaN only where score.py itself
    would give a non-finite value or raise (Z_Mag = 0, Z_Ang = 0). The margin
    ``z_mag_margin_log10 = log10(Z_Mag / (storage_eps * max|chain|))`` shows a
    normaliser that sits below the resolution of the stored states (<= 0)
    instead of filtering it; ``storage_eps`` defaults to the chain's own dtype.
    """
    def _norm2(v):
        """`v.norm(p=2)` in the chain's dtype, with a double fallback only where
        that norm under/overflows (float-first keeps parity with score.py)."""
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

    # No floor beyond score.py's own (decision 1); the margin below discloses
    # a normaliser near the storage resolution instead.
    _chain_scale = max((float(c.abs().max()) for c in chain), default=0.0)
    if not math.isfinite(z_mag) or z_mag == 0.0:
        mag_norm_pair = [float("nan")] * L
    else:
        mag_norm_pair = [m / z_mag for m in mags]
    if not math.isfinite(z_ang) or z_ang == 0.0:
        ang_norm_pair = [float("nan")] * L
    else:
        ang_norm_pair = [a / z_ang for a in angs]
    if storage_eps is None:
        storage_eps = float(torch.finfo(chain[0].dtype).eps)
    if _chain_scale > 0 and z_mag > 0 and math.isfinite(z_mag):
        z_margin = math.log10(z_mag / (storage_eps * _chain_scale))
    else:
        z_margin = float("-inf") if _chain_scale > 0 else float("nan")

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

    # CoE-C (Eq. 6/7) as score.py compute_CoE_C: normalised magnitude as radius
    # AND normalised angle as phase (the paper's prose and Algorithm 2 differ;
    # the released code is followed).
    if np.all(np.isfinite(mag_norm_pair)) and np.all(np.isfinite(ang_norm_pair)):
        x = np.array([mn * np.cos(an) for mn, an in zip(mag_norm_pair, ang_norm_pair)])
        y = np.array([mn * np.sin(an) for mn, an in zip(mag_norm_pair, ang_norm_pair)])
        coe_c = float(np.sqrt(x.mean() ** 2 + y.mean() ** 2))
        # Documented alias kept for compatibility (emitted as _meta_).
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
        "z_mag_margin_log10": z_margin,
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

        # The deepest block's raw output is hooked only for the _meta_ blockout
        # disclosure; the headline chain is the raw HF tuple, as in inference.py.
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
                # Full chain h_0 .. h_L at the chosen position, from the raw HF
                # tuple including the post-final-norm entry, as inference.py does
                # (post-final-norm=False is deliberate: the released code wins).
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
                    # No usable hook: same headline chain, no blockout disclosure.
                    _used_hook[0] = False
                    chain = [h[0, pos].detach().float().cpu() for h in _hs]
                    _chain_blockout = None
                if len(chain) < 2:
                    continue
                chain_length = len(chain) - 1
                _eps = float(torch.finfo(_hs[0].dtype).eps)
                sample_results.append(_coe_from_chain(chain, storage_eps=_eps))
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
            # Paper Eq. 3. The normalised magnitude is _meta_: coe_r is exactly
            # magnitude - angle, so the three would carry two degrees of freedom.
            "_meta_mean_normalized_magnitude": _mean("mag_norm_mean"),
            "mean_normalized_angle": _mean("ang_norm_mean"),

            # Paper Eq. 5 + Eq. 7 — the headline output-free scores.
            "coe_r": _mean("coe_r"),
            "coe_r_std": _std("coe_r"),
            "coe_c": _mean("coe_c"),
            "coe_c_std": _std("coe_c"),
            # _meta_: a documented alias, bit-identical to coe_c.
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
            # _meta_: equals n_layers exactly.
            "_meta_chain_length_per_sample": int(chain_length or 0),
            # False by construction: the headline chain ends at ln_f(...), as in
            # inference.py; the hooked raw block output feeds only *_blockout.
            "_meta_last_entry_is_block_output": False,
            "_meta_blockout_hook_fired": bool(_used_hook[0]),
            # Disclosure: coe_r / coe_c with the raw deepest block output in
            # place of ln_f(...); it flips the sign on gpt2, pythia and distilgpt2.
            "_meta_coe_r_blockout": (
                float(np.mean([r["coe_r"] for r in _blockout_results]))
                if _blockout_results else float("nan")),
            # Smallest log10(Z_Mag / (eps_storage * max|chain|)) over samples:
            # <= 0 means a normaliser below the stored states' resolution.
            "_meta_z_mag_margin_log10_min": float(np.min(
                [r["z_mag_margin_log10"] for r in sample_results])),
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
