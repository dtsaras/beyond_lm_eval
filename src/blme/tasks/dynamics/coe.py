"""Chain-of-Embedding (CoE) — Wang et al., ICLR 2025 (arXiv:2410.13640).

Given the chain of hidden states
``H = h^(0), h^(1), …, h^(L)`` at a fixed token (``h^(0)`` is the
embedding output, ``h^(L)`` is the final transformer block output),
CoE measures how the representation *changes* between adjacent layers:

  * Magnitude change   ``M(h_l, h_{l+1}) = ‖h_{l+1} − h_l‖₂``               (Eq. 1)
  * Angle change       ``A(h_l, h_{l+1}) = arccos(<h_{l+1}, h_l>/(‖.‖·‖.‖))`` (Eq. 2)

The paper's headline "output-free" scores:

  * ``Mag̃ = (1/L) Σ M(h_l,h_{l+1}) / M(h_0, h_L)``                         (Eq. 3)
  * ``Ãng = (1/L) Σ A(h_l,h_{l+1}) / A(h_0, h_L)``                          (Eq. 3)
  * ``CoE-R = (1/L) Σ [M(h_l,h_{l+1})/M(h_0,h_L) − A(h_l,h_{l+1})/A(h_0,h_L)]`` (Eq. 5)
  * ``CoE-C = |(1/L) Σ C(h_l,h_{l+1})|`` where
              ``C(h_l,h_{l+1}) = M · exp(i · A)``                             (Eq. 7)

Note on token position (departure from the paper): Wang et al. run
``model.generate`` and mean-pool ``h^(l)`` across every generated
output token. BLME is comparing *base and instruct models without
generating* — we read the final token of the prompt instead. The two
give highly correlated but not identical signals; we flag
``token_position`` in the result so reviewers can see the choice.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def _angle_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Arccos of the clamped cosine similarity — Eq. 2."""
    an = F.normalize(a, p=2, dim=-1)
    bn = F.normalize(b, p=2, dim=-1)
    c = float(torch.dot(an, bn).item())
    if c >= 1.0 - 1e-12:
        return 0.0
    if c <= -1.0 + 1e-12:
        return float(np.pi)
    return float(torch.acos(torch.tensor(c)).item())


def _coe_from_chain(chain: list[torch.Tensor]) -> dict:
    """Compute all Wang et al. 2025 CoE statistics for one sample.

    ``chain`` is ``[h_0, h_1, …, h_L]`` with ``L ≥ 1``. Returns a dict
    containing per-pair magnitudes + angles, the Eq. 3 normalised
    means, and the Eq. 5 / Eq. 7 scalar scores.
    """
    L = len(chain) - 1
    mags = []
    angs = []
    for i in range(L):
        mags.append(float((chain[i + 1] - chain[i]).norm(p=2).item()))
        angs.append(_angle_between(chain[i], chain[i + 1]))

    # End-to-end normalisers Z_Mag, Z_Ang (Eq. 3).
    z_mag = float((chain[-1] - chain[0]).norm(p=2).item())
    z_ang = _angle_between(chain[0], chain[-1])

    # Guard degenerate cases where the representation doesn't move at
    # all: Z_Mag → 0 makes Mag̃ undefined, Z_Ang → 0 makes Ãng and the
    # Eq. 7 complex phase undefined. Return NaN for those fields.
    if z_mag <= 1e-12:
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
    # (Fixed 2026-06-22: previously used RAW magnitude and RAW angle, which
    # matched neither the paper code nor the paper prose.)
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
                chain = [
                    h[0, pos].detach().float().cpu()
                    for h in out.hidden_states
                ]
                if len(chain) < 2:
                    continue
                chain_length = len(chain) - 1
                sample_results.append(_coe_from_chain(chain))

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
            "mean_normalized_magnitude": _mean("mag_norm_mean"),
            "mean_normalized_angle": _mean("ang_norm_mean"),

            # Paper Eq. 5 + Eq. 7 — the headline output-free scores.
            "coe_r": _mean("coe_r"),
            "coe_r_std": _std("coe_r"),
            "coe_c": _mean("coe_c"),
            "coe_c_std": _std("coe_c"),
            "normalized_coe_c": _mean("normalized_coe_c"),
            "normalized_coe_c_std": _std("normalized_coe_c"),

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
            "per_sample_normalized_coe_c": [
                r["normalized_coe_c"] for r in sample_results
            ],

            # Metadata so reviewers can interpret the numbers.
            "axis": "layers",
            "chain_length_per_sample": int(chain_length or 0),
            "token_position": token_position,
            "note": (
                "Token position is the last-token hidden state of the "
                "prompt, not a mean over generated output tokens as in "
                "the paper — used to keep the metric output-free for "
                "base models that cannot be safely generated with."
            ),
        }
