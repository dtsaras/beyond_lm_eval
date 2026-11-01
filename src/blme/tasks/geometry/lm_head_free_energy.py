"""LM-head spectral free energy BFE-W (Wilson, NeurIPS 2026).

Reference: "Why Heavy-Tailed Weight Spectra Predict Model Quality",
arXiv:2610.13123; code josephwilsonmaths/heavytail
(``util/pythia_spectra.py``: the eigenvalues of W.T @ W of the output
unembedding in float32; ``util/pythia_ww.py::bfe_w`` and
``util/qwen_benchmarks.py::compute_free_energy``).

With e the eigenvalues and lambda = 1 / mean(e), the release's per-eigenvalue
free energy is

    BFE-W = lambda/2 mean(e) - 1/2 mean(log e) + 1/2 log(2 pi / lambda)
          = 1/2 log(mean(e) / geomean(e)) + 1/2 + 1/2 log(2 pi),

half the log ratio of the arithmetic to the geometric mean of the spectrum
plus a constant: scale-free, larger for a more unequal (heavier-tailed)
spectrum.
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_lm_head

logger = logging.getLogger("blme")


def head_eigenvalues(W, dtype=torch.float32, chunk=16384):
    """``eigvalsh(W.T @ W)`` of the (vocab, d) head in ``dtype``, ascending.
    float32 forms the Gram in one product, as the release does; float64
    accumulates it over row chunks so no float64 copy of the head is made."""
    W = W.detach()
    if dtype == torch.float32:
        W32 = W.to(torch.float32)
        G = W32.T @ W32
    else:
        G = torch.zeros(W.shape[1], W.shape[1], dtype=dtype, device=W.device)
        for i in range(0, W.shape[0], chunk):
            Wc = W[i:i + chunk].to(dtype)
            G += Wc.T @ Wc
    return torch.linalg.eigvalsh(G).cpu().numpy().astype(np.float64)


def bfe_w(e):
    """``pythia_ww.bfe_w(e, jitter=0.0)``: NaN where the release raises (a
    non-positive or non-finite eigenvalue)."""
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    if not len(e) or not np.all(np.isfinite(e)) or e.mean() <= 0 or np.any(e <= 0):
        return float("nan"), float("nan")
    lam = 1.0 / e.mean()
    energy = lam / 2 * e.mean() - .5 * np.log(e).mean() + .5 * np.log(2 * np.pi / lam)
    return float(energy), float(lam)


@register_task("geometry_lm_head_free_energy")
class LMHeadFreeEnergyTask(DiagnosticTask):
    """BFE-W of the output unembedding's spectrum (Wilson 2026)."""

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2610.13123 (Wilson, NeurIPS 2026, Why Heavy-Tailed Weight Spectra Predict "
                 "Model Quality)",
        "code": "josephwilsonmaths/heavytail@80f7f8008e2adf62014a058e74641c6ab97c41aa",
        "llm_precedent": ("arXiv:2610.13123 (Pythia 410M-12B embed_out, Qwen3.5 lm_head: "
                          "BFE-W against LAMBADA and MMLU-Pro)"),
    }
    ADAPTATION_NOTE = (
        "As released: the eigenvalues of W.T @ W of the output unembedding (the tied input "
        "embedding where the model ties them), in float32, and BFE-W at lambda = 1 / mean(e) "
        "with no jitter (pythia_ww.bfe_w; the Qwen script's 1e-20 jitter changes nothing for a "
        "full-rank head). A non-positive eigenvalue gives NaN where the release raises. The "
        "float32 Gram loses the smallest eigenvalues of an ill-conditioned head (gpt2: 12% on "
        "the smallest, 8.8e-5 on BFE-W), so the float64 value is _meta_. BFE-W is unchanged by "
        "rescaling or rotating the head and by inserted blocks; zero-padding the width adds "
        "zero eigenvalues, where it is undefined."
    )
    HEADLINE_KEYS = ("lm_head_bfe_w",)
    SIZE_INVARIANT = {
        "lm_head_bfe_w": (
            ("insert_identity_blocks",),
            "tests/tasks/parity/test_lm_head_free_energy_parity.py::"
            "test_headline_invariant_under_identity_blocks"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running LM-head spectral free energy (BFE-W)...")
        head = get_lm_head(model)
        if head is None or getattr(head, "weight", None) is None:
            return {"error": "Could not find the output unembedding"}
        W = head.weight
        energy, lam = bfe_w(head_eigenvalues(W))
        e64 = head_eigenvalues(W, torch.float64)
        energy64, _ = bfe_w(e64)
        return {
            "lm_head_bfe_w": energy,
            "_meta_lm_head_bfe_w_float64": energy64,
            "_meta_lm_head_half_log_am_gm": energy64 - 0.5 - 0.5 * math.log(2 * math.pi),
            "_meta_lm_head_bfe_lambda": lam,
            "_meta_lm_head_min_eigenvalue_float64": float(e64.min()),
            "_meta_n_eigenvalues": int(e64.size),
            "_meta_head_shape": [int(s) for s in W.shape],
        }
