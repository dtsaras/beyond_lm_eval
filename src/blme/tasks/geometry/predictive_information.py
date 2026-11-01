"""Predictive information I(Z_l; Y) across depth: the mutual information
between each layer's state and the true next token (Chang et al., COLM 2026).

Reference: "The Generalization Ridge: Information Flow in Natural Language
Generation", arXiv:2507.05387; code chili-lab/InfoRidge
(``mutual_information/analysis.py::analyze_batch``, ``utils.py``).

The estimator is the matrix-based Renyi entropy at alpha = 1 (von Neumann):
rows are L2-normalised, the Gram matrix is a Gaussian kernel (sigma = 1)
divided by its trace, H = -sum lambda log lambda over its eigenvalues
(floored at 1e-12 and renormalised), and
I(Z; Y) = H(Z) + H(Y) - H(Z, Y), with the joint density the trace-normalised
Hadamard product of the two Grams. Z_l is the state at the last context
position at HF's hidden_states[l], l = 1..L; Y is hidden_states[0] (the
embedding output) at the target position.
"""
import logging
import math
import random

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from .utils import safe_unit_rows_torch

logger = logging.getLogger("blme")

# utils.compute_von_neumann_entropy_from_density floors the eigenvalues of the
# unit-trace density at 1e-12; the trace fixes the scale, so this is relative.
_EIG_FLOOR = 1e-12
_MIN_CONTEXT = 8


def gaussian_gram(x, sigma=1.0):
    """``utils.gaussian_gram_matrix`` on (N, D) rows."""
    sq = (x ** 2).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (x @ x.T)
    return torch.exp(-torch.clamp(d2, min=0.0) / (2 * sigma ** 2))


def von_neumann_entropy(rho):
    eig = torch.clamp(torch.linalg.eigvalsh(rho), min=_EIG_FLOOR)
    eig = eig / eig.sum()
    return float(-(eig * torch.log(eig)).sum())


def gram_and_entropy(x, sigma=1.0):
    """``analysis._compute_gram_and_entropy``: unit rows, Gaussian Gram, entropy
    of the trace-normalised Gram."""
    g = gaussian_gram(safe_unit_rows_torch(x.float()), sigma)
    return g, von_neumann_entropy(g / torch.trace(g))


def mutual_information(gz, hz, gy, hy):
    """``H_Z + H_Y - compute_joint_entropy(gram_z, gram_y)``."""
    joint = (gz / torch.trace(gz)) * (gy / torch.trace(gy))
    return hz + hy - von_neumann_entropy(joint / torch.trace(joint))


def split_points(lengths, seed, min_context=_MIN_CONTEXT):
    """One target position per sequence, uniform over [min_context, T - 1];
    None for a sequence too short to have one."""
    rng = random.Random(seed)
    return [rng.randint(min_context, n - 1) if n > min_context else None for n in lengths]


@register_task("geometry_predictive_information")
class PredictiveInformationTask(DiagnosticTask):
    """Matrix-based I(Z_l; Y) profile of the next-token target (InfoRidge)."""

    TOKEN_POLICY = ("paper_protocol:Chang2026 the last context position of each sequence "
                    "against the embedding of the true next token")
    SOURCES = {
        "paper": "arXiv:2507.05387 (Chang et al., COLM 2026, The Generalization Ridge)",
        "code": "chili-lab/InfoRidge@2cde64367d2f46d089ac6f9eaa4fb1e61b3e45e7",
        "llm_precedent": ("arXiv:2507.05387 (Qwen-2.5-0.5B, Llama-3.1-8B, GPT-2: last prompt "
                          "token against the first answer token)"),
    }
    ADAPTATION_NOTE = (
        "The estimator follows the release (unit rows, Gaussian kernel sigma = 1, "
        "trace-normalised Grams, von Neumann entropy with its 1e-12 eigenvalue floor, float32), "
        "and so do the readouts: Z_l at HF's hidden_states 1..L (the last after the final norm) "
        "on the last context token, Y at hidden_states[0] (the embedding output, position "
        "embedding included where the model adds one) on the target. The release scores "
        "question/answer pairs; BLME splits each of the first 1000 corpus passages at one "
        "position drawn uniformly from [8, T - 1] (seed 0), the tokens before it the context "
        "and the token at it the target (class 2), and runs context and target in one "
        "forward pass, which causal attention makes equal to the release's two. N is pinned "
        "at 1000 because the MI is bounded by log N. predictive_info_peak and "
        "predictive_info_last_layer are unchanged when identity blocks duplicate inner "
        "layers; predictive_info_peak_depth (argmax / L) moves by design."
    )
    HEADLINE_KEYS = ("predictive_info_peak", "predictive_info_peak_depth",
                     "predictive_info_last_layer")
    _INV = "tests/tasks/invariance/test_geometry_predictive_information.py::"
    SIZE_INVARIANT = dict.fromkeys(
        ("predictive_info_peak", "predictive_info_last_layer"),
        (("insert_identity_blocks",), _INV + "test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Predictive Information I(Z_l; Y)...")
        num_samples = int(self.config.get("num_samples", 1000))
        max_length = int(self.config.get("max_length", 512))
        sigma = float(self.config.get("sigma", 1.0))
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        device = next(model.parameters()).device

        texts = [s["text"] if isinstance(s, dict) else str(s) for s in list(dataset)[:num_samples]]
        encoded = [tokenizer(t, truncation=True, max_length=max_length)["input_ids"] for t in texts]
        targets = split_points([len(ids) for ids in encoded],
                               int(self.config.get("split_seed", 0)))

        zs, ys = None, []
        with torch.no_grad():
            for ids, t in zip(encoded, targets):
                if t is None:
                    continue
                out = model(input_ids=torch.tensor([ids[:t + 1]], device=device),
                            output_hidden_states=True, use_cache=False)
                # post-final-norm: the release's layers are HF's hidden_states
                # 1..L, the last after the final norm; hidden_states[0] is Y.
                hs = out.hidden_states
                if zs is None:
                    zs = [[] for _ in range(len(hs) - 1)]
                for l in range(1, len(hs)):
                    zs[l - 1].append(hs[l][0, t - 1].float().cpu())
                ys.append(hs[0][0, t].float().cpu())
        n = len(ys)
        if n < 2:
            return {"error": f"Fewer than 2 sequences longer than {_MIN_CONTEXT} tokens"}

        gy, hy = gram_and_entropy(torch.stack(ys), sigma)
        profile = []
        for z in zs:
            gz, hz = gram_and_entropy(torch.stack(z), sigma)
            profile.append(mutual_information(gz, hz, gy, hy))
        L = len(profile)
        peak = int(np.argmax(profile))
        return {
            "predictive_info_profile": profile,
            "predictive_info_peak": float(profile[peak]),
            "predictive_info_peak_depth": (peak + 1) / L,
            "predictive_info_last_layer": float(profile[-1]),
            "_meta_predictive_info_peak_layer": peak + 1,
            "_meta_H_Y": hy,
            "_meta_log_n": math.log(n),
            "_meta_n_sequences": n,
            "_meta_n_too_short": len(texts) - n,
        }
