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

Where the model adds a position embedding to the token embedding (GPT-2's
wpe), Y carries position too; the profile with Y the token embedding alone is
``_meta_predictive_info_profile_token_embedding``. Special tokens (an appended
EOS, as OLMo's tokenizer adds) are never drawn as targets.
"""
import logging
import math
import random

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions, relative_depth
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


def split_points(lengths, seed, min_context=_MIN_CONTEXT, allowed=None):
    """One target position per sequence, uniform over [min_context, T - 1];
    None for a sequence too short to have one. ``allowed[i]`` (the positions
    of that range that may be a target) narrows the draw; with every position
    allowed it is the same draw as without."""
    rng = random.Random(seed)
    out = []
    for i, n in enumerate(lengths):
        cand = range(min_context, n) if allowed is None else allowed[i]
        out.append(cand[rng.randrange(len(cand))] if n > min_context and len(cand) else None)
    return out


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
    SIZE_INVARIANT = {
        "predictive_info_peak": (("identity_blocks_interior",), _INV + "test_headline_invariant"),
        "predictive_info_last_layer": (
            ("identity_blocks_interior", "identity_blocks_leading", "identity_blocks_trailing"),
            _INV + "test_headline_invariant"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Predictive Information I(Z_l; Y)...")
        num_samples = int(self.config.get("num_samples", 1000))
        max_length = int(self.config.get("max_length", 512))
        max_length = min(max_length, get_max_positions(model, max_length))
        sigma = float(self.config.get("sigma", 1.0))
        if dataset is None:
            from ...cache import load_default_corpus
            dataset = load_default_corpus(num_samples)
        device = next(model.parameters()).device

        texts = [s["text"] if isinstance(s, dict) else str(s) for s in list(dataset)[:num_samples]]
        encoded = [tokenizer(t, truncation=True, max_length=max_length)["input_ids"] for t in texts]
        # A special token is never a target: an EOS-appending tokenizer (OLMo)
        # would otherwise offer its EOS as the next token of a passage.
        special = set(getattr(tokenizer, "all_special_ids", None) or [])
        allowed = [[p for p in range(_MIN_CONTEXT, len(ids)) if ids[p] not in special]
                   for ids in encoded]
        n_special_excluded = sum(max(0, len(ids) - _MIN_CONTEXT) - len(a)
                                 for ids, a in zip(encoded, allowed))
        targets = split_points([len(ids) for ids in encoded],
                               int(self.config.get("split_seed", 0)), allowed=allowed)

        # Y without the position embedding (GPT-2's wpe), read from the input
        # embedding module at the target token.
        try:
            emb = model.get_input_embeddings()
        except (AttributeError, NotImplementedError):
            emb = None
        emb_dev = next(emb.parameters()).device if emb is not None else device
        zs, ys, ys_tok = None, [], []
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
                if emb is not None:
                    ys_tok.append(emb(torch.tensor([ids[t]], device=emb_dev))[0].float().cpu())
        n = len(ys)
        if n < 2:
            return {"error": f"Fewer than 2 sequences longer than {_MIN_CONTEXT} tokens"}

        gy, hy = gram_and_entropy(torch.stack(ys), sigma)
        profile, hzs = [], []
        for z in zs:
            gz, hz = gram_and_entropy(torch.stack(z), sigma)
            hzs.append(hz)
            profile.append(mutual_information(gz, hz, gy, hy))
        L = len(profile)
        peak = int(np.argmax(profile))
        # Token-embedding-only twin: equal to the profile unless the model adds
        # something to the token embedding before block 1 (a position
        # embedding, or a scale). tests/tasks/test_predictive_information_targets.py
        y_is_tok = len(ys_tok) == n and all(torch.equal(a, b) for a, b in zip(ys, ys_tok))
        if y_is_tok or len(ys_tok) != n:
            profile_tok, hy_tok = (profile, hy) if y_is_tok else (None, float("nan"))
        else:
            gy_t, hy_tok = gram_and_entropy(torch.stack(ys_tok), sigma)
            profile_tok = [
                mutual_information(gaussian_gram(safe_unit_rows_torch(torch.stack(z).float()),
                                                 sigma), hz, gy_t, hy_tok)
                for z, hz in zip(zs, hzs)]
        return {
            "_meta_predictive_info_profile_token_embedding": profile_tok,
            "_meta_predictive_info_peak_token_embedding": (
                float(max(profile_tok)) if profile_tok else float("nan")),
            "_meta_predictive_info_last_layer_token_embedding": (
                float(profile_tok[-1]) if profile_tok else float("nan")),
            "_meta_H_Y_token_embedding": hy_tok,
            "_meta_y_is_token_embedding": bool(y_is_tok),
            "_meta_n_special_target_positions_excluded": int(n_special_excluded),
            "predictive_info_profile": profile,
            "predictive_info_peak": float(profile[peak]),
            "predictive_info_peak_depth": relative_depth(peak + 1, L),
            "predictive_info_last_layer": float(profile[-1]),
            "_meta_predictive_info_peak_layer": peak + 1,
            "_meta_H_Y": hy,
            "_meta_log_n": math.log(n),
            "_meta_n_sequences": n,
            "_meta_n_too_short": len(texts) - n,
        }
