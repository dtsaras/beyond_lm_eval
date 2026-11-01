"""Rogue dimensions: how much of the representations' mean cosine similarity
(their anisotropy) a few coordinates carry (Timkey & van Schijndel 2021).

Reference: "All Bark and No Bite: Rogue Dimensions in Transformer Language
Models Obscure Representational Quality", EMNLP 2021 (arXiv:2109.04404);
code wtimkey/rogue-dimensions, ``replication.ipynb`` (Sections 2 and 3.1).

The release embeds the first 128 tokens (no special tokens) of each of 150
Wikipedia articles, keeps every token at every hidden state, and estimates
each coordinate's contribution to the cosine similarity of random token pairs,
CC_i = E[u_i v_i / (||u|| ||v||)] over 500000 random distinct pairs. Their sum
is the mean cosine similarity (the anisotropy); the paper reports the shares
CC_i / sum_j CC_j of the three largest. BLME computes the expectation over
all distinct pairs exactly: with unit rows u_a, S = sum_a u_a and
Q_i = sum_a u_ai^2, CC_i = (S_i^2 - Q_i) / (N (N - 1)).
"""
import logging

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions
from .utils import safe_unit_rows_torch

logger = logging.getLogger("blme")


def _sums(X):
    """Sufficient statistics of the unit rows of ``X``: (sum, sum of squares, n)."""
    U = safe_unit_rows_torch(torch.as_tensor(X).to(torch.float64))
    return U.sum(dim=0), (U * U).sum(dim=0), U.shape[0]


def _contributions(S, Q, n):
    return ((S * S - Q) / (n * (n - 1))).cpu().numpy()


def cosine_contributions(X):
    """Exact all-distinct-pairs CC_i of the rows of ``X`` (float64)."""
    return _contributions(*_sums(X))


def top_shares(cc, k=3):
    """Shares of the anisotropy carried by the k largest CC_i, largest first
    (the notebook's argsort over the signed contributions)."""
    aniso = float(cc.sum())
    top = np.sort(cc)[::-1][:k]
    if not aniso > 0:
        return [float("nan")] * k, aniso
    return [float(v / aniso) for v in top], aniso


@register_task("geometry_rogue_dimensions")
class RogueDimensionsTask(DiagnosticTask):
    """Share of the anisotropy carried by the top coordinates, per layer."""

    TOKEN_POLICY = ("paper_protocol:Timkey2021 every token of the first 128 of each document, "
                    "no special tokens")
    SOURCES = {
        "paper": "arXiv:2109.04404 (Timkey & van Schijndel, EMNLP 2021)",
        "code": "wtimkey/rogue-dimensions@84f40096f1be5fa9dc40a8c85cd9a916d93c9675",
        "llm_precedent": "arXiv:2109.04404 (GPT-2, XLNet, BERT, RoBERTa; every token, every layer)",
    }
    ADAPTATION_NOTE = (
        "The release's readout: the first 128 tokens of each of 150 articles, tokenized "
        "without special tokens, every token at every HF hidden state (the embedding output "
        "included, the last after the final norm). Articles are the first 150 WikiText-103 "
        "train articles in the library's order, where the release samples Wikipedia dumps "
        "(class 2). The release estimates CC_i from 500000 random distinct token pairs; BLME "
        "computes the same expectation over all distinct pairs exactly, which removes the "
        "Monte Carlo noise (class 2). Headline: the largest top-1 and top-3 shares over the "
        "layers and the top-1 share of the output representation. Duplicated layers (identity "
        "blocks), zero-padded coordinates and rescaling leave them unchanged; a rotation of the "
        "basis does not, since rogue dimensions are coordinates by definition. The shares "
        "without each document's first token, a sink in many models, are _meta_."
    )
    HEADLINE_KEYS = ("rogue_top1_share_max", "rogue_top3_share_max", "rogue_top1_share_last")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("insert_identity_blocks",),
         "tests/tasks/invariance/test_geometry_rogue_dimensions.py::test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Rogue Dimensions...")
        num_samples = int(self.config.get("num_samples", 150))
        context_length = int(self.config.get("context_length", 128))
        context_length = min(context_length, get_max_positions(model, context_length))
        if self.config.get("use_dataset", False):
            texts = [d["text"] if isinstance(d, dict) else str(d) for d in list(dataset)[:num_samples]]
        else:
            from ...cache import load_long_documents
            texts = [d["text"] for d in load_long_documents(num_samples, split="train")]
        device = next(model.parameters()).device

        # Per layer: sums over all tokens and over each document's first token,
        # so the all-pairs CC needs no stored states.
        acc, n_tok, n_doc = None, 0, 0
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, add_special_tokens=False, truncation=True,
                                max_length=context_length)["input_ids"]
                if len(ids) < 2:
                    continue
                out = model(input_ids=torch.tensor([ids], device=device),
                            output_hidden_states=True, use_cache=False)
                # post-final-norm: the release stacks HF's hidden_states, the
                # embedding output first and the post-norm output state last.
                hs = out.hidden_states
                if acc is None:
                    acc = [[0.0, 0.0, 0.0, 0.0] for _ in hs]
                for l, h in enumerate(hs):
                    S, Q, _ = _sums(h[0].float().cpu())
                    S0, Q0, _ = _sums(h[0, :1].float().cpu())
                    a = acc[l]
                    a[0], a[1], a[2], a[3] = a[0] + S, a[1] + Q, a[2] + S0, a[3] + Q0
                n_tok += len(ids)
                n_doc += 1
        if acc is None:
            return {"error": "No document has two tokens"}

        top1, top3, aniso, top1_nosink = [], [], [], []
        for S, Q, S0, Q0 in acc:
            shares, a = top_shares(_contributions(S, Q, n_tok))
            top1.append(shares[0])
            top3.append(float(np.sum(shares)))
            aniso.append(a)
            top1_nosink.append(top_shares(_contributions(S - S0, Q - Q0, n_tok - n_doc))[0][0])
        return {
            "rogue_top1_share_max": float(np.nanmax(top1)),
            "rogue_top3_share_max": float(np.nanmax(top3)),
            "rogue_top1_share_last": top1[-1],
            "rogue_top1_share_profile": top1,
            "rogue_top3_share_profile": top3,
            "anisotropy_profile": aniso,
            "_meta_rogue_top1_share_profile_without_first_token": top1_nosink,
            "_meta_n_tokens": n_tok,
            "_meta_n_documents": n_doc,
        }
