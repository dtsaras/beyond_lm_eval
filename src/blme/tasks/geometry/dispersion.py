"""Representation dispersion: the mean pairwise cosine distance between the
final representations of text segments (Li et al. 2026).

Reference: "On the Predictive Power of Representation Dispersion in Language
Models", arXiv:2506.24106 (ICLR 2026); code yanhong-lbh/rep_dispersion
(``data.make_token_chunks``, ``embeddings.extract_lm_embeddings``,
``metrics.average_pairwise_cosine_distance``).

The release joins the WikiText-103 validation rows with blank lines, tokenizes
the stream without special tokens, cuts it into 512-token chunks (the last
partial one dropped; a seeded sample of 10000 if there are more), and takes
each chunk's last-token state from HF's final hidden state. Dispersion is
mean_{i<j} (1 - cos(x_i, x_j)) over those vectors, in float32.
"""

import logging
import random

import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


def wikitext_validation_rows():
    """The release's ``load_texts("wikitext", split="validation")`` rows: every
    non-blank row, unstripped (on the raw WikiText-103 the library caches)."""
    from ...cache import WIKITEXT_DATASET, _load_split
    path, name, split = WIKITEXT_DATASET
    rows = [str(r["text"]) for r in _load_split(path, name, split)]
    return [t for t in rows if t and not t.isspace()]


def token_chunks(tokenizer, texts, chunk_len, n_chunks, seed):
    """``make_token_chunks(concatenate=True, drop_last=True)``."""
    ids = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
    chunks = [ids[s:s + chunk_len] for s in range(0, len(ids), chunk_len)]
    chunks = [c for c in chunks if len(c) == chunk_len]
    if n_chunks is not None and n_chunks < len(chunks):
        chunks = [chunks[i] for i in random.Random(seed).sample(range(len(chunks)), n_chunks)]
    return chunks


def average_pairwise_cosine_distance(x):
    """``metrics.average_pairwise_cosine_distance`` (i < j pairs). The release
    floors the norm at 1e-12; the float32 tiny floor gives the same value for
    every row with a norm above 1e-12 and does not set a scale."""
    x = x.float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    dist = 1.0 - x @ x.T
    n = dist.shape[0]
    if n < 2:
        return 0.0
    mask = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
    return float(dist[mask].mean().item())


@register_task("geometry_representation_dispersion")
class RepresentationDispersionTask(DiagnosticTask):
    """Mean pairwise cosine distance of the last-token states of 512-token chunks."""

    TOKEN_POLICY = "paper_protocol:Li2026 the last token of each 512-token chunk"
    SOURCES = {
        "paper": "arXiv:2506.24106 (Li et al., ICLR 2026)",
        "code": "yanhong-lbh/rep_dispersion@dae432db4da1c0935bf48ccc5200e5ec647fb90a",
        "llm_precedent": ("arXiv:2506.24106 (Llama-3.x, Gemma-2, Mistral, Phi-2, Qwen-2.5: "
                          "the last token of 512-token WikiText chunks)"),
    }
    ADAPTATION_NOTE = (
        "Follows the release: the WikiText-103 validation rows joined with blank lines, "
        "tokenized without special tokens, every full 512-token chunk (a seeded sample of "
        "10000 when there are more), the last token's state from HF's final hidden state "
        "(after the final norm), and the mean pairwise cosine distance over i < j in float32. "
        "The rows are the raw WikiText-103 where the release reads the tokenized variant "
        "(<unk> for rare words, ' @-@ ' hyphens), which the library does not cache (class 2). "
        "The paper relates dispersion to perplexity within a model across segment bins; "
        "BLME reports the model-level dispersion over all chunks. Without special tokens a "
        "model trained with a BOS sees none, as in the release; _meta_tokenizer_adds_bos "
        "flags those models (Gemma is known to degrade without it). Cosine distances are "
        "unchanged by identity blocks, rescaling or an isometric embedding of the final state."
    )
    HEADLINE_KEYS = ("representation_dispersion",)
    SIZE_INVARIANT = {
        "representation_dispersion": (
            ("insert_identity_blocks",),
            "tests/tasks/invariance/test_geometry_representation_dispersion.py::"
            "test_headline_invariant"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Representation Dispersion...")
        chunk_len = int(self.config.get("chunk_len", 512))
        n_chunks = self.config.get("n_chunks", 10000)
        seed = int(self.config.get("chunk_seed", 42))
        batch_size = int(self.config.get("batch_size", 1))
        if self.config.get("use_dataset", False):
            texts = [d["text"] if isinstance(d, dict) else str(d) for d in dataset]
        else:
            texts = wikitext_validation_rows()
        chunks = token_chunks(tokenizer, texts, chunk_len,
                              None if n_chunks is None else int(n_chunks), seed)
        if len(chunks) < 2:
            return {"error": f"Fewer than 2 chunks of {chunk_len} tokens"}

        device = next(model.parameters()).device
        vecs = []
        with torch.no_grad():
            for s in range(0, len(chunks), batch_size):
                ids = torch.tensor(chunks[s:s + batch_size], device=device)
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                # post-final-norm: the release's layer_idx=-1, HF's final hidden
                # state after the final norm.
                vecs.append(out.hidden_states[-1][:, -1, :].detach().cpu().float())
        x = torch.cat(vecs, dim=0)
        with_bos = tokenizer("a")["input_ids"]
        return {
            "representation_dispersion": average_pairwise_cosine_distance(x),
            "_meta_n_chunks": len(chunks),
            "_meta_tokenizer_adds_bos": len(with_bos) > len(tokenizer.encode("a", add_special_tokens=False)),
        }
