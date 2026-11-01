"""Zipfian symmetry of the embedding tables: centrality and isotropy weighted
by how often each token is used (Yokoi et al., NeurIPS 2024).

Reference: "Zipfian Whitening", arXiv:2411.00680; code cl-tohoku/zipfian-whitening
(``src/calc_isotropy_score.py``: ``calc_zipfian_isotropy_score`` and
``calc_uniform_isotropy_score``).

With p(w) the unigram probability of token w and W its embedding table
(rows of unused tokens removed):

* Sym1 = 1 - ||E_p[w]|| / E_p[||w||]: 1 when the table is centred under p;
* Sym2 = H(lambda) / log d, the entropy of the normalised spectrum lambda of the
  p-weighted centred table sqrt(p) (W - E_p[w]): 1 when it is isotropic.

The uniform versions put p = 1/n on the same rows. Yokoi et al. argue that
the frequency-weighted versions are the ones the geometry of a model reflects.
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_embeddings, get_lm_head

logger = logging.getLogger("blme")


def corpus_ids(tokenizer, texts):
    """Every token id of the texts (no special tokens), concatenated."""
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    return np.concatenate([np.asarray(i, dtype=np.int64) for i in ids]) if ids else np.zeros(0, np.int64)


def token_counts(ids, n_rows):
    """Unigram counts of the ids that have a row in an ``n_rows`` table."""
    return np.bincount(ids[ids < n_rows], minlength=int(n_rows))


def symmetry(W, p):
    """``calc_zipfian_isotropy_score(W, p)`` in float64, plus the spectral
    entropy H before its division by log d. Zero eigenvalues are skipped in H
    (the release would return NaN; a full-rank table has none). The release
    takes log d of an integer tensor, i.e. in float32; here it is float64."""
    W = torch.as_tensor(W, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    mu = p @ W
    sym1 = 1.0 - float(torch.linalg.norm(mu) / (p @ torch.linalg.norm(W, dim=1)))
    lam = torch.linalg.svdvals((W - mu) * torch.sqrt(p)[:, None]) ** 2
    q = lam / lam.sum()
    q = q[q > 0]
    H = float(-(q * torch.log(q)).sum())
    return sym1, H / math.log(W.shape[1]), H


@register_task("geometry_zipfian_symmetry")
class ZipfianSymmetryTask(DiagnosticTask):
    """Zipfian Sym1 / Sym2 of the input embedding and the unembedding."""

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2411.00680 (Yokoi et al., NeurIPS 2024, Zipfian Whitening)",
        "code": "cl-tohoku/zipfian-whitening@fa44d297f3fd54020e1195212b76614def81f8f9",
        "llm_precedent": ("arXiv:2411.00680 Sec. 5.2 (argues that causal-LM heads fall in the "
                          "Zipfian family; the experiments are on static word embeddings)"),
    }
    ADAPTATION_NOTE = (
        "The release's Sym1 and Sym2 (calc_zipfian_isotropy_score), and the uniform versions, "
        "on the rows of the tokens that occur in the corpus. The release takes p(w) from "
        "English Wikipedia word counts for static word embeddings; BLME counts each model's "
        "own tokens (no special tokens) over every passage of the library's corpus, the same "
        "text for every model (class 2), and scores both the input embedding and the output "
        "unembedding (identical for tied models). float64 SVD. Sym1 is unchanged by rescaling, "
        "rotation and zero-padding of the table; Sym2's division by log d is not "
        "padding-invariant, so the undivided spectral entropy is _meta_ (the survey's twin). "
        "Inserted blocks do not touch either table."
    )
    HEADLINE_KEYS = ("zipf_sym1_embedding", "zipf_sym2_embedding",
                     "zipf_sym1_unembedding", "zipf_sym2_unembedding")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("insert_identity_blocks",),
         "tests/tasks/parity/test_zipfian_symmetry_parity.py::"
         "test_headline_invariant_under_identity_blocks"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Zipfian symmetry...")
        if self.config.get("use_dataset", False):
            texts = [d["text"] if isinstance(d, dict) else str(d) for d in dataset]
        else:
            from ...cache import default_corpus_size, load_default_corpus
            texts = [d["text"] for d in load_default_corpus(default_corpus_size())]
        ids = corpus_ids(tokenizer, texts)
        tables = {"embedding": get_embeddings(model)}
        head = get_lm_head(model)
        tables["unembedding"] = head
        out, used = {}, {}
        for name, module in tables.items():
            W = module if torch.is_tensor(module) else getattr(module, "weight", None)
            if W is None:
                return {"error": f"Could not find the {name} table"}
            W = W.detach()
            counts = token_counts(ids, W.shape[0])
            keep = np.nonzero(counts)[0]
            if keep.size < 2:
                return {"error": "Fewer than two corpus tokens have rows in the table"}
            rows = W[torch.as_tensor(keep, device=W.device)].to(torch.float64).cpu()
            p = counts[keep] / counts[keep].sum()
            s1, s2, H = symmetry(rows, p)
            u1, u2, uH = symmetry(rows, np.full(keep.size, 1.0 / keep.size))
            out[f"zipf_sym1_{name}"] = s1
            out[f"zipf_sym2_{name}"] = s2
            out[f"uniform_sym1_{name}"] = u1
            out[f"uniform_sym2_{name}"] = u2
            out[f"_meta_zipf_spectral_entropy_{name}"] = H
            out[f"_meta_uniform_spectral_entropy_{name}"] = uH
            used[name] = int(keep.size)
        out["_meta_n_used_tokens"] = used["embedding"]
        out["_meta_n_corpus_tokens"] = int(ids.size)
        emb = tables["embedding"] if torch.is_tensor(tables["embedding"]) else tables["embedding"].weight
        out["_meta_tied_embeddings"] = bool(
            getattr(head, "weight", None) is not None and emb.data_ptr() == head.weight.data_ptr())
        return out
