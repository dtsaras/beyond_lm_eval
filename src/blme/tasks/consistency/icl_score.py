"""Olsson in-context learning score: how much the preceding context lowers the
loss within a document.

Reference: Olsson et al. 2022, "In-context Learning and Induction Heads",
arXiv:2209.11895: "the loss of the 500th token in the context minus the
average loss of the 50th token in the context, averaged over dataset
examples". There is no released code; the score is the formula.

Each context is the start of one whole WikiText-103 article (train split,
the library's nested order), so every tokenizer sees the same text. Articles
are kept by byte length (>= 4096 bytes, which holds 501 tokens for any
tokenizer under 8 bytes per token), not by token count, so every model scores
the same articles.

* ``icl_score``: mean over contexts of NLL(token 500) - NLL(token 50), counted
  over content tokens (a BOS is context, not a counted token).
* ``icl_score_byte_anchored``: the same difference at the tokens covering
  bytes 2000 and 200 of the article (about tokens 500 and 50 at 4 bytes per
  token), so the two positions sit at the same text for every tokenizer.
"""

import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions

logger = logging.getLogger("blme")

_EARLY, _LATE = 50, 500
_EARLY_BYTE, _LATE_BYTE = 200, 2000
_MIN_BYTES = 4096


def _byte_spans(text, offsets):
    """Byte [start, end) of each token from its character offsets."""
    cum = np.concatenate([[0], np.cumsum([len(ch.encode("utf-8")) for ch in text])])
    return [(int(cum[a]), int(cum[b])) for a, b in offsets]


def _covering(spans, content, byte):
    """Input index of the first content token ending after ``byte``."""
    for j in content:
        if spans[j][1] > byte:
            return j
    return None


def _token_nll(logits, ids, j):
    """-log p(ids[j] | ids[:j]) in float32."""
    return float(-logits[j - 1].float().log_softmax(dim=-1)[ids[j]])


@register_task("consistency_icl_score")
class ICLScoreTask(DiagnosticTask):
    """Olsson et al. (2022) in-context learning score."""

    TOKEN_POLICY = ("paper_protocol:Olsson2022 the 50th and 500th content token of each "
                    "context")
    SOURCES = {
        "paper": "arXiv:2209.11895 (Olsson et al. 2022, In-context Learning and Induction Heads)",
        "code": "",
        "llm_precedent": "arXiv:2209.11895 (34 transformer language models)",
    }
    ADAPTATION_NOTE = (
        "The score is Olsson et al.'s definition, NLL(500th token) - NLL(50th token) averaged "
        "over contexts. Their contexts are training-distribution sequences; BLME's are the "
        "starts of the first 2000 WikiText-103 train articles of at least 4096 bytes, so the "
        "same text reaches every model (class 2). Single-token losses are noisy, hence N = 2000 "
        "(gpt2: standard error 0.30 nats at 200 contexts; _meta_icl_score_se). "
        "Tokens are counted from the first content "
        "token, a BOS being context. icl_score_byte_anchored is a BLME twin (class 2): the "
        "same difference at the tokens covering bytes 200 and 2000, because the 500th token "
        "of one tokenizer is a different word of the text than another's. Both are "
        "functions of the model's output distribution, so identity blocks leave them "
        "unchanged; the losses are still in per-token units of each tokenizer."
    )
    HEADLINE_KEYS = ("icl_score", "icl_score_byte_anchored")
    _INV = "tests/tasks/invariance/test_consistency_icl_score.py::"
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS, (("insert_identity_blocks",), _INV + "test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Olsson ICL score...")
        num_samples = int(self.config.get("num_samples", 2000))
        max_length = int(self.config.get("max_length", 1024))
        max_length = min(max_length, get_max_positions(model, max_length))
        min_bytes = int(self.config.get("min_bytes", _MIN_BYTES))
        early, late = int(self.config.get("early_token", _EARLY)), int(self.config.get("late_token", _LATE))
        early_b = int(self.config.get("early_byte", _EARLY_BYTE))
        late_b = int(self.config.get("late_byte", _LATE_BYTE))
        if dataset is None or self.config.get("use_long_documents", True):
            from ...cache import load_long_documents
            pool = load_long_documents(4 * num_samples, split=self.config.get("split", "train"))
        else:
            pool = list(dataset)
        texts = [d["text"] if isinstance(d, dict) else str(d) for d in pool]
        texts = [t for t in texts if len(t.encode("utf-8")) >= min_bytes][:num_samples]
        if not texts:
            return {"error": f"No document of at least {min_bytes} bytes"}
        device = next(model.parameters()).device

        tok_diff, byte_diff, nll = [], [], {k: [] for k in ("early", "late", "early_b", "late_b")}
        n_short = 0
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, truncation=True, max_length=max_length,
                                return_offsets_mapping=True, return_special_tokens_mask=True)
                ids = enc["input_ids"]
                content = [j for j, m in enumerate(enc["special_tokens_mask"]) if not m]
                spans = _byte_spans(text, enc["offset_mapping"])
                pos = (content[early - 1] if len(content) >= late else None,
                       content[late - 1] if len(content) >= late else None,
                       _covering(spans, content, early_b),
                       _covering(spans, content, late_b))
                if any(p is None or p < 1 for p in pos):
                    n_short += 1
                    continue
                logits = model(input_ids=torch.tensor([ids], device=device),
                               use_cache=False).logits[0]
                for key, j in zip(nll, pos):
                    nll[key].append(_token_nll(logits, ids, j))
                tok_diff.append(nll["late"][-1] - nll["early"][-1])
                byte_diff.append(nll["late_b"][-1] - nll["early_b"][-1])
        if not tok_diff:
            return {"error": f"No context reaches token {late} and byte {late_b} "
                             f"within {max_length} tokens"}

        n = len(tok_diff)
        return {
            "icl_score": math.fsum(tok_diff) / n,
            "icl_score_byte_anchored": math.fsum(byte_diff) / n,
            "_meta_icl_score_se": float(np.std(tok_diff, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
            "_meta_icl_score_byte_anchored_se": (float(np.std(byte_diff, ddof=1) / math.sqrt(n))
                                                 if n > 1 else float("nan")),
            "_meta_nll_token_50": math.fsum(nll["early"]) / n,
            "_meta_nll_token_500": math.fsum(nll["late"]) / n,
            "_meta_nll_byte_200": math.fsum(nll["early_b"]) / n,
            "_meta_nll_byte_2000": math.fsum(nll["late_b"]) / n,
            "_meta_n_contexts": n,
            "_meta_n_contexts_too_short": n_short,
        }
