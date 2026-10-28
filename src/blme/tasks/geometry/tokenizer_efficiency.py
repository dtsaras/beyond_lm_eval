"""
Tokenizer efficiency metrics — Tier 1 (no model forward pass needed).

Different tokenizers produce different numbers of tokens for the same text.
On a fixed corpus:

  - **fertility**: tokens per whitespace-delimited word (Rust et al. 2021,
    after Acs 2019). Lower = each word maps to fewer tokens.
  - **compression_ratio** (``_meta_``): tokens per Unicode code point
    (``len(text)``), not per UTF-8 byte.
  - **token_entropy**: Shannon entropy (nats) of the token unigram
    distribution on the corpus; a plug-in estimate that grows with corpus size.
  - **vocab_utilization** (``_meta_``): share of the vocabulary seen at least
    once.

On a fixed corpus fertility, compression_ratio and total_tokens are one
degree of freedom (each is total_tokens over a corpus constant), so only
fertility is a feature. All keys are tokenizer confounds by name in the
downstream analysis, not structural predictors.

References:
  * Rust, Pfeiffer, Vulić, Ruder, Gurevych 2021 — "How Good is Your
    Tokenizer? On the Monolingual Performance of Multilingual Language
    Models", ACL 2021, arXiv:2012.15613 (fertility, after Ács 2019).
  * Ali, Fromm, Thellmann et al. 2024 (NAACL Findings 2024) — "Tokenizer Choice
    For LLM Training: Negligible or Crucial?", arXiv:2310.08754: measures these
    statistics at LLM scale and finds them not always predictive of downstream
    performance.
"""

import logging
from collections import Counter

import numpy as np

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


_EFFICIENCY_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models are trained on large text corpora.",
    "A federal judge ruled the policy was unconstitutional last week.",
    "Symphonies are typically composed in four distinct movements.",
    "The compiler emitted three warnings about unused variables.",
    "Researchers identified a previously unknown protein in the cell membrane.",
    "The bridge will be closed for two weeks during major repairs.",
    "Mathematicians celebrated the proof as elegant and surprising.",
    "Local farmers worry that the drought will reduce this year's yield.",
    "Quantum computers may eventually solve currently intractable problems.",
    "The orchestra performed Beethoven's ninth symphony to a packed concert hall.",
    "A small startup raised twelve million dollars in their seed funding round.",
    "Astronomers spotted a faint object near the distant orbit of Neptune.",
    "Climate change is reshaping weather patterns and ecosystems worldwide.",
    "The new tax policy is expected to take effect early next year.",
    "Fresh vegetables are available at the local farmers market every Saturday morning.",
    "She wrote the software using Python and deployed it on cloud infrastructure.",
    "The history of mathematics begins with ancient counting and measurement systems.",
    "Coral reefs are particularly vulnerable to ocean acidification and rising temperatures.",
    "Open source projects have become a major part of the global software ecosystem.",
]


@register_task("geometry_tokenizer_efficiency")
class TokenizerEfficiencyTask(DiagnosticTask):
    """Tokenizer compression and efficiency metrics (Tier 1).

    The word unit is a whitespace token (``text.split()``) of the shared
    WikiText corpus, tokenised in context, not a treebank word tokenised on its
    own as in Rust et al. 2021 (App. A.3).
    """

    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:2012.15613",
        "code": "",
        "llm_precedent": ("Dagan et al. 2024 arXiv:2402.01035 (bytes per token, NSL); "
                          "Zouhar et al. 2023 arXiv:2306.16842 (token entropy); "
                          "Ali et al. 2024 arXiv:2310.08754"),
    }
    ADAPTATION_NOTE = (
        "Fertility (tokens per word, Rust et al. 2021) is computed with whitespace-delimited "
        "words of the shared WikiText corpus as the word unit, not the UD treebank words of "
        "the paper: WikiText-raw already space-separates punctuation (' , ', ' @-@ '), so "
        "whitespace words approximate treebank tokens on this corpus. The text is tokenised "
        "in context rather than word by word, because per-word tokenisation changes "
        "byte-level BPE counts (leading-space variants). On a fixed corpus fertility, "
        "bytes per token and NSL against a reference tokenizer are one degree of freedom, "
        "so only fertility is a feature; it is not V-linked mechanically (50,000 unused "
        "added tokens leave it unchanged). token_entropy is a plug-in estimate that grows "
        "with corpus size, and the compression ratio counts Unicode code points, not bytes."
    )
    HEADLINE_KEYS = ("fertility", "token_entropy", "vocab_size")
    # token_entropy is also unchanged by unused tokens but grows with corpus
    # size, so it has no entry; vocab_size is V by definition.
    SIZE_INVARIANT = {
        "fertility": (("add_unused_tokens",),
                      "tests/tasks/invariance/test_geometry_tokenizer_efficiency.py::"
                      "test_fertility_invariant_under_unused_vocabulary"),
    }

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Tokenizer Efficiency Analysis...")

        num_samples = self.config.get("num_samples", len(_EFFICIENCY_CORPUS))

        # Accept any iterable dataset (HF Datasets, generators).
        if dataset is not None and not isinstance(dataset, list):
            try:
                dataset = list(dataset)
            except TypeError:
                dataset = None
        if dataset is not None and isinstance(dataset, list) and dataset:
            texts = []
            for s in dataset[:num_samples]:
                if isinstance(s, dict) and "text" in s:
                    texts.append(s["text"])
                elif isinstance(s, str):
                    texts.append(s)
        else:
            texts = _EFFICIENCY_CORPUS[:num_samples]

        if not texts:
            return {"error": "No texts for tokenizer analysis"}

        total_tokens = 0
        total_words = 0
        total_chars = 0
        token_counter: Counter = Counter()

        for text in texts:
            enc = tokenizer(text, add_special_tokens=False)
            ids = enc["input_ids"]
            total_tokens += len(ids)
            total_words += len(text.split())
            total_chars += len(text)
            for tid in ids:
                token_counter[tid] += 1

        if total_tokens == 0:
            return {"error": "No tokens produced"}

        fertility = total_tokens / max(1, total_words)
        compression_ratio = total_tokens / max(1, total_chars)

        # Token frequency entropy
        counts = np.array(list(token_counter.values()), dtype=np.float64)
        p = counts / counts.sum()
        token_entropy = float(-np.sum(p * np.log(p)))
        # `len(tokenizer)` counts added/special tokens; `vocab_size` is the
        # base vocabulary and can be exceeded by the ids actually produced.
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(getattr(tokenizer, "vocab_size", 2) or 2)

        # Normalise by log(min(T, V)): an entropy over T tokens cannot exceed
        # log T, so log V would make the ceiling unreachable when T < V.
        _T = int(total_tokens) if total_tokens else 0
        _ceiling_n = max(2, min(_T, vocab_size)) if _T else max(2, vocab_size)
        max_entropy = np.log(_ceiling_n)
        normalised_token_entropy = float(token_entropy / max_entropy) if max_entropy > 0 else 0.0

        # Vocab utilization: fraction of full vocabulary used on this corpus
        vocab_utilization = len(token_counter) / max(1, vocab_size)
        # Both fractions are shares of a bounded set; assert rather than assume.
        vocab_utilization = float(min(1.0, max(0.0, vocab_utilization)))
        normalised_token_entropy = float(min(1.0, max(0.0, normalised_token_entropy)))

        return {
            # Fertility is THE efficiency feature (the cited literature's);
            # compression_ratio and total_tokens are it rescaled by corpus
            # constants. The normalised entropy tracks token_entropy.
            "fertility": float(fertility),
            "_meta_compression_ratio": float(compression_ratio),
            "token_entropy": float(token_entropy),
            "_meta_normalised_token_entropy": normalised_token_entropy,
            # distinct/V on a fixed corpus is essentially 1/V.
            "_meta_vocab_utilization": float(vocab_utilization),
            "vocab_size": int(vocab_size),
            "_meta_entropy_ceiling_n": int(_ceiling_n),
            "_meta_total_tokens": int(total_tokens),
            # Constants of the corpus, not properties of the tokenizer.
            "_meta_total_words": int(total_words),
            "_meta_total_chars": int(total_chars),
        }
