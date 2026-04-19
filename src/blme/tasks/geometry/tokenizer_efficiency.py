"""
Tokenizer efficiency metrics — Tier 1 (no model forward pass needed).

Different tokenizers produce different numbers of tokens for the same
text. Models with efficient tokenizers require fewer forward steps,
affecting both compute cost and effective context length. These metrics
capture the tokenizer's intrinsic properties:

  - **fertility**: average number of tokens per whitespace-delimited word.
    Lower = more efficient (each word maps to fewer tokens).
  - **compression_ratio**: tokens per UTF-8 character. Lower = better
    compression.
  - **token_entropy**: Shannon entropy of the token frequency distribution
    on the evaluation corpus. Higher = more uniform token usage; lower =
    skewed toward a few dominant tokens.
  - **vocab_utilization**: fraction of the vocabulary that appears at least
    once in the evaluation corpus.

References:
  * Rust, Pfeiffer, Vulić, Ruder, Gurevych 2021 — "How Good is Your
    Tokenizer? On the Monolingual Performance of Multilingual Language
    Models", ACL 2021, arXiv:2012.15613. Introduces fertility as a
    cross-tokenizer comparison metric.
  * Rajput, Chamberlain, Reese et al. 2024 — "Tokenizer Choice For LLM
    Training: Negligible or Crucial?", arXiv:2310.08754. Evidence that
    tokenizer efficiency is correlated with downstream capability.
  * fertility, compression_ratio, total_tokens, and vocab_size enter
    BLME's top-25 partial predictors (`docs/TOP_PREDICTORS.md` §2) as
    tokenizer-confound signals — all ~+0.71 partial ρ with composite
    benchmark capability beyond scale.
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
    """Tokenizer compression and efficiency metrics (Tier 1)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Tokenizer Efficiency Analysis...")

        num_samples = self.config.get("num_samples", len(_EFFICIENCY_CORPUS))

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
        # Normalise by log(vocab_size) so it's in [0, 1]
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
        max_entropy = np.log(max(2, vocab_size))
        normalised_token_entropy = float(token_entropy / max_entropy) if max_entropy > 0 else 0.0

        # Vocab utilization: fraction of full vocabulary used on this corpus
        vocab_utilization = len(token_counter) / max(1, vocab_size)

        return {
            "fertility": float(fertility),
            "compression_ratio": float(compression_ratio),
            "token_entropy": float(token_entropy),
            "normalised_token_entropy": normalised_token_entropy,
            "vocab_utilization": float(vocab_utilization),
            "vocab_size": int(vocab_size),
            "total_tokens": int(total_tokens),
            "total_words": int(total_words),
            "total_chars": int(total_chars),
        }
