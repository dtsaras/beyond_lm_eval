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
    Models", ACL 2021, arXiv:2012.15613. Popularizes fertility (introduced by Ács 2019) as a
    cross-tokenizer comparison metric.
  * Ali, Fromm, Thellmann et al. 2024 (NAACL Findings 2024) — "Tokenizer Choice For LLM
    Training: Negligible or Crucial?", arXiv:2310.08754. NOTE: the paper's own
    finding is that tokenizer choice matters but fertility/parity "are not
    always predictive of the model's downstream performance" — cited here
    as the study that MEASURES these statistics at LLM scale, not as
    evidence they predict capability (mischaracterization fixed, audit
    2026-07-16).
  * NOTE (2026-09-05): an independent review measured that **every one of the
    ten keys this task emits is already excluded** by the analysis pipeline's
    `is_excluded_predictor` — `TOKENIZER_PROXY_TERMS` catches them by name. So
    the claim below describes a state that no longer obtains: none of these
    columns reaches the feature matrix, and `docs/TOP_PREDICTORS.md` §2 needs
    the same correction. The task remains a useful DIAGNOSTIC of tokenizer
    confounding; it is not a source of structural predictors.

  * (historical) fertility and vocab_size enter BLME's top-25 partial predictors
    (`docs/TOP_PREDICTORS.md` §2) as tokenizer-confound signals — +0.71 to
    +0.77 partial ρ with composite benchmark capability beyond scale.
    `compression_ratio` and `total_tokens` were also listed there; they are
    `fertility` rescaled by corpus constants (pearson 1.000000000000 over 9
    tokenizers, since the corpus is fixed), so they are now `_meta_` and the
    top-25 entry should count ONE tokenizer-efficiency degree of freedom, not
    three (red-team review, 2026-09-05).
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

        # Accept any iterable dataset (audit 2026-07-16: the isinstance-
        # list gate silently ignored HF Datasets/generators and fell back
        # to the internal corpus).
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
        # `len(tokenizer)`, NOT `tokenizer.vocab_size`.
        #
        # `vocab_size` is the BASE vocabulary and excludes added/special
        # tokens, while the ids actually produced can exceed it -- so both
        # fractions below could leave [0, 1] and the emitted `vocab_size` would
        # understate the real symbol set. `len(tokenizer)` is the usable id
        # count (adversarial review, 2026-09-04).
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(getattr(tokenizer, "vocab_size", 2) or 2)

        # Normalise by log(min(T, V)), not log(V).
        #
        # An empirical entropy over T observed tokens cannot exceed log(T), so
        # dividing by log(V) when T < V makes the ceiling unreachable and turns
        # the "normalised" value into a sample-size readout: at T = 10k and
        # V = 50k the maximum attainable is log(10k)/log(50k) = 0.851, and the
        # gap differs by model because V does.
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
            # ONE degree of freedom, not four. The corpus is FIXED across
            # models, so `total_words` and `total_chars` are constants and
            # `fertility = total_tokens/total_words`,
            # `compression_ratio = total_tokens/total_chars` are `total_tokens`
            # divided by those constants. Measured over 9 tokenizers at the
            # shipped config (red-team review, 2026-09-05):
            #
            #   total_words  = 1400  for every model, std EXACTLY 0
            #   total_chars  = 7192  for every model, std EXACTLY 0
            #   pearson(fertility, compression_ratio)   = 1.000000000000
            #   pearson(fertility, total_tokens)        = 1.000000000000
            #   pearson(normalised_token_entropy,
            #           token_entropy)                  = 0.9901
            #
            # The module docstring above claimed fertility, compression_ratio,
            # total_tokens AND vocab_size were four separate top-25 partial
            # predictors at +0.71 to +0.77. Three of those four are the same
            # number rescaled, and a sparse selection splits one coefficient
            # across them. `fertility` is kept as THE tokenizer-efficiency
            # feature -- it is the one the cited literature names -- and the
            # other two are demoted to provenance.
            "fertility": float(fertility),
            "_meta_compression_ratio": float(compression_ratio),
            "token_entropy": float(token_entropy),
            "_meta_normalised_token_entropy": normalised_token_entropy,
            # `_meta_`: `len(distinct tokens seen) / vocab_size` on a FIXED
            # corpus, so the numerator is near-constant and the column is
            # essentially 1/vocab_size -- pearson **+0.9955** with it over
            # 13 tokenizers (independent Opus review, 2026-09-05).
            # `vocab_size` is already blocked; this was the same variable
            # inverted.
            "_meta_vocab_utilization": float(vocab_utilization),
            "vocab_size": int(vocab_size),
            "_meta_entropy_ceiling_n": int(_ceiling_n),
            "_meta_total_tokens": int(total_tokens),
            # Constants of the corpus, not properties of the tokenizer.
            "_meta_total_words": int(total_words),
            "_meta_total_chars": int(total_chars),
        }
