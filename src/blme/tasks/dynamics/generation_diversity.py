"""
Generation diversity metrics — Li et al. 2016 (Distinct-n); Zhu et al. 2018
(Self-BLEU).

For each prompt, generate N completions (with temperature > 0) and measure:

  - **Distinct-n** (n=1,2,3): fraction of *unique* n-grams in the
    generated text, averaged over completions. Higher = more diverse
    vocabulary usage. Li et al. 2016 ("A Diversity-Promoting Objective
    Function for Neural Conversation Models").

  - **Self-BLEU**: for each completion, compute BLEU against all *other*
    completions from the same prompt, then average. Lower = more diverse
    (each completion is unlike the others). Zhu et al. 2018 ("Texygen: A
    Benchmarking Platform for Text Generation Models"). Uses a simple
    n-gram overlap implementation (used only when NLTK is unavailable;
    the primary path uses NLTK sentence_bleu, a declared dependency).

  - **Entropy collapse**: Shannon entropy of the softmax distribution at
    each generated token position, averaged. If entropy drops sharply
    over positions, the model is "degenerating" (becoming too confident
    and repetitive as generation progresses). Report the entropy at the
    first and last quarter of generated tokens, and the delta.
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


_DIVERSITY_PROMPTS: List[str] = [
    "Once upon a time in a quiet village,",
    "The scientist carefully examined the results and concluded that",
    "In the year 2050, technology had advanced to the point where",
    "The most important thing about learning is",
    "Climate change will likely cause",
    "The history of mathematics begins with",
    "When asked about the meaning of life, she said",
    "A good software engineer should always",
]


def _distinct_n(tokens: List[int], n: int) -> float:
    """Distinct-n (Li et al. 2016): distinct n-grams / total generated tokens.

    Li et al. 2016 (Sec. 5.2) and the ACL-2022 refinement define Distinct-n as
    the number of distinct n-grams divided by the TOTAL number of generated
    tokens (len(tokens)), not by the n-gram count (len-n+1) — the token-count
    denominator is what "avoids favoring long sentences". Identical for n=1.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(tokens)


def _distinct_n_corpus(completions: List[List[int]], n: int) -> float:
    """Collection-level Distinct-n, exactly as Li et al. 2016 define it:

        |union of n-gram TYPES over the whole response collection|
        --------------------------------------------------------
                 total tokens generated in the collection

    One ratio for the collection, not the mean of per-completion ratios (ten
    duplicate [1, 2] completions give 0.1 here and 1.0 as a per-completion
    mean); `mean_distinct_n` is the per-completion statistic.
    """
    types = set()
    total_tokens = 0
    for comp in completions:
        total_tokens += len(comp)
        for i in range(len(comp) - n + 1):
            types.add(tuple(comp[i:i + n]))
    if total_tokens == 0:
        return float("nan")
    return len(types) / total_tokens


def _word_tokenize(text: str):
    """Texygen's unit: ``nltk.word_tokenize`` (SelfBleu.py:38, :52).

    Returns ``(words, tokenizer_name)``. Without NLTK's punkt_tab data the
    Treebank tokenizer runs on the whole string (``preserve_line=True``), which
    differs only in not splitting sentence-internal periods; without NLTK a
    regex word/punctuation split is used. The name actually used is reported.
    """
    try:
        import nltk
    except ImportError:
        return re.findall(r"\w+|[^\w\s]", text), "regex_word_punct"
    try:
        return nltk.word_tokenize(text), "nltk_word_tokenize"
    except LookupError:
        return nltk.word_tokenize(text, preserve_line=True), "nltk_treebank_no_punkt"


def _trim_completion(tokens: List[int], eos_token_id: Optional[int],
                     pad_token_id: Optional[int]) -> List[int]:
    """Drop EOS/pad tokens and anything generated after EOS."""
    trimmed = []
    for token in tokens:
        if eos_token_id is not None and token == eos_token_id:
            break
        if pad_token_id is not None and token == pad_token_id:
            continue
        trimmed.append(token)
    return trimmed


def _ngram_overlap(ref_ngrams: Counter, hyp_ngrams: Counter, n: int) -> float:
    """Precision of n-gram overlap (clipped counts)."""
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    total = sum(hyp_ngrams.values())
    return overlap / max(1, total)


def _self_bleu_single(hypothesis: List[int], references: List[List[int]],
                      max_n: int = 4) -> float:
    """Self-BLEU of one hypothesis against all other completions, matching
    the Texygen reference (Zhu et al. 2018): NLTK ``sentence_bleu`` with
    ``SmoothingFunction().method1`` and uniform weights ``(1/max_n,)*max_n``,
    including NLTK's brevity penalty.

    Texygen's SelfBleu calls exactly this; the previous hand-rolled version
    returned a hard 0.0 whenever any higher-order precision was 0 (common
    for short/diverse completions), systematically understating Self-BLEU.
    Falls back to the smoothed hand-rolled estimator only if NLTK is absent.
    """
    if not references or len(hypothesis) < 1:
        return 0.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return _self_bleu_single_fallback(hypothesis, references, max_n)
    weights = tuple(1.0 / max_n for _ in range(max_n))
    return float(sentence_bleu(
        references, hypothesis, weights,
        smoothing_function=SmoothingFunction().method1,
    ))


# Texygen reports Self-BLEU at n-gram orders 2-5 (docs/evaluation.md, the
# paper's result tables); its SelfBleu class defaults to 3. BLME emits all four
# and keeps order 4 as mean_self_bleu (also Holtzman et al. 2020's Self-BLEU4).
SELF_BLEU_ORDERS = (2, 3, 4, 5)
SELF_BLEU_HEADLINE_ORDER = 4


def _self_bleu_orders(hypothesis: List, references: List[List],
                      orders=SELF_BLEU_ORDERS) -> Dict[int, float]:
    """Self-BLEU of one hypothesis at several orders in one NLTK pass: the
    modified precisions p_1..p_max are shared, so this equals one
    ``sentence_bleu`` call per order (uniform weights, method1)."""
    if not references or len(hypothesis) < 1:
        return {n: 0.0 for n in orders}
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return {n: _self_bleu_single_fallback(hypothesis, references, n) for n in orders}
    weights = [tuple(1.0 / n for _ in range(n)) for n in orders]
    vals = sentence_bleu(references, hypothesis, weights,
                         smoothing_function=SmoothingFunction().method1)
    if not isinstance(vals, list):
        vals = [vals] * len(orders)
    return {n: float(v) for n, v in zip(orders, vals)}


def _self_bleu_single_fallback(hypothesis: List[int], references: List[List[int]],
                               max_n: int = 4) -> float:
    """NLTK-free fallback for Self-BLEU (used only when nltk is unavailable).

    Geometric mean of clipped n-gram precisions with add-epsilon (method1-
    style) smoothing on zero precisions, so it no longer hard-zeros — a close
    approximation to the Texygen/NLTK value but not byte-identical.
    """
    if not references or len(hypothesis) < 1:
        return 0.0
    ref_pool = [Counter() for _ in range(max_n)]
    for ref in references:
        for n in range(1, max_n + 1):
            ng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            for k, v in ng.items():
                ref_pool[n - 1][k] = max(ref_pool[n - 1][k], v)

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))
        p = _ngram_overlap(ref_pool[n - 1], hyp_ng, n)
        # NLTK method1 smoothing: 0.1 / denominator for zero counts.
        denom = max(1, sum(hyp_ng.values()))
        if p <= 0 and n == 1:
            # NLTK short-circuits to 0 when there is no unigram match at all.
            return 0.0
        precisions.append(p if p > 0 else 0.1 / denom)
    log_avg = np.mean([np.log(p) for p in precisions])
    # Brevity penalty, NLTK convention: r is the closest reference length,
    # ties going to the shorter.
    c = len(hypothesis)
    r = min((len(ref) for ref in references),
            key=lambda rl: (abs(rl - c), rl))
    bp = 1.0 if c > r else (float(np.exp(1.0 - r / c)) if c > 0 else 0.0)
    return float(bp * np.exp(log_avg))


@register_task("dynamics_generation_diversity")
class GenerationDiversityTask(DiagnosticTask):
    """Distinct-n, Self-BLEU, and entropy collapse over generated tokens.

    Self-BLEU and distinct-n are computed over WORDS (each completion decoded
    and split by nltk.word_tokenize, as Texygen does), so the tokenizer does not
    enter them; the subword-id versions are _meta_."""

    # Statistics of generated continuations; prompt tokens never enter.
    TOKEN_POLICY = "not_token_based"
    SOURCES = {
        "paper": "arXiv:1510.03055; arXiv:1802.01886",
        "code": "geek-ai/Texygen@3104e22ac75f3cc2070da2bf5e2da6d2bef149ad",
        "llm_precedent": "Holtzman et al. 2020 (arXiv:1904.09751): Self-BLEU4 of GPT-2 "
                         "generations",
    }
    HEADLINE_KEYS = (
        "distinct_1", "distinct_2", "distinct_3", "mean_distinct_1", "mean_distinct_2",
        "mean_distinct_3", "mean_self_bleu", "mean_phrase_repetition_rate",
        "mean_sampling_entropy_first_quarter", "mean_sampling_entropy_last_quarter",
        "sampling_entropy_collapse_delta",
    )
    # Word units make the n-gram statistics independent of the tokenizer (its
    # vocabulary) for a fixed text; the collection-size drift of corpus
    # distinct-n is handled by pinning N = 500 for every model, not normalised.
    # The phrase-repetition and sampling-entropy keys have no direct test.
    SIZE_INVARIANT = dict.fromkeys(
        ("mean_self_bleu", "distinct_1", "distinct_2", "distinct_3",
         "mean_distinct_1", "mean_distinct_2", "mean_distinct_3"),
        (("tokenizer_swap(gpt2, pythia-160m)",),
         "tests/tasks/invariance/test_dynamics_generation_diversity.py::"
         "test_word_unit_metrics_do_not_depend_on_the_tokenizer"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        from ...utils import isolated_torch_rng
        with isolated_torch_rng(model):
            return self._evaluate(model, tokenizer, dataset, cache)

    def _evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Generation Diversity Analysis...")

        # Collection size: Texygen scores 500 sentences (SelfBleu sample_size), so
        # n_completions (default 500) is split as evenly as possible over the
        # prompts. An explicit n_samples_per_prompt overrides the split.
        n_completions = int(self.config.get("n_completions", 500))
        n_per_prompt = self.config.get("n_samples_per_prompt", None)
        self_bleu_sample_size = int(self.config.get("self_bleu_sample_size", 500))
        temperature = self.config.get("temperature", 0.8)
        max_new_tokens = self.config.get("max_new_tokens", 32)
        num_prompts = self.config.get("num_prompts", len(_DIVERSITY_PROMPTS))
        seed = self.config.get("seed", None)

        if dataset is not None and isinstance(dataset, list) and dataset:
            if isinstance(dataset[0], dict) and "prompt" in dataset[0]:
                prompts = [d["prompt"] for d in dataset[:num_prompts]]
            elif isinstance(dataset[0], str):
                prompts = list(dataset)[:num_prompts]
            else:
                prompts = _DIVERSITY_PROMPTS[:num_prompts]
        else:
            prompts = _DIVERSITY_PROMPTS[:num_prompts]

        if n_per_prompt is not None:
            counts = [int(n_per_prompt)] * len(prompts)
        else:
            q, r = divmod(n_completions, max(1, len(prompts)))
            counts = [q + (i < r) for i in range(len(prompts))]

        device = next(model.parameters()).device

        all_distinct1 = []
        all_distinct2 = []
        all_distinct3 = []
        all_completions = []
        all_words = []
        units_used = set()
        sub_distinct = {1: [], 2: [], 3: []}
        _n_ok_prompts = [0]
        _n_failed_prompts = [0]
        all_entropy_first_q = []
        all_entropy_last_q = []
        all_entropy_delta = []
        all_token_rep_rate = []     # fraction of tokens seen earlier in same sequence
        all_phrase_rep_rate = []    # fraction of 4-grams that are repeated

        with torch.no_grad():
            if seed is not None:
                # generate() takes no generator: seed the global RNG once,
                # before the loop (a per-prompt reseed correlates prompts).
                torch.manual_seed(int(seed))
            for prompt, n_samples in zip(prompts, counts):
                if n_samples < 1:
                    continue
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                prompt_len = enc["input_ids"].shape[1]

                try:
                    generate_kwargs = {
                        "input_ids": enc["input_ids"].expand(n_samples, -1),
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": 1.0,
                        # Pinned: generate() would otherwise take top_k=50 from
                        # the model's generation config.
                        "top_k": int(self.config.get("top_k", 0)) or 0,
                        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                    }
                    outputs = model.generate(**generate_kwargs)
                except Exception as e:
                    logger.info(f"  generate() failed: {e}")
                    _n_failed_prompts[0] += 1
                    continue
                _n_ok_prompts[0] += 1

                gen_ids = outputs.sequences[:, prompt_len:]  # (N, L)
                completions = [
                    _trim_completion(
                        row.tolist(),
                        getattr(tokenizer, "eos_token_id", None),
                        getattr(tokenizer, "pad_token_id", None),
                    )
                    for row in gen_ids
                ]

                # Word units (Texygen): decode, then nltk.word_tokenize. A
                # tokenizer without decode() leaves the ids as the units.
                _dec = getattr(tokenizer, "decode", None)
                words = []
                for comp in completions:
                    if callable(_dec):
                        w, unit_name = _word_tokenize(_dec(comp, skip_special_tokens=True))
                    else:
                        w, unit_name = list(comp), "token_ids"
                    words.append(w)
                    units_used.add(unit_name)
                all_words.extend(words)

                # Distinct-n per completion, averaged
                for comp in words:
                    all_distinct1.append(_distinct_n(comp, 1))
                    all_distinct2.append(_distinct_n(comp, 2))
                    all_distinct3.append(_distinct_n(comp, 3))
                for comp in completions:
                    sub_distinct[1].append(_distinct_n(comp, 1))
                    sub_distinct[2].append(_distinct_n(comp, 2))
                    sub_distinct[3].append(_distinct_n(comp, 3))

                # Self-BLEU is computed over the whole collection below, as
                # Texygen scores each sentence against every other one.
                all_completions.extend(completions)

                # Sequence-level repetition: per completion, fraction of
                # tokens that appeared earlier in the same sequence, and
                # fraction of 4-grams that are duplicates of earlier 4-grams.
                for comp in completions:
                    if len(comp) >= 2:
                        seen = set()
                        rep_count = 0
                        for t in comp:
                            if t in seen:
                                rep_count += 1
                            seen.add(t)
                        all_token_rep_rate.append(rep_count / len(comp))
                    if len(comp) >= 4:
                        seen_4g = set()
                        rep_4g = 0
                        total_4g = 0
                        for j in range(len(comp) - 3):
                            ng = tuple(comp[j:j + 4])
                            total_4g += 1
                            if ng in seen_4g:
                                rep_4g += 1
                            seen_4g.add(ng)
                        if total_4g > 0:
                            all_phrase_rep_rate.append(rep_4g / total_4g)

                # Entropy collapse from the per-step scores. Mask positions
                # after each sequence's EOS/pad so uneven completion lengths
                # do not bias the quarter summaries.
                if hasattr(outputs, "scores") and outputs.scores:
                    scores_stack = torch.stack(outputs.scores, dim=1).float()  # (N, L, V)
                    log_probs = F.log_softmax(scores_stack, dim=-1)
                    probs = log_probs.exp()
                    # entr(p) = -p log p with entr(0) = 0: processed scores hold
                    # -inf entries, where -(p * log p) would be NaN.
                    per_token_H = torch.special.entr(probs).sum(dim=-1)  # (N, L)
                    for row_idx, comp in enumerate(completions):
                        valid_len = len(comp)
                        if valid_len < 4:
                            continue
                        h_row = per_token_H[row_idx, :valid_len].cpu().numpy()
                        q1 = float(h_row[:valid_len // 4].mean())
                        q4 = float(h_row[3 * valid_len // 4:].mean())
                        all_entropy_first_q.append(q1)
                        all_entropy_last_q.append(q4)
                        all_entropy_delta.append(q4 - q1)

        if not all_distinct1:
            return {"error": "No completions generated"}

        # Texygen aggregation: every completion against every other one.
        def _self_bleu_means(units, orders=SELF_BLEU_ORDERS):
            vals = {n: [] for n in orders}
            for i, comp in enumerate(units):
                refs = [c for j, c in enumerate(units) if j != i]
                if refs:
                    for n, v in _self_bleu_orders(comp, refs, orders).items():
                        vals[n].append(v)
            return {n: float(np.mean(v)) if v else float("nan") for n, v in vals.items()}
        # Texygen's get_bleu_fast scores the first sample_size (500) sentences.
        self_bleu = _self_bleu_means(all_words[:self_bleu_sample_size])
        mean_self_bleu = self_bleu[SELF_BLEU_HEADLINE_ORDER]

        return {
            # Li et al.'s own collection-level ratio (see _distinct_n_corpus),
            # over words.
            "distinct_1": _distinct_n_corpus(all_words, 1),
            "distinct_2": _distinct_n_corpus(all_words, 2),
            "distinct_3": _distinct_n_corpus(all_words, 3),
            "mean_distinct_1": float(np.mean(all_distinct1)),
            "mean_distinct_2": float(np.mean(all_distinct2)),
            "mean_distinct_3": float(np.mean(all_distinct3)),
            "mean_self_bleu": mean_self_bleu,
            # The unit actually used, and the subword-id forms (tokenizer-
            # dependent: the same 48 strings span Self-BLEU 0.084-0.161).
            "_meta_diversity_unit": "+".join(sorted(units_used)),
            "_meta_distinct_1_subword": _distinct_n_corpus(all_completions, 1),
            "_meta_distinct_2_subword": _distinct_n_corpus(all_completions, 2),
            "_meta_distinct_3_subword": _distinct_n_corpus(all_completions, 3),
            "_meta_mean_distinct_1_subword": float(np.mean(sub_distinct[1])),
            "_meta_mean_distinct_2_subword": float(np.mean(sub_distinct[2])),
            "_meta_mean_distinct_3_subword": float(np.mean(sub_distinct[3])),
            "_meta_mean_self_bleu_subword": _self_bleu_means(
                all_completions[:self_bleu_sample_size],
                (SELF_BLEU_HEADLINE_ORDER,))[SELF_BLEU_HEADLINE_ORDER],
            "_meta_self_bleu_n_scored": min(len(all_words), self_bleu_sample_size),
            "_meta_self_bleu_gram": SELF_BLEU_HEADLINE_ORDER,
            **{f"_meta_self_bleu_{n}": v for n, v in self_bleu.items()
               if n != SELF_BLEU_HEADLINE_ORDER},
            # _meta_: equals 1 - _meta_mean_distinct_1_subword by definition.
            "_meta_mean_token_repetition_rate": float(np.mean(all_token_rep_rate)) if all_token_rep_rate else float("nan"),
            "mean_phrase_repetition_rate": float(np.mean(all_phrase_rep_rate)) if all_phrase_rep_rate else float("nan"),
            # Entropy of the SAMPLING distribution (temperature applied, full
            # vocabulary since top_k is pinned to 0), not the model's own.
            "mean_sampling_entropy_first_quarter": float(np.mean(all_entropy_first_q)) if all_entropy_first_q else float("nan"),
            "mean_sampling_entropy_last_quarter": float(np.mean(all_entropy_last_q)) if all_entropy_last_q else float("nan"),
            "sampling_entropy_collapse_delta": float(np.mean(all_entropy_delta)) if all_entropy_delta else float("nan"),
            # Counts actually generated, not requested.
            "n_prompts": int(_n_ok_prompts[0]),
            "n_samples_per_prompt": (int(n_per_prompt) if n_per_prompt is not None
                                     else max(counts) if counts else 0),
            "_meta_n_prompts_requested": len(prompts),
            "_meta_n_prompts_failed": int(_n_failed_prompts[0]),
            "_meta_n_completions": len(all_completions),
            "_meta_top_k": int(self.config.get("top_k", 0)) or 0,
            "temperature": temperature,
            "seed": int(seed) if seed is not None else None,
            "completion_filter": "trim_at_eos_drop_special",
            "distinct_n_scope": "corpus (distinct_n) + per_completion_mean (mean_distinct_n)",
            "self_bleu_scope": "all_pairs_over_collection",
        }
