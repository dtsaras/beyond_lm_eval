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

    This is one ratio for the collection, NOT the mean of per-completion
    ratios. The two are very different when a model is degenerate: ten
    duplicate [1, 2] completions give 0.100000 here and 1.000000 as a
    per-completion mean, because each completion is internally diverse while
    the collection contains only two distinct types (adversarial review,
    2026-09-04). `mean_distinct_n` is retained under its own name as the
    per-completion statistic.
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
        # NLTK method1 smoothing: 0.1 / denominator for zero counts (audit
        # 2026-07-16: the old hard 1e-9 gave 0.004 where NLTK gives 0.33 —
        # reintroducing the hard-zero understatement this fallback exists
        # to avoid).
        denom = max(1, sum(hyp_ng.values()))
        if p <= 0 and n == 1:
            # NLTK/Texygen short-circuit to 0 when there is NO unigram match at
            # all -- method-1 smoothing is applied to the HIGHER-order zeros,
            # not to a hypothesis that shares no token with any reference.
            # Smoothing it anyway returned 0.0451801 for [7,8,9,10] against
            # [1,2,3,4] where the reference gives 0.0 (adversarial review round
            # 2, 2026-09-04).
            return 0.0
        precisions.append(p if p > 0 else 0.1 / denom)
    log_avg = np.mean([np.log(p) for p in precisions])
    # BREVITY PENALTY -- BLEU is not a bare precision geometric mean.
    # Without it a short hypothesis that is a prefix of a long reference
    # scores a perfect 1.0: [1,2,3,4] against [1,2,3,4,5,6,7,8] returned
    # 1.000000 where Texygen/NLTK give 0.367879, a factor of 2.718
    # (adversarial review, 2026-09-04). NLTK's convention: r is the closest
    # reference length, ties going to the shorter.
    c = len(hypothesis)
    r = min((len(ref) for ref in references),
            key=lambda rl: (abs(rl - c), rl))
    bp = 1.0 if c > r else (float(np.exp(1.0 - r / c)) if c > 0 else 0.0)
    return float(bp * np.exp(log_avg))


@register_task("dynamics_generation_diversity")
class GenerationDiversityTask(DiagnosticTask):
    """Distinct-n, Self-BLEU, and entropy collapse over generated tokens.

    PROTOCOL NOTE (audit 2026-07-16): Self-BLEU is computed over model
    SUBWORD TOKEN IDS; Texygen computes it over nltk.word_tokenize word
    n-grams. The kernel is parity-verified, but absolute Self-BLEU values
    are not comparable across models with different tokenizers — interpret
    within-model or across models sharing a tokenizer."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Generation Diversity Analysis...")

        n_samples = self.config.get("n_samples_per_prompt", 6)
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

        device = next(model.parameters()).device

        all_distinct1 = []
        all_distinct2 = []
        all_distinct3 = []
        all_self_bleu = []
        all_completions = []
        _n_ok_prompts = [0]
        _n_failed_prompts = [0]
        all_entropy_first_q = []
        all_entropy_last_q = []
        all_entropy_delta = []
        all_token_rep_rate = []     # fraction of tokens seen earlier in same sequence
        all_phrase_rep_rate = []    # fraction of 4-grams that are repeated

        with torch.no_grad():
            if seed is not None:
                # transformers' generate() has no per-call generator kwarg;
                # seed the global RNG once BEFORE the loop (audit 2026-07-16:
                # the old per-prompt reseed replayed the identical RNG stream
                # for every prompt, correlating completions across prompts).
                torch.manual_seed(int(seed))
            for prompt in prompts:
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                prompt_len = enc["input_ids"].shape[1]

                try:
                    generate_kwargs = {
                        "input_ids": enc["input_ids"].expand(n_samples, -1),
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": 1.0,
                        # PINNED. `generate()` falls back to the model's
                        # generation config, and transformers 5.2.0 silently
                        # supplies top_k=50 -- an unrecorded truncation of the
                        # sampling distribution that decides what "diversity"
                        # even means here, and that can change between library
                        # versions (adversarial review round 2, 2026-09-04).
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

                # Distinct-n per completion, averaged
                for comp in completions:
                    all_distinct1.append(_distinct_n(comp, 1))
                    all_distinct2.append(_distinct_n(comp, 2))
                    all_distinct3.append(_distinct_n(comp, 3))

                # Self-BLEU is deferred to the whole collection (below):
                # Texygen scores each generated sentence against every OTHER
                # sentence in the document, not only its five siblings from
                # the same prompt. Restricting to siblings understated the
                # current GPT-2 value at 0.046494 against 0.065155 under the
                # released aggregation (adversarial review, 2026-09-04).
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
                    # torch.special.entr(p) = -p*log(p) with entr(0) := 0.
                    #
                    # The naive -(probs * log_probs).sum() is NaN here, always.
                    # generate() returns PROCESSED scores, so the sampling
                    # logits-processors have already set masked positions to
                    # -inf: 9.6M of 9.65M entries on gpt2. exp(-inf) is 0 and
                    # 0 * -inf is NaN, which poisons every token's entropy.
                    # The result was not an empty list but a list OF NaNs, so
                    # the `if ... else nan` guard below could not catch it and
                    # all three entropy metrics were NaN for all 66 models.
                    # entr() takes the p -> 0 limit correctly (2026-08-26).
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
        for i, comp in enumerate(all_completions):
            refs = [c for j, c in enumerate(all_completions) if j != i]
            if refs:
                all_self_bleu.append(_self_bleu_single(comp, refs))

        return {
            # Li et al.'s own collection-level ratio (see _distinct_n_corpus).
            "distinct_1": _distinct_n_corpus(all_completions, 1),
            "distinct_2": _distinct_n_corpus(all_completions, 2),
            "distinct_3": _distinct_n_corpus(all_completions, 3),
            "mean_distinct_1": float(np.mean(all_distinct1)),
            "mean_distinct_2": float(np.mean(all_distinct2)),
            "mean_distinct_3": float(np.mean(all_distinct3)),
            "mean_self_bleu": float(np.mean(all_self_bleu)),
            # _meta_: `mean_distinct_1 + mean_token_repetition_rate == 1` to
            # 1.1e-16 on every model -- the repetition rate is 1 minus the
            # distinct-unigram rate BY DEFINITION. Emitting both puts one degree
            # of freedom in the feature matrix twice (adversarial review,
            # 2026-09-03).
            "_meta_mean_token_repetition_rate": float(np.mean(all_token_rep_rate)) if all_token_rep_rate else float("nan"),
            "mean_phrase_repetition_rate": float(np.mean(all_phrase_rep_rate)) if all_phrase_rep_rate else float("nan"),
            # NAMED for what they measure. These are the entropy of the
            # STALE-NUMBER CORRECTION (2026-09-04): every figure in the note
            # below was measured when `generate()` silently applied top_k=50.
            # `top_k` is now PINNED TO 0, so the support is the FULL vocabulary
            # -- measured 50257/50257 on gpt2, not 50 -- and the log(50) = 3.91
            # cap no longer exists. The entropy is still the SAMPLING
            # distribution's (temperature 0.8 is still applied), not the
            # model's own, so the naming point stands; the magnitudes do not.
            #
            # SAMPLING distribution actually used for generation (top-50,
            # T = 0.8), not the model's own next-token distribution: 99.90% of
            # the vocabulary is -inf and the support is exactly 50, so the value
            # is 2.21 nats where the model's own entropy is 4.37, the range is
            # compressed 2.7-3.7x, and it is capped at log(50) = 3.91
            # (adversarial review, 2026-09-03). Reading these as "the model's
            # entropy" would be wrong.
            "mean_sampling_entropy_first_quarter": float(np.mean(all_entropy_first_q)) if all_entropy_first_q else float("nan"),
            "mean_sampling_entropy_last_quarter": float(np.mean(all_entropy_last_q)) if all_entropy_last_q else float("nan"),
            "sampling_entropy_collapse_delta": float(np.mean(all_entropy_delta)) if all_entropy_delta else float("nan"),
            # ACTUAL, not requested. A prompt whose generate() raised was
            # skipped while the output still reported the full requested
            # counts, so a run with one failure of two prompts still claimed
            # 2 prompts x 2 samples (adversarial review round 2, 2026-09-04).
            "n_prompts": int(_n_ok_prompts[0]),
            "n_samples_per_prompt": n_samples,
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
