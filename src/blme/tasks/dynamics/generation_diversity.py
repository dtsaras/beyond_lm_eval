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
    n-gram overlap implementation (no external NLTK dependency).

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
        precisions.append(p if p > 0 else 1e-9)  # method1-style smoothing
    log_avg = np.mean([np.log(p) for p in precisions])
    return float(np.exp(log_avg))


@register_task("dynamics_generation_diversity")
class GenerationDiversityTask(DiagnosticTask):
    """Distinct-n, Self-BLEU, and entropy collapse over generated tokens."""

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
        all_entropy_first_q = []
        all_entropy_last_q = []
        all_entropy_delta = []
        all_token_rep_rate = []     # fraction of tokens seen earlier in same sequence
        all_phrase_rep_rate = []    # fraction of 4-grams that are repeated

        with torch.no_grad():
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
                        "pad_token_id": tokenizer.eos_token_id or tokenizer.pad_token_id or 0,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                    }
                    if seed is not None:
                        generator = torch.Generator(device=device)
                        generator.manual_seed(int(seed))
                        generate_kwargs["generator"] = generator
                    outputs = model.generate(**generate_kwargs)
                except Exception as e:
                    logger.info(f"  generate() failed: {e}")
                    continue

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

                # Self-BLEU: for each completion, BLEU against the rest
                for i, comp in enumerate(completions):
                    refs = [c for j, c in enumerate(completions) if j != i]
                    all_self_bleu.append(_self_bleu_single(comp, refs))

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
                    per_token_H = -(probs * log_probs).sum(dim=-1)  # (N, L)
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

        return {
            "mean_distinct_1": float(np.mean(all_distinct1)),
            "mean_distinct_2": float(np.mean(all_distinct2)),
            "mean_distinct_3": float(np.mean(all_distinct3)),
            "mean_self_bleu": float(np.mean(all_self_bleu)),
            "mean_token_repetition_rate": float(np.mean(all_token_rep_rate)) if all_token_rep_rate else float("nan"),
            "mean_phrase_repetition_rate": float(np.mean(all_phrase_rep_rate)) if all_phrase_rep_rate else float("nan"),
            "mean_entropy_first_quarter": float(np.mean(all_entropy_first_q)) if all_entropy_first_q else float("nan"),
            "mean_entropy_last_quarter": float(np.mean(all_entropy_last_q)) if all_entropy_last_q else float("nan"),
            "entropy_collapse_delta": float(np.mean(all_entropy_delta)) if all_entropy_delta else float("nan"),
            "n_prompts": len(prompts),
            "n_samples_per_prompt": n_samples,
            "temperature": temperature,
            "seed": int(seed) if seed is not None else None,
            "completion_filter": "trim_at_eos_drop_special",
            "distinct_n_scope": "per_completion_mean",
        }
