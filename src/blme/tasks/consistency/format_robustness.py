"""
Prompt-format robustness — Sclar et al. 2023, arXiv:2310.11324.

For each (question, expected_answer) pair we render the prompt in N
different surface formats (different separators, casing, punctuation,
labels) and compare the model's behavior across formats. Sclar et al.
showed that the spread of accuracy across formats often exceeds the
gap between modern models on standard benchmarks — meaning format
sensitivity is a major confound.

We measure two complementary signals on a small bundled QA dataset:

  1. **NLL spread** of the expected answer continuation across formats.
     A robust model should give the same answer with similar log-prob
     regardless of format. Reported as `mean_nll_std_across_formats`.

  2. **Top-1 agreement rate**: fraction of items for which the argmax
     next-token-after-prompt is the same across all formats. Reported
     as `top1_agreement_rate`. Closer to 1.0 = more robust.

Both are pure intrinsic measurements (no external benchmark needed):
all questions and answers are bundled with the task.
"""

import logging
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# Bundled (question, expected answer) pairs. The answer is short, so the
# format-sensitivity signal stays focused on the prompt rendering rather
# than the generation length.
_QA_BUNDLE: List[Tuple[str, str]] = [
    ("What is the capital of France", "Paris"),
    ("What is the capital of Japan", "Tokyo"),
    ("Who painted the Mona Lisa", "Leonardo"),
    ("What is two plus two", "4"),
    ("What color is the sky on a clear day", "blue"),
    ("What is the largest planet in our solar system", "Jupiter"),
    ("Who wrote the play Hamlet", "Shakespeare"),
    ("What gas do plants absorb from the air", "carbon"),
    ("What is the chemical symbol for water", "H2O"),
    ("How many continents are there on Earth", "seven"),
    ("Who was the first president of the United States", "Washington"),
    ("What is the speed of light in a vacuum approximately", "300"),
]


# Each format is a callable that takes (question, answer) and returns
# (prompt_text, answer_text). The model is given prompt_text and we score
# answer_text as the continuation.
def _f1(q, a): return (f"Q: {q}?\nA:", " " + a)
def _f2(q, a): return (f"Question: {q}?\nAnswer:", " " + a)
def _f3(q, a): return (f"{q}?", " " + a)
def _f4(q, a): return (f"[Q] {q}?\n[A]", " " + a)
def _f5(q, a): return (f"User: {q}?\nAssistant:", " " + a)
def _f6(q, a): return (f"q: {q}?\na:", " " + a)
def _f7(q, a): return (f"Question - {q}?\nAnswer -", " " + a)
def _f8(q, a): return (f"### {q}?\n### Answer:", " " + a)


_FORMATS: List[Callable[[str, str], Tuple[str, str]]] = [
    _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8,
]


@register_task("consistency_format_robustness")
class FormatRobustnessTask(DiagnosticTask):
    """Prompt-format robustness diagnostic (Sclar et al. 2023)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Prompt-Format Robustness Analysis...")

        num_samples = self.config.get("num_samples", len(_QA_BUNDLE))

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and {"question", "answer"} <= set(dataset[0])
        ):
            qa = [(d["question"], d["answer"]) for d in dataset[:num_samples]]
        else:
            qa = _QA_BUNDLE[:num_samples]

        device = next(model.parameters()).device
        n_formats = len(_FORMATS)

        # nll_matrix[i_qa][i_format] = mean NLL over the answer tokens
        nll_matrix = np.full((len(qa), n_formats), np.nan, dtype=np.float64)
        # next_token_matrix[i_qa][i_format] = argmax token id at the position
        # immediately after the prompt
        next_token_matrix = np.full((len(qa), n_formats), -1, dtype=np.int64)

        with torch.no_grad():
            for qi, (q, a) in enumerate(qa):
                for fi, fmt in enumerate(_FORMATS):
                    prompt, ans = fmt(q, a)
                    full = prompt + ans

                    enc_full = tokenizer(full, return_tensors="pt").to(device)
                    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
                    full_ids = enc_full["input_ids"][0]
                    prompt_len = enc_prompt["input_ids"].shape[1]
                    if full_ids.shape[0] <= prompt_len:
                        continue

                    out = model(**enc_full)
                    logits = out.logits[0]

                    # Score the answer tokens
                    pred_logits = logits[prompt_len - 1: -1]
                    targets = full_ids[prompt_len:]
                    if pred_logits.shape[0] != targets.shape[0] or pred_logits.shape[0] == 0:
                        continue
                    losses = F.cross_entropy(pred_logits, targets, reduction="none")
                    nll_matrix[qi, fi] = float(losses.mean().item())

                    # Top-1 next token after the prompt (zero-context QA mode)
                    next_token_logits = logits[prompt_len - 1]
                    next_token_matrix[qi, fi] = int(next_token_logits.argmax().item())

        # Per-question stats across formats
        per_q_std = np.nanstd(nll_matrix, axis=1)
        per_q_mean = np.nanmean(nll_matrix, axis=1)
        # Coefficient of variation (per question), guarding against zero mean
        per_q_cv = np.where(per_q_mean > 1e-6, per_q_std / per_q_mean, 0.0)

        # Top-1 agreement: fraction of questions where all formats agree
        # on the most-likely next token (excluding -1 / failures).
        agreement_count = 0
        valid_q = 0
        for qi in range(len(qa)):
            row = next_token_matrix[qi]
            mask = row >= 0
            if mask.sum() < 2:
                continue
            valid_q += 1
            if len(set(row[mask].tolist())) == 1:
                agreement_count += 1
        top1_agreement_rate = (agreement_count / valid_q) if valid_q else float("nan")

        return {
            "n_questions": len(qa),
            "n_formats": n_formats,
            "mean_nll_std_across_formats": float(np.nanmean(per_q_std)),
            "mean_nll_cv_across_formats": float(np.nanmean(per_q_cv)),
            "max_nll_std_across_formats": float(np.nanmax(per_q_std)),
            "top1_agreement_rate": float(top1_agreement_rate),
            "mean_nll_overall": float(np.nanmean(nll_matrix)),
        }
