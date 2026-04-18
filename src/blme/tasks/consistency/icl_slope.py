"""
In-context learning slope — measures how much the model improves at
predicting a target fact when given increasing numbers of in-context
demonstrations (0-shot, 1-shot, 2-shot, 4-shot).

The slope of NLL vs. number of demonstrations quantifies the model's
ability to learn from context. Models with steeper slopes are better
few-shot learners — a strong predictor of downstream benchmark
performance (Brown et al. 2020; Min et al. 2022).

Reported metrics:
  - **mean_nll_per_k**: NLL at each shot count {0, 1, 2, 4}.
  - **icl_slope**: linear regression slope of NLL vs. k. More negative =
    stronger ICL ability.
  - **icl_gain**: NLL(0-shot) - NLL(4-shot). Positive = improvement.
  - **icl_relative_gain**: icl_gain / NLL(0-shot). Fraction of baseline
    uncertainty removed by 4 demonstrations.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# Each entry: (demonstrations, query, target_continuation)
# Demonstrations are (input, output) pairs for the same task.
_ICL_BUNDLE: List[Dict] = [
    {
        "demos": [
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Italy", "Rome"),
            ("Germany", "Berlin"),
        ],
        "query": "Spain",
        "target": " Madrid",
        "template": "Country: {inp} -> Capital: {out}",
        "query_template": "Country: {inp} -> Capital:",
    },
    {
        "demos": [
            ("dog", "animal"),
            ("rose", "flower"),
            ("oak", "tree"),
            ("salmon", "fish"),
        ],
        "query": "eagle",
        "target": " bird",
        "template": "{inp} is a {out}",
        "query_template": "{inp} is a",
    },
    {
        "demos": [
            ("happy", "sad"),
            ("hot", "cold"),
            ("big", "small"),
            ("fast", "slow"),
        ],
        "query": "light",
        "target": " dark",
        "template": "The opposite of {inp} is {out}.",
        "query_template": "The opposite of {inp} is",
    },
    {
        "demos": [
            ("2 + 3", "5"),
            ("7 + 1", "8"),
            ("4 + 4", "8"),
            ("6 + 3", "9"),
        ],
        "query": "5 + 2",
        "target": " 7",
        "template": "{inp} = {out}",
        "query_template": "{inp} =",
    },
]


def _build_prompt(item: Dict, k: int) -> Tuple[str, str]:
    """Build a prompt with k demonstrations + query. Returns (prompt, target)."""
    parts = []
    for inp, out in item["demos"][:k]:
        parts.append(item["template"].format(inp=inp, out=out))
    parts.append(item["query_template"].format(inp=item["query"]))
    prompt = "\n".join(parts)
    return prompt, item["target"]


@register_task("consistency_icl_slope")
class ICLSlopeTask(DiagnosticTask):
    """In-context learning slope (NLL vs. number of demonstrations)."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running In-Context Learning Slope Analysis...")

        shot_counts = self.config.get("shot_counts", [0, 1, 2, 4])

        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and "demos" in dataset[0]
        ):
            items = list(dataset)
        else:
            items = list(_ICL_BUNDLE)

        device = next(model.parameters()).device

        # nll_by_k[k] = list of per-item NLLs
        nll_by_k: Dict[int, List[float]] = {k: [] for k in shot_counts}

        from ..common import score_continuation

        with torch.no_grad():
            for item in items:
                for k in shot_counts:
                    prompt, target = _build_prompt(item, k)
                    res = score_continuation(model, tokenizer, prompt, target)
                    if res is None:
                        continue
                    mean_nll, _n_ans_tok, _ans_ids = res
                    nll_by_k[k].append(float(mean_nll))

        mean_nll = {k: float(np.mean(v)) if v else float("nan")
                    for k, v in nll_by_k.items()}

        # ICL slope: linear fit of NLL vs k
        ks = np.array([k for k in shot_counts if not np.isnan(mean_nll[k])], dtype=np.float64)
        nlls = np.array([mean_nll[k] for k in shot_counts if not np.isnan(mean_nll[k])], dtype=np.float64)

        if len(ks) >= 2:
            slope = float(np.polyfit(ks, nlls, 1)[0])
        else:
            slope = float("nan")

        # ICL gain: 0-shot minus max-shot
        nll_0 = mean_nll.get(shot_counts[0], float("nan"))
        nll_max = mean_nll.get(shot_counts[-1], float("nan"))
        icl_gain = (nll_0 - nll_max) if not (np.isnan(nll_0) or np.isnan(nll_max)) else float("nan")
        icl_rel = (icl_gain / nll_0) if (not np.isnan(icl_gain) and nll_0 > 0) else float("nan")

        return {
            **{f"mean_nll_{k}shot": mean_nll[k] for k in shot_counts},
            "icl_slope": slope,
            "icl_gain": float(icl_gain) if not np.isnan(icl_gain) else float("nan"),
            "icl_relative_gain": float(icl_rel) if not np.isnan(icl_rel) else float("nan"),
            "n_items": len(items),
            "shot_counts": shot_counts,
        }
