"""
Loss-based membership separability proxy.

A membership inference attack asks: "Was this sample in the model's
training data?" The simplest and most effective MIA for language models
(Carlini et al. 2021) thresholds on per-sample loss: training-set
members tend to have lower loss than held-out text.

When callers provide comparable member/non-member examples (for example
paired via ``pair_id`` / ``comparison_id`` / ``calibration_group``), the
loss split can be interpreted as a calibrated membership-inference
diagnostic. Since the default data is not actual training-set membership,
it is reported as a domain/frequency separability proxy:
  - "Members": common factual sentences likely present in standard
    web-crawl corpora (Wikipedia, news, etc.)
  - "Non-members": niche / unusual text unlikely to appear verbatim
    in pretraining data (obscure technical jargon, synthetic text)

Reported metrics:
  - **mia_auroc**: AUROC of using negative-NLL as a membership score.
    1.0 = perfect discrimination (extreme memorization), 0.5 = no signal.
  - **loss_gap**: mean NLL on non-members minus mean NLL on members.
    Positive = model does better on members (consistent with memorization).
  - **mean_loss_member / non_member**: raw per-group NLL.
  - **counterfactual_gap**: for each member sentence, also compute NLL
    on a word-shuffled version. The mean gap (shuffled - original) is the
    memorization intensity for that passage.

References:
  - Yeom, Giacomelli, Fredrikson, Jha, "Privacy Risk in ML: Analyzing the
    Connection to Overfitting", IEEE CSF 2018. arXiv:1709.01604.
  - Carlini, Tramer, Wallace et al., "Extracting Training Data from Large
    Language Models", USENIX 2021. arXiv:2012.07805.
"""

import logging
import random
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


_MEMBERS: List[str] = [
    "The United Nations was established in 1945 after World War II.",
    "Photosynthesis is the process by which plants convert sunlight into food.",
    "The Great Wall of China is one of the most famous structures in the world.",
    "Albert Einstein published his theory of general relativity in 1915.",
    "The human body contains approximately 206 bones.",
    "Water covers about 71 percent of the Earth surface.",
    "The speed of light in a vacuum is approximately 300000 kilometers per second.",
    "DNA carries the genetic instructions used in growth and reproduction.",
    "The French Revolution began in 1789 and ended with Napoleon's rise.",
    "Shakespeare wrote approximately 37 plays during his lifetime.",
    "The Amazon rainforest produces about 20 percent of the world's oxygen.",
    "Newton's laws of motion describe the relationship between force and acceleration.",
]

_NON_MEMBERS: List[str] = [
    "Quaternionic Koszul duality provides a derived equivalence for hypercomplex categories.",
    "The zygodactyl toe arrangement in jacamars facilitates perch-gleaning foraging behaviour.",
    "Ruthenium-catalysed olefin metathesis under Schrock conditions yields E-selectivity.",
    "Paleoproterozoic banded iron formations record the Great Oxidation Event at 2.4 Ga.",
    "Ergodic Ramsey theory generalises Szemeredi's theorem via ultrafilter combinatorics.",
    "The pedunculate oak Quercus robur exhibits marcescent leaf retention in juvenile trees.",
    "Magnetohydrodynamic Kelvin-Helmholtz instabilities dominate the jovian magnetopause.",
    "Fourier-Mukai transforms on abelian varieties relate to derived autoequivalences.",
    "Alpine tarns above the firn line exhibit oligotrophic dimictic stratification.",
    "Trilobite pygidial morphology distinguishes Redlichiida from Ptychopariida at ordinal level.",
    "Stochastic quantisation of Yang-Mills fields via Parisi-Wu theory requires Langevin dynamics.",
    "Chalcogenide phase-change materials exhibit threshold switching via Ovshinsky effect.",
]


def _normalise_membership_label(label: Any) -> Optional[int]:
    if label in (1, "1", True, "member", "members", "train", "training"):
        return 1
    if label in (0, "0", False, "non_member", "non-member", "nonmember", "heldout", "test"):
        return 0
    return None


def _has_comparable_pairs(dataset: List[dict]) -> bool:
    """Return True when labels are paired within a shared comparison group."""
    for key in ("pair_id", "comparison_id", "calibration_group"):
        if not all(key in item for item in dataset):
            continue
        grouped = defaultdict(set)
        for item in dataset:
            label = _normalise_membership_label(item.get("label"))
            if label is not None:
                grouped[item[key]].add(label)
        return bool(grouped) and all(labels == {0, 1} for labels in grouped.values())
    return False


def _compute_nll(model, tokenizer, text: str, device) -> float:
    """Mean per-token NLL for a single text."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 3:
        return float("nan")
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), reduction="mean")
    return float(loss.item())


@register_task("consistency_membership_inference")
class MembershipInferenceTask(DiagnosticTask):
    """Loss-based separability proxy; calibrated MIA only with comparable labels."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Membership Separability Analysis...")

        require_comparable = bool(self.config.get("require_comparable_membership_data", False))
        has_comparable_data = False
        used_default_proxy = False

        if dataset is not None and isinstance(dataset, list) and dataset and all(
            isinstance(item, dict) and {"text", "label"} <= set(item) for item in dataset
        ):
            has_comparable_data = _has_comparable_pairs(dataset)
            if require_comparable and not has_comparable_data:
                return {
                    "error": (
                        "Comparable member/non-member examples are required. "
                        "Provide paired labels with pair_id, comparison_id, or calibration_group."
                    )
                }
            members = [
                d["text"] for d in dataset if _normalise_membership_label(d["label"]) == 1
            ]
            non_members = [
                d["text"] for d in dataset if _normalise_membership_label(d["label"]) == 0
            ]
        else:
            if require_comparable:
                return {
                    "error": (
                        "Comparable member/non-member examples are required; "
                        "the bundled defaults are only a domain/frequency separability proxy."
                    )
                }
            members = list(_MEMBERS)
            non_members = list(_NON_MEMBERS)
            used_default_proxy = True

        if len(members) < 2 or len(non_members) < 2:
            return {"error": "Need at least 2 members and 2 non-members"}

        device = next(model.parameters()).device

        member_losses = [_compute_nll(model, tokenizer, t, device) for t in members]
        nonmember_losses = [_compute_nll(model, tokenizer, t, device) for t in non_members]

        # Filter NaN
        member_losses = [x for x in member_losses if not np.isnan(x)]
        nonmember_losses = [x for x in nonmember_losses if not np.isnan(x)]

        if len(member_losses) < 2 or len(nonmember_losses) < 2:
            return {"error": "Too few valid samples after NaN filtering"}

        mean_member = float(np.mean(member_losses))
        mean_nonmember = float(np.mean(nonmember_losses))
        loss_gap = mean_nonmember - mean_member

        # AUROC: use negative loss as "membership score" (higher = more likely member)
        scores = [-l for l in member_losses] + [-l for l in nonmember_losses]
        labels = [1] * len(member_losses) + [0] * len(nonmember_losses)
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(labels, scores))
        except Exception:
            # Manual AUROC via Mann-Whitney U
            n1, n0 = len(member_losses), len(nonmember_losses)
            concordant = sum(1 for m in member_losses for nm in nonmember_losses if m < nm)
            ties = sum(0.5 for m in member_losses for nm in nonmember_losses if m == nm)
            auroc = float((concordant + ties) / (n1 * n0))

        # Counterfactual memorization: for each member, compute NLL on a
        # word-shuffled version. The gap measures how much the model
        # benefits from the exact word order (a memorization proxy).
        rng = random.Random(0)
        cf_gaps = []
        for text in members:
            words = text.split()
            if len(words) < 4:
                continue
            shuffled = list(words)
            rng.shuffle(shuffled)
            shuffled_text = " ".join(shuffled)
            orig_nll = _compute_nll(model, tokenizer, text, device)
            shuf_nll = _compute_nll(model, tokenizer, shuffled_text, device)
            if not (np.isnan(orig_nll) or np.isnan(shuf_nll)):
                cf_gaps.append(shuf_nll - orig_nll)

        if has_comparable_data:
            score_semantics = "calibrated_membership_inference"
            warning = None
        elif used_default_proxy:
            score_semantics = "domain_frequency_separability_proxy"
            warning = (
                "Bundled defaults compare common factual sentences to niche text; "
                "this is not membership inference against a known training set."
            )
        else:
            score_semantics = "uncalibrated_loss_separability_proxy"
            warning = (
                "Provided member/non-member labels are not paired or calibrated; "
                "interpret AUROC as loss separability, not membership inference."
            )

        result = {
            "score_semantics": score_semantics,
            "is_calibrated_membership_inference": bool(has_comparable_data),
            "separability_auroc": auroc,
            "loss_gap": loss_gap,
            "mean_loss_member": mean_member,
            "mean_loss_nonmember": mean_nonmember,
            "counterfactual_gap": float(np.mean(cf_gaps)) if cf_gaps else float("nan"),
            "n_members": len(member_losses),
            "n_nonmembers": len(nonmember_losses),
            # Legacy alias retained for downstream compatibility.
            "mia_auroc": auroc,
        }
        if warning is not None:
            result["diagnostic_warning"] = warning
        return result
