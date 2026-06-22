"""
Position sensitivity score inspired by "lost in the middle" probes.

Inserts a key fact at varying relative positions inside a distractor passage
and measures the NLL of a short recall continuation that depends on the
inserted fact. Models that exhibit "lost in the middle" behavior show high
NLL when the fact is in the middle of the context and low NLL when it is at
the start or end.

Reported metrics:
- mean NLL per relative position {0.0, 0.25, 0.5, 0.75, 1.0}
- lost-in-middle NLL depth: middle - best edge (positive = higher middle NLL)
- position_spread: max NLL - min NLL across positions
- position_argmin: relative position of best recall (0.0 ... 1.0)

The metric is purely intrinsic (no labels): all needles are bundled with
the task as `(passage, key_fact, recall_continuation)` triples and we use
shifted-NLL on the continuation tokens.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...registry import register_task
from ...tasks.base import DiagnosticTask

logger = logging.getLogger("blme")


# Each entry: (distractor_passage, key_fact, recall_continuation)
# - distractor_passage: long-ish text NOT containing the answer
# - key_fact: a single sentence that introduces the answer
# - recall_continuation: a short sentence whose plausibility depends on
#   knowing the key fact (we measure NLL on these tokens)
_NEEDLE_BUNDLE: List[Tuple[str, str, str]] = [
    (
        "The moon is Earth's only natural satellite. It causes the tides "
        "and stabilizes the planet's axial tilt. Tides ebb and flow in "
        "complex patterns that vary by region and lunar phase. Coastal "
        "ecosystems depend on these regular fluctuations. Scientists have "
        "studied lunar geology for decades through both telescopes and "
        "robotic landers. Many cultures throughout history have created "
        "myths and rituals tied to the lunar cycle.",
        "The first human to walk on the moon was Neil Armstrong on July 20, 1969.",
        " The first human to walk on the moon was Neil Armstrong.",
    ),
    (
        "The Pacific Ocean is the largest of Earth's oceans. It contains "
        "thousands of islands and supports a vast diversity of marine life. "
        "The ocean's currents play a major role in regulating global climate "
        "patterns. Many fishing communities along the Pacific Rim depend on "
        "its resources. Coral reefs are particularly vulnerable to ocean "
        "acidification and rising temperatures. Migratory species cross the "
        "Pacific in patterns that have been observed for centuries.",
        "The deepest point in the Pacific Ocean is the Mariana Trench at about 10,994 meters.",
        " The deepest point in the Pacific Ocean is the Mariana Trench.",
    ),
    (
        "Photosynthesis is the process by which green plants convert "
        "sunlight into chemical energy. It is essential for life on Earth. "
        "Plants absorb light through pigments in their leaves and produce "
        "oxygen as a byproduct. The process is sensitive to changes in "
        "temperature and water availability. Crops grown in different "
        "climates rely on adapted varieties to survive local conditions. "
        "Agricultural research has long focused on improving photosynthetic efficiency.",
        "The primary pigment used in photosynthesis is chlorophyll, which gives plants their green color.",
        " The primary pigment used in photosynthesis is chlorophyll.",
    ),
    (
        "The Roman Empire spanned three continents at its peak. It left "
        "lasting influence on language, law, and architecture across Europe "
        "and beyond. Roman engineers built roads, aqueducts, and amphitheaters "
        "that still stand today. The empire's military strength came from "
        "highly disciplined legions. Trade routes connected the Mediterranean "
        "to distant regions. The empire eventually divided into eastern and "
        "western halves before its decline.",
        "The Roman Empire was officially founded in 27 BC by Augustus Caesar.",
        " The Roman Empire was founded by Augustus Caesar.",
    ),
    (
        "Quantum mechanics is the branch of physics that describes nature at "
        "the smallest scales. It explains the behavior of subatomic particles "
        "and the structure of atoms. Many of its predictions have been "
        "confirmed by experiment, even though they often defy classical "
        "intuition. Quantum effects underlie technologies like lasers and "
        "transistors. Researchers continue to explore applications in "
        "computing and cryptography.",
        "The Heisenberg uncertainty principle states that the position and momentum of a particle cannot both be known precisely.",
        " The Heisenberg uncertainty principle relates position and momentum.",
    ),
    (
        "Music has been part of human culture for tens of thousands of years. "
        "Different societies have developed unique instruments and styles. "
        "Recorded music in the twentieth century transformed how people "
        "consume and share songs. Genres continue to evolve as new "
        "technologies enable creative experimentation. Live performance "
        "remains an important form of artistic expression. Music education "
        "is widely valued in schools around the world.",
        "Ludwig van Beethoven composed his ninth symphony entirely after he had become deaf.",
        " Beethoven composed his ninth symphony while deaf.",
    ),
    (
        "Computers have transformed nearly every aspect of modern life. "
        "From scientific research to everyday communication, they have "
        "made many tasks faster and more accessible. Software engineers "
        "design programs that run on a wide variety of devices. Operating "
        "systems coordinate the underlying hardware and applications. "
        "Open source projects have become a major part of the software "
        "ecosystem worldwide.",
        "The first general-purpose electronic computer was ENIAC, built in 1945 at the University of Pennsylvania.",
        " The first general-purpose electronic computer was ENIAC.",
    ),
    (
        "Mountains form through geological processes that take millions of "
        "years. Tectonic plate collisions, volcanic activity, and erosion "
        "all play roles in shaping mountain ranges. Mountainous regions "
        "are home to unique ecosystems and species adapted to thin air "
        "and cold temperatures. Many cultures regard mountains as sacred. "
        "Climbing has become a popular activity but presents serious risks.",
        "The tallest mountain on Earth measured from sea level is Mount Everest at 8,849 meters.",
        " The tallest mountain on Earth is Mount Everest.",
    ),
    (
        "Languages evolve continuously over time, influenced by migration, "
        "trade, and cultural exchange. New words enter the lexicon while "
        "older ones fall out of use. Linguists study these patterns to "
        "understand human cognition and history. Some languages are "
        "spoken by billions while others are endangered with only a "
        "handful of remaining speakers. Preservation efforts try to "
        "document and revive minority languages.",
        "Mandarin Chinese is the most spoken native language in the world, with over a billion speakers.",
        " The most spoken native language in the world is Mandarin Chinese.",
    ),
    (
        "Climate change is reshaping weather patterns and ecosystems "
        "worldwide. Rising global temperatures have led to more frequent "
        "extreme weather events. Polar ice is melting and sea levels are "
        "rising. Many species are struggling to adapt to the rapid pace "
        "of change. International agreements have set goals to reduce "
        "greenhouse gas emissions, but progress has been uneven across "
        "different countries.",
        "The greenhouse gas most responsible for human-caused climate change is carbon dioxide.",
        " The greenhouse gas most responsible for climate change is carbon dioxide.",
    ),
]


@register_task("consistency_position_sensitivity")
class PositionSensitivityTask(DiagnosticTask):
    """Position-dependent continuation-NLL proxy."""

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Position Sensitivity (Lost in the Middle) Analysis...")

        positions = self.config.get("positions", [0.0, 0.25, 0.5, 0.75, 1.0])
        num_samples = self.config.get("num_samples", len(_NEEDLE_BUNDLE))

        # Allow user override; otherwise use the bundled needle dataset.
        if dataset is not None and isinstance(dataset, list) and dataset and (
            isinstance(dataset[0], dict) and {"passage", "fact", "recall"} <= set(dataset[0])
        ):
            triples = [(d["passage"], d["fact"], d["recall"]) for d in dataset[:num_samples]]
        else:
            triples = _NEEDLE_BUNDLE[:num_samples]

        device = next(model.parameters()).device

        # nll_by_pos[pos_idx] = list of per-needle continuation NLLs
        nll_by_pos = {p: [] for p in positions}

        with torch.no_grad():
            for passage, fact, recall in triples:
                # Tokenize the passage as a list of words and re-join — we
                # want to insert the fact at WORD boundaries to keep the
                # text grammatical at all positions.
                words = passage.split(" ")
                if len(words) < 4:
                    continue

                for rel_pos in positions:
                    word_idx = int(round(rel_pos * len(words)))
                    word_idx = max(0, min(len(words), word_idx))
                    prefix = " ".join(words[:word_idx])
                    suffix = " ".join(words[word_idx:])
                    if prefix and suffix:
                        full_context = prefix + " " + fact + " " + suffix
                    elif prefix:
                        full_context = prefix + " " + fact
                    else:
                        full_context = fact + " " + suffix
                    full_text = full_context + recall

                    enc_full = tokenizer(full_text, return_tensors="pt").to(device)
                    enc_ctx = tokenizer(full_context, return_tensors="pt").to(device)
                    full_ids = enc_full["input_ids"][0]
                    ctx_len = enc_ctx["input_ids"].shape[1]

                    if full_ids.shape[0] <= ctx_len:
                        continue
                    if full_ids.shape[0] > 1024:
                        # Truncate from the left so the recall stays at the
                        # tail; this only matters for very long passages.
                        cut = full_ids.shape[0] - 1024
                        full_ids = full_ids[cut:]
                        ctx_len = max(0, ctx_len - cut)
                        enc_full["input_ids"] = full_ids.unsqueeze(0)
                        if "attention_mask" in enc_full:
                            enc_full["attention_mask"] = enc_full["attention_mask"][:, cut:]

                    out = model(**enc_full)
                    logits = out.logits[0]  # (T, V)
                    # Continuation tokens occupy positions [ctx_len, T)
                    # Their predictions come from logits at positions
                    # [ctx_len - 1, T - 1).
                    pred_logits = logits[ctx_len - 1: -1]
                    targets = full_ids[ctx_len:]
                    if pred_logits.shape[0] != targets.shape[0] or pred_logits.shape[0] == 0:
                        continue
                    losses = F.cross_entropy(pred_logits, targets, reduction="none")
                    mean_nll = float(losses.mean().item())
                    nll_by_pos[rel_pos].append(mean_nll)

        per_pos_mean = {f"nll_at_{p}": (float(np.mean(v)) if v else float("nan"))
                        for p, v in nll_by_pos.items()}
        # Convenience aliases
        nll_arr = [per_pos_mean[f"nll_at_{p}"] for p in positions]
        if not any(np.isnan(nll_arr)):
            n = len(positions)
            mid_idx = n // 2
            start_end_min = min(nll_arr[0], nll_arr[-1])
            u_curve_depth = float(nll_arr[mid_idx] - start_end_min)
            position_spread = float(max(nll_arr) - min(nll_arr))
            position_argmin = float(positions[int(np.argmin(nll_arr))])
        else:
            u_curve_depth = float("nan")
            position_spread = float("nan")
            position_argmin = float("nan")

        return {
            **per_pos_mean,
            "mean_nll_across_positions": float(np.nanmean(nll_arr)),
            "diagnostic_semantics": "position_conditioned_continuation_nll_proxy",
            "lost_in_middle_nll_depth": u_curve_depth,
            "position_nll_spread": position_spread,
            "best_recall_position": position_argmin,
            # Legacy aliases retained for downstream compatibility.
            "u_curve_depth": u_curve_depth,
            "position_spread": position_spread,
            "position_argmin": position_argmin,
            "n_needles": len(triples),
        }
