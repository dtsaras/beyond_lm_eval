"""dynamics_interpolation.convexity_gap recomputed from its definition.

For every pair of contexts, take the last-token post-norm hidden states h1,
h2; the midpoint (h1 + h2) / 2 rescaled to (|h1| + |h2|) / 2; decode it, and
the endpoints, with the LM head; the gap is mean H(mid) - (mean H(h1) +
mean H(h2)) / 2, with H the Shannon entropy of the softmax.
"""
import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.trajectories import LatentInterpolationTask  # noqa: E402


def _entropy(logits):
    p = torch.softmax(logits.double(), dim=-1)
    return float(-(p * torch.log(p.clamp_min(1e-300))).sum())


def test_convexity_gap_matches_its_definition_over_every_distinct_pair():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    data = fd.dataset(5)
    r = LatentInterpolationTask({"num_samples": 5, "num_pairs": 10, "seed": 0}).evaluate(
        model, tok, data)
    assert r["_meta_num_pairs_used"] == 10
    with torch.no_grad():
        hs = [model(**tok(d["text"], return_tensors="pt", truncation=True, max_length=128),
                    output_hidden_states=True).hidden_states[-1][0, -1].float() for d in data]
        mids, ends = [], []
        for i, j in itertools.combinations(range(5), 2):
            h1, h2 = hs[i], hs[j]
            m = (h1 + h2) / 2
            m = m * (((h1.norm() + h2.norm()) / 2) / m.norm())
            mids.append(_entropy(model.lm_head(m)))
            ends.append((_entropy(model.lm_head(h1)) + _entropy(model.lm_head(h2))) / 2)
    ref = sum(mids) / len(mids) - sum(ends) / len(ends)
    assert r["convexity_gap"] == pytest.approx(ref, rel=1e-5, abs=1e-6)
