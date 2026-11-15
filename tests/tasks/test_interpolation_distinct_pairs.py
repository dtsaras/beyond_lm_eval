"""dynamics_interpolation: context pairs are distinct and drawn without
replacement, capped at the n(n-1)/2 available."""
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.dynamics.trajectories import LatentInterpolationTask  # noqa: E402

N = 5


@pytest.fixture(scope="module")
def tiny():
    return fd.tiny_gpt2(), fd.tokenizer()


def _run(model, tok, data, **cfg):
    return LatentInterpolationTask(config={"num_samples": len(data), "seed": 0, **cfg}).evaluate(
        model, tok, data)


def test_more_pairs_than_exist_uses_each_distinct_pair_once(tiny):
    """5 passages have 10 distinct pairs; asking for 50 uses each once, so the
    gap is the uniform mean of the 10 per-pair gaps (with replacement, pairs
    repeated and the mean was weighted by the repeats)."""
    model, tok = tiny
    data = fd.dataset(N)
    r = _run(model, tok, data, num_pairs=50)
    per_pair = [_run(model, tok, [data[i], data[j]], num_pairs=1)["convexity_gap"]
                for i, j in combinations(range(N), 2)]
    assert r["convexity_gap"] == pytest.approx(float(np.mean(per_pair)), rel=1e-9, abs=1e-12)
    assert r["_meta_num_pairs_used"] == 10
    assert r["_meta_num_distinct_pairs"] == 10
    assert r["_meta_num_pairs_requested"] == 50


def test_drawn_pairs_are_distinct_unordered_pairs():
    from blme.tasks.dynamics.trajectories import _distinct_pairs
    for n in (2, 3, 7, 40):
        total = n * (n - 1) // 2
        got = _distinct_pairs(n, total, random.Random(1))
        assert sorted(tuple(sorted(p)) for p in got) == list(combinations(range(n), 2))
    big = _distinct_pairs(10_000, 500, random.Random(0))
    assert len({tuple(sorted(p)) for p in big}) == 500
    assert all(0 <= a < 10_000 and 0 <= b < 10_000 and a != b for a, b in big)
