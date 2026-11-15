"""geometry_intrinsic_dim reproduces Facco et al. (2017) on their released data.

The TWO-NN repository (efacco/TWO-NN, cloned under
~/.cache/blme-refrepos) ships the benchmark sets of the paper. The Swiss roll
(2500 points in R^3) is published at d = 2.01. evaluate()'s static-embedding
route estimates the ID of a model's input-embedding rows, so a carrier model
whose embedding matrix holds the 2500 points runs the shipped estimator
(two nearest neighbours, top 10% of mu discarded, linear fit through the
origin) unchanged. Skips without the clone (fails under BLME_REQUIRE_REFS=1).
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

_REPO = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) / "TWO-NN"


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


class _EmbeddingCarrier(torch.nn.Module):
    def __init__(self, X):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(torch.as_tensor(X), freeze=True)

    def get_input_embeddings(self):
        return self.emb


def test_swiss_roll_reproduces_the_published_dimension():
    """Published 2.01; ours 1.9585 (-2.6%). The tolerance 0.1 is 2.5 standard
    errors of a 2500-point Two-NN estimate (about d / sqrt(N) = 0.04)."""
    path = _REPO / "datasets" / "swissroll"
    if not path.is_file():
        _missing(path)
    X = np.loadtxt(path)
    assert X.shape == (2500, 3)
    res = IntrinsicDimensionTask({"layerwise": False}).evaluate(_EmbeddingCarrier(X), None, None)
    assert res["_meta_intrinsic_dim_mode"] == "static_embeddings"
    assert res["_meta_sample_size"] == 2500
    assert res["intrinsic_dimension"] == pytest.approx(2.01, abs=0.1)
