"""Empty barcodes give NaN, not 0.0; homology's _norm twins ignore repeated points."""
import math

import numpy as np
import pytest

pytest.importorskip("ripser")

from blme.tasks.topology.homology import _lifespan_summary  # noqa: E402
from blme.tasks.topology.persistence_entropy import _persistence_entropy  # noqa: E402


def test_entropy_of_an_empty_barcode_is_nan_and_of_one_bar_is_zero():
    assert math.isnan(_persistence_entropy([]))
    assert math.isnan(_persistence_entropy([0.0, 0.0]))       # no positive bar
    assert _persistence_entropy([0.7]) == 0.0
    assert _persistence_entropy([1.0, 1.0]) == pytest.approx(1.0)


def test_mean_h1_lifetime_without_loops_is_nan():
    dgms = [np.array([[0.0, 1.0], [0.0, np.inf]]), np.empty((0, 2))]
    s = _lifespan_summary(dgms)
    assert s["num_loops_h1"] == 0 and math.isnan(s["mean_persistence_h1"])


def test_homology_norm_twins_ignore_repeated_points(monkeypatch):
    from blme.tasks.topology import homology as H
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.c_[np.cos(th), np.sin(th), np.zeros(40)]
    clouds = {0: circle, 1: np.vstack([circle, circle[:10]])}

    def fake_points(*a, **k):
        return {0: clouds[k["_which"]]}, {"source": "test", "last_is_block": True, "n_layers": 1,
                                          "positions": [0] * len(clouds[k["_which"]]),
                                          "n_docs": len(clouds[k["_which"]]), "cache_rejected": None}
    out = {}
    for which in (0, 1):
        monkeypatch.setattr(H, "last_token_points",
                            lambda *a, _w=which, **k: fake_points(*a, _which=_w, **k))
        r = H.PersistentHomologyTask({"num_samples": 50}).evaluate(None, None, [{"text": "x"}])
        out[which] = r
    assert out[1]["_meta_layer_0_n_duplicate_points"] == 10
    for k in ("mean_persistence_h0_norm", "max_persistence_h0_norm", "mean_persistence_h1_norm"):
        key = f"_meta_layer_0_{k}"
        assert out[0][key] == pytest.approx(out[1][key], rel=1e-12), k
