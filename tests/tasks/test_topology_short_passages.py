"""Topology tasks on a corpus whose passages are all shorter than the 20-word
window: no point can be placed, and every task returns an error dict instead
of raising (persistence_landscape raised ``min() iterable argument is empty``)."""
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("ripser")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402

SHORT = [{"text": f"only a few words here, passage {i}."} for i in range(8)]


@pytest.mark.parametrize("name", ["persistence_landscape", "betti_curve",
                                  "persistence_entropy"])
def test_no_window_point_gives_an_error_not_an_exception(name):
    from blme.tasks.topology.betti_curve import BettiCurveTask
    from blme.tasks.topology.persistence_entropy import PersistenceEntropyTask
    from blme.tasks.topology.persistence_landscape import PersistenceLandscapeTask
    cls = {"persistence_landscape": PersistenceLandscapeTask, "betti_curve": BettiCurveTask,
           "persistence_entropy": PersistenceEntropyTask}[name]
    model = fd.tiny_gpt2()
    res = cls(config={"num_samples": len(SHORT)}).evaluate(
        model, fd.tokenizer(), SHORT, cache=None)
    assert set(res) == {"error"}, res
