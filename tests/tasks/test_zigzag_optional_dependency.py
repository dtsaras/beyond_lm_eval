"""dionysus is an optional extra (blme[topology]); without it the zigzag task errors.

topology_zigzag_persistence is a paper-core task whose barcode comes from
dionysus, and there is no substitute construction. When dionysus cannot be
imported, the public ``blme.core.evaluate`` must report an error for the task
and publish none of its readouts, in every environment: unlike
tests/tasks/test_zigzag_protocol.py, this file does not skip when dionysus is
missing, which is exactly the environment it describes.
"""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("ripser")       # checked first; without it the task names ripser

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

TASK = "topology_zigzag_persistence"
CONFIG = {"num_samples": len(fd.TEXTS), "points_readout": "passage"}


def _absent(monkeypatch):
    """dionysus is not installed: ``import dionysus`` raises ImportError."""
    monkeypatch.setitem(sys.modules, "dionysus", None)


def _broken(monkeypatch):
    """dionysus is installed but its extension cannot load (a missing Boost .so)."""
    monkeypatch.delitem(sys.modules, "dionysus", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dionysus" or name.startswith("dionysus."):
            raise ImportError("libboost_serialization.so.1.74.0: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize("simulate", [_absent, _broken], ids=["absent", "broken"])
def test_public_evaluate_reports_the_missing_dependency(monkeypatch, simulate):
    from blme import core

    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    monkeypatch.setattr(core, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    simulate(monkeypatch)
    out = core.evaluate(model_args="pretrained=unused", tasks=[TASK], device="cpu",
                        dataset=fd.dataset(), task_configs={TASK: CONFIG})

    result = out["results"][TASK]
    assert set(result) == {"error"}, result           # no proxy readouts, no zz_* keys
    assert "dionysus" in result["error"]


@pytest.mark.xfail(strict=True, reason=(
    "0.2.6 message says 'pip install dionysus==2.1.1' (equivalent, still correct); naming the "
    "extra needs an edit to src/blme/tasks/topology/zigzag.py, which is receipt-pinned"))
def test_the_error_names_the_extra(monkeypatch):
    from blme.tasks.topology.zigzag import ZigzagPersistenceTask

    _absent(monkeypatch)
    result = ZigzagPersistenceTask(CONFIG).evaluate(fd.tiny_llama(n_layers=3), fd.tokenizer(), fd.dataset())
    assert "blme[topology]" in result["error"]


def test_strict_mode_fails_on_the_missing_dependency(monkeypatch):
    from blme import core

    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    monkeypatch.setattr(core, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    _absent(monkeypatch)
    with pytest.raises(RuntimeError, match="dionysus"):
        core.evaluate(model_args="pretrained=unused", tasks=[TASK], device="cpu",
                      dataset=fd.dataset(), task_configs={TASK: CONFIG}, fail_on_task_error=True)
