"""Key hygiene of the topology tasks (P7): depth-role names, _meta_ prefixes.

Absolute layer-index names (``layer_11_*``) mean the deepest block of a
12-layer model and the middle of a 22-layer one; in the 66-model study only
``layer_0_*`` was a shared column. Feature keys are named by depth role and
the absolute-index copies are ``_meta_``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))
import _family_d as fd  # noqa: E402

pytest.importorskip("ripser")

ABS = re.compile(r"(^|_)layer_\d+_")


@pytest.fixture(scope="module")
def states():
    return fd.inv.states_from_model(fd.tiny_gpt2(), fd.tokenizer(), fd.TEXTS)


def _run(task, S, **cfg):
    return fd.inv.run_task_on_states(task, S, {"num_samples": S.n_docs, **cfg}, force=True)


def _features(res):
    return [k for k in res if not k.startswith("_meta_")]


def test_persistence_entropy_keys_are_named_by_depth_role(states):
    res = _run("topology_persistence_entropy", states)
    assert not [k for k in _features(res) if ABS.search(k)]
    for r in ("first", "mid", "last"):
        assert f"{r}_layer_pe_h1" in res and f"{r}_layer_pe_h1_norm" in res
    L = states.n_layers
    assert res[f"_meta_layer_{L - 1}_pe_h1"] == res["last_layer_pe_h1"]
