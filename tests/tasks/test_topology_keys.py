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


def test_landscape_status_keys_start_with_meta(states):
    """30 keys were named layer_<l>_<h0|h1>_meta_*: a reader testing
    startswith("_meta_") ingested string/bool status flags as features."""
    res = _run("topology_persistence_landscape", states)
    assert not [k for k in res if "_meta_" in k and not k.startswith("_meta_")]
    L = states.n_layers
    for l in sorted({0, L // 2, L - 1}):
        for dim in ("h0", "h1"):
            for k in ("last_entry_is_block_output", "prescale", "landscape_max_status",
                      "landscape_quadrature", "landscape_n_bars"):
                assert f"_meta_layer_{l}_{dim}_{k}" in res, (l, dim, k)


def test_persistence_entropy_h0_family_is_meta(states):
    """Raw H0 entropy of N points is log2(N - 1) to three digits; E / log2(n)
    passes every direct test but does not separate models (dossier: between/
    within SD 0.5-1.3). Feature keys before the fix included all of them."""
    res = _run("topology_persistence_entropy", states)
    feats = set(_features(res))
    assert feats == {f"{r}_layer_{k}" for r in ("first", "mid", "last")
                     for k in ("pe_h1", "pe_h1_norm", "num_features_h1")}, feats
    for r in ("first", "mid", "last"):
        for k in ("pe_h0", "pe_h0_full", "pe_h0_norm", "num_features_h0"):
            assert f"_meta_{r}_layer_{k}" in res
    assert "_meta_pe_simplification_ratio" in res
