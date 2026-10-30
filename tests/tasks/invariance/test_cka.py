"""geometry_cka: token policy, key hygiene and size invariance, driven
through the real ``evaluate()`` on injected hidden states."""
from __future__ import annotations

import math

import numpy as np
import pytest

from _family_c import TEXTS, inv, load, scale_position0

TASK = "geometry_cka"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


@pytest.fixture(scope="module")
def gpt2_states():
    return inv.states_from_model(*load("gpt2"), TEXTS[:6])


def test_headline_ignores_position0_rows(tiny_states):
    """Position 0 is excluded from every headline: scaling the position-0 row
    of every document by 1e3 (a synthetic sink) must not move one of them."""
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    keys = [k for k in base if not k.startswith("_meta_")
            and isinstance(base[k], float)]
    assert keys
    inv.assert_invariant(base, sunk, keys, rtol=1e-6)


def test_position0_and_massive_shares_are_derived(tiny_states, gpt2_states):
    """The sink diagnostics are measured, not constants: gpt2's real position-0
    sink dominates the mid-layer centred SS, the synthetic one too, and the
    inclusive companion moves while the headline does not."""
    g = inv.run_task_on_states(TASK, gpt2_states)
    assert g["_meta_position0_share"] > 0.5
    assert 0.0 <= g["_meta_massive_token_share"] <= 1.0
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, scale_position0(tiny_states, 1e3))
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]
    inv.assert_moves(base, sunk, "_meta_incl_pos0_early_late_cka", 1e-3)
    # exclude_sinks keeps every row but position 0 and the massive ones.
    assert base["_meta_rows_policy"] == "exclude_sinks"
    assert base["_meta_rows_n_rows"] == (sum(tiny_states.lengths) - tiny_states.n_docs
                                         - base["_meta_sink_n_massive_dropped"])


def test_massive_row_at_one_middle_depth_is_dropped_at_every_layer(gpt2_states, monkeypatch):
    """A first-period-style sink (row 3 of every document x1000 at block L/2
    only) moves exclude_pos0's adjacent minimum; exclude_sinks drops that token
    at every layer, giving the row-deleted corpus's value bit for bit, and with
    no massive row it equals exclude_pos0 exactly."""
    from _sinks import check_state_task
    check_state_task(TASK, CKATask, gpt2_states, monkeypatch, "min_adjacent_cka",
                     keys=list(CKATask.HEADLINE_KEYS))


def test_max_offdiag_duplicate_is_not_emitted(tiny_states):
    """max_offdiag_cka equalled max_adjacent_cka on 65 of 66 study models (the
    most similar pair is adjacent); the duplicate column is dropped."""
    r = inv.run_task_on_states(TASK, tiny_states)
    assert "max_offdiag_cka" not in r
    off = [k for k in r if "max_offdiag" in k]
    assert off == [], off


@pytest.fixture(scope="module")
def tiny_identity_states(tiny):
    """tiny-random-gpt2 with exact identity blocks after blocks 1 and 3."""
    model, tok = tiny
    return inv.states_from_model(inv.insert_identity_blocks(model, [1, 3]), tok, TEXTS)


def test_headline_is_depth_invariant_under_identity_insertion(
        tiny_states, tiny_identity_states):
    """P4: inserting function-preserving identity blocks must not move any
    headline key; the uniform layer means it replaced do move (known negative)."""
    base = inv.run_task_on_states(TASK, tiny_states)
    ins = inv.run_task_on_states(TASK, tiny_identity_states)
    assert ins["_meta_n_layers"] == base["_meta_n_layers"] + 2
    keys = [k for k in base if not k.startswith("_meta_")
            and isinstance(base[k], float)]
    assert {"early_late_cka", "min_offdiag_cka", "min_adjacent_cka"} <= set(keys)
    inv.assert_invariant(base, ins, keys, rtol=1e-9)
    for k in ("_meta_avg_adjacent_cka", "_meta_mean_offdiag_cka",
              "_meta_max_adjacent_cka"):
        inv.assert_moves(base, ins, k, 1e-3)


DEBIASED = ("early_late_cka_debiased", "min_offdiag_cka_debiased",
            "min_adjacent_cka_debiased")


def test_debiased_twin_has_no_sample_size_floor(tiny_states):
    """N variation on a row-permutation null (the second layer's rows shuffled
    across the cloud, so the population CKA is 0): the biased estimator's
    d_eff/N floor falls ~4x from 3 to 12 documents, while the debiased twin
    stays at 0 at every N."""
    from _family_c import keep_layers, permute_layer_rows
    L = tiny_states.n_layers
    null = permute_layer_rows(keep_layers(tiny_states, [0, L // 2]), 1, seed=0)
    small = [inv.run_task_on_states(TASK, inv.subsample_docs(null, 3, seed=s))
             for s in (0, 1)]
    full = inv.run_task_on_states(TASK, null)
    biased_small = np.mean([r["early_late_cka"] for r in small])
    assert biased_small - full["early_late_cka"] > 0.1      # known negative
    for r in small + [full]:
        for k in DEBIASED:
            assert abs(r[k]) < 0.03, (k, r[k])


@pytest.mark.parametrize("transform", ["zero_pad", "isometry", "rescale"])
def test_headline_is_width_and_scale_invariant(tiny_states, transform):
    """Zero-padding, a random isometric embedding (x2 width) and a global
    power-of-two rescale leave every headline key in place."""
    St = {"zero_pad": lambda S: inv.zero_pad_docs(S, 2),
          "isometry": lambda S: inv.embed_isometric_docs(S, 2, 3),
          "rescale": lambda S: inv.rescale_docs(S, 2.0)}[transform](tiny_states)
    base = inv.run_task_on_states(TASK, tiny_states)
    keys = [k for k in base if not k.startswith("_meta_") and isinstance(base[k], float)]
    inv.assert_invariant(base, inv.run_task_on_states(TASK, St), keys, rtol=1e-5)


# --------------------------------------------------------------------------
# SIZE_INVARIANT declarations, checked against their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.cka import CKATask  # noqa: E402

_NEG_N = {'early_late_cka_debiased': 'early_late_cka', 'min_offdiag_cka_debiased': 'min_offdiag_cka', 'min_adjacent_cka_debiased': 'min_adjacent_cka'}


@pytest.fixture(scope="module")
def null_states(tiny_states):
    from _family_c import keep_layers, permute_layer_rows
    L = tiny_states.n_layers
    return permute_layer_rows(keep_layers(tiny_states, [0, L // 2]), 1, seed=0)


@pytest.mark.parametrize("key", sorted(CKATask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny_states, tiny_identity_states, null_states):
    """Each declared transform's direct test, plus the known negative: the
    uniform layer mean under identity blocks, the biased twin under N."""
    from _family_c import check_size_invariant_entry
    negatives = {"identity_blocks": "_meta_avg_adjacent_cka"}
    if key in _NEG_N:
        negatives["N"] = _NEG_N[key]
    check_size_invariant_entry(TASK, CKATask, key, tiny_states, tiny_identity_states,
                               null_states=null_states, negatives=negatives)


def test_declarations():
    assert CKATask.TOKEN_POLICY == "exclude_sinks"
    assert set(CKATask.SIZE_INVARIANT) <= set(CKATask.HEADLINE_KEYS)
    assert {"paper", "code", "llm_precedent"} <= set(CKATask.SOURCES)
