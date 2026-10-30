"""geometry_matrix_entropy: the cache window gate, the gated float64 row
normalisation, key hygiene and the size-invariant raw entropy."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_matrix_entropy"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


def test_cache_route_honours_max_length(tiny):
    """The shared cache tokenises at 512; with max_length 8 (which truncates
    every TEXTS document) the task must self-collect at 8 tokens and match
    its own no-cache run, not read the 512-token cache."""
    model, tok = tiny
    cfg = {"max_length": 8}
    via_cache = inv.run_task_on_model(TASK, model, tok, TEXTS, cfg, use_cache=True)
    no_cache = inv.run_task_on_model(TASK, model, tok, TEXTS, {**cfg, "use_cache": False})
    assert via_cache["matrix_entropy"] == no_cache["matrix_entropy"]
    assert via_cache["_meta_matrix_entropy_source"] == "self_collected"
    assert via_cache["_meta_matrix_entropy_max_length"] == 8
    full = inv.run_task_on_model(TASK, model, tok, TEXTS, {}, use_cache=True)
    assert full["_meta_matrix_entropy_source"] == "shared_cache"
    assert full["matrix_entropy"] != via_cache["matrix_entropy"]


def test_rows_spanning_50_orders_survive_row_normalisation():
    """Wei et al. normalise every row, so the centred row-normalised
    covariance of the fixture is exactly I_2 / 2 and H / ln 2 = 1 (both
    released implementations return 1.0). The global power-of-two rescale
    and float32 cast used to flush the 1e-25 rows, leaving two collinear rows
    and 9.97e-29. The float64 row path runs only when rows were lost."""
    from adversarial_fixtures import wide_row_magnitude_span
    from blme.tasks.geometry.matrix_entropy import _matrix_entropy
    for label, X, want in wide_row_magnitude_span():
        got = _matrix_entropy(torch.tensor(X, dtype=torch.float32))
        assert got["entropy_normalized"] == pytest.approx(want, rel=1e-6), label
        assert got["n_rows_rescued"] == 2
    ordinary = _matrix_entropy(torch.randn(24, 16, generator=torch.Generator().manual_seed(0)))
    assert ordinary["n_rows_rescued"] == 0


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def test_headline_keeps_position0_and_reports_the_value_without_it(tiny_states):
    """Wei et al.'s released code scores every token of the text, so a sink
    at position 0 of the post-norm state moves the headline; the
    position-0-free value is _meta_ and does not move."""
    def sink(x):
        y = np.array(x, dtype=np.float64, copy=True)
        y[0] *= 1e3
        return y
    base = inv.run_task_on_states(TASK, tiny_states)
    sunk = inv.run_task_on_states(TASK, tiny_states.map(sink))
    inv.assert_moves(base, sunk, "matrix_entropy", 1e-2)
    inv.assert_invariant(base, sunk, ["_meta_matrix_entropy_excl_pos0",
                                      "_meta_matrix_entropy_raw_excl_pos0"], rtol=1e-6)
    assert base["_meta_position0_share"] < 0.5 < sunk["_meta_position0_share"]


@pytest.fixture(scope="module")
def tiny_identity_states(tiny):
    """tiny-random-gpt2 with exact identity blocks after blocks 1 and 3."""
    model, tok = tiny
    return inv.states_from_model(inv.insert_identity_blocks(model, [1, 3]), tok, TEXTS)


def test_layer_aggregates_are_meta(tiny_states, tiny_identity_states):
    """P4/P7: the per-layer dicts (absolute layer index) and their uniform
    means are _meta_; the means move under identity blocks while the post-norm
    headline does not."""
    base = inv.run_task_on_states(TASK, tiny_states)
    assert not {"mean_matrix_entropy", "mean_matrix_entropy_raw",
                "layer_matrix_entropies", "layer_matrix_entropies_normalized"} & set(base)
    ins = inv.run_task_on_states(TASK, tiny_identity_states)
    inv.assert_invariant(base, ins, ["matrix_entropy", "matrix_entropy_raw"], rtol=1e-9)
    inv.assert_moves(base, ins, "_meta_mean_matrix_entropy_raw", 1e-4)


def test_own_forward_drops_an_appended_eos(tiny):
    """A tokenizer that appends EOS to every sequence (OLMo-1B's does): the
    task's own forward strips it, as the shared cache does, so the result
    equals the plain tokenizer's."""
    model, tok = tiny

    class AppendEOS:
        eos_token_id, sep_token_id = tok.eos_token_id, None

        def __call__(self, text, **kw):
            enc = tok(text, **kw)
            enc["input_ids"] = torch.cat([enc["input_ids"],
                                          torch.tensor([[tok.eos_token_id]])], 1)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            return enc

    cfg = {"use_cache": False}
    plain = inv.run_task_on_model(TASK, model, tok, TEXTS, cfg, use_cache=False)
    eos = inv.run_task_on_model(TASK, model, AppendEOS(), TEXTS, cfg, use_cache=False)
    assert eos["matrix_entropy"] == plain["matrix_entropy"]
    assert eos["_meta_n_trailing_special_stripped"] == len(TEXTS)
    assert plain["_meta_n_trailing_special_stripped"] == 0



# --------------------------------------------------------------------------
# Declarations and their direct tests
# --------------------------------------------------------------------------

from blme.tasks.geometry.matrix_entropy import MatrixEntropyTask  # noqa: E402

_WIDTH = {"zero_pad": lambda S: inv.zero_pad_docs(S, 2),
          "isometry": lambda S: inv.embed_isometric_docs(S, 2, 3),
          "rescale": lambda S: inv.rescale_docs(S, 2.0)}


@pytest.mark.parametrize("key", sorted(MatrixEntropyTask.SIZE_INVARIANT))
def test_size_invariant_entry(key, tiny_states, tiny_identity_states):
    """Raw H is unchanged by zero-padding, isometric embedding, rescaling and
    identity blocks; the headline H / ln d falls by exactly ln d / ln 2d under
    zero-padding (known negative)."""
    transforms, node = MatrixEntropyTask.SIZE_INVARIANT[key]
    assert node.endswith(f"test_size_invariant_entry[{key}]")
    base = inv.run_task_on_states(TASK, tiny_states)
    d = tiny_states.d
    for t in transforms:
        S = tiny_identity_states if t == "identity_blocks" else _WIDTH[t](tiny_states)
        other = inv.run_task_on_states(TASK, S)
        inv.assert_invariant(base, other, [key], rtol=1e-5)
        if t == "zero_pad":
            assert other["matrix_entropy"] / base["matrix_entropy"] == pytest.approx(
                math.log(d) / math.log(2 * d), rel=1e-5)


def test_declarations(tiny_states):
    cls = MatrixEntropyTask
    assert cls.TOKEN_POLICY.startswith("paper_protocol:")
    assert {"paper", "code", "llm_precedent"} <= set(cls.SOURCES)
    assert set(cls.SIZE_INVARIANT) <= set(cls.HEADLINE_KEYS)
    assert set(cls.HEADLINE_KEYS) <= set(inv.run_task_on_states(TASK, tiny_states))
