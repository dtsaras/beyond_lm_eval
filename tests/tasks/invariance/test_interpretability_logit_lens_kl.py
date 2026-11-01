"""Mechanism tests for interpretability_logit_lens_kl, through the real evaluate().

An identity block maps x to x exactly (``insert_identity_blocks``), so its
decode equals the previous layer's and KL(p_l || p_l+1) = 0.0: the churn sum is
unchanged when one is inserted after any block. A leading identity block (before
block 0) would expose the embedding output as a new first decode, which the
release's block-output profile does not contain, so none is inserted there. The
normalized-depth AUC and the two depths are averages on (l + 1) / L and move.
"""
from __future__ import annotations

import copy

import pytest
import torch

import _family_d as fd

inv = fd.inv
TASK = "interpretability_logit_lens_kl"

from blme.tasks.interpretability.logit_lens_kl import LogitLensKLTask  # noqa: E402
from invariance import model_transforms as mt  # noqa: E402


@pytest.fixture(scope="module")
def texts():
    from blme.cache import load_default_corpus
    return [d["text"] for d in load_default_corpus(24)]


def _model(arch):
    return fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=4)


def _run(model, texts, **cfg):
    return inv.run_task_on_model(TASK, model, fd.tokenizer(), texts, config=cfg or None,
                                 use_cache=False)


def _verify_ids():
    tok = fd.tokenizer()
    return [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_churn_is_exact_under_identity_blocks(arch, texts):
    model = _model(arch)
    base = _run(model, texts)
    L0 = base["_meta_n_layers"]
    after = [0, 1, 1, L0 - 1]
    ins = _run(inv.insert_identity_blocks(model, after, verify_on=_verify_ids()), texts)
    assert ins["_meta_n_layers"] == L0 + len(after)
    # Same token rows on both sides: the sink masks agree.
    for k in ("_meta_n_tokens", "_meta_sink_n_massive_dropped"):
        assert ins[k] == base[k]
    assert base["kl_churn"] > 0
    inv.assert_invariant(base, ins, sorted(LogitLensKLTask.SIZE_INVARIANT), rtol=0.0)
    # Per transition: the original terms, bit for bit, and an exact 0.0 for
    # every transition into an inserted block.
    layout = inv.identity_layout(L0, after)
    expect = []
    for j in range(1, len(layout)):
        kind, i = layout[j]
        expect.append(0.0 if kind == "identity" else base["kl_next_profile"][i - 1])
    assert ins["kl_next_profile"] == expect


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_later_identity_blocks_give_zero_kl_from_that_depth(arch, texts):
    """Known positive: blocks k..L-1 made identity (residual-writing projections
    zeroed in place), so from block k-1 on every decode is the output's."""
    model = copy.deepcopy(_model(arch))
    a = mt._arch(model)
    blocks = a.get_blocks(model)
    L, k = len(blocks), 2
    for b in list(blocks)[k:]:
        mt._zero_(a.attn_out(b))
        mt._zero_(a.mlp_out(b))
    r = _run(model, texts)
    assert r["_meta_n_layers"] == L
    assert r["kl_to_final_profile"][k - 1:] == [0.0] * (L - k + 1)
    assert r["_meta_kl_final_to_layer_profile"][k - 1:] == [0.0] * (L - k + 1)
    assert r["kl_next_profile"][k - 1:] == [0.0] * (L - k)
    assert r["top1_agreement_profile"][k - 1:] == [1.0] * (L - k + 1)
    assert all(v > 0 for v in r["kl_to_final_profile"][:k - 1])
    # Every position is decided by block k-1 at the latest.
    assert r["first_agreement_depth"] <= r["decision_depth"] <= k / L
    assert r["kl_churn"] == pytest.approx(sum(r["kl_next_profile"][:k - 1]), rel=1e-12)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_auc_and_depths_move_under_identity_blocks(arch, texts):
    """Known negative. Appended identity blocks add zero-KL layers at the end:
    Hu et al.'s summed AUC is unchanged, the normalized AUC and the depths
    (s + 1) / L shrink by exactly L0 / L. Blocks inserted early duplicate the
    early, high-KL layers and raise the normalized AUC."""
    model = _model(arch)
    base = _run(model, texts)
    L0 = base["_meta_n_layers"]
    tail = _run(inv.insert_identity_blocks(model, [L0 - 1] * 3, verify_on=_verify_ids()), texts)
    s = L0 / (L0 + 3)
    assert tail["_meta_kl_to_final_auc_sum"] == base["_meta_kl_to_final_auc_sum"]
    for key in ("kl_to_final_auc", "decision_depth", "first_agreement_depth"):
        assert tail[key] == pytest.approx(base[key] * s, rel=1e-12)
        inv.assert_moves(base, tail, key, min_rel=0.3)
    head = _run(inv.insert_identity_blocks(model, [0, 0, 0], verify_on=_verify_ids()), texts)
    inv.assert_moves(base, head, "kl_to_final_auc", min_rel=0.1)
    assert head["kl_to_final_auc"] > base["kl_to_final_auc"]
    inv.assert_invariant(base, head, ["kl_churn"], rtol=0.0)


def test_decode_is_float32_and_float64_on_a_bf16_model(texts):
    """The lens runs in float32 (log-softmax in float64) at every size; only the
    final norm and head are promoted, and the model is handed back in bf16."""
    model = fd.tiny_gpt2().to(torch.bfloat16)
    r = _run(model, texts[:4])
    assert r["_meta_model_dtype"] == "bfloat16"
    assert r["_meta_decode_dtype"] == "float32"
    assert r["_meta_log_softmax_dtype"] == "float64"
    # ln_f weight + bias, and the head weight it shares with wte, per document.
    assert r["_meta_n_lens_params_promoted"] == 3 * r["_meta_n_docs"]
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert r["kl_to_final_profile"][-1] == 0.0
