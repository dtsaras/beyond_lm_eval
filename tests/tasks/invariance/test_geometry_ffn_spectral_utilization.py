"""geometry_ffn_spectral_utilization under size transforms.

* Identity blocks (the known negative): an inserted copy of block i runs its
  MLP on block i's output, not on its input, so it adds a new layer to every
  profile. The original layers' entries are bit-identical and the medians
  move: nothing is invariant under insertion, hence SIZE_INVARIANT = {}.
* FFN-width zero-padding (dead neurons): the raw hard and soft ranks and eDim
  (D cancels in 1 + (D - 1) SUI) stay to float rounding; everything divided
  by D moves.
* Residual-width zero-padding (Llama): d_ff is unchanged, and so is every key.
"""
from __future__ import annotations

import copy

import pytest
import torch

import _family_d as fd

inv = fd.inv

from invariance.model_transforms import zero_pad_width  # noqa: E402
from blme.tasks.geometry.ffn_spectral import FFNSpectralUtilizationTask  # noqa: E402

CFG = {"use_dataset": True, "max_length": 64, "n_tokens": 10 ** 6}
PROFILES = ("ffn_hard_rank_profile", "ffn_soft_rank_profile", "ffn_sui_profile",
            "ffn_spectral_concentration_profile")


def _model(arch):
    return fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=3)


def _run(model, tok):
    with inv.limit_threads(1):
        return FFNSpectralUtilizationTask(CFG).evaluate(model, tok, fd.dataset())


def _ids(tok):
    return [tok(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]


def zero_pad_ffn_width(model, k=2):
    """FFN width d_ff -> k d_ff with dead neurons (GPT-2 and Llama). The extra
    c_fc / gate and up outputs have zero weights and biases, so their
    post-activations are gelu_new(0) = 0 and SiLU(0) * 0 = 0, and the down
    projection gets zero input columns. Function-preserving to float rounding."""
    cfg = copy.deepcopy(model.config)
    if cfg.model_type == "gpt2":
        cfg.n_inner = k * (cfg.n_inner or 4 * cfg.n_embd)
    else:
        cfg.intermediate_size = k * cfg.intermediate_size
    new = type(model)(cfg).eval()
    src = model.state_dict()
    with torch.no_grad():
        for name, p in new.state_dict().items():
            s = src[name]
            p.zero_()
            p[tuple(slice(0, n) for n in s.shape)] = s
    return new


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_identity_blocks_add_new_layers_so_nothing_is_invariant(arch):
    model, tok = _model(arch), fd.tokenizer()
    after = [0, 0, 0]
    deep = inv.insert_identity_blocks(model, after, verify_on=_ids(tok))
    base, ins = _run(model, tok), _run(deep, tok)
    n = len(base["ffn_sui_profile"])
    layout = inv.identity_layout(n, after)
    orig = [j for j, (kind, _) in enumerate(layout) if kind == "orig"]
    new = [j for j, (kind, _) in enumerate(layout) if kind == "identity"]
    for k in PROFILES:
        assert len(ins[k]) == n + len(after)
        assert [ins[k][j] for j in orig] == base[k], k
        for j in new:
            assert all(abs(ins[k][j] - v) > 1e-6 * abs(v) for v in base[k]), (k, j)
    assert FFNSpectralUtilizationTask.SIZE_INVARIANT == {}
    for key in FFNSpectralUtilizationTask.HEADLINE_KEYS:
        inv.assert_moves(base, ins, key, 1e-6)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_raw_ranks_survive_ffn_width_padding_and_sui_does_not(arch):
    model, tok = _model(arch), fd.tokenizer()
    wide = zero_pad_ffn_width(model, k=2)
    assert inv.max_logit_delta(model, wide, _ids(tok)) < 1e-5
    base, pad = _run(model, tok), _run(wide, tok)
    assert pad["_meta_ffn_width_profile"] == [2 * w for w in base["_meta_ffn_width_profile"]]
    assert pad["_meta_n_nonzero_eigenvalues_profile"] == base["_meta_n_nonzero_eigenvalues_profile"]
    inv.assert_invariant(base, pad, [
        "ffn_hard_rank_median", "ffn_soft_rank_median", "ffn_edim_median",
        "ffn_hard_rank_profile", "ffn_soft_rank_profile", "ffn_edim_profile",
        "_meta_ffn_spectral_centroid_profile",
        "_meta_ffn_hard_rank_median_without_position0",
        "_meta_ffn_soft_rank_median_without_position0"], rtol=1e-5)
    for key in ("ffn_sui_median", "ffn_soft_rank_normalized_median",
                "ffn_hard_rank_normalized_median", "ffn_spectral_concentration_median"):
        inv.assert_moves(base, pad, key, 0.05)


def test_every_key_survives_residual_width_padding():
    model, tok = fd.tiny_llama(n_layers=3), fd.tokenizer()
    wide = zero_pad_width(model, k=4, verify_on=_ids(tok), atol=1e-4)
    base, pad = _run(model, tok), _run(wide, tok)
    assert pad["_meta_ffn_width_profile"] == base["_meta_ffn_width_profile"]
    inv.assert_invariant(base, pad, sorted(k for k in base if k.startswith("ffn_")), rtol=1e-5)
