"""geometry_predictive_information: a token-embedding-only Y profile (GPT-2's
Y carries wpe), and special tokens are never targets (OLMo's tokenizer
appends EOS)."""
import math
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.geometry import predictive_information as PI  # noqa: E402

CFG = {"num_samples": len(fd.TEXTS)}


class _AppendsEos:
    """The tiny gpt2 tokenizer, appending its EOS as OLMo's tokenizer does."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, text, **kw):
        enc = self._tok(text, **kw)
        enc["input_ids"] = list(enc["input_ids"]) + [self._tok.eos_token_id]
        return enc


def _run(model, tok):
    return PI.PredictiveInformationTask(config=CFG).evaluate(model, tok, fd.dataset())


def test_appended_eos_is_never_a_target():
    """With the EOS excluded, the candidate targets are those of the plain
    tokenizer, so the draws and every value are the same bit for bit."""
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    plain = _run(model, tok)
    eos = _run(model, _AppendsEos(tok))
    assert eos["predictive_info_profile"] == plain["predictive_info_profile"]
    assert eos["_meta_n_special_target_positions_excluded"] == len(fd.TEXTS)
    assert plain["_meta_n_special_target_positions_excluded"] == 0


def test_token_embedding_profile_drops_gpt2_position_embedding():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    r = _run(model, tok)
    assert r["_meta_y_is_token_embedding"] is False
    prof = r["_meta_predictive_info_profile_token_embedding"]
    assert len(prof) == len(r["predictive_info_profile"]) and all(map(math.isfinite, prof))
    assert prof != r["predictive_info_profile"]
    # H(Y) of the token-only readout, from wte rows at the same targets.
    enc = [tok(t["text"], truncation=True, max_length=512)["input_ids"] for t in fd.dataset()]
    tgt = PI.split_points([len(e) for e in enc], 0)
    wte = model.transformer.wte.weight.detach().float()
    Y = torch.stack([wte[e[t]] for e, t in zip(enc, tgt) if t is not None])
    assert r["_meta_H_Y_token_embedding"] == PI.gram_and_entropy(Y)[1]


def test_without_position_embedding_the_twin_is_the_profile():
    r = _run(fd.tiny_llama(), fd.tokenizer())
    assert r["_meta_y_is_token_embedding"] is True
    assert r["_meta_predictive_info_profile_token_embedding"] == r["predictive_info_profile"]


def test_split_points_unchanged_when_every_position_is_allowed():
    lengths = [5, 9, 30, 100, 8, 64]
    allowed = [list(range(PI._MIN_CONTEXT, n)) for n in lengths]
    import random
    rng = random.Random(3)
    old = [rng.randint(PI._MIN_CONTEXT, n - 1) if n > PI._MIN_CONTEXT else None for n in lengths]
    assert PI.split_points(lengths, 3) == old
    assert PI.split_points(lengths, 3, allowed=allowed) == old
