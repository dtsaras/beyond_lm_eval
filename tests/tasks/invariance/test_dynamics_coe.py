"""dynamics_coe: release arithmetic, generation protocol and depth invariance."""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics_fixtures import (dataset, inv, tiny_gpt2, token_batches,  # noqa: E402
                               tokenizer)

from blme.tasks.dynamics.coe import ChainOfEmbeddingTask, _coe_from_chain  # noqa: E402

_SCORE_PY = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser(
    "~/.cache/blme-refrepos")), "Chain-of-Embedding", "score.py")


def _release_score():
    if not os.path.exists(_SCORE_PY):
        pytest.skip(f"reference not available: {_SCORE_PY}")
    spec = importlib.util.spec_from_file_location("coe_release_score", _SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"reference failed to load: {exc}")
    return mod


def test_no_floor_beyond_the_release():
    """DECISIONS item 1: the chain whose Z_Mag (sqrt 2) sits ~6 orders below the
    float32 resolution of its coordinates (1e13) is scored as score.py scores it,
    and the margin shows it. Pre-fix BLME's relative floor returned NaN."""
    chain = [torch.tensor(v, dtype=torch.float64)
             for v in ([1.0, 0.0], [1e13, 1e13], [0.0, 1.0])]
    out = _coe_from_chain(chain, storage_eps=float(np.finfo(np.float32).eps))
    assert math.isfinite(out["coe_r"]) and math.isfinite(out["coe_c"])
    assert out["coe_r"] > 1e12
    assert out["z_mag_margin_log10"] == pytest.approx(
        math.log10(math.sqrt(2) / (float(np.finfo(np.float32).eps) * 1e13)), rel=1e-9)
    assert out["z_mag_margin_log10"] < 0
    ref = _release_score().CoEScoreInfo([c.numpy() for c in chain])
    assert out["coe_r"] == pytest.approx(ref.compute_CoE_R(), rel=1e-9)
    assert out["coe_c"] == pytest.approx(ref.compute_CoE_C(), rel=1e-9)


GEN_CFG = {"num_samples": 3, "max_new_tokens": 8}


def _run(model, tok, cfg=GEN_CFG):
    with inv.limit_threads(1):
        out = ChainOfEmbeddingTask(config=dict(cfg)).evaluate(model, tok, dataset())
    assert "error" not in out, out
    return out


def _reference_chains(model, tok, text, max_new_tokens):
    """The release's own route: generate(output_hidden_states=True), then per
    layer the mean over steps of hidden_states[step][layer][0, -1] (the state
    that emits each output token; the release's [0][0] at step 0 instead reads
    prompt position 0)."""
    from transformers import GenerationConfig
    ids = tok(text, return_tensors="pt")["input_ids"]
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                       generation_config=GenerationConfig(
                           do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id),
                       output_hidden_states=True, return_dict_in_generate=True)
    hs = g.hidden_states
    n_layers = len(hs[1])
    emit = [torch.stack([hs[t][j][0, -1].double() for t in range(len(hs))]).mean(0)
            for j in range(n_layers)]
    release = [torch.stack([hs[t][j][0, 0].double() for t in range(len(hs))]).mean(0)
               for j in range(n_layers)]
    return emit, release


def test_chain_pools_the_states_that_emit_generated_tokens():
    """Greedy generation, then per layer the mean over output tokens of the
    emitting state (paper Sec. 2.1). Checked against generate()'s own cached
    hidden states, the route inference.py takes; the [0][0] variant is the
    _meta_ disclosure. Pre-fix the task read the last prompt token only."""
    tok = tokenizer()
    model = tiny_gpt2()
    out = _run(model, tok)
    texts = [d["text"] for d in dataset()[:GEN_CFG["num_samples"]]]
    emit_r, rel_r = [], []
    with inv.limit_threads(1):
        for t in texts:
            emit, release = _reference_chains(model, tok, t, GEN_CFG["max_new_tokens"])
            emit_r.append(_coe_from_chain(emit)["coe_r"])
            rel_r.append(_coe_from_chain(release)["coe_r"])
    assert out["per_sample_coe_r"] == pytest.approx(emit_r, rel=1e-5, abs=1e-7)
    assert out["_meta_coe_r_release_pos0"] == pytest.approx(float(np.mean(rel_r)),
                                                            rel=1e-5, abs=1e-7)
    assert out["token_position"] == "generated_mean"
    assert out["_meta_n_generated_tokens_mean"] <= GEN_CFG["max_new_tokens"]
    assert out["coe_r"] != out["_meta_coe_r_last_prompt"]


# tiny gpt2 has 5 blocks; identity copies are inserted MID-STACK only: one
# appended after the last block would expose its raw output before ln_f as an
# extra chain point (HF's last entry is post-norm), which is not neutral.
MID_LAYOUTS = {"early": [0, 0], "late": [3, 3], "uniform": [0, 1, 2, 3]}


@pytest.mark.parametrize("where", sorted(MID_LAYOUTS))
def test_coe_r_sum_invariant_under_identity_blocks(where):
    """L * coe_r is bit-identical under mid-stack identity blocks; coe_r itself
    (Eq. 5's mean over L transitions) falls exactly as L / L'."""
    tok = tokenizer()
    model = tiny_gpt2()
    deeper = inv.insert_identity_blocks(model, MID_LAYOUTS[where],
                                        verify_on=token_batches(tok))
    base, other = _run(model, tok), _run(deeper, tok)
    inv.assert_invariant(base, other, ["coe_r_sum"], rtol=0.0)
    L0, L1 = base["_meta_chain_length_per_sample"], other["_meta_chain_length_per_sample"]
    assert L1 == L0 + len(MID_LAYOUTS[where])
    assert other["coe_r"] == pytest.approx(base["coe_r"] * L0 / L1, rel=1e-12)
    inv.assert_moves(base, other, "coe_r", 0.1)


def test_zero_normaliser_is_nan_as_in_the_release():
    """A chain that returns to its start: Z_Mag = 0 (score.py divides by zero)."""
    chain = [torch.tensor(v, dtype=torch.float64)
             for v in ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0])]
    out = _coe_from_chain(chain)
    assert math.isnan(out["coe_r"]) and math.isnan(out["coe_c"])
