"""dynamics_coe: every headline key through evaluate() against the release.

The chains are rebuilt the way the release's inference.py collects them
(greedy generate(output_hidden_states=True); per layer the mean over output
tokens of the emitting state, as tests/tasks/invariance/test_dynamics_coe.py
pins), then scored by Alsace08/Chain-of-Embedding score.py's own
CoEScoreInfo: compute_CoE_Ang's mean normalised angle, compute_CoE_R,
compute_CoE_C, L * CoE-R for coe_r_sum, and the raw per-transition
magnitudes / angles (the normalised ones times their normaliser). Skips
without the clone (fails under BLME_REQUIRE_REFS=1).
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))
from dynamics_fixtures import dataset, inv, tiny_gpt2, tokenizer  # noqa: E402

from blme.tasks.dynamics.coe import ChainOfEmbeddingTask  # noqa: E402

_SCORE_PY = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "Chain-of-Embedding" / "score.py"
CFG = {"num_samples": 4, "max_new_tokens": 8}


def _release():
    if not _SCORE_PY.is_file():
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_SCORE_PY}")
        pytest.skip(f"reference not available: {_SCORE_PY}")
    spec = importlib.util.spec_from_file_location("coe_release_score_values", _SCORE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _emit_chain(model, tok, text):
    from transformers import GenerationConfig
    ids = tok(text, return_tensors="pt")["input_ids"]
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                       generation_config=GenerationConfig(
                           do_sample=False, num_beams=1, max_new_tokens=CFG["max_new_tokens"],
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id),
                       output_hidden_states=True, return_dict_in_generate=True)
    hs = g.hidden_states
    return [torch.stack([hs[t][j][0, -1].double() for t in range(len(hs))]).mean(0).numpy()
            for j in range(len(hs[1]))]


def test_every_coe_key_matches_score_py():
    score = _release()
    model, tok = tiny_gpt2(), tokenizer()
    with inv.limit_threads(1):
        res = ChainOfEmbeddingTask(config=dict(CFG)).evaluate(model, tok, dataset())
        chains = [_emit_chain(model, tok, d["text"]) for d in dataset()[:CFG["num_samples"]]]
    assert "error" not in res, res
    ang, r, c, r_sum, mags, angs = [], [], [], [], [], []
    for h in chains:
        info = score.CoEScoreInfo(h)
        mag_n, _, _ = info.compute_CoE_Mag()
        ang_n, ang_ave, _ = info.compute_CoE_Ang()
        z_mag = np.linalg.norm(h[-1] - h[0])
        z_ang = math.acos(np.dot(h[-1], h[0]) / (np.linalg.norm(h[-1]) * np.linalg.norm(h[0])))
        ang.append(ang_ave)
        r.append(info.compute_CoE_R())
        c.append(info.compute_CoE_C())
        r_sum.append(len(mag_n) * info.compute_CoE_R())
        mags += [m * z_mag for m in mag_n]
        angs += [a * z_ang for a in ang_n]
    want = {"coe_r": np.mean(r), "coe_c": np.mean(c), "mean_normalized_angle": np.mean(ang),
            "coe_r_std": np.std(r), "coe_c_std": np.std(c), "coe_r_sum": np.mean(r_sum),
            "mean_magnitude_change": np.mean(mags), "std_magnitude_change": np.std(mags),
            "mean_angle_change": np.mean(angs), "std_angle_change": np.std(angs)}
    for key, v in want.items():
        assert res[key] == pytest.approx(v, rel=1e-5, abs=1e-7), key
