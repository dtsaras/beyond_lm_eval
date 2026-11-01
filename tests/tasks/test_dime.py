"""geometry_dime: known positives, the data rule, RNG isolation and the
optional nlpaug import."""
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from _nlpaug_side import require_nlpaug  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks.geometry import dime as DM  # noqa: E402


def test_known_positive_on_clouds():
    """Identical views share every neighbourhood, noisy views most, unrelated
    views none: DiME orders them and is about 0 for the last, within log N."""
    g = torch.Generator().manual_seed(0)
    n, d = 400, 512
    w = torch.randn(4, d, generator=g, dtype=torch.float64)
    a = torch.randn(n, 4, generator=g, dtype=torch.float64) @ w
    close = a + 0.3 * torch.randn(n, 4, generator=g, dtype=torch.float64) @ w
    unrelated = torch.randn(n, 4, generator=g, dtype=torch.float64) @ w
    (same,), _, _ = DM.dime_profiles([a], [a.clone()], 10, 0)
    (near,), _, _ = DM.dime_profiles([a], [close], 10, 0)
    (none,), _, _ = DM.dime_profiles([a], [unrelated], 10, 0)
    assert math.log(n) > same > near > 0.1 and abs(none) < 0.01
    assert same > 0.5


def _texts(n):
    return [d["text"] for d in load_default_corpus(n)]


def test_known_positive_through_evaluate(monkeypatch):
    """The views of a prompt must land in the same row: unaugmented views (the
    same text twice) give a clearly positive DiME at every hidden state, and
    pairing each passage with the next one gives about 0."""
    require_nlpaug()
    texts = _texts(40)
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cfg = {"num_samples": 40, "max_length": 64}
    monkeypatch.setattr(DM, "augment_view", lambda aug, text, seed, i, v: text)
    same = DM.DiMETask(cfg).evaluate(model, tok, [{"text": t} for t in texts])
    nxt = {t: texts[(i + 1) % len(texts)] for i, t in enumerate(texts)}
    monkeypatch.setattr(DM, "augment_view", lambda aug, text, seed, i, v: nxt[text] if v else text)
    shifted = DM.DiMETask(cfg).evaluate(model, tok, [{"text": t} for t in texts])
    assert all(x > 3 * abs(y) for x, y in zip(same["dime_profile"], shifted["dime_profile"]))
    assert max(abs(y) for y in shifted["dime_profile"]) < 0.02
    assert same["_meta_n_views_augmented"] == 0


def test_dropped_passages_are_not_refilled():
    """The first num_samples passages, short ones dropped: a longer corpus
    changes nothing, and each view is seeded by its passage index."""
    require_nlpaug()
    texts = _texts(40)
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    cfg = {"num_samples": 16, "max_length": 64, "min_tokens": 200}
    kept = sum(len(tok(t, add_special_tokens=False)["input_ids"]) >= 200 for t in texts[:16])
    assert 2 <= kept < 16
    short = DM.DiMETask(cfg).evaluate(model, tok, [{"text": t} for t in texts[:16]])
    long = DM.DiMETask(cfg).evaluate(model, tok, [{"text": t} for t in texts])
    assert short == long
    assert short["_meta_n_prompts"] == kept and short["_meta_n_too_short"] == 16 - kept


def test_evaluate_leaves_the_global_rngs_alone():
    require_nlpaug()
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    r = DM.DiMETask({"num_samples": 12, "max_length": 64}).evaluate(
        model, tok, [{"text": t} for t in _texts(12)])
    assert "error" not in r
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert after[0] == before[0]
    assert all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(after[1], before[1]))
    assert torch.equal(after[2], before[2])


def test_seeds_are_disclosed_and_live():
    require_nlpaug()
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    data = [{"text": t} for t in _texts(24)]
    base = DM.DiMETask({"num_samples": 24, "max_length": 64}).evaluate(model, tok, data)
    moved = DM.DiMETask({"num_samples": 24, "max_length": 64, "seed_offset": 1}).evaluate(
        model, tok, data)
    assert (base["_meta_augment_seed"], base["_meta_perm_seed"]) == (0, 0)
    assert (moved["_meta_augment_seed"], moved["_meta_perm_seed"]) == (1, 1)
    assert moved["dime_peak"] != base["dime_peak"]


def test_blme_imports_without_nlpaug():
    code = ("import sys; sys.modules['nlpaug'] = None; import blme.tasks.geometry.dime; "
            "assert sys.modules['nlpaug'] is None")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_missing_nlpaug_is_an_error_result(monkeypatch):
    monkeypatch.setitem(sys.modules, "nlpaug", None)
    r = DM.DiMETask({}).evaluate(fd.tiny_gpt2(), fd.tokenizer(), [{"text": "x"}])
    assert r == {"error": DM.NLPAUG_MISSING}
