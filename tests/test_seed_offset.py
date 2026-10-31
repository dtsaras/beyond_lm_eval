"""``seed_offset`` moves every hard-coded random stream behind a paper-core headline.

Each listed task adds the config key ``seed_offset`` (default 0) to the
constant seed of such a stream. Two checks per task, through the real
``evaluate()``: ``seed_offset=0`` reproduces a run without the key exactly,
and ``seed_offset=1`` moves at least one HEADLINE_KEY, so the stream is live.
"""
from __future__ import annotations

import functools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

transformers = pytest.importorskip("transformers")
pytest.importorskip("sklearn")

import blme  # noqa: E402
from invariance.driver import run_task_on_model, run_task_on_states  # noqa: E402

CHAT = ("{% for m in messages %}<|user|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
_WORDS = ("the river bank market story light water stone city night road "
          "garden window music paper").split()


def _texts(n, length=10, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(_WORDS, size=length)).capitalize() + "." for _ in range(n)]


def _states(n_docs, T, d, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    return {li: [rng.standard_normal((T, d)).astype(np.float32) for _ in range(n_docs)]
            for li in range(n_layers)}


@functools.lru_cache(maxsize=None)
def _tiny():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257,
                                  n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    return model, transformers.GPT2TokenizerFast.from_pretrained("gpt2")


def _on_model(task, texts, base=None):
    def run(extra):
        model, tok = _tiny()
        return run_task_on_model(task, model, tok, texts, {**(base or {}), **extra},
                                 use_cache=False)
    return run


def _on_states(task, states, base=None):
    def run(extra):
        return run_task_on_states(task, states, {**(base or {}), **extra})
    return run


@functools.lru_cache(maxsize=None)
def _refusal_splits():
    """A ``refusal_splits_dir`` in the reference layout, 12 train / 8 val per class."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="blme_refusal_splits_"))
    for c, kind in enumerate(("harmful", "harmless")):
        for s, (split, n) in enumerate((("train", 12), ("val", 8))):
            rows = [{"instruction": f"{kind} {split} {i}: {t}"}
                    for i, t in enumerate(_texts(n, 6, seed=10 * c + s))]
            (d / f"{kind}_{split}.json").write_text(json.dumps(rows))
    return str(d)


def _refusal(extra):
    from blme.tasks.config_loader import resolve_task_config
    from blme.tasks.repe.refusal import RefusalDirectionTask

    class _Loose(RefusalDirectionTask):
        # Filters opened so a random model keeps a candidate
        # (as tests/tasks/test_refusal_direction_zero_final.py).
        KL_THRESHOLD, INDUCE_THRESHOLD = 1e9, -1e9

    model, _ = _tiny()
    tok = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    tok.chat_template = CHAT
    cfg = resolve_task_config("repe_refusal_direction", {
        "refusal_splits_dir": _refusal_splits(), "n_train": 4, "n_val": 3, "seed": 42,
        **extra})
    with torch.no_grad():
        return _Loose(config=cfg).evaluate(model, tok, None)


RUNNERS = {
    # > 500 sequences, so the 500-query draw is a strict subset.
    "geometry_lid": _on_states("geometry_lid", _states(600, 3, 8, 2)),
    # 4d+1 = 33 of ~440 kept rows enter the fixed-gamma headline.
    "geometry_mp_bulk_deviation": _on_states(
        "geometry_mp_bulk_deviation", _states(40, 12, 8, 3)),
    "geometry_intrinsic_dim": _on_model(
        "geometry_intrinsic_dim", _texts(2), {"layerwise": False, "sample_size": 3000}),
    "geometry_contextualization": _on_model("geometry_contextualization", _texts(40)),
    "geometry_prediction_alignment": _on_model("geometry_prediction_alignment", _texts(12)),
    "geometry_categories": _on_model("geometry_categories", _texts(2)),
    "geometry_mahalanobis": _on_model("geometry_mahalanobis", _texts(40, 12),
                                      {"id_corpus": "caller"}),
    "repe_refusal_direction": _refusal,
}


@functools.lru_cache(maxsize=None)
def _result(task, offset):
    res = RUNNERS[task]({} if offset is None else {"seed_offset": offset})
    assert isinstance(res, dict) and "error" not in res, (task, res)
    return res


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.integer):
        return int(x)
    return x


def _headline_values(task, res):
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    _register_all_tasks()
    out = {}
    for key in get_task(task).HEADLINE_KEYS:
        pat = re.compile(re.sub(r"\\\{\w+\\\}", r"[^.]+", re.escape(key)))
        out.update({k: v for k, v in res.items() if pat.fullmatch(k)})
    return out


def test_imports_the_tree_under_test():
    assert Path(blme.__file__).resolve().is_relative_to(SRC.resolve())


@pytest.mark.parametrize("task", sorted(RUNNERS))
def test_offset_zero_is_the_default(task):
    assert _canon(_result(task, 0)) == _canon(_result(task, None))


@pytest.mark.parametrize("task", sorted(RUNNERS))
def test_offset_one_moves_a_headline(task):
    base = _headline_values(task, _result(task, None))
    moved = _headline_values(task, _result(task, 1))
    finite = {k for k, v in base.items()
              if isinstance(v, (int, float)) and math.isfinite(v)}
    assert finite, f"{task}: no finite headline value to compare"
    changed = sorted(k for k in finite if _canon(base[k]) != _canon(moved.get(k)))
    assert changed, f"{task}: seed_offset=1 moved no headline key"
