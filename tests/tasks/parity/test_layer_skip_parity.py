"""Parity: causality_layer_skip vs the RELEASED Csordas et al. analysis
(robertcsordas/llm_effective_depth@393e5c7).

The release traces with nnsight 0.4 on pre-5 transformers, where a decoder
layer returns a tuple; under transformers 5 a Llama layer returns a tensor, so
its ``layer.output[0]`` indexes the batch and the script cannot run here. The
test loads the release's own ``merge_io`` and ``get_future``
(analyze_future_effects.py) and ``effect_layer_score`` (skip_max_depth.py)
from the clone, and drives ``test_effect``'s loop, transcribed line for line,
with HF forward hooks standing in for nnsight's ``layer.inputs`` /
``layer.output``. Skips when the clone is absent (fails under
BLME_REQUIRE_REFS=1).
"""
import ast
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.causality import layer_skip as LS  # noqa: E402
from blme.tasks.common import get_layers  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "llm_effective_depth")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


@pytest.fixture(scope="module")
def release():
    """The released pure functions, compiled from the clone's source (the
    modules themselves import nnsight and matplotlib at top level)."""
    analysis = os.path.join(_REPO, "analysis")
    if not os.path.isdir(analysis):
        _missing(analysis)
    ns = {"torch": torch, "Optional": Optional}
    for fname, names in (("analyze_future_effects.py", {"merge_io", "get_future"}),
                         ("skip_max_depth.py", {"effect_layer_score"})):
        path = os.path.join(analysis, fname)
        tree = ast.parse(open(path).read())
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {d.name for d in defs} == names
        exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)
    return ns


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in LS.LayerSkipEffectTask.SOURCES["code"]


def _release_test_effect(model, ids, positions, F):
    """analyze_future_effects.test_effect(llm, prompt, positions, "layer")."""
    layers = get_layers(model)
    st = {"skip": None, "t": None, "log": []}

    def hook(i):
        def fn(module, args, kwargs, output):
            h_in = args[0] if args and torch.is_tensor(args[0]) else kwargs["hidden_states"]
            out = output[0] if isinstance(output, tuple) else output
            if st["skip"] == i:   # intervene_layer(part="layer")
                out = F["merge_io"](h_in, out, st["t"], 1)
            st["log"].append(out.detach().cpu().float() - h_in.detach().cpu().float())
            if st["skip"] == i:
                return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
        return fn

    handles = [b.register_forward_hook(hook(i), with_kwargs=True) for i, b in enumerate(layers)]
    gf = F["get_future"]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits
            residual_log = torch.cat(st["log"], dim=0)
            outputs = logits.detach().float().softmax(dim=-1).cpu()
            all_diffs, all_out_diffs = [], []
            for t in positions:
                diffs, out_diffs = [], []
                for lskip in range(len(layers)):
                    st.update(skip=lskip, t=t, log=[])
                    out_logits = model(input_ids=ids, use_cache=False).logits
                    new_logs = torch.cat(st["log"], dim=0).float()
                    relative_diffs = ((gf(residual_log, t) - gf(new_logs, t)).norm(dim=-1)
                                      / gf(residual_log, t).norm(dim=-1).clamp(min=1e-6))
                    diffs.append(relative_diffs.max(dim=-1).values)
                    out_diffs.append((gf(out_logits.detach(), t).float().softmax(dim=-1).cpu()
                                      - gf(outputs, t)).norm(dim=-1).max(dim=-1).values)
                all_diffs.append(torch.stack(diffs, dim=0))
                all_out_diffs.append(torch.stack(out_diffs, dim=0))
            dall = torch.stack(all_diffs, dim=0).max(dim=0).values
            dall_out = torch.stack(all_out_diffs, dim=0).max(dim=0).values
    finally:
        for h in handles:
            h.remove()
    return dall, dall_out


def _release_run(model, tok, texts, F, max_length, n_chunks=4):
    """run()/skip_max_depth main(): random.seed(123), per-prompt split points,
    element-wise max over prompts, the later-block reduce and the depth score."""
    state = random.getstate()
    random.seed(123)
    try:
        d_max, dout_max = torch.zeros([1]), torch.zeros([1])
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"]
            positions = list(range(8, ids.shape[1] - 4, 8))
            random.shuffle(positions)
            positions = positions[:n_chunks]
            diff_now, diff_out = _release_test_effect(model, ids, positions, F)
            d_max = torch.max(d_max, diff_now)
            dout_max = torch.max(dout_max, diff_out)
    finally:
        random.setstate(state)
    L = d_max.shape[0]
    reduce = d_max.sum(-1) / torch.arange(L - 1, -1, -1).clamp(min=1)
    return d_max, dout_max.squeeze(-1), reduce, F["effect_layer_score"]([reduce, dout_max])


def _texts(n):
    from blme.cache import load_default_corpus
    return [d["text"] for d in load_default_corpus(n)]


@pytest.mark.parametrize("n_tokens", [5, 12, 13, 20, 41, 64, 300, 512])
def test_split_positions_match_release(n_tokens):
    rng_ref = random.Random(123)
    rng = random.Random(123)
    for _ in range(3):
        ref = list(range(8, n_tokens - 4, 8))
        rng_ref.shuffle(ref)
        assert LS.split_positions(n_tokens, 4, rng) == ref[:4]


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_prompt_effects_match_release(release, arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=4)
    ids = fd.tokenizer()(_texts(1)[0], return_tensors="pt", truncation=True,
                         max_length=80)["input_ids"]
    positions = LS.split_positions(ids.shape[1], 4, random.Random(5))
    assert len(positions) == 4
    mat, out = LS.prompt_effects(model, get_layers(model), ids, positions)
    ref_mat, ref_out = _release_test_effect(model, ids, positions, release)
    torch.testing.assert_close(mat, ref_mat, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(out, ref_out.squeeze(-1), rtol=1e-6, atol=1e-9)
    # Blocks before the skipped one, and the skipped one itself on the later
    # tokens, are untouched: exactly 0 on and below the diagonal.
    assert torch.all(torch.tril(mat) == 0)


@pytest.mark.parametrize("arch", ["gpt2", "llama"])
def test_evaluate_matches_release(release, arch):
    model = fd.tiny_gpt2() if arch == "gpt2" else fd.tiny_llama(n_layers=4)
    tok, texts = fd.tokenizer(), _texts(3)
    r = LS.LayerSkipEffectTask({"num_samples": 3, "max_length": 96}).evaluate(
        model, tok, [{"text": t} for t in texts])
    d_max, dout, reduce, (score_future, score_output) = _release_run(model, tok, texts, release, 96)
    torch.testing.assert_close(torch.tensor(r["_meta_skip_future_effect_matrix"]), d_max,
                               rtol=1e-6, atol=1e-7)
    assert r["skip_output_effect_per_layer"] == pytest.approx(dout.tolist(), rel=1e-6, abs=1e-9)
    assert r["skip_future_effect_per_layer"] == pytest.approx(reduce.tolist(), rel=1e-6)
    # The paper's 1-based depth score is the release's 0-based one plus 1.
    assert r["skip_depth_score_future"] == pytest.approx(score_future + 1, rel=1e-6)
    assert r["skip_depth_score_output"] == pytest.approx(score_output + 1, rel=1e-6)
    L = r["_meta_n_layers"]
    assert r["skip_depth_score_output_ratio"] == pytest.approx((score_output + 1) / L, rel=1e-6)
    assert r["_meta_n_prompts"] == 3 and r["_meta_n_split_points"] == 12


def test_short_prompts_are_dropped_not_replaced():
    model, tok = fd.tiny_gpt2(), fd.tokenizer()
    texts = ["Too short.", *_texts(2)]
    r = LS.LayerSkipEffectTask({"num_samples": 2, "max_length": 64}).evaluate(
        model, tok, [{"text": t} for t in texts])
    assert r["_meta_n_prompts"] == 1 and r["_meta_n_prompts_too_short"] == 1


def _gpt2():
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    return model, tok


def test_gpt2_reproduces_the_papers_shape():
    """Known positive (Csordas et al. Sec. 3): the second half of the blocks
    matters less to later computation than the first; gpt2's block 0 (its
    extended embedding) dominates the output; the last block has no effect on
    later tokens, by construction."""
    model, tok = _gpt2()
    r = LS.LayerSkipEffectTask({"num_samples": 4, "max_length": 256}).evaluate(
        model, tok, [{"text": t} for t in _texts(4)])
    e, out = r["skip_future_effect_per_layer"], r["skip_output_effect_per_layer"]
    half = len(e) // 2
    assert sum(e[:half]) > 2 * sum(e[half:])
    assert max(out) == out[0]
    assert e[-1] == 0.0 and out[-1] == 0.0
    assert r["skip_depth_score_future_ratio"] < 0.5
