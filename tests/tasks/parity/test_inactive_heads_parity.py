"""Parity: interpretability_inactive_heads vs the RELEASED inactive-heads code
(psandovalsegura/inactive-heads@9dd884c).

The release patches the attention classes of transformers 4.48: its
monkey_patch_llama.py imports FlashAttentionKwargs from modeling_llama, which
transformers 5 no longer exports, so the modules cannot be imported here. The
test compiles the release's own ``my_eager_attention_forward``
(monkey_patch_llama.py) and ``get_model_sizes``, ``get_model_accessors``,
``get_dormant_proportion`` and ``get_head_scores`` (monkey_patch_utils.py)
from the clone, imports monkey_patch_head_types.py (HonorHeads,
x_intercept_for_cdf) as is, and runs the release's attention function as the
model's eager attention, which is what its patched classes call. Transcribed:
the module attributes PaddingSaverHFLM sets (pad_idxs, fake_batch), the GPT-2
scaling the release's caller would pass, the NeoX accessor entry the release's
tables lack, and create-CDF.py's two CDF lines (top-level script statements).
Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).
"""
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.common import get_layers  # noqa: E402
from blme.tasks.interpretability import inactive_heads as IH  # noqa: E402

_ROOT = os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))
_REPO = os.path.join(_ROOT, "inactive-heads")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


def _compile(fname, names, ns):
    path = os.path.join(_REPO, fname)
    tree = ast.parse(open(path).read())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {d.name for d in defs} == set(names)
    exec(compile(ast.Module(body=defs, type_ignores=[]), path, "exec"), ns)


@pytest.fixture(scope="module")
def release():
    path = os.path.join(_REPO, "monkey_patch_head_types.py")
    if not os.path.isfile(path):
        _missing(path)
    spec = importlib.util.spec_from_file_location("monkey_patch_head_types", path)
    ht = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ht)
    from transformers import GPT2LMHeadModel, LlamaForCausalLM, Olmo2ForCausalLM, Qwen2ForCausalLM
    from transformers.models.llama.modeling_llama import repeat_kv
    ns = {"torch": torch, "nn": nn, "Optional": Optional, "repeat_kv": repeat_kv,
          "GPT2LMHeadModel": GPT2LMHeadModel, "LlamaForCausalLM": LlamaForCausalLM,
          "Olmo2ForCausalLM": Olmo2ForCausalLM, "Qwen2ForCausalLM": Qwen2ForCausalLM}
    ns.update({k: getattr(ht, k) for k in dir(ht) if not k.startswith("_")})
    _compile("monkey_patch_llama.py", ["my_eager_attention_forward"], ns)
    _compile("monkey_patch_utils.py", ["get_model_sizes", "get_model_accessors",
                                       "get_dormant_proportion", "get_head_scores"], ns)
    return types.SimpleNamespace(ht=ht, **ns)


def test_sources_pin_the_cloned_commit():
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ("@" + head) in IH.InactiveHeadsTask.SOURCES["code"]


def _release_cdf(scores):
    """create-CDF.py: the sorted flattened scores against (1..n) / n."""
    head_scores = scores.flatten().float().cpu()
    sorted_scores = torch.sort(head_scores).values
    return torch.stack((sorted_scores, torch.arange(1, len(sorted_scores) + 1) / len(sorted_scores)),
                       dim=1)


# --------------------------------------------------------------------------
# The release's functions on synthetic inputs
# --------------------------------------------------------------------------

def test_honor_score_matches_the_release_attention(release):
    """A padded GQA batch through the release's HonorHeads branch against the
    per-sequence score on the unpadded o_proj input."""
    g = torch.Generator().manual_seed(0)
    B, H, KV, S, D = 3, 6, 2, 11, 8
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, KV, S, D, generator=g)
    v = torch.randn(B, KV, S, D, generator=g) * torch.tensor([0.05, 1.0]).view(1, KV, 1, 1)
    mask = torch.full((S, S), torch.finfo(torch.float32).min).triu(1).view(1, 1, S, S)
    lengths = [S, S - 3, 4]
    mod = types.SimpleNamespace(num_key_value_groups=H // KV, training=False, fake_batch=False,
                                pad_idxs=lengths, head_scores=[], dormant_masks=[])
    tau = 0.6
    out, _ = release.my_eager_attention_forward(
        mod, q, k, v, mask, scaling=D ** -0.5, track_head_type=release.HonorHeads(tau),
        record_head_scores=True)
    ref = mod.head_scores[0]
    x = out.reshape(B, S, H * D)               # what o_proj receives
    ours = torch.cat([IH.honor_scores(IH.head_output_norms(x[b:b + 1, :n], H, D).mean(dim=-2))
                      for b, n in enumerate(lengths)])
    assert float((ours - ref).abs().max()) < 1e-6
    # The quiet value head makes its query heads inactive on some sequences.
    assert torch.equal(ours < tau, mod.dormant_masks[0])
    assert mod.dormant_masks[0].any() and not mod.dormant_masks[0].all()


def test_cdf_thresholds_match_the_release(release):
    g = torch.Generator().manual_seed(1)
    layers = [torch.rand(50, 4, generator=g) * 2 for _ in range(3)]
    layers[1][:10] = 0.75                         # ties: flat CDF steps
    ref_cdf = _release_cdf(torch.cat(layers))
    ref = [float(release.ht.x_intercept_for_cdf(ref_cdf, p)) for p in IH.CDF_SHARES]
    assert IH.cdf_thresholds(layers) == ref
    assert ref[0] == float(torch.cat(layers).min())
    for y in (-0.5, 0.0, 0.004, 0.2, 0.31, 0.999, 1.0, 2.0):
        assert float(IH.x_intercept_for_cdf(ref_cdf, y)) == float(
            release.ht.x_intercept_for_cdf(ref_cdf, y))


# --------------------------------------------------------------------------
# evaluate() against the release run as the model's attention
# --------------------------------------------------------------------------

_NEOX_ACCESSORS = (lambda m: m.gpt_neox.layers, lambda layer: layer.attention)


def _release_run(release, model, texts, tok, tau):
    """Run every text through the model with the release's attention function
    as its eager attention; return the release's head scores, model_proportion
    and per-layer proportions."""
    import transformers.models.gpt2.modeling_gpt2 as m_gpt2
    import transformers.models.gpt_neox.modeling_gpt_neox as m_neox
    import transformers.models.llama.modeling_llama as m_llama
    family = {"gpt2": m_gpt2, "gpt_neox": m_neox, "llama": m_llama}[model.config.model_type]
    ns = dict(vars(release))
    if model.config.model_type == "gpt_neox":
        cfg = model.config
        ns["get_model_accessors"] = lambda m: _NEOX_ACCESSORS
        ns["get_model_sizes"] = lambda m: (cfg.num_hidden_layers, cfg.num_attention_heads,
                                           cfg.num_attention_heads,
                                           cfg.hidden_size // cfg.num_attention_heads,
                                           cfg.hidden_size)
        for f in ("get_dormant_proportion", "get_head_scores"):
            fn = release.__dict__[f]
            ns[f] = types.FunctionType(fn.__code__, ns, f, fn.__defaults__)
    get_layers_fn, get_attn_fn = ns["get_model_accessors"](model)
    attns = [get_attn_fn(layer) for layer in get_layers_fn(model)]
    head = release.HonorHeads(tau)

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        if scaling is None:                       # GPT-2 scales inside its eager function
            scaling = value.size(-1) ** -0.5 if module.scale_attn_weights else 1.0
        return release.my_eager_attention_forward(
            module, query, key, value, attention_mask, scaling=scaling, track_head_type=head,
            zero_track_head_type=False, record_head_scores=True, dropout=dropout, **kw)

    for a in attns:
        a.head_scores, a.dormant_masks, a.fake_batch = [], [], False
        if not hasattr(a, "num_key_value_groups"):
            a.num_key_value_groups = 1
    orig = family.eager_attention_forward
    family.eager_attention_forward = attention
    try:
        with torch.no_grad():
            for t in texts:
                ids = tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
                for a in attns:
                    a.pad_idxs = [ids.shape[1]]
                model(input_ids=ids, use_cache=False)
    finally:
        family.eager_attention_forward = orig
    n_heads = ns["get_model_sizes"](model)[1]
    # evaluate_attention_heads_drop.py: layers_to_proportion, model_proportion.
    per_layer = [torch.cat(a.dormant_masks, dim=0).sum().item() / (len(texts) * n_heads)
                 for a in attns]
    return (ns["get_head_scores"](model), float(np.mean(per_layer)), per_layer,
            ns["get_dormant_proportion"](model))


_MODELS = {"gpt2": fd.tiny_gpt2, "llama": fd.tiny_llama, "neox": fd.tiny_neox}


@pytest.mark.parametrize("arch", sorted(_MODELS))
@pytest.mark.parametrize("tau", [0.5, 1.0])
def test_evaluate_matches_the_release(release, arch, tau):
    """gpt2 (Conv1D c_proj), Llama (GQA o_proj) and NeoX (dense): the scores,
    the fraction, the per-layer shares and the CDF thresholds."""
    model, tok = _MODELS[arch](), fd.tokenizer()
    texts = list(fd.TEXTS)
    ref_scores, ref_fraction, ref_per_layer, ref_f32 = _release_run(release, model, texts, tok, tau)
    targets, _, _ = IH.attention_targets(model, get_layers(model))
    norms, _ = IH.collect_head_norms(model, tok, texts, targets, 512)
    ours = torch.cat([IH.honor_scores(a) for a in norms])
    assert ours.shape == ref_scores.shape
    assert float((ours - ref_scores).abs().max()) < 1e-6

    r = IH.InactiveHeadsTask({"num_samples": len(texts), "tau": tau}).evaluate(
        model, tok, [{"text": t} for t in texts])
    assert r["inactive_head_fraction"] == pytest.approx(ref_fraction, abs=1e-15)
    assert r["inactive_head_fraction_per_layer"] == pytest.approx(ref_per_layer, abs=1e-15)
    # get_dormant_proportion: the same count divided in float32.
    assert r["inactive_head_fraction"] == pytest.approx(ref_f32, abs=1e-7)
    ref_thr = [float(release.ht.x_intercept_for_cdf(_release_cdf(ref_scores), p))
               for p in IH.CDF_SHARES]
    assert r["_meta_cdf_thresholds"] == pytest.approx(ref_thr, abs=1e-6)
    assert r["inactive_head_threshold_5pct"] == r["_meta_cdf_thresholds"][1]
    if tau == 1.0:                                # a non-trivial split of the heads
        assert 0.05 < r["inactive_head_fraction"] < 0.95


def _notebook_honor_rows():
    """The notebook's stored Figure-4 data: per model, (threshold, % zeroed,
    MMLU accuracy) for the HONOR score."""
    nb = json.load(open(os.path.join(_REPO, "plot-accuracy-vs-percent-of-heads-zeroed.ipynb")))
    text = "".join("".join(o.get("text", "")) for o in nb["cells"][7].get("outputs", []))
    rows = re.findall(r"\[Model=(.*?), Task=mmlu, HeadType=HONOR, Len=7\] (.*)", text)
    return nb, {m: ast.literal_eval(v) for m, v in rows}


def test_default_tau_is_the_release_operating_point():
    """The shipped tau against the release's own results: the threshold at
    which zeroing HONOR heads leaves MMLU 1 point under baseline (Table 2's
    criterion), read with the notebook's get_intersection. The baseline is the
    HONOR run at its 0% threshold (the notebook reads RandomHeads:0.0.json,
    whose accuracy is not stored in the outputs)."""
    if not os.path.isdir(_REPO):
        _missing(_REPO)
    import yaml
    nb, rows = _notebook_honor_rows()
    assert len(rows) == 14
    ns = {}
    exec("".join(nb["cells"][5]["source"]), ns)
    taus = []
    for values in rows.values():
        ts, xs, ys = zip(*sorted(values, key=lambda v: v[1]))
        taus.append(ns["get_intersection"](ts, ys, values[0][2] - 1))
    taus = np.array(taus)
    bulk = taus[(taus > 0.45) & (taus < 0.58)]
    assert len(bulk) == 13
    assert abs(np.median(taus) - 0.5379) < 1e-3
    defaults = Path(IH.__file__).resolve().parents[1] / "configs" / "defaults.yaml"
    tau = yaml.safe_load(defaults.read_text())["interpretability_inactive_heads"]["tau"]
    assert bulk.min() < tau < bulk.max()


# --------------------------------------------------------------------------
# Architecture coverage: the hook reads each family's attention output
# --------------------------------------------------------------------------

def _tiny(kind):
    t = transformers
    common = dict(vocab_size=1000, hidden_size=32, intermediate_size=64,
                  max_position_embeddings=128, attn_implementation="eager")
    torch.manual_seed(0)
    if kind == "gpt2":
        return fd.tiny_gpt2()
    if kind == "neox":
        return fd.tiny_neox()
    if kind == "llama_gqa":
        return fd.tiny_llama()
    if kind == "qwen2":
        return t.Qwen2ForCausalLM(t.Qwen2Config(num_hidden_layers=2, num_attention_heads=4,
                                                num_key_value_heads=2, **common))
    if kind == "qwen3":
        return t.Qwen3ForCausalLM(t.Qwen3Config(num_hidden_layers=2, num_attention_heads=4,
                                                num_key_value_heads=2, head_dim=8, **common))
    if kind == "olmo2":
        return t.Olmo2ForCausalLM(t.Olmo2Config(num_hidden_layers=2, num_attention_heads=4,
                                                num_key_value_heads=4, **common))
    if kind == "gemma2":     # head_dim * heads = 48 != hidden_size = 32
        return t.Gemma2ForCausalLM(t.Gemma2Config(num_hidden_layers=2, num_attention_heads=2,
                                                  num_key_value_heads=1, head_dim=24, **common))
    if kind == "gemma3":
        return t.Gemma3ForCausalLM(t.Gemma3TextConfig(
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=24,
            sliding_window=4, layer_types=["sliding_attention", "sliding_attention",
                                           "full_attention"], **common))
    if kind == "phi3":
        return t.Phi3ForCausalLM(t.Phi3Config(num_hidden_layers=2, num_attention_heads=4,
                                              num_key_value_heads=2, pad_token_id=0, **common))
    if kind == "qwen3_5":    # two linear-attention layers, then two gated softmax layers
        return t.Qwen3_5ForCausalLM(t.Qwen3_5TextConfig(
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            use_cache=False, layer_types=["linear_attention", "linear_attention",
                                          "full_attention", "full_attention"], **common))
    raise KeyError(kind)


_KINDS = ("gpt2", "neox", "llama_gqa", "qwen2", "qwen3", "olmo2", "gemma2", "gemma3", "phi3",
          "qwen3_5")


@pytest.mark.parametrize("kind", _KINDS)
def test_hook_reads_each_familys_attention_output(kind):
    """The o_proj-input norms equal the norms of the attention output the
    model's own eager function returns (times the output gate on Qwen3.5),
    and only softmax layers are scored."""
    model = _tiny(kind).eval()
    layers = get_layers(model)
    targets, n_non_softmax, n_no_proj = IH.attention_targets(model, layers)
    attn_mods = {i: IH._find_attn_module(layers[i]) for i, *_ in targets}
    family = sys.modules[type(next(iter(attn_mods.values()))).__module__]
    seen, gates = [], []
    orig = family.eager_attention_forward

    def recording(*a, **k):
        out = orig(*a, **k)
        seen.append(out[0].detach())
        return out

    hooks = []
    if kind == "qwen3_5":                      # Qwen3_5Attention.forward's gate
        for i in sorted(attn_mods):
            m = attn_mods[i]
            hooks.append(m.q_proj.register_forward_hook(
                lambda mod, inp, out, m=m: gates.append(torch.chunk(
                    out.view(*out.shape[:-1], -1, m.head_dim * 2), 2, dim=-1)[1]
                    .reshape(*out.shape[:-1], -1))))
    ids = fd.tokenizer()(fd.TEXTS[0], return_tensors="pt")["input_ids"]
    family.eager_attention_forward = recording
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        family.eager_attention_forward = orig
        for h in hooks:
            h.remove()
    assert len(seen) == len(targets)
    norms, _ = IH.collect_head_norms(model, fd.tokenizer(), [fd.TEXTS[0]], targets, 512)
    for j, (_, _, n_heads, head_dim) in enumerate(targets):
        x = seen[j].reshape(*seen[j].shape[:-2], -1)
        if gates:
            x = x * torch.sigmoid(gates[j])
        expected = IH.head_output_norms(x, n_heads, head_dim).mean(dim=-2)
        assert torch.allclose(norms[j], expected, rtol=1e-5, atol=0)
    expect_skipped = 2 if kind == "qwen3_5" else 0
    assert (n_non_softmax, n_no_proj) == (expect_skipped, 0)
    r = IH.InactiveHeadsTask({"num_samples": 2}).evaluate(
        model, fd.tokenizer(), [{"text": t} for t in fd.TEXTS[:2]])
    assert r["_meta_n_layers_skipped_non_softmax"] == expect_skipped
    assert r["_meta_n_layers_scored"] == len(layers) - expect_skipped
    assert r["_meta_n_heads"] == model.config.num_attention_heads
