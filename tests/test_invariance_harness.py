"""Tests for the invariance harness in ``tests/invariance/``.

Known positives: quantities that must not move under a transform that leaves
the model's function (or the cloud's geometry) unchanged. Known negatives: the
harness must detect quantities known to move. Opt-in slow tests (real
pythia-160m / TinyLlama checkpoints, gpt2 weight padding) run with
BLME_SLOW_INVARIANCE=1.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import blme  # noqa: E402
import invariance as inv  # noqa: E402
from invariance import model_transforms as mt  # noqa: E402

transformers = pytest.importorskip("transformers")

SLOW = pytest.mark.skipif(not os.environ.get("BLME_SLOW_INVARIANCE"),
                          reason="slow; set BLME_SLOW_INVARIANCE=1")

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
    "Volcanoes form where tectonic plates diverge or where one plate subducts.",
    "The orchestra tuned their instruments while the audience took their seats.",
    "A prime number has exactly two distinct positive divisors.",
    "Rivers carry sediment downstream and deposit it in deltas near the sea.",
    "The committee will vote on the proposal at its meeting next Thursday.",
    "Bees communicate the location of flowers through a waggle dance.",
]
EPS32 = float(np.finfo(np.float32).eps)


def test_blme_is_imported_from_this_tree():
    assert Path(blme.__file__).resolve().is_relative_to(SRC.resolve()), blme.__file__


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    """Tiny matmuls on a many-core host are dominated by thread start-up."""
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))
    yield
    torch.set_num_threads(n)


def _load(name):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32).eval()
    except OSError as e:
        pytest.skip(f"{name} not in the local HF cache: {e}")
    return model, tok


@pytest.fixture(scope="module")
def tiny_gpt2():
    return _load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def gpt2():
    return _load("gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny_gpt2):
    return inv.states_from_model(*tiny_gpt2, TEXTS)


@pytest.fixture(scope="module")
def gpt2_states(gpt2):
    return inv.states_from_model(*gpt2, TEXTS[:6])


def _tiny_random(kind):
    from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                              LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM)
    torch.manual_seed(0)
    common = dict(vocab_size=500, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=4, num_attention_heads=4)
    if kind == "neox_parallel":
        return GPTNeoXForCausalLM(GPTNeoXConfig(**common, use_parallel_residual=True)).eval()
    if kind == "neox_sequential":
        return GPTNeoXForCausalLM(GPTNeoXConfig(**common, use_parallel_residual=False)).eval()
    if kind == "llama":
        return LlamaForCausalLM(LlamaConfig(**common, num_key_value_heads=2)).eval()
    if kind == "llama_tied":
        return LlamaForCausalLM(LlamaConfig(**common, num_key_value_heads=2,
                                            tie_word_embeddings=True)).eval()
    if kind == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(**common, num_key_value_heads=2, head_dim=16)).eval()
    raise ValueError(kind)


ARCHS = ["tiny_gpt2", "neox_parallel", "neox_sequential", "llama", "llama_tied", "qwen3"]


@pytest.fixture(params=ARCHS)
def arch_model(request, tiny_gpt2):
    if request.param == "tiny_gpt2":
        return request.param, tiny_gpt2[0]
    return request.param, _tiny_random(request.param)


def _ids(model, lengths=(7, 19, 33), seed=1):
    g = torch.Generator().manual_seed(seed)
    V = min(model.config.vocab_size, 1000)
    return [torch.randint(0, V, (n,), generator=g) for n in lengths]


# --------------------------------------------------------------------------
# Cloud transforms
# --------------------------------------------------------------------------

def _pairwise(X):
    X = np.asarray(X, dtype=np.float64)
    sq = (X * X).sum(1)
    return sq[:, None] + sq[None, :] - 2 * X @ X.T


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_embed_isometric_preserves_gram(backend, k):
    rng = np.random.default_rng(0)
    H = rng.standard_normal((50, 16))
    Hb = torch.from_numpy(H) if backend == "torch" else H
    E = inv.embed_isometric(Hb, k, seed=3)
    E = E.numpy() if backend == "torch" else E
    assert E.shape == (50, 16 * k) and E.dtype == np.float64
    np.testing.assert_allclose(E @ E.T, H @ H.T, rtol=0, atol=1e-12 * np.abs(H @ H.T).max())
    M = inv.random_isometry(16, k, 3)
    np.testing.assert_allclose(M @ M.T, np.eye(16), atol=1e-13)


def test_embed_isometric_same_map_for_numpy_and_torch_and_seeded():
    H = np.random.default_rng(1).standard_normal((10, 8)).astype(np.float32)
    a = inv.embed_isometric(H, 2, seed=5)
    b = inv.embed_isometric(torch.from_numpy(H), 2, seed=5).numpy()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, inv.embed_isometric(H, 2, seed=6))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_zero_pad_subsample_rescale(backend):
    H = np.arange(24, dtype=np.float32).reshape(6, 4)
    Hb = torch.from_numpy(H) if backend == "torch" else H
    Z = inv.zero_pad(Hb, 3)
    Zn = Z.numpy() if backend == "torch" else Z
    assert Zn.shape == (6, 12) and Zn.dtype == np.float32
    np.testing.assert_array_equal(Zn[:, :4], H)
    assert not Zn[:, 4:].any()
    S1 = inv.subsample_rows(Hb, 4, seed=0)
    S2 = inv.subsample_rows(Hb, 4, seed=0)
    S1n = S1.numpy() if backend == "torch" else S1
    np.testing.assert_array_equal(S1n, S2.numpy() if backend == "torch" else S2)
    rows = [int(r[0]) // 4 for r in S1n]
    assert len(rows) == 4 and rows == sorted(set(rows))
    R = inv.rescale(Hb, 0.5)
    np.testing.assert_array_equal(R.numpy() if backend == "torch" else R, H.astype(np.float64) * 0.5)


def test_docs_variants_share_one_transform_across_docs_and_layers():
    rng = np.random.default_rng(2)
    docs = {0: [rng.standard_normal((5, 6)), rng.standard_normal((3, 6))],
            1: [rng.standard_normal((5, 6)), rng.standard_normal((3, 6))]}
    E = inv.embed_isometric_docs(docs, 3, seed=1)
    before = _pairwise(np.concatenate([inv.pool_docs(docs[0]), inv.pool_docs(docs[1])]))
    after = _pairwise(np.concatenate([inv.pool_docs(E[0]), inv.pool_docs(E[1])]))
    np.testing.assert_allclose(after, before, atol=1e-11)
    sub = inv.subsample_docs(docs, 1, seed=0)
    assert len(sub[0]) == 1 and sub[0][0].shape == sub[1][0].shape
    S = inv.States(blocks=docs, final_norm=docs[1], embedding=docs[0])
    Sz = inv.zero_pad_docs(S, 2)
    assert Sz.d == 12 and Sz.final_norm[0].shape == (5, 12) and Sz.embedding[1].shape == (3, 12)
    assert inv.subsample_docs(S, 1, seed=0).n_docs == 1


def test_states_validates_layout():
    with pytest.raises(ValueError):
        inv.States(blocks={1: [np.zeros((2, 3))]})
    with pytest.raises(ValueError):
        inv.States(blocks={0: [np.zeros((2, 3))], 1: [np.zeros((3, 3))]})


# --------------------------------------------------------------------------
# Stub-cache driver
# --------------------------------------------------------------------------

def test_states_from_model_matches_cache_layout(tiny_gpt2, tiny_states):
    S = tiny_states
    L = tiny_gpt2[0].config.n_layer
    assert S.n_layers == L and S.n_docs == len(TEXTS) and S.d == 32
    # final_norm is ln_f(deepest raw block), as HF's last hidden_states entry.
    ln_f = tiny_gpt2[0].transformer.ln_f
    with torch.no_grad():
        refit = ln_f(torch.from_numpy(S.blocks[L - 1][0])).numpy()
    np.testing.assert_array_equal(refit, S.final_norm[0])
    assert S.embedding is not None and len(S.embedding) == S.n_docs


def test_stub_model_reports_depth_and_width():
    from blme.tasks.common import get_layers, get_num_layers
    from blme.tasks.geometry.utils import collected_layer_count
    m = inv.StubModel(6, 48)
    assert m.config.num_hidden_layers == 6 and m.config.n_layer == 6
    assert m.config.hidden_size == 48 and m.config.n_embd == 48
    assert len(get_layers(m)) == 6 and get_num_layers(m) == 6
    assert collected_layer_count(m) == 5
    with pytest.raises(inv.TaskForwardAttempted):
        m(input_ids=torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(inv.TaskForwardAttempted):
        get_layers(m)[0](torch.zeros(1))
    assert m.forward_calls.n == 2


# geometry_lid takes one point per document, and its k=20 neighbours need
# more than the 12 documents here.
_DRIVER_CONFIG = {"geometry_lid": {"k": 5}}


@pytest.mark.parametrize("task", sorted(inv.STATE_DRIVABLE_TASKS))
def test_driver_reproduces_a_real_cache_bit_for_bit(task, tiny_gpt2, tiny_states):
    """The stub path and a real populated cache give identical results."""
    stub = inv.run_task_on_states(task, tiny_states, _DRIVER_CONFIG.get(task))
    real = inv.run_task_on_model(task, *tiny_gpt2, TEXTS, _DRIVER_CONFIG.get(task))
    assert _same(stub, real), _diff(stub, real)


def _same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _diff(a, b, pre=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in set(a) | set(b) for d in (
            [f"{pre}{k}: missing"] if k not in a or k not in b else _diff(a[k], b[k], f"{pre}{k}."))]
    return [] if _same(a, b) else [f"{pre.rstrip('.')}: {a!r} != {b!r}"]


def test_profiles_keep_full_depth():
    """RSA/HSIC see every injected layer, not the 2 a depth-less config gives."""
    rng = np.random.default_rng(0)
    L = 6
    S = {li: [rng.standard_normal((40, 8)) + li for _ in range(3)] for li in range(L)}
    hsic = inv.run_task_on_states("geometry_hsic", S)
    assert len(hsic["_meta_adjacent_hsic"]) == L - 1
    rsa = inv.run_task_on_states("geometry_rsa", S)
    assert rsa["_meta_last_entry_is_block_output"] is True
    assert rsa["rsa_early_late_corrdist"] != rsa["rsa_adjacent_min_corrdist"]


def test_legacy_cache_trim_does_not_depend_on_the_stub_config():
    """On a cache without the deepest-block capture, HSIC drops exactly the
    post-norm entry whatever the config says. It used to trim to
    collected_layer_count(model) -- 2 when the config lacked
    num_hidden_layers/n_layer (tests/tasks/invariance/test_hsic.py)."""
    from blme.tasks.geometry.mutual_info import HSICDependenceTask
    rng = np.random.default_rng(0)
    L = 6
    S = inv.States(blocks={li: [rng.standard_normal((40, 8)) + li for _ in range(3)]
                           for li in range(L)})
    lengths = {}
    for name, attrs in (("depth", {}), ("no depth", {"num_hidden_layers": None, "n_layer": None})):
        m = inv.StubModel(L, 8, attrs)
        cache = inv.InjectedStateCache(S, m, inv.StubTokenizer())
        cache._last_block_raw = None          # a cache built before the hook existed
        r = HSICDependenceTask({"num_samples": 3}).evaluate(m, None, None, cache=cache)
        lengths[name] = r["_meta_n_layers"] - 1
    assert lengths == {"depth": L - 2, "no depth": L - 2}


@pytest.mark.parametrize("task", ["geometry_correlation_dimension", "topology_homology",
                                  "geometry_weight_norms", "geometry_phd_dimension",
                                  "geometry_perplexity"])
def test_unlisted_tasks_are_refused(task, tiny_states):
    with pytest.raises(inv.TaskNotDrivable, match=task):
        inv.run_task_on_states(task, tiny_states)


@pytest.mark.parametrize("task", ["topology_homology", "geometry_phd_dimension",
                                  "geometry_effective_depth"])
def test_forced_self_forward_tasks_are_caught(task, tiny_states):
    """A forced run still fails loudly when the task tries its own forward,
    including when the task swallows the exception."""
    with pytest.raises(inv.TaskNotDrivable):
        inv.run_task_on_states(task, tiny_states, force=True)


def test_drivability_partition_covers_geometry_and_topology():
    from blme.core import _register_all_tasks
    from blme.registry import list_tasks
    _register_all_tasks()
    names = {t for t in list_tasks() if t.startswith(("geometry_", "topology_"))}
    drivable = set(inv.STATE_DRIVABLE_TASKS)
    undrivable = set(inv.NOT_STATE_DRIVABLE)
    assert not drivable & undrivable
    assert names <= drivable | undrivable, sorted(names - drivable - undrivable)
    assert (drivable | undrivable) <= set(list_tasks())


def test_dict_input_with_final_norm_slot(tiny_states):
    """Single-layer IsoScore reads the post-final-norm slot; layerwise LID the
    raw deepest block."""
    S = tiny_states
    as_dict = inv.run_task_on_states("geometry_isoscore", S.blocks, final_norm=S.final_norm)
    assert as_dict == inv.run_task_on_states("geometry_isoscore", S)
    no_final = inv.run_task_on_states("geometry_isoscore", S.blocks)
    assert no_final["isoscore"] != as_dict["isoscore"]
    lw = {"layerwise": True, "k": 5}
    assert (inv.run_task_on_states("geometry_lid", S, lw)["lid_mean"]
            == inv.run_task_on_states("geometry_lid", S.blocks, lw)["lid_mean"])


# --------------------------------------------------------------------------
# Known positives: cloud transforms through the real evaluate()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 4])
def test_isoscore_effective_dims_invariant_under_zero_padding(gpt2_states, k):
    base = inv.run_task_on_states("geometry_isoscore", gpt2_states)
    pad = inv.run_task_on_states("geometry_isoscore", inv.zero_pad_docs(gpt2_states, k))
    d = gpt2_states.d
    assert pad["_meta_d_model"] == k * d
    inv.assert_invariant(base, pad, ["isoscore_effective_dims"], rtol=1e-9)
    # The raw score carries 1/(d-1): it falls by exactly (d-1)/(kd-1) ~ 1/k.
    ratio = pad["isoscore"] / base["isoscore"]
    assert ratio == pytest.approx((d - 1) / (k * d - 1), rel=1e-9)
    assert ratio == pytest.approx(1 / k, rel=2e-3)
    inv.assert_moves(base, pad, "isoscore", min_rel=0.4)


@pytest.mark.parametrize("mode", [{"k": 4}, {"layerwise": True, "k": 4}],
                         ids=["single", "layerwise"])
@pytest.mark.parametrize("k", [2, 4])
def test_lid_bit_identical_under_zero_padding(gpt2_states, k, mode):
    base = inv.run_task_on_states("geometry_lid", gpt2_states, mode)
    pad = inv.run_task_on_states("geometry_lid", inv.zero_pad_docs(gpt2_states, k), mode)
    keys = [key for key in base if key.startswith("lid_") or key == "layer_lid_mean"]
    assert len(keys) >= 4
    inv.assert_invariant(base, pad, keys, rtol=0.0)


@pytest.mark.parametrize("mode", [{"k": 4}, {"layerwise": True, "k": 4}],
                         ids=["single", "layerwise"])
def test_lid_invariant_under_random_orthogonal_embedding(gpt2_states, mode):
    """Random-orthogonal embedding changes the float32 rounding the task's own
    cast applies, so equality is to rounding (measured <= 7.6e-8). Position 0 is
    excluded: two documents opening with the same token give position-0 rows
    that differ only by rounding, and LID amplifies that to ~1e-4."""
    S = gpt2_states.map(lambda x: x[1:])
    base = inv.run_task_on_states("geometry_lid", S, mode)
    keys = [key for key in base if key.startswith("lid_") or key == "layer_lid_mean"]
    for seed in (1, 2):
        emb = inv.run_task_on_states("geometry_lid", inv.embed_isometric_docs(S, 2, seed), mode)
        inv.assert_invariant(base, emb, keys, rtol=1e-6)


ISOMETRY_INVARIANT_TASKS = [
    "geometry_collapse", "geometry_hsic", "geometry_intrinsic_dim", "geometry_lipschitz",
    "geometry_svd", "geometry_trajectory_curvature", "geometry_vendi_score",
    "topology_zigzag_persistence",
]


def _feature_keys(result):
    return [k for k, v in result.items() if not k.startswith("_meta_")
            and (isinstance(v, (int, float)) and not isinstance(v, bool)
                 or isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v))]


@pytest.mark.parametrize("task", ISOMETRY_INVARIANT_TASKS)
def test_isometry_and_scale_invariant_tasks(task, tiny_states):
    """Tasks whose every feature key is invariant to zero-padding, a random
    isometric embedding and a power-of-two rescale (N=338 >> 2d=64)."""
    base = inv.run_task_on_states(task, tiny_states)
    keys = _feature_keys(base)
    assert keys
    for St in (inv.zero_pad_docs(tiny_states, 2), inv.embed_isometric_docs(tiny_states, 2, 3),
               inv.rescale_docs(tiny_states, 2.0)):
        inv.assert_invariant(base, inv.run_task_on_states(task, St), keys, rtol=1e-6)


# --------------------------------------------------------------------------
# Known positives: function-preserving model transforms
# --------------------------------------------------------------------------

def test_identity_insertion_is_exact(arch_model):
    name, m = arch_model
    L = len(mt._arch(m).get_blocks(m))
    new = inv.insert_identity_blocks(m, [-1, 0, L - 1, L - 1])
    assert len(mt._arch(new).get_blocks(new)) == L + 4
    assert new.config.num_hidden_layers == L + 4
    assert len(mt._arch(m).get_blocks(m)) == L          # input untouched
    assert inv.max_logit_delta(m, new, _ids(m)) == 0.0
    inv.insert_identity_blocks(m, [1], verify_on=_ids(m))


def test_identity_layout():
    assert inv.identity_layout(3, [-1, 1, 1]) == [
        ("identity", 0), ("orig", 0), ("orig", 1), ("identity", 1), ("identity", 1), ("orig", 2)]
    with pytest.raises(ValueError):
        inv.identity_layout(3, [3])


def test_redecompose_is_exact_or_refused(arch_model):
    name, m = arch_model
    if name == "neox_parallel":
        with pytest.raises(NotImplementedError, match="parallel residual"):
            inv.redecompose(m, 1)
        return
    new = inv.redecompose(m, 1, verify_on=_ids(m))
    assert len(mt._arch(new).get_blocks(new)) == len(mt._arch(m).get_blocks(m)) + 1
    assert inv.max_logit_delta(m, new, _ids(m)) == 0.0


@pytest.mark.parametrize("c", [0.5, 2.0, 4.0])
def test_residual_gauge_with_rescaled_eps_is_bit_exact(arch_model, c):
    """With every residual-reading norm's eps scaled by c^2 the gauge is exact,
    so the plain gauge's error is entirely the norm eps."""
    name, m = arch_model
    new = inv.residual_scale_gauge(m, c, rescale_eps=True)
    assert inv.max_logit_delta(m, new, _ids(m)) == 0.0


def test_residual_gauge_scales_every_residual_state(gpt2, gpt2_states):
    model, tok = gpt2
    g = inv.residual_scale_gauge(model, 2.0, rescale_eps=True)
    Sg = inv.states_from_model(g, tok, TEXTS[:6])
    for li in range(gpt2_states.n_layers):
        for a, b in zip(gpt2_states.blocks[li], Sg.blocks[li]):
            np.testing.assert_array_equal(b, 2.0 * a)
    for a, b in zip(gpt2_states.final_norm, Sg.final_norm):   # after the norm: unchanged
        np.testing.assert_array_equal(a, b)
    assert not inv.is_tied(g) and inv.is_tied(model)


def test_residual_gauge_plain_tolerance_on_gpt2(gpt2):
    """Plain gauge (eps untouched) on gpt2 text: measured 2.8e-3 at c=2 and
    3.5e-3 at c=4 on logits up to 198."""
    model, tok = gpt2
    ids = [tok(t, return_tensors="pt").input_ids for t in TEXTS[:3]]
    for c, atol in ((2.0, 5e-3), (4.0, 6e-3)):
        delta = inv.max_logit_delta(model, inv.residual_scale_gauge(model, c), ids)
        assert 0.0 < delta <= atol, (c, delta)


@pytest.mark.parametrize("D", [2.0, 0.1, None], ids=["2", "0.1", "lognormal"])
def test_final_norm_unembedding_gauge(arch_model, D):
    """Exact for a power-of-two D; otherwise float32 rounding of D*gamma and
    W_U/D (measured <= 2.4 eps32 * max|logit|)."""
    name, m = arch_model
    new = inv.final_norm_unembedding_gauge(m, D, seed=0)
    ids = _ids(m)
    delta = inv.max_logit_delta(m, new, ids)
    if D == 2.0:
        assert delta == 0.0
    else:
        scale = max(float(x.abs().max()) for x in inv.logits(m, ids))
        assert delta <= 8 * EPS32 * scale, (delta, scale)
    a = mt._arch(new)
    assert not torch.equal(a.final_norm(new).weight, mt._arch(m).final_norm(m).weight)


def test_pad_vocabulary(arch_model):
    name, m = arch_model
    ids = _ids(m)
    V = m.config.vocab_size
    ref = inv.logits(m, ids)
    masked = inv.pad_vocabulary(m, 5, kind="masked")
    assert masked.config.vocab_size == V + 5
    assert mt._arch(masked).unembed(masked).weight.shape[0] == V + 5
    for a, b in zip(ref, inv.logits(masked, ids)):
        assert torch.equal(a, b[..., :V])
        assert torch.isneginf(b[..., V:]).all()
        assert torch.equal(torch.log_softmax(a, -1), torch.log_softmax(b, -1)[..., :V])
    for kind in ("zero", "small_random"):
        pad = inv.pad_vocabulary(m, 5, kind=kind)
        assert pad.config.vocab_size == V + 5
        assert inv.is_tied(pad) == inv.is_tied(m)
        lp = [torch.log_softmax(b, -1)[..., :V] for b in inv.logits(pad, ids)]
        shift = max(float((torch.log_softmax(a, -1) - b).abs().max()) for a, b in zip(ref, lp))
        assert shift > 1e-3, (kind, shift)       # not function-preserving
        assert inv.max_logit_delta(m, pad, ids) == 0.0


def test_precision_transforms(arch_model):
    name, m = arch_model
    w = inv.bf16_weights_fp32_math(m)
    for p, q in zip(m.parameters(), w.parameters()):
        assert q.dtype == torch.float32
        assert torch.equal(q, q.to(torch.bfloat16).float())
    low = inv.run_with_dtype(m, torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in low.parameters())
    for (n0, b0), (n1, b1) in zip(m.named_buffers(), low.named_buffers()):
        assert b0.dtype == b1.dtype, n0       # buffers (e.g. RoPE inv_freq) untouched
    split = inv.precision_error_split(m, _ids(m))
    assert all(np.isfinite(v) and v > 0 for v in split.values()), split
    assert split["total"] <= split["storage"] + split["arithmetic"] + 1e-6


# --------------------------------------------------------------------------
# Known negatives
# --------------------------------------------------------------------------

def test_uniform_layer_mean_moves_under_identity_insertion(tiny_gpt2, tiny_states):
    """lipschitz._meta_relative_change_mean is a uniform mean over the L - 1
    block-to-block updates (the embedding update is reported separately): each
    inserted identity block adds a zero update, so the mean falls by exactly
    (L - 1) / (L - 1 + n_inserted). The max does not move."""
    model, tok = tiny_gpt2
    new = inv.insert_identity_blocks(model, [1, 2])
    S_new = inv.states_from_model(new, tok, TEXTS)
    base = inv.run_task_on_states("geometry_lipschitz", tiny_states)
    ins = inv.run_task_on_states("geometry_lipschitz", S_new)
    n_tr = tiny_states.n_layers - 1
    inv.assert_moves(base, ins, "_meta_relative_change_mean", min_rel=0.2)
    assert ins["_meta_relative_change_mean"] / base["_meta_relative_change_mean"] == pytest.approx(
        n_tr / (n_tr + 2), rel=1e-12)
    inv.assert_invariant(base, ins, ["relative_change_max"], rtol=0.0)


def test_layer_profile_under_identity_insertion(tiny_gpt2, tiny_states):
    """Inserted entries duplicate their predecessor bit for bit, so the deepest
    value is invariant while the uniform layer mean moves."""
    model, tok = tiny_gpt2
    after = [0, 0, 3]
    S_new = inv.states_from_model(inv.insert_identity_blocks(model, after), tok, TEXTS)
    lw = {"layerwise": True, "k": 5}
    base = inv.run_task_on_states("geometry_lid", tiny_states, lw)
    ins = inv.run_task_on_states("geometry_lid", S_new, lw)
    expected = [base["layer_lid_mean"][i]
                for _, i in inv.identity_layout(tiny_states.n_layers, after)]
    assert ins["layer_lid_mean"] == expected
    inv.assert_invariant(base, ins, ["lid_mean", "lid_first_layer"], rtol=0.0)
    m0 = inv.uniform_layer_mean(base["layer_lid_mean"])
    m1 = inv.uniform_layer_mean(ins["layer_lid_mean"])
    assert inv.rel_change(m0, m1) > 1e-3


def test_weight_norms_normalized_stable_rank_moves_under_zero_padding(tiny_gpt2):
    """sr/min(m,n) halves under k=2 zero-padding; raw stable rank, Frobenius
    and spectral norms do not move."""
    from blme.tasks.geometry.weight_norms import WeightNormProfileTask
    model, _ = tiny_gpt2
    padded = inv.zero_pad_weight_matrices(model, 2)
    base = WeightNormProfileTask({}).evaluate(model, None, None)
    pad = WeightNormProfileTask({}).evaluate(padded, None, None)
    inv.assert_moves(base, pad, "mean_stable_rank_normalized", min_rel=0.4)
    assert pad["mean_stable_rank_normalized"] / base["mean_stable_rank_normalized"] == \
        pytest.approx(0.5, rel=1e-12)
    inv.assert_invariant(base, pad, ["_meta_mean_stable_rank", "stable_rank_per_layer",
                                     "mean_frobenius_norm", "mean_spectral_norm"], rtol=1e-12)
    with pytest.raises(RuntimeError, match="weight-only"):
        padded(input_ids=torch.zeros(1, 2, dtype=torch.long))


@pytest.mark.xfail(strict=True, reason=(
    "library: geometry_weight_norms reports 0.0 for blocks it measured nothing "
    "in (weight_norms.py: `... if frob_vals else 0.0`); geometry_spectral "
    "returns NaN in the same case. Unreachable on standard HF checkpoints."))
def test_weight_norms_does_not_fabricate_zeros_for_unmeasured_blocks():
    class NoMatrices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.LayerNorm(8) for _ in range(3)])
    from blme.tasks.geometry.weight_norms import WeightNormProfileTask
    r = WeightNormProfileTask({}).evaluate(NoMatrices(), None, None)
    assert "error" in r or math.isnan(r["mean_stable_rank_normalized"]), r


def test_compare_helpers_detect_movement():
    a = {"x": 1.0, "p": [1.0, 2.0], "n": {"y": 3.0}}
    b = {"x": 1.0 + 1e-3, "p": [1.0, 2.0], "n": {"y": 3.0}}
    with pytest.raises(AssertionError, match="x"):
        inv.assert_invariant(a, b, ["x"], rtol=1e-6)
    inv.assert_invariant(a, b, ["p", "n.y"], rtol=0.0)
    with pytest.raises(AssertionError, match="did not move"):
        inv.assert_moves(a, b, "p", min_rel=1e-9)
    assert inv.assert_moves(a, b, "x", min_rel=1e-4) == pytest.approx(1e-3)
    assert inv.rel_change(float("nan"), float("nan")) == 0.0
    assert inv.rel_change(0.0, 1.0) == math.inf


# --------------------------------------------------------------------------
# Slow: real checkpoints
# --------------------------------------------------------------------------

@SLOW
@pytest.mark.parametrize("name", ["gpt2", "EleutherAI/pythia-160m",
                                  "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"])
def test_model_transforms_on_real_checkpoints(name):
    model, tok = _load(name)
    ids = [tok(t, return_tensors="pt").input_ids for t in TEXTS[:2]]
    L = len(mt._arch(model).get_blocks(model))
    assert inv.max_logit_delta(model, inv.insert_identity_blocks(model, [0, L // 2]), ids) == 0.0
    if mt._arch(model).is_sequential(mt._arch(model).get_blocks(model)[0]):
        assert inv.max_logit_delta(model, inv.redecompose(model, L // 2), ids) == 0.0
    assert inv.max_logit_delta(model, inv.residual_scale_gauge(model, 2.0, rescale_eps=True), ids) == 0.0
    assert inv.max_logit_delta(model, inv.final_norm_unembedding_gauge(model, 2.0), ids) == 0.0
    masked = inv.pad_vocabulary(model, 3, kind="masked")
    assert inv.max_logit_delta(model, masked, ids) == 0.0


@SLOW
def test_weight_norms_zero_padding_on_gpt2(gpt2):
    from blme.tasks.geometry.weight_norms import WeightNormProfileTask
    model, _ = gpt2
    base = WeightNormProfileTask({}).evaluate(model, None, None)
    pad = WeightNormProfileTask({}).evaluate(inv.zero_pad_weight_matrices(model, 2), None, None)
    assert pad["mean_stable_rank_normalized"] / base["mean_stable_rank_normalized"] == \
        pytest.approx(0.5, rel=1e-12)
    inv.assert_invariant(base, pad, ["_meta_mean_stable_rank"], rtol=1e-12)
