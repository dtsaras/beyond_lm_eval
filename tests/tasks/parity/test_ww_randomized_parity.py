"""Parity: geometry_ww_randomized vs the RELEASED WeightWatcher
(CalculatedContent/WeightWatcher@a5940f7), loaded from the clone under a
private package name (the venv's weightwatcher 0.7.7 is a different release;
its functions used here are textually identical, but parity is pinned to the
clone).

* per function, on Gaussian, low-rank + noise and Student-t (3 dof) matrices
  of tall, wide, square, 10:3 and M < 100 shapes: ``combined_eigenvalues``,
  ``random_eigenvalues`` after ``np.random.seed``, ``jensen_shannon_distance``,
  ``mp_fit``, ``rescale_eigenvalues`` + ``detX_constraint`` and ``WWFit`` must
  equal BLME's kernels exactly;
* per matrix: ``analyze(randomize=True, mp_fit=True, detX=True)`` on a model
  holding the matrix as its one Linear, after ``np.random.seed(seed)``, must
  equal ``ww_matrix_metrics(W, RandomState(seed))`` in every column;
* through ``evaluate()``: on tiny-random-gpt2 (HF Conv1D), a tiny Llama (every
  matrix has M < 100, the 5-replica path) and gpt2's first two blocks, every
  per-matrix ``_meta_`` column must equal the clone's ``analyze()`` on the same
  modules after ``np.random.seed(shuffle_seed)``.

The release is imported with CUDA reported absent: with CUDA it reads weights
as float16 and runs float32 torch SVDs, while BLME always follows its CPU path.
Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1). The known
positives and the seed / population checks need no reference.
"""
import contextlib
import copy
import importlib.util
import io
import math
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

pytest.importorskip("pandas")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "invariance"))

import _family_d as fd  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.geometry import ww_randomized as R  # noqa: E402

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "WeightWatcher")
_PKG = os.path.join(_REPO, "weightwatcher")
_NAME = "_blme_parity_ww_ref"
_COLUMNS = ("rand_distance", "ww_softrank", "num_spikes", "mp_softrank", "sigma_mp", "bulk_min",
            "bulk_max", "detX_num", "num_pl_spikes", "ERG_gap")
_SEED = 42


@pytest.fixture(scope="module")
def ref():
    if not os.path.isfile(os.path.join(_PKG, "weightwatcher.py")):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    spec = importlib.util.spec_from_file_location(_NAME, os.path.join(_PKG, "__init__.py"),
                                                  submodule_search_locations=[_PKG])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = pkg
    try:
        # WW_powerlaw sets warnings.simplefilter("ignore") at import; RMT_Util
        # picks its SVD and weight dtype by torch.cuda.is_available().
        with warnings.catch_warnings(), mock.patch.object(torch.cuda, "is_available", lambda: False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(pkg)
        yield _namespace(pkg)
    finally:
        for n in [n for n in sys.modules if n == _NAME or n.startswith(_NAME + ".")]:
            sys.modules.pop(n, None)


def _namespace(pkg):
    return types.SimpleNamespace(
        WeightWatcher=pkg.WeightWatcher, ww=sys.modules[_NAME + ".weightwatcher"],
        rmt=sys.modules[_NAME + ".RMT_Util"], plaw=sys.modules[_NAME + ".WW_powerlaw"],
        DEFAULT_PARAMS=sys.modules[_NAME + ".constants"].DEFAULT_PARAMS)


def test_sources_pin_the_cloned_commit(ref):
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert R.WWRandomizedTask.SOURCES["code"].endswith("@" + head)


@contextlib.contextmanager
def _global_seed(seed):
    """Seed the global numpy RNG for the release and restore it afterwards."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _matrix(kind, n, m, seed):
    g = np.random.default_rng(seed)
    if kind == "gaussian":
        W = g.standard_normal((n, m)) * 0.02
    elif kind == "student_t3":
        W = g.standard_t(3, (n, m)) * 0.02
    else:                                   # rank-8 signal plus noise
        U = np.linalg.qr(g.standard_normal((n, 8)))[0]
        V = np.linalg.qr(g.standard_normal((m, 8)))[0]
        W = U @ np.diag(np.linspace(3, 1, 8)) @ V.T * 0.3 + g.standard_normal((n, m)) * 0.02
    return W.astype(np.float32)


_SHAPES = [("tall", 256, 64), ("wide", 64, 256), ("square", 200, 200), ("10:3", 300, 90),
           ("m_below_100", 40, 16)]
_CASES = [(f"{kind}-{label}", kind, n, m) for label, n, m in _SHAPES
          for kind in ("gaussian", "low_rank_plus_noise", "student_t3")]


@pytest.mark.parametrize("label,kind,n,m", _CASES, ids=[c[0] for c in _CASES])
def test_release_functions_equal_blme_kernels(ref, label, kind, n, m):
    W32 = _matrix(kind, n, m, seed=len(label))
    N, M = max(W32.shape), min(W32.shape)
    params = ref.DEFAULT_PARAMS.copy()
    params["svd_method"] = "accurate"
    w = ref.WeightWatcher()
    ev_ref = w.combined_eigenvalues([W32], N, M, M, params)[0]
    ev = R._esd(W32.astype(float))
    np.testing.assert_array_equal(ev, ev_ref)

    with _global_seed(7):
        rand_ref = w.random_eigenvalues([W32], M, 1, params)
    Wr = W32.astype(float).flatten()
    np.random.RandomState(7).shuffle(Wr)
    rand = R._esd(Wr.reshape(W32.shape))
    np.testing.assert_array_equal(rand, rand_ref)

    assert R.jensen_shannon_distance(ev, rand) == ref.rmt.jensen_shannon_distance(ev_ref, rand_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        mp_ref = w.mp_fit(ev_ref, N, M, 1, "x", 0, 0, False, False, "", "b", 0.5, True, ev_ref)
    mp = R.mp_fit(ev, N, M)
    assert (mp["num_spikes"], mp["sigma_mp"], mp["mp_softrank"], mp["bulk_min"], mp["bulk_max"],
            mp["W_scale"]) == tuple(float(v) if i else int(v) for i, v in enumerate(mp_ref))

    dx_ref = ref.rmt.detX_constraint(ref.rmt.rescale_eigenvalues(ev_ref)[0], rescale=False)
    assert R.detx_constraint(ev)[:2] == tuple(int(v) for v in dx_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        fit = ref.plaw.WWFit(ev_ref[ev_ref > 1e-5])
    assert R.wwfit(ev[ev > 1e-5]) == (float(fit.xmin), float(fit.alpha), float(fit.D))
    assert R.num_pl_spikes(ev) == len(ev_ref[ev_ref >= fit.xmin])


def _analyze(ref, model, names=None, seed=_SEED):
    """``analyze(randomize, mp_fit, detX)`` on the named modules (WW layer ids
    are named_modules positions), after ``np.random.seed(seed)``."""
    ids = [i for i, (n, _) in enumerate(model.named_modules()) if names is None or n in set(names)]
    with _global_seed(seed), contextlib.redirect_stdout(io.StringIO()):
        return ref.WeightWatcher(model=model).analyze(
            layers=ids, randomize=True, mp_fit=True, detX=True, svd_method="accurate")


class _One(torch.nn.Module):
    def __init__(self, W32):
        super().__init__()
        self.lin = torch.nn.Linear(W32.shape[1], W32.shape[0], bias=False)
        self.lin.weight.data = torch.from_numpy(W32.copy())


@pytest.mark.parametrize("label,kind,n,m", _CASES, ids=[c[0] for c in _CASES])
def test_release_pipeline_on_one_matrix_equals_ww_matrix_metrics(ref, label, kind, n, m):
    """apply_esd, apply_powerlaw, apply_mp_fit, apply_random_esd (its first call
    drawing 5 copies when M < 100) and apply_detX, as analyze() runs them."""
    W32 = _matrix(kind, n, m, seed=len(label) + 100)
    df = _analyze(ref, _One(W32), names=["lin"])
    assert len(df) == 1
    ours = R.ww_matrix_metrics(W32, np.random.RandomState(_SEED))
    for col in _COLUMNS:
        assert ours[col] == df[col].iloc[0], col


def _assert_columns_equal(out, df, n=None):
    names = out["_meta_modules"][:n]
    assert list(df["longname"]) == names
    for col in _COLUMNS:
        np.testing.assert_array_equal(np.asarray(out[f"_meta_{col}"][:len(names)], dtype=float),
                                      df[col].to_numpy(dtype=float), err_msg=col)


def _tiny_gpt2():
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")


@pytest.mark.parametrize("build", [_tiny_gpt2, fd.tiny_llama], ids=["tiny_gpt2_conv1d", "tiny_llama"])
def test_evaluate_equals_release_analyze(ref, build):
    model = build()
    out = R.WWRandomizedTask(resolve_task_config("geometry_ww_randomized")).evaluate(model, None, None)
    df = _analyze(ref, model, out["_meta_modules"])
    _assert_columns_equal(out, df)
    # The headline means of the release's own columns.
    assert out["rand_distance_mean"] == pytest.approx(float(df["rand_distance"].mean()), rel=1e-12)
    assert out["ww_softrank_mean"] == pytest.approx(float(df["ww_softrank"].mean()), rel=1e-12)
    assert out["mp_softrank_mean"] == pytest.approx(float(df["mp_softrank"].mean()), rel=1e-12)
    assert out["spike_fraction_mean"] == pytest.approx(
        float((df["num_spikes"] / np.asarray(out["_meta_n_evals"])).mean()), rel=1e-12)
    # Every other module the release analyses is an embedding.
    with contextlib.redirect_stdout(io.StringIO()):
        desc = ref.WeightWatcher(model=model).describe()
    rest = desc[~desc["longname"].isin(out["_meta_modules"])]
    assert len(desc) == len(out["_meta_modules"]) + len(rest) and len(rest) > 0
    assert all("embedding" in str(t).lower() for t in rest["layer_type"])


def test_evaluate_equals_release_analyze_on_gpt2(ref):
    """gpt2's first two blocks (8 matrices; the shuffle stream of the first
    matrices does not depend on the later ones). The vocabulary head (50257
    rows) is past max_N, so both skip it."""
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    out = R.WWRandomizedTask({}).evaluate(model, None, None)
    assert out["_meta_n_matrices"] == 48 and out["_meta_n_above_max_N"] == 1
    assert "lm_head" not in out["_meta_modules"]
    df = _analyze(ref, model, out["_meta_modules"][:8])
    _assert_columns_equal(out, df, n=8)


def _shrunk(fraction, seed, n=512, m=128):
    """A Gaussian matrix with a random ``fraction`` of its singular values x 0.1."""
    g = np.random.default_rng(seed)
    U, s, Vt = np.linalg.svd(g.standard_normal((n, m)) / np.sqrt(n), full_matrices=False)
    idx = g.choice(s.size, int(fraction * s.size), replace=False)
    s[idx] *= 0.1
    return (U @ np.diag(s) @ Vt).astype(np.float32)


def test_release_fit_locks_onto_a_near_zero_cluster(ref):
    """Known fragility of the released MP fit, reproduced, not fixed: with 10%
    of the singular values shrunk by 10x, the KDE fit settles on the cluster
    near 0 (sigma ~0.2 instead of ~1) and counts most of the bulk as spikes;
    at 5% it still fits the bulk (see the lower-edge known positive)."""
    ev = R._esd(_shrunk(0.10, seed=0).astype(float))
    with contextlib.redirect_stdout(io.StringIO()):
        mp_ref = ref.WeightWatcher().mp_fit(ev, 512, 128, 1, "x", 0, 0, False, False, "", "b", 0.5,
                                            True, ev)
    mp = R.mp_fit(ev, 512, 128)
    assert (mp["num_spikes"], mp["sigma_mp"]) == (mp_ref[0], mp_ref[1])
    assert mp["sigma_mp"] < 0.3 and mp["num_spikes"] > 100
    assert R.mp_fit(R._esd(_shrunk(0.05, seed=0).astype(float)), 512, 128)["num_spikes"] == 0


def _underflowing_spectrum():
    """Sorted eigenvalues whose largest suffix product is about e^1200 after
    the mean-1 rescaling: 1200 values near e, 2400 near e^-0.5."""
    ramp = np.linspace(0, 1e-3, 3600)
    return np.sort(np.concatenate([np.full(1200, np.e), np.full(2400, np.exp(-0.5))]) + ramp)


def test_release_detx_underflows_on_a_wide_spectrum(ref):
    """Known defect of the release, reproduced in detX_num and corrected in the
    detX_num_logsum twin: np.prod multiplies an ascending suffix smallest
    first, so once its log passes ~745 the partial product underflows to 0 and
    the scan stops at a shorter suffix than the det X = 1 crossing. Measured
    (2026-11): no gpt2 matrix is affected; every Qwen3-8B-Base 4096 x 4096 q/o
    projection tested (layers 0, 10, 35; log of the largest suffix product
    956-1056) is off by -14 to -84 (one by +1), its k/v projections by 0."""
    ev = _underflowing_spectrum()
    with np.errstate(over="ignore", under="ignore"):
        rel = ref.rmt.detX_constraint(ref.rmt.rescale_eigenvalues(ev)[0], rescale=False)
    assert R.detx_constraint(ev) == tuple(int(v) for v in rel)
    resc = R._rescale_eigenvalues(ev)
    exact, logs = len(ev), []
    for k in range(1, len(ev)):
        logs.append(math.log(resc[-k]))
        if math.fsum(logs) < 0:
            exact = k
            break
    assert R.detx_num_logsum(ev) == exact
    assert R.detx_constraint(ev)[0] < exact - 100


def test_detx_twin_equals_the_release_without_underflow():
    for kind, n, m in (("gaussian", 300, 90), ("low_rank_plus_noise", 256, 64),
                       ("student_t3", 200, 200)):
        ev = R._esd(_matrix(kind, n, m, seed=3).astype(float))
        assert R.detx_num_logsum(ev) == R.detx_constraint(ev)[0]


def _planted(n, m, rank, scale, seed):
    g = np.random.default_rng(seed)
    U = np.linalg.qr(g.standard_normal((n, rank)))[0]
    V = np.linalg.qr(g.standard_normal((m, rank)))[0]
    return (U @ np.diag(np.full(rank, scale)) @ V.T
            + g.standard_normal((n, m)) / np.sqrt(n)).astype(np.float32)


@pytest.mark.parametrize("seed", range(4))
def test_known_positive_planted_correlation_and_its_shuffle(seed):
    """A rank-16 signal (singular values 3) on unit-variance noise, 512 x 128,
    and an element shuffle of the same matrix. Measured (2026-11): planted
    rand_distance 0.264-0.271, ww_softrank 0.39-0.41, 16 spikes; shuffled
    rand_distance 0.004-0.007, ww_softrank 1.00-1.05, 0 spikes."""
    W = _planted(512, 128, 16, 3.0, seed)
    Ws = W.flatten()
    np.random.default_rng(100 + seed).shuffle(Ws)
    a = R.ww_matrix_metrics(W, np.random.RandomState(seed))
    b = R.ww_matrix_metrics(Ws.reshape(W.shape), np.random.RandomState(seed))
    assert a["rand_distance"] > 0.2 and b["rand_distance"] < 0.01
    assert a["ww_softrank"] < 0.5 and b["ww_softrank"] > 0.9
    assert (a["num_spikes"], b["num_spikes"]) == (16, 0)


@pytest.mark.parametrize("shape", [(512, 128), (128, 512), (300, 300), (1024, 256), (200, 64)])
def test_known_positive_gaussian_has_no_spikes_outside_the_bulk(shape):
    """iid Gaussian matrices, 5 seeds: no eigenvalue above the fitted MP edge
    plus the Tracy-Widom margin. Measured: mp_softrank 0.994-1.064,
    ww_softrank 0.95-1.054, rand_distance <= 0.011, frac_below_bulk_min <= 0.016
    (NaN for the square shape, whose lower edge is 0)."""
    for seed in range(5):
        W = (np.random.default_rng(seed).standard_normal(shape) * 0.02).astype(np.float32)
        r = R.ww_matrix_metrics(W, np.random.RandomState(seed))
        assert r["num_spikes"] == 0
        assert 0.95 < r["mp_softrank"] < 1.1 and 0.9 < r["ww_softrank"] < 1.1
        assert r["rand_distance"] < 0.02
        if shape[0] == shape[1]:
            assert "frac_below_bulk_min" not in r and r["bulk_min"] == 0.0
        else:
            assert r["frac_below_bulk_min"] < 0.02


@pytest.mark.parametrize("k", [1, 3, 5])
def test_known_positive_planted_spikes_are_counted(k):
    for seed in range(3):
        assert R.ww_matrix_metrics(_planted(512, 128, k, 4.0, seed),
                                   np.random.RandomState(seed))["num_spikes"] == k


def test_known_positive_lower_edge_counts_shrunk_singular_values():
    """Staats et al.'s small end: 5% of a Gaussian matrix's singular values
    shrunk by 10x fall below the fitted MP lower edge (measured 0.047-0.055),
    against <= 0.008 unshrunk."""
    for seed in range(3):
        base = R.ww_matrix_metrics(_shrunk(0.0, seed), np.random.RandomState(seed))
        r = R.ww_matrix_metrics(_shrunk(0.05, seed), np.random.RandomState(seed))
        assert base["frac_below_bulk_min"] < 0.01
        assert abs(r["frac_below_bulk_min"] - 0.05) < 0.01


def _assert_same_result(a, b):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], list) and a[k] and isinstance(a[k][0], str):
            assert a[k] == b[k], k
        else:
            np.testing.assert_array_equal(np.asarray(a[k], float), np.asarray(b[k], float), err_msg=k)


def test_shuffle_seed_drives_only_the_nulls_and_leaves_global_rngs_alone():
    model = fd.tiny_llama()
    np.random.seed(123)
    torch.manual_seed(123)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    a = R.WWRandomizedTask({"shuffle_seed": 1}).evaluate(model, None, None)
    b = R.WWRandomizedTask({"shuffle_seed": 1}).evaluate(model, None, None)
    c = R.WWRandomizedTask({"shuffle_seed": 2}).evaluate(model, None, None)
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1])
    assert after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    _assert_same_result(a, b)
    assert a["_meta_rand_distance"] != c["_meta_rand_distance"]
    for key in ("_meta_num_spikes", "_meta_mp_softrank", "_meta_bulk_min", "_meta_detX_num",
                "_meta_ERG_gap", "_meta_frac_below_bulk_min"):
        np.testing.assert_array_equal(np.asarray(a[key], float), np.asarray(c[key], float),
                                      err_msg=key)
    assert (a["_meta_shuffle_seed"], c["_meta_shuffle_seed"]) == (1, 2)


def test_default_seed_and_bf16_weights_cast_up():
    """A bf16 model gives exactly what its float32 copy gives: the weights are
    cast up before any arithmetic."""
    cfg = resolve_task_config("geometry_ww_randomized")
    assert cfg["shuffle_seed"] == R.DEFAULT_SHUFFLE_SEED == 42
    model = fd.tiny_llama().to(torch.bfloat16)
    up = copy.deepcopy(model).to(torch.float32)
    _assert_same_result(R.WWRandomizedTask(cfg).evaluate(model, None, None),
                        R.WWRandomizedTask(cfg).evaluate(up, None, None))


class _Gated(torch.nn.Module):
    """WeightWatcher's layer_supported gates (M < 8, M > 15000, N > 50000) and a
    torch conv kernel."""

    def __init__(self):
        super().__init__()
        self.ok = torch.nn.Linear(16, 32, bias=False)
        self.narrow = torch.nn.Linear(7, 64, bias=False)         # M = 7
        self.long = torch.nn.Linear(8, 50001, bias=False)        # N = 50001
        self.conv = torch.nn.Conv1d(16, 16, 3)


def test_release_layer_gates_and_conv_kernels_are_counted():
    torch.manual_seed(0)
    out = R.WWRandomizedTask({}).evaluate(_Gated(), None, None)
    assert out["_meta_modules"] == ["ok"]
    assert (out["_meta_n_below_min_evals"], out["_meta_n_above_max_N"],
            out["_meta_n_conv_kernels_skipped"], out["_meta_n_above_max_evals"]) == (1, 1, 1, 0)
