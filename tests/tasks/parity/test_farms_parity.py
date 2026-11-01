"""Parity: geometry_farms_alpha vs the RELEASED FARMS code
(HUST-AI-HYZ/FARMS@e0a2ded, ``LLM_Pruning/lib``).

``sampling.fixed_sampling`` (the window grid) and ``esd_utils_farms.analysis_esd``
(``fix_fingers='xmin_peak'``) are run on the same matrices and pooled
eigenvalues as BLME's kernels, on square, 1:4 and 4:1 matrices, matrices
narrower and wider than the window and a 225-window grid, with Gaussian,
Student-t and power-law-spectrum entries. alpha and D must agree to float32
tolerance and the selected xmin index (read from the release's
``torch.argmin``) must be the same. ``net_esd_estimator_farms`` in the
release's LLM configuration is run on the blocks of a tiny Llama, a tiny
Qwen2 and tiny-random-gpt2 (whose Conv1D weights the release cannot see: it
is run on a copy with each Conv1D replaced by the equivalent nn.Linear).
Skips when the clone is absent (fails under BLME_REQUIRE_REFS=1).

The aspect-ratio known positive and the window-geometry checks need no
reference.
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
from unittest import mock

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers.pytorch_utils import Conv1D  # noqa: E402

from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.geometry import farms as F  # noqa: E402
from blme.tasks.geometry.spectral_signature import _hill_alpha_xmin_mid  # noqa: E402

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "FARMS")
_LIB = os.path.join(_REPO, "LLM_Pruning", "lib")
# scripts/llm_pruning_main.txt, the alpha_peak_farms rows (step_size is unused
# when sampling_ops_per_dim is set).
_LLM_CONFIG = dict(fix_fingers="xmin_peak", use_sliding_window=True, num_row_samples=2000,
                   Q_ratio=1.0, step_size=15, sampling_ops_per_dim=15)
_RESULT_KEYS = ("alpha", "spectral_norm", "D", "longname", "eigs", "norm", "alpha_hat",
                "stable_rank", "norm_stable_rank")


@pytest.fixture(scope="module")
def ref():
    paths = {sub: os.path.join(_LIB, sub + ".py") for sub in ("sampling", "esd_utils_farms")}
    if not all(os.path.isfile(p) for p in paths.values()):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_REPO}")
        pytest.skip(f"reference not available: {_REPO}")
    # esd_utils_farms does `from .sampling import *`: load both as submodules of
    # a synthetic package under a name nothing else uses.
    pkg_name = "_blme_parity_farms_ref_lib"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [_LIB]
    names = [pkg_name] + [f"{pkg_name}.{sub}" for sub in paths]
    sys.modules[pkg_name] = pkg
    mods = {}
    try:
        for sub, path in paths.items():
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.{sub}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            mods[sub] = mod
        yield types.SimpleNamespace(**mods)
    finally:
        for n in names:
            sys.modules.pop(n, None)


def test_sources_pin_the_cloned_commit(ref):
    head = subprocess.run(["git", "-C", _REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert F.FARMSAlphaTask.SOURCES["code"].endswith("@" + head)


def _release_eigs(ref, W, num_row_samples, q_ratio=1.0, ops=15):
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.sampling.fixed_sampling(W.detach().float().clone(), False, 0.5,
                                           num_row_samples, q_ratio, ops)


def _release_fit(ref, eigs):
    """``(alpha, D, selected index)`` of ``analysis_esd(fix_fingers='xmin_peak')``;
    the index is the value its ``torch.argmin(Ds)`` returns."""
    picked = []
    real_argmin = torch.argmin

    def spy(*args, **kwargs):
        r = real_argmin(*args, **kwargs)
        picked.append(int(r))
        return r

    res = {k: [] for k in _RESULT_KEYS}
    with mock.patch.object(torch, "argmin", spy):
        res = ref.esd_utils_farms.analysis_esd(
            results=res, eigs=eigs, EVALS_THRESH=0.00001, bins=100, fix_fingers="xmin_peak",
            xmin_pos=2, filter_zeros=False, device="cpu")
    return res["alpha"][0], res["D"][0], picked[-1]


def _release_on_blocks(ref, blocks, **overrides):
    cfg = dict(_LLM_CONFIG, **overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        res = ref.esd_utils_farms.net_esd_estimator_farms(blocks, **cfg)
    return dict(zip(res["longname"], res["alpha"]))


def _matrix(kind, rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "gaussian":
        return torch.randn(rows, cols, generator=g) * 0.02
    if kind == "student_t3":
        z = torch.randn(rows, cols, generator=g)
        chi2 = (torch.randn(3, rows, cols, generator=g) ** 2).sum(0)
        return z / torch.sqrt(chi2 / 3) * 0.02
    k = min(rows, cols)              # power-law singular values sigma_i = i^-0.7
    U, _ = torch.linalg.qr(torch.randn(rows, k, generator=g))
    V, _ = torch.linalg.qr(torch.randn(cols, k, generator=g))
    return (U * torch.arange(1, k + 1, dtype=torch.float32) ** -0.7) @ V.T


# (label, rows, cols, num_row_samples)
_SHAPES = [
    ("square", 256, 256, 64),
    ("1:4", 128, 512, 64),
    ("4:1", 512, 128, 64),
    ("narrower_than_window", 48, 96, 64),
    ("wider_than_window", 700, 200, 64),
    ("llm_window_capped_at_256", 256, 1024, 2000),
    ("225_windows", 600, 600, 128),
]
_CASES = [(f"{kind}-{label}", kind, r, c, w) for label, r, c, w in _SHAPES
          for kind in ("gaussian", "student_t3", "power_law")]


@pytest.mark.parametrize("label,kind,rows,cols,window", _CASES, ids=[c[0] for c in _CASES])
def test_windows_and_fit_match_the_release(ref, label, kind, rows, cols, window):
    W = _matrix(kind, rows, cols, seed=len(label))
    eigs_ref = _release_eigs(ref, W, window)
    eigs, n_windows = F._pooled_window_eigs(W, window, 1.0, 15)
    assert eigs.shape == eigs_ref.shape
    torch.testing.assert_close(eigs, eigs_ref, rtol=1e-6, atol=1e-6 * float(eigs_ref.max()))
    r, c, rs, cs = F._window_grid(rows, cols, window, 1.0, 15)
    assert n_windows == len(rs) * len(cs) and eigs.numel() == n_windows * min(r, c)

    a_ref, d_ref, i_ref = _release_fit(ref, eigs_ref)
    fit = F._xmin_peak_fit(eigs_ref)
    assert fit["xmin_index"] == i_ref
    assert fit["alpha"] == pytest.approx(a_ref, rel=1e-5)
    assert fit["D"] == pytest.approx(d_ref, abs=1e-6)


def test_window_grid_examples():
    """The window side is capped by the matrix's narrow side; the grid is at
    most 15 x 15 (fixed_sampling)."""
    assert F._window_grid(4096, 4096, 2000, 1.0, 15) == (
        2000, 2000, [149 * i for i in range(15)], [149 * i for i in range(15)])
    r, c, rs, cs = F._window_grid(1024, 4096, 2000, 1.0, 15)     # a GQA k_proj at d = 4096
    assert (r, c, len(rs), len(cs)) == (1024, 1024, 1, 15)
    r, c, rs, cs = F._window_grid(768, 768, 2000, 1.0, 15)       # gpt2: one window
    assert (r, c, rs, cs) == (768, 768, [0], [0])
    r, c, rs, cs = F._window_grid(768, 768, 512, 1.0, 15)        # the w512 twin on gpt2
    assert (r, c, len(rs), len(cs), rs[-1]) == (512, 512, 15, 15, 252)


def _tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=500, hidden_size=64, intermediate_size=176,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2)
    return transformers.LlamaForCausalLM(cfg).eval()


def _tiny_qwen2():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=500, hidden_size=96, intermediate_size=224,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def _tiny_gpt2():
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"tiny-random-gpt2 not cached: {e}")


def _linear_equivalent(blocks):
    """A copy of ``blocks`` with every HF Conv1D (weight (in, out)) replaced by the
    nn.Linear computing the same map (weight (out, in) = W.T), names unchanged."""
    blocks = copy.deepcopy(blocks)
    for parent in list(blocks.modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, Conv1D):
                nx, nf = child.weight.shape
                lin = torch.nn.Linear(nx, nf, bias=False)
                lin.weight.data = child.weight.detach().t().contiguous().clone()
                setattr(parent, cname, lin)
    return blocks


def _blme_alphas(model, num_row_samples=2000):
    names, mats, counts = F._farms_matrices(model)
    out = {}
    with torch.no_grad():
        for n, W in zip(names, mats):
            fit = F._farms_alpha(W, num_row_samples, 1.0, 15)
            if fit is not None:
                out[n] = fit["alpha"]
    return out, counts


@pytest.mark.parametrize("build,prefix", [(_tiny_llama, "model.layers."), (_tiny_qwen2, "model.layers."),
                                          (_tiny_gpt2, "transformer.h.")],
                         ids=["llama", "qwen2", "gpt2_conv1d"])
@pytest.mark.parametrize("window", [2000, 512], ids=["llm_config", "w512"])
def test_population_and_alphas_equal_the_release(ref, build, prefix, window):
    model = build()
    blocks = model.model.layers if prefix == "model.layers." else model.transformer.h
    is_gpt2 = prefix == "transformer.h."
    rel = _release_on_blocks(ref, _linear_equivalent(blocks) if is_gpt2 else blocks,
                             num_row_samples=window)
    ours, counts = _blme_alphas(model, window)
    assert list(ours) == [prefix + n for n in rel]
    assert len(rel) == (4 if is_gpt2 else 7) * len(blocks)
    assert counts["conv1d_included"] == (4 * len(blocks) if is_gpt2 else 0)
    for n, a in rel.items():
        assert ours[prefix + n] == pytest.approx(a, rel=1e-5), n


def test_release_cannot_see_conv1d(ref):
    """The departure exists: on the original GPT-2 blocks the release finds nothing."""
    assert _release_on_blocks(ref, _tiny_gpt2().transformer.h) == {}


class _Blocks(torch.nn.Module):
    """A model whose blocks hold a 1x4 Linear: 4 pooled eigenvalues, fewer than
    eigs_num_thresh = 5."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        blk = torch.nn.Module()
        blk.big = torch.nn.Linear(64, 64)
        blk.tiny = torch.nn.Linear(4, 1)
        blk.wide = torch.nn.Linear(64, 200)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([blk])


def test_matrices_with_fewer_than_5_eigenvalues_are_skipped_as_in_the_release(ref):
    model = _Blocks()
    rel = _release_on_blocks(ref, model.model.layers)
    assert list(rel) == ["0.big", "0.wide"]
    out = F.FARMSAlphaTask({}).evaluate(model, None, None)
    assert out["_meta_farms_modules"] == ["model.layers.0.big", "model.layers.0.wide"]
    assert out["_meta_n_skipped_fewer_than_5_eigenvalues"] == 1
    assert out["_meta_n_matrices"] == 3
    for n, a in zip(out["_meta_farms_modules"], out["_meta_farms_alphas"]):
        assert a == pytest.approx(rel[n.replace("model.layers.", "")], rel=1e-5)


def test_evaluate_reports_the_release_mean_and_the_w512_twin(ref):
    model = _tiny_llama()
    out = F.FARMSAlphaTask(resolve_task_config("geometry_farms_alpha")).evaluate(model, None, None)
    rel = _release_on_blocks(ref, model.model.layers)
    rel512 = _release_on_blocks(ref, model.model.layers, num_row_samples=512)
    assert out["farms_alpha_mean"] == pytest.approx(float(np.mean(list(rel.values()))), rel=1e-5)
    assert out["farms_alpha_mean_w512"] == pytest.approx(float(np.mean(list(rel512.values()))),
                                                         rel=1e-5)
    assert out["_meta_n_matrices"] == out["_meta_n_matrices_fitted"] == 14
    assert out["_meta_window_config"] == [2000, 1.0, 15]
    assert out["_meta_window_config_w512"] == [512, 1.0, 15]
    assert out["_meta_n_conv1d_included"] == 0
    assert len(out["_meta_farms_alphas"]) == len(out["_meta_farms_modules"]) == 14


def test_defaults_are_the_release_llm_configuration():
    cfg = resolve_task_config("geometry_farms_alpha")
    assert (cfg["num_row_samples"], cfg["q_ratio"], cfg["sampling_ops_per_dim"]) == (2000, 1.0, 15)
    out = F.FARMSAlphaTask(dict(cfg, num_row_samples=16)).evaluate(_tiny_llama(), None, None)
    assert out["_meta_window_config"] == [16, 1.0, 15]
    assert set(out["_meta_farms_window_rows"]) == {16}
    assert out["_meta_window_config_w512"] == [512, 1.0, 15]


def test_transposition_keeps_the_pooled_eigenvalues_when_the_grid_is_symmetric():
    """Q = 1: the grid of W^T is the transposed grid of W, so the pooled
    eigenvalues agree to SVD rounding (the claim behind measuring a Conv1D as
    W.T). Q != 1 makes the windows of W and W^T different shapes."""
    W = _matrix("student_t3", 96, 300, seed=5)
    for window in (32, 2000):
        a, na = F._pooled_window_eigs(W, window, 1.0, 15)
        b, nb = F._pooled_window_eigs(W.t(), window, 1.0, 15)
        assert na == nb and a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5 * float(a.max()))
    a, _ = F._pooled_window_eigs(W, 32, 0.5, 15)
    b, _ = F._pooled_window_eigs(W.t(), 32, 0.5, 15)
    assert not torch.allclose(a, b, rtol=1e-2, atol=1e-3 * float(a.max()))


def _student_t3(g, rows, cols):
    z = torch.randn(rows, cols, generator=g)
    chi2 = (torch.randn(3, rows, cols, generator=g) ** 2).sum(0)
    return z / torch.sqrt(chi2 / 3) / 16.0


@pytest.mark.parametrize("dist", ["gaussian", "student_t3"])
def test_known_positive_fixed_windows_remove_the_aspect_ratio_bias(dist):
    """Known positive (the paper's point): iid matrices of one entry variance
    and aspect ratios 1, 4 and 16 (256 x 256, 256 x 1024, 256 x 4096), 8 seeds.

    Measured (2026-11, CPU, these seeds): the whole-matrix PL_Alpha_Hill
    (``_hill_alpha_xmin_mid``) rises with the aspect ratio in every seed (by at
    least 3.64 Gaussian, 2.60 Student-t), means 2.120 / 3.460 / 5.910 (Gaussian)
    and 2.052 / 3.228 / 5.103 (Student-t, 3 dof); FARMS with a fixed 64 x 64
    window (225 windows per matrix) has means 5.834 / 6.408 / 6.172 (Gaussian;
    no trend, spread 0.574 = 0.15 of the Hill spread 3.790) and 3.389 / 3.320 /
    3.403 (Student-t; spread 0.084 = 0.03 of 3.051). On a Gaussian bulk the
    xmin_peak fit is noisy (per-seed standard deviation 0.9-1.7), so only the
    means are compared there.
    """
    seeds = 8
    hill, farms = {1: [], 4: [], 16: []}, {1: [], 4: [], 16: []}
    for seed in range(seeds):
        g = torch.Generator().manual_seed(seed)
        per_seed = []
        for r in (1, 4, 16):
            W = (torch.randn(256, 256 * r, generator=g) / 16.0 if dist == "gaussian"
                 else _student_t3(g, 256, 256 * r))
            per_seed.append(_hill_alpha_xmin_mid(W)[0])
            hill[r].append(per_seed[-1])
            farms[r].append(F._farms_alpha(W, 64, 1.0, 15)["alpha"])
        assert per_seed[0] < per_seed[1] < per_seed[2]
        assert per_seed[2] - per_seed[0] > 2.0
    h = [float(np.mean(hill[r])) for r in (1, 4, 16)]
    f = [float(np.mean(farms[r])) for r in (1, 4, 16)]
    hill_spread, farms_spread = max(h) - min(h), max(f) - min(f)
    assert hill_spread > 2.5
    assert farms_spread < (0.3 if dist == "gaussian" else 0.1) * hill_spread


def test_isometric_embedding_and_zero_padding_change_the_matrix_alpha():
    """What the ADAPTATION_NOTE says is not invariant: a rotated embedding of a
    matrix changes what its windows hold; a zero-padded matrix has windows with
    exactly-zero eigenvalues, where the release's histc raises and BLME
    returns NaN with a reason."""
    torch.manual_seed(1)
    W = torch.randn(48, 80) / 9.0
    a = F._farms_alpha(W, 2000, 1.0, 15)["alpha"]
    Q, _ = torch.linalg.qr(torch.randn(120, 48))
    R, _ = torch.linalg.qr(torch.randn(80, 80))
    assert abs(F._farms_alpha(Q @ W @ R, 2000, 1.0, 15)["alpha"] - a) > 0.1
    padded = torch.zeros(96, 160)
    padded[:48, :80] = W
    fit = F._farms_alpha(padded, 2000, 1.0, 15)
    assert math.isnan(fit["alpha"]) and "histc" in fit["reason"]


def test_rounding_of_the_smallest_eigenvalue_can_move_the_release_alpha(ref):
    """Known fragility of the released fit, reproduced, not fixed: gpt2's
    h.6.mlp.c_proj, measured as BLME measures it (the Linear equivalent of the
    Conv1D weight) with 512 x 512 windows, pools 115,200 eigenvalues whose two
    highest log10-histogram bins hold 8131 and 8130. Scaling the smallest
    eigenvalue (1.9e-10; the largest is 166) by under 1%, the size of the
    difference between float32 SVDs of the windows (it depends on the BLAS
    thread count), shifts the bin range so the other bin wins: alpha is 3.2748
    or 2.8572, in the release and in BLME alike. Which side the measured
    spectrum falls on is itself BLAS-dependent, so the test scans a +-2% grid
    for both. The near-tie and the xmin at the 1.5 p cap are what
    ``_meta_n_peak_bin_near_tie`` and ``_meta_n_xmin_at_candidate_cap`` count."""
    try:
        gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"gpt2 not cached: {e}")
    W = gpt2.transformer.h[6].mlp.c_proj.weight.detach().t()
    eigs, _ = F._pooled_window_eigs(W, 512, 1.0, 15)
    branches = {}
    for k in range(-15, 16):
        nudged = eigs.clone()
        nudged[0] = nudged[0] * (1 + 0.0013 * k)
        fit = F._xmin_peak_fit(nudged)
        assert fit["peak_near_tie"] and fit["xmin_at_cap"]
        branches.setdefault(fit["alpha"] > 3.0, (fit["alpha"], nudged))
    assert set(branches) == {False, True}
    assert branches[False][0] == pytest.approx(2.857, abs=5e-3)
    assert branches[True][0] == pytest.approx(3.275, abs=5e-3)
    for _, e in branches.values():
        ours = F._xmin_peak_fit(e)
        a_ref, _, i_ref = _release_fit(ref, e)
        assert ours["xmin_index"] == i_ref and ours["alpha"] == pytest.approx(a_ref, rel=1e-5)


def test_candidate_lower_bound_is_in_log10_units_as_released(ref):
    """The released lower bound log10(0.95 p) is negative, so no bound, while
    the peak p is below 1/0.95, and excludes eigenvalues once p is larger:
    rescaling a matrix by 64 (eigenvalues by 4096) moves the histogram rigidly
    in log10 and keeps the upper end 1.5 p in place, but cuts the candidate
    range from below. The release does the same (same index and alpha)."""
    W = _matrix("student_t3", 128, 512, seed=2)
    eigs = F._pooled_window_eigs(W, 64, 1.0, 15)[0]
    small, big = F._xmin_peak_fit(eigs), F._xmin_peak_fit(eigs * 4096)
    assert small["candidate_range"][0] == 0 < big["candidate_range"][0]
    assert abs(small["candidate_range"][1] - big["candidate_range"][1]) <= 2
    a_ref, _, i_ref = _release_fit(ref, eigs * 4096)
    assert big["xmin_index"] == i_ref and big["alpha"] == pytest.approx(a_ref, rel=1e-5)
