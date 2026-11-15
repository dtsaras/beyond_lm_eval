"""dynamics_fractal: the pooled headline estimators vs the released notebook.

``_pooled_fractal_params`` gives hurst_exponent, joseph_exponent and
self_similarity_exponent. The reference is google-research/fractals_language
``fractal_patterns_in_language.ipynb`` (get_hurst_exponent,
get_joseph_exponent, get_self_sim_exponent, power_law_fit), taken from the
notebook with ``ast``. The notebook draws random offsets per document; here
``np.random.randint`` is scripted to the offsets BLME enumerates (every
disjoint window for R/S, every lag-tau pair for J and S), so both pool the
same increments. Each notebook call gets one scale, which also sidesteps its
Joseph accumulator that is never reset across scales (noted in fractal.py).
BLME's scales are read off its own fits; the per-scale statistic and the
fitted slope must then agree. Skips without the clone (fails under
BLME_REQUIRE_REFS=1).
"""
import ast
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest

import blme.tasks.dynamics.fractal as FR

_NB = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "google-research" / "fractals_language" / "fractal_patterns_in_language.ipynb"
_FUNCS = ("power_law_fit", "get_hurst_exponent", "get_joseph_exponent", "get_self_sim_exponent")


def _missing(what):
    if os.environ.get("BLME_REQUIRE_REFS") == "1":
        pytest.fail(f"reference not available: {what}")
    pytest.skip(f"reference not available: {what}")


class _ScriptedNumpy(types.ModuleType):
    """numpy, with ``random.randint`` returning a scripted offset sequence."""

    def __init__(self):
        super().__init__("np_scripted")
        self.offsets = iter(())
        self.random = types.SimpleNamespace(randint=lambda lo, hi: next(self.offsets))

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture(scope="module")
def nb():
    if not _NB.is_file():
        _missing(_NB)
    sklearn_lm = pytest.importorskip("sklearn.linear_model")
    nodes = []
    for cell in json.loads(_NB.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            body = ast.parse("".join(cell["source"])).body
        except SyntaxError:
            continue                      # IPython magics
        nodes += [n for n in body if isinstance(n, ast.FunctionDef) and n.name in _FUNCS]
        nodes += [n for n in body if isinstance(n, ast.Assign)
                  and all(isinstance(t, ast.Name) and t.id.startswith("DEFAULT_") for t in n.targets)]
    assert sorted({n.name for n in nodes if isinstance(n, ast.FunctionDef)}) == sorted(_FUNCS)
    snp = _ScriptedNumpy()
    ns = {"np": snp, "sklearn": types.SimpleNamespace(linear_model=sklearn_lm)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_NB), "exec"), ns)
    ns["_snp"] = snp
    return ns


def _streams(seed, n_docs=6, n_tok=1500):
    """Per-document NLL streams with long memory (an AR(1) on top of noise)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_docs):
        e = rng.standard_normal(n_tok)
        z = np.empty(n_tok)
        z[0] = e[0]
        for t in range(1, n_tok):
            z[t] = 0.9 * z[t - 1] + e[t]
        out.append(3.0 + np.abs(z))
    return out


def _blme_fits(streams, monkeypatch, eps):
    """Run the pooled estimator, recording each np.polyfit it makes."""
    fits = []
    real = np.polyfit

    def spy(x, y, deg, *a, **k):
        fits.append((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
        return real(x, y, deg, *a, **k)

    monkeypatch.setattr(FR.np, "polyfit", spy)
    out = FR._pooled_fractal_params(streams, eps=eps)
    monkeypatch.undo()
    assert len(fits) == 3, "hurst, joseph and self-similarity fits"
    return out, fits


def _scales(log_x):
    s = np.rint(np.exp(log_x)).astype(int)
    assert np.allclose(np.log(s), log_x, rtol=0, atol=1e-12)
    return [int(v) for v in s]


@pytest.mark.parametrize("seed,eps", [(0, FR.PAPER_HOLDER_EPS), (1, FR.PAPER_HOLDER_EPS), (2, 5e-2)])
def test_pooled_estimators_match_the_notebook(nb, monkeypatch, seed, eps):
    streams = _streams(seed)
    out, fits = _blme_fits(streams, monkeypatch, eps)
    xs = np.stack([(z - z.mean()) / z.std() for z in streams])     # per-document standardisation
    Xs = np.cumsum(xs, axis=1)
    n_doc, N = xs.shape
    snp = nb["_snp"]

    # Hurst: mean R/S over every disjoint window of every document.
    scales = _scales(fits[0][0])
    sr = []
    for n in scales:
        snp.offsets = iter([c * n for _ in range(n_doc) for c in range(N // n)])
        sr.append(nb["get_hurst_exponent"](xs, scales=[n], samples_per_doc=N // n)[1][0])
    np.testing.assert_allclose(fits[0][1], np.log(sr), rtol=1e-12, atol=1e-12)
    want_h = nb["power_law_fit"](np.asarray(scales, float), np.asarray(sr))[1]
    assert out["hurst_exponent"] == pytest.approx(want_h, rel=1e-9)

    # Joseph: std of all lag-tau increments of the integrated process.
    taus = _scales(fits[1][0])
    nb["get_joseph_exponent"].__globals__["X"] = Xs          # the notebook reads a global X
    sig = []
    for tau in taus:
        snp.offsets = iter([o for _ in range(n_doc) for o in range(N - tau)])
        sig.append(nb["get_joseph_exponent"](xs, scales=[tau], samples_per_doc=N - tau)[1][0])
    np.testing.assert_allclose(fits[1][1], np.log(sig), rtol=1e-12, atol=1e-12)
    want_j = nb["power_law_fit"](np.asarray(taus, float), np.asarray(sig))[1]
    assert out["joseph_exponent"] == pytest.approx(want_j, rel=1e-9)

    # Self-similarity: P(|X_{t+tau} - X_t| < eps) over every pair. BLME drops a
    # lag with no hit (the notebook would take log 0); the lags it kept are used.
    taus_s = _scales(fits[2][0])
    p0 = []
    for tau in taus_s:
        snp.offsets = iter([o for _ in range(n_doc) for o in range(N - tau)])
        p0.append(nb["get_self_sim_exponent"](Xs, scales=[tau], samples_per_doc=N - tau, eps=eps)[1][0])
    np.testing.assert_allclose(fits[2][1], np.log(p0), rtol=1e-12, atol=1e-12)
    want_s = -nb["power_law_fit"](np.asarray(taus_s, float), np.asarray(p0))[1]
    assert out["self_similarity_exponent"] == pytest.approx(want_s, rel=1e-9)
    assert len(taus_s) >= 3


def test_pooled_joseph_follows_the_brownian_bridge_law_for_white_noise(monkeypatch):
    """iid increments standardised per document (as the notebook's cell 34
    does) integrate to a Brownian bridge: E (X_{t+tau} - X_t)^2 =
    tau (1 - tau / N), so J is the slope of 0.5 log(tau (1 - tau/N)), below
    0.5 at the protocol's largest lags (about 0.44 here)."""
    rng = np.random.default_rng(3)
    N = 1024
    out, fits = _blme_fits([rng.standard_normal(N) for _ in range(400)], monkeypatch,
                           FR.PAPER_HOLDER_EPS)
    taus = np.asarray(_scales(fits[1][0]), dtype=np.float64)
    law = 0.5 * np.log(taus * (1.0 - taus / N))
    np.testing.assert_allclose(fits[1][1], law, atol=0.03)
    want = np.polyfit(np.log(taus), law, 1)[0]
    assert out["joseph_exponent"] == pytest.approx(want, abs=0.015)
    assert want < 0.47
