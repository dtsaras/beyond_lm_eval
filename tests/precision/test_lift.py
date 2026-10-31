"""The fp64 lift harness (tests/precision/lift.py): what it lifts, what it
flags, and that it agrees with an independent float64 computation."""
import os
import re
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from precision import lift as L  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "blme")


@pytest.fixture
def blme_module():
    """A throwaway module under ``blme.`` so the lift swaps its bindings."""
    made = []

    def make(code):
        name = f"blme._lift_test_{len(made)}"
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        exec("import numpy as np\nimport torch\n" + code, mod.__dict__)
        made.append(name)
        return mod
    yield make
    for name in made:
        sys.modules.pop(name, None)


def _values():
    torch.manual_seed(1)
    return (torch.randn(200, 64) * 1e3 + 1e4).to(torch.bfloat16)  # bf16-exact


def _run(f, X):
    return L.shipped_and_lifted(lambda: f(X.float()), lambda: f(X.double()))


def test_the_lift_equals_an_independent_float64_computation(blme_module):
    mod = blme_module('''
def f(x):
    x = x.float()
    r = torch.randn(x.shape[1], generator=torch.Generator().manual_seed(0))
    return float((x @ r).sum(dtype=torch.float32) / x.shape[0])
''')
    X = _values()
    shipped, lifted, cert = _run(mod.f, X)
    r = torch.randn(64, generator=torch.Generator().manual_seed(0)).double()
    assert cert.ok, cert
    assert lifted == (X.double() @ r).mean().item()
    assert shipped != lifted


def test_a_bf16_reduction_shows_up_as_shipped_minus_lifted(blme_module):
    mod = blme_module('''
def f(x, defect):
    y = x.float().sum(dim=1)
    if defect:
        y = y.to(torch.bfloat16)
    return float(y.sum() / x.shape[0])
''')
    X = _values()
    gaps = {}
    for defect in (False, True):
        s, l, cert = L.shipped_and_lifted(lambda: mod.f(X.float(), defect),
                                          lambda: mod.f(X.double(), defect))
        assert cert.ok, cert
        gaps[defect] = abs(s - l) / abs(l)
    assert gaps[False] < 1e-6 < gaps[True]


def test_numpy_casts_and_finfo(blme_module):
    mod = blme_module('''
def f(x):
    a = x.float().numpy().astype(np.float32)
    return a.dtype, float(torch.finfo(x.dtype).eps), float(np.finfo(np.float32).eps)
''')
    shipped, lifted, cert = _run(mod.f, _values())
    assert cert.ok, cert
    assert shipped[0] == np.float32 and lifted[0] == np.float64
    # Tolerances are the shipped ones, so the lift changes precision only.
    assert lifted[1:] == shipped[1:] == (float(torch.finfo(torch.float32).eps),) * 2


def test_a_narrowing_the_lift_cannot_reach_is_flagged(blme_module):
    mod = blme_module('''
def f(x, buf):
    buf.copy_(x * 2)
    return float(buf.sum())
''')
    X = _values()
    buf = torch.empty(X.shape, dtype=torch.float32)  # allocated outside the analysis
    _, _, cert = L.shipped_and_lifted(lambda: mod.f(X.float(), buf),
                                      lambda: mod.f(X.double(), buf))
    assert not cert.ok and cert.low_ops


def test_a_dtype_dependent_branch_is_flagged(blme_module):
    mod = blme_module('''
def f(x):
    if x.dtype == torch.float64:
        return float(x.sum())
    return float((x * 1).sum())
''')
    _, _, cert = _run(mod.f, _values())
    assert not cert.same_algorithm and cert.first_divergence is not None and not cert.ok


def test_library_boundaries_are_audited():
    sklearn = pytest.importorskip("sklearn.linear_model")
    X = np.random.default_rng(0).standard_normal((40, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    def f():
        return sklearn.LogisticRegression().fit(X, y).coef_
    _, _, cert = L.shipped_and_lifted(f, f)
    assert ("sklearn.LogisticRegression.fit", "float32") in cert.narrow_boundaries


def _integer_distances(seed, n=60):
    X = np.random.default_rng(seed).standard_normal((n, 3))
    D = np.round(np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) * 8)
    np.fill_diagonal(D, 0)
    return D  # exact in float32 and float64


@pytest.mark.parametrize("seed", range(5))
def test_the_ripser_reference_agrees_on_exact_inputs(seed):
    ripser = pytest.importorskip("ripser")
    pytest.importorskip("gudhi")
    D = _integer_distances(seed)
    got = L._ripser_fp64(D, distance_matrix=True, maxdim=1)["dgms"]
    want = ripser.ripser(D, distance_matrix=True, maxdim=1)["dgms"]
    for d in (0, 1):
        assert sorted(map(tuple, got[d])) == sorted(map(tuple, want[d]))
    assert np.array_equal(got[0], want[0])  # H0 in ripser's order


def test_ripser_imported_by_name_is_replaced_in_the_lift(blme_module):
    pytest.importorskip("ripser")
    pytest.importorskip("gudhi")
    mod = blme_module('''
from ripser import ripser
def f(D):
    return float(sum(d - b for b, d in ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]))
''')
    D = _integer_distances(0)
    s, l, cert = L.shipped_and_lifted(lambda: mod.f(D), lambda: mod.f(D))
    assert cert.fixed_precision == ["ripser.ripser"] == cert.substituted and cert.ok
    assert s == l


# Narrowings written so the proxies cannot see them (string dtypes, names
# imported from numpy/torch). Each is listed with why it is not analysis.
_STATIC_ALLOW = {
    ("models/wrapper.py", '"float16"'): "model loading",
    ("models/wrapper.py", '"bfloat16"'): "model loading",
    ("models/wrapper.py", '"float32"'): "model loading",
    ("tasks/_precision.py", '"bfloat16"'): "docstring",
    ("tasks/geometry/isotropy.py", '"float32"'): "the model's precision for a disclosed threshold",
    ("tasks/geometry/utils.py", '"bfloat16"'): "offload serialisation tag",
}
_STATIC = re.compile(r"""["'](?:float32|float16|bfloat16|f4|<f4|f2)["']"""
                     r"""|from (?:numpy|torch) import [^#\n]*\b(?:float32|float16|bfloat16|half|finfo)\b""")


def test_no_narrowing_hides_from_the_lift():
    found = set()
    for root, _, files in os.walk(SRC):
        for fn in files:
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, SRC)
                for line in open(path, encoding="utf-8"):
                    if line.lstrip().startswith("#"):
                        continue
                    for m in _STATIC.finditer(line):
                        found.add((rel, m.group(0)))
    assert found <= set(_STATIC_ALLOW), sorted(found - set(_STATIC_ALLOW))
