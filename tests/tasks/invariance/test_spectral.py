"""geometry_spectral: WeightWatcher's alpha is the headline; the fixed-tail
Hill proxy (whose tail length counts the zero-padded width) and the
tail_fraction config echo are _meta_."""
from __future__ import annotations

import pytest

from _family_c import inv, load

TASK = "geometry_spectral"
HILL = ("avg_alpha", "min_alpha", "max_alpha", "std_alpha", "median_alpha")


def _run(model, **cfg):
    from blme.tasks.geometry.spectral import WeightSpectralTask
    return WeightSpectralTask(cfg).evaluate(model, None, None)


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")[0]


@pytest.fixture(scope="module")
def base(tiny):
    return _run(tiny)


@pytest.fixture(scope="module")
def padded(tiny):
    return _run(inv.zero_pad_weight_matrices(tiny, 2))


def test_hill_proxy_is_meta_and_moves_under_zero_padding(base, padded):
    """Zero-padding every block matrix to (2m, 2n) keeps each operator; the
    official alpha does not move, the Hill proxy does (k = 20% of a width that
    now counts the padded zeros), so the proxy is not a headline."""
    for k in HILL:
        assert k not in base and f"_meta_{k}" in base
    inv.assert_invariant(base, padded, ["avg_alpha_official"], rtol=1e-9)
    inv.assert_moves(base, padded, "_meta_avg_alpha", 0.05)
