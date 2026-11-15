"""topology_zigzag_persistence: the headline scalars vs the release's post-processing.

zz_true_adjacent_mean is Z_1(l, l + 1) (Eq. 7, footnote 6) averaged over
adjacent layers, from the effective barcode that
test_zigzag_reference_construction pins to the ZigZagLLMs construction. The
reference is RitAreaSciencePark/ZigZagLLMs src/plots/post_process.py
POSTPROCESS (find_betti_layers, find_eff_pis, find_ph_sim) on raw zigzag
pairs: Z_1(l, l + 1) = div(PHSim[l, l + 1], beta_1[l]). BLME's scalars are
computed on the effective bars of the same pairs (eff = i // 2 + i % 2 after
start_ind, dropping eff_b == eff_d, as the release's extract_pairs does).
zz_true_mean_bar_len is the mean effective lifetime. Skips without the clone
(fails under BLME_REQUIRE_REFS=1).
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from blme.tasks.topology.zigzag import _bar_scalars

_PP = Path(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos"))) \
    / "ZigZagLLMs" / "src" / "plots" / "post_process.py"
L = 5                                   # layers; the zigzag has 2L - 1 complexes


@pytest.fixture(scope="module")
def release():
    if not _PP.is_file():
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {_PP}")
        pytest.skip(f"reference not available: {_PP}")
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("zz_post_process", _PP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _effective(pairs, start_ind=1):
    out = []
    for b, d in pairs:
        b, d = b - start_ind, d - start_ind
        eb, ed = b // 2 + b % 2, d // 2 + d % 2
        if eb != ed:
            out.append((eb, ed))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_adjacent_mean_and_bar_length_match_post_process(release, seed):
    g = np.random.default_rng(seed)
    n_steps = 2 * L - 1
    pairs = []
    for _ in range(12):                 # raw (birth, death) in zigzag steps, start_ind = 1
        b = int(g.integers(1, n_steps + 1))
        d = int(g.integers(b + 1, n_steps + 2))
        pairs.append((b, d))
    pp = release.POSTPROCESS([[], pairs], num_layers=n_steps, start_ind=1, zigzag=True)
    summ = pp.compute_persistence_summaries()
    betti, psim = summ["bettis"][1], summ["phsim"][1]
    assert betti.shape == (L,) and psim.shape == (L, L)
    want_adj = np.mean([release.div(psim[l, l + 1], betti[l]) for l in range(L - 1)])

    bars = _effective(pairs)
    got = _bar_scalars([(float(b), float(d)) for b, d in bars], L)
    assert got["zz_true_adjacent_mean"] == pytest.approx(want_adj, rel=1e-12, abs=1e-15)
    assert got["zz_true_mean_bar_len"] == pytest.approx(np.mean([d - b for b, d in bars]), rel=1e-12)
    assert 0.0 < want_adj < 1.0
