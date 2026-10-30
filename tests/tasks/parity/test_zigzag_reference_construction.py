"""BLME's zigzag barcode equals the ZigZagLLMs construction (rule 1).

The reference is re-typed verbatim from RitAreaSciencePark/ZigZagLLMs@2d10c07:
``src/zigzag/zigzag_DL.py`` (generate_simplex_tree, compute_layers_with_
intersection, compute_filtration_times, compute_zigzag_persistence), ``utils/
fclaux.py:ranges`` and ``src/plots/post_process.py`` (extract_pairs with
start_ind = 1 and the effective map eff = i // 2 + i % 2, dropping eff_b ==
eff_d). The clouds are seeded random walks across layers, so the kNN graphs
have no ties (sklearn and cKDTree order equal-distance neighbours differently).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

dionysus = pytest.importorskip("dionysus")
gd = pytest.importorskip("gudhi")
sk_neighbors = pytest.importorskip("sklearn.neighbors")

from blme.tasks.topology.zigzag import _bar_scalars, _effective_bars  # noqa: E402


def _ranges(i):
    for _, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1] + 1, b[-1][1] + 2


def _reference_barcode(reps, knn, dim=4):
    sID, simplices, padded, seen = {}, [], [], set()
    for i in range(len(reps)):
        padded.append([])
        G = sk_neighbors.kneighbors_graph(reps[i], n_neighbors=knn).toarray()
        S = gd.SimplexTree()
        ind = np.array(np.where(G == 1)).T
        for k in range(len(reps[i])):
            S.insert([k])
        for k in range(len(ind)):
            S.insert(list(ind[k]))
        S.expansion(dim)
        for s in S.get_skeleton(dim):
            if tuple(s[0]) not in seen:
                sID[tuple(s[0])] = len(simplices)
                simplices.append(s[0])
                seen.add(tuple(s[0]))
            padded[i].append(sID[tuple(s[0])])
    layers = []
    for i in range(2 * len(padded) - 1):
        if i % 2 == 1:
            layers.append(list(set(padded[i // 2]).intersection(padded[(i + 1) // 2])))
        else:
            layers.append(padded[i // 2])
    f = dionysus.Filtration(simplices)
    app = np.zeros((len(layers), len(simplices)), dtype=int)
    for k in range(len(layers)):
        app[k, layers[k]] = 1
    times = [list(_ranges(np.where(app[:, i] == 1)[0])) for i in range(len(simplices))]
    times = [list(np.array([list(t) for t in ts]).flatten()) for ts in times]
    _, dgms, _ = dionysus.zigzag_homology_persistence(f, times)
    return dgms


def _effective(dgm, start_ind=1):
    out = []
    for p in dgm:
        b = int(p.birth) - start_ind
        d = (10 ** 9) if not np.isfinite(p.death) else int(p.death) - start_ind
        eb, ed = b // 2 + b % 2, d // 2 + d % 2
        if eb != ed:
            out.append((eb, ed))
    return sorted(out)


def _random_walk(n_layers, n, d, step, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    out = []
    for _ in range(n_layers):
        out.append(X.copy())
        X = X + step * rng.standard_normal(X.shape)
    return out


@pytest.mark.parametrize("seed,k", [(0, 4), (1, 4), (2, 5), (3, 3)])
def test_h1_barcode_equals_the_reference_effective_barcode(seed, k):
    clouds = _random_walk(6, 40, 8, 0.6, seed)
    ref = _effective(_reference_barcode(clouds, k)[1])
    bars, L = _effective_bars(clouds, k)
    assert L == 6
    assert sorted((int(b), int(d)) for b, d in bars) == ref
    assert _bar_scalars(bars, L)["zz_true_n_bars"] == len(ref) > 0
