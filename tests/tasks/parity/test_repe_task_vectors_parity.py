"""repe_task_vectors: the shipped kernel vs the OFFICIAL RepE `ClusterMeanRepReader`.

Task: representation-engineering "task / reading vector" (Zou et al. 2023
"Representation Engineering", arXiv:2310.01405; Ilharco et al. 2023 "Editing
Models with Task Arithmetic", arXiv:2212.04089).

Shipped kernel: ``TaskVectorGeometryTask._summarise``
(src/blme/tasks/repe/task_vectors.py), called here on the fixture's matrices
as one layer. It forms v = mean_pos - mean_neg in float32 and reports its norm
(``layer_task_vector_norms``, which decides ``max_norm_layer_fraction``) and the
agreement of the per-pair differences d_i = h_i+ - h_i-
(``consensus_energy_rel``, ``pair_direction_cos_mean``).

OFFICIAL reference kernel (RepE, github.com/andyzoujm/representation-engineering,
commit 5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e):
    repe/rep_readers.py  class ClusterMeanRepReader.get_rep_directions (196-216)
        line 212  H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
        line 213  H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)
        line 215  directions[layer] = H_pos_mean - H_neg_mean

The fixture embeds synthetic paired activations (two Gaussians shifted along a
KNOWN unit direction d*) and the direction captured by RUNNING the real RepE
ClusterMeanRepReader class (a float64 run). The shipped kernel does not return
v, so parity is pinned on what it does return: ||v|| equals the released
direction's norm to rel 1e-6 and the float32 computation bit for bit, and the
consensus headline equals its closed form in that released direction,
    mean_{i != j} <d_i, d_j> = (N ||v||^2 - mean_i ||d_i||^2) / (N - 1).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "repe_task_vectors.json"
)

TOL = 1e-9


@pytest.fixture(scope="module")
def fx():
    with open(_FIXTURE) as f:
        return json.load(f)


def _shipped(pos, neg, layers=1):
    """The fixture's matrices as ``layers`` identical layers of float32 states,
    through the kernel evaluate() calls."""
    from blme.tasks.repe.task_vectors import TaskVectorGeometryTask
    P = [[torch.tensor(r, dtype=torch.float32) for r in pos] for _ in range(layers)]
    N = [[torch.tensor(r, dtype=torch.float32) for r in neg] for _ in range(layers)]
    return TaskVectorGeometryTask._summarise(P, N, list(range(len(pos))), layers)


def _cosabs(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))


def test_fixture_provenance(fx):
    assert fx["reference"].startswith("RepE ClusterMeanRepReader")
    assert fx["repe_repo_commit"] == "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e"
    assert fx["verdict"] == "PARITY"


def test_task_vector_norm_matches_cluster_mean_reader(fx):
    """||mean_pos - mean_neg|| from the shipped kernel == the norm of the
    direction RepE's ClusterMeanRepReader returned (float32 states)."""
    ref = np.array(fx["ref_direction"], np.float64)
    out = _shipped(fx["pos"], fx["neg"])
    assert float(np.linalg.norm(ref)) == pytest.approx(fx["ref_norm"], rel=1e-12)
    assert out["layer_task_vector_norms"] == [pytest.approx(fx["ref_norm"], rel=1e-6)]
    # The reading vector is the shifted direction, not noise: |cos(ref, d*)|.
    assert _cosabs(ref, fx["dstar"]) == pytest.approx(fx["abs_cos_blme_dstar"], abs=1e-9)
    assert fx["abs_cos_blme_dstar"] > 0.9


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_task_vector_norm_is_the_float32_computation_bit_for_bit(fx, offset):
    """The kernel forms v = mean_pos - mean_neg in float32 and reports its
    float32 norm; double is used only where float32 under/overflowed. The
    rel 1e-6 parity above cannot see a float64 promotion (the two norms agree
    to 2e-8 on the fixture), so the float32 arithmetic is pinned here, bit for
    bit. A common offset of 1000 separates the third candidate, float64 means
    with a float32 norm, which coincides with float32 on the bare fixture."""
    pos = (np.array(fx["pos"], np.float64) + offset).tolist()
    neg = (np.array(fx["neg"], np.float64) + offset).tolist()
    P = torch.tensor(pos, dtype=torch.float32)
    N = torch.tensor(neg, dtype=torch.float32)
    f32 = float(torch.linalg.vector_norm(P.mean(dim=0) - N.mean(dim=0)).item())
    v64 = P.double().mean(dim=0) - N.double().mean(dim=0)
    f64 = float(torch.linalg.vector_norm(v64).item())
    mixed = float(torch.linalg.vector_norm(v64.float()).item())
    assert f32 != f64
    if offset:
        assert f32 != mixed

    got = _shipped(pos, neg)["layer_task_vector_norms"][0]
    assert got == f32
    assert got == float(np.float32(got))      # a float32 value, not a double


def test_consensus_headline_is_its_closed_form_in_the_released_direction(fx):
    """consensus_energy_rel = mean_{i != j} <d_i, d_j> / mean ||h||^2, and the
    numerator equals (N ||v||^2 - mean ||d_i||^2) / (N - 1) with v the released
    direction. The plain ||v||^2 / mean ||h||^2 is a different, biased number."""
    pos = np.array(fx["pos"], np.float64)
    neg = np.array(fx["neg"], np.float64)
    ref = np.array(fx["ref_direction"], np.float64).ravel()
    n = pos.shape[0]
    d = pos - neg
    h2 = float(np.mean(np.sum(np.concatenate([pos, neg]) ** 2, axis=1)))
    u = (n * float(ref @ ref) - float(np.mean(np.sum(d * d, axis=1)))) / (n - 1)
    out = _shipped(fx["pos"], fx["neg"])
    assert out["consensus_energy_rel"] == pytest.approx(u / h2, rel=1e-5)
    assert out["layer_consensus_energy_rel"] == [out["consensus_energy_rel"]]
    assert abs(float(ref @ ref) / h2 - u / h2) > 1e-3

    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    G = dn @ dn.T
    pw = G[np.triu_indices(n, k=1)]
    assert out["pair_direction_cos_mean"] == pytest.approx(float(pw.mean()), rel=1e-5)
    assert 0.0 < out["pair_direction_cos_mean"] < 0.5


def test_sign_flip_symmetry(fx):
    """Swapping pos<->neg negates every d_i: the vector norm and both
    agreement headlines are unchanged."""
    a = _shipped(fx["pos"], fx["neg"])
    b = _shipped(fx["neg"], fx["pos"])
    for key in ("layer_task_vector_norms", "consensus_energy_rel", "pair_direction_cos_mean"):
        assert b[key] == pytest.approx(a[key], rel=1e-6), key


def test_max_norm_layer_fraction_follows_the_released_norm(fx):
    """Three layers holding the fixture at scales 1, 3, 2: the largest reading
    vector is at layer 1 of 3, depth fraction 1/2."""
    from blme.tasks.repe.task_vectors import TaskVectorGeometryTask
    scales = (1.0, 3.0, 2.0)
    P = [[c * torch.tensor(r, dtype=torch.float32) for r in fx["pos"]] for c in scales]
    N = [[c * torch.tensor(r, dtype=torch.float32) for r in fx["neg"]] for c in scales]
    out = TaskVectorGeometryTask._summarise(P, N, list(range(len(fx["pos"]))), 3)
    assert out["layer_task_vector_norms"] == [
        pytest.approx(c * fx["ref_norm"], rel=1e-6) for c in scales]
    assert out["max_norm_layer_fraction"] == 0.5
    # Scale-free headlines do not see the per-layer scale.
    assert out["layer_consensus_energy_rel"] == [
        pytest.approx(out["layer_consensus_energy_rel"][0], rel=1e-5)] * 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
