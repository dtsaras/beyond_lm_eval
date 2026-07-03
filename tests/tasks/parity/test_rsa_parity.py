"""
Numeric-parity test: BLME geometry_rsa vs rsatoolbox (rsagroup/rsatoolbox).

BLME kernel (src/blme/tasks/geometry/rsa.py, RepresentationalSimilarityTask.evaluate,
lines ~80-99):
    per-layer RDM : scipy.spatial.distance.pdist(X, metric="euclidean")
    cross-layer   : scipy.stats.spearmanr(rdm_a, rdm_b)  -> the RSA feature

Official reference (rsatoolbox 0.3.2):
    rsatoolbox.data.Dataset
    rsatoolbox.rdm.calc_rdm(method='euclidean')
    rsatoolbox.rdm.compare(method='spearman')

Apples-to-apples note: rsatoolbox method='euclidean' returns SQUARED euclidean
distance / n_features (see calc_rdm_euclidean). BLME uses PLAIN euclidean. Both
RSA values must still match exactly because Spearman rank correlation is invariant
to the strictly-monotonic transform d -> d^2 / n_features on non-negative distances.
The test verifies (a) the RSA statistic matches to 1e-12 and (b) the RDM
construction relationship holds (rsatoolbox RDM == BLME pdist^2 / n_features).
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

rsatoolbox = pytest.importorskip("rsatoolbox")
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare


def _toy_inputs():
    rng = np.random.default_rng(20240624)
    n_cond, n_feat = 8, 6
    X1 = rng.standard_normal((n_cond, n_feat))
    Q, _ = np.linalg.qr(rng.standard_normal((n_feat, n_feat)))
    X2 = X1 @ Q + 0.5 * rng.standard_normal((n_cond, n_feat))
    return X1, X2, n_feat


def _blme_rsa(X1, X2):
    """Replicate BLME's exact RSA code path (pdist euclidean + spearmanr)."""
    rdm1 = pdist(X1, metric="euclidean")
    rdm2 = pdist(X2, metric="euclidean")
    rho, _ = spearmanr(rdm1, rdm2)
    return float(rho)


def _rsatoolbox_rsa(X1, X2):
    R1 = calc_rdm(Dataset(X1), method="euclidean")
    R2 = calc_rdm(Dataset(X2), method="euclidean")
    return float(compare(R1, R2, method="spearman")[0, 0])


def test_rsa_statistic_parity():
    X1, X2, _ = _toy_inputs()
    blme = _blme_rsa(X1, X2)
    official = _rsatoolbox_rsa(X1, X2)
    assert abs(blme - official) <= 1e-12, (
        f"RSA mismatch: BLME={blme} rsatoolbox={official} diff={abs(blme - official)}"
    )


def test_rdm_construction_relationship():
    """rsatoolbox euclidean RDM == BLME plain-euclidean RDM squared / n_features."""
    X1, _, n_feat = _toy_inputs()
    rdm_blme = pdist(X1, metric="euclidean")
    rdm_rsatoolbox = calc_rdm(Dataset(X1), method="euclidean").get_vectors()[0]
    assert np.max(np.abs(rdm_rsatoolbox - rdm_blme ** 2 / n_feat)) <= 1e-12


def test_rsa_invariant_to_monotonic_rdm_transform():
    """The Spearman RSA is unchanged whether RDMs are plain or squared euclidean."""
    X1, X2, n_feat = _toy_inputs()
    plain = _blme_rsa(X1, X2)
    sq1 = pdist(X1, metric="euclidean") ** 2 / n_feat
    sq2 = pdist(X2, metric="euclidean") ** 2 / n_feat
    squared, _ = spearmanr(sq1, sq2)
    assert abs(plain - float(squared)) <= 1e-12
