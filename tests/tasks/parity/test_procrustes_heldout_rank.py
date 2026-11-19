"""Held-out rank is fixed by the training spectrum and the cache precision."""
import numpy as np
import pytest
import torch

from blme.tasks._tokens import DocList
from blme.tasks.geometry.procrustes_linearity import _document_heldout_profile, _gram_rank_errors, _rank_errors


@pytest.mark.parametrize("shape", [(90, 12), (12, 24)])
@pytest.mark.parametrize("spectrum", ["full", "weak", "duplicate", "constant"])
def test_float64_gram_backfill_matches_direct_svd(shape, spectrum):
    gen = torch.Generator().manual_seed(7234)
    x, y, test, target = [torch.randn(*shape, generator=gen, dtype=torch.float64) for _ in range(4)]
    if spectrum == "weak":
        x[:, -5:] *= 1e-4
    if spectrum == "duplicate":
        x[:, -4:] = x[:, :4]
    if spectrum == "constant":
        x.fill_(1.)
    cuts = [np.sqrt(2**-23), .001, .01, .1]
    direct = _rank_errors(x, y, test, target, cuts)
    gram = _gram_rank_errors(x, y, test, target, cuts, "cpu")
    assert gram[0] == pytest.approx(direct[0], rel=1e-12)
    for c in cuts:
        assert gram[1][c][0] == direct[1][c][0]
        assert gram[1][c][1] == pytest.approx(direct[1][c][1], rel=1e-9)


def heldout(x, y, xt, yt):
    # Position zero is excluded. Document zero is the held-out parent.
    def documents(train, test):
        return DocList([torch.cat((a[:1], a)) for a in (test, train)], doc_ids=[0, 1])

    result = _document_heldout_profile({0: documents(x, xt), 1: documents(y, yt)}, [0, 1], [0])
    return result["procrustes_linearity_heldout_r2_mean"], result["_meta_procrustes_heldout"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_weak_training_direction_cannot_explode_on_unseen_documents(dtype):
    # Directions either side of the fixed cutoff. The final direction carries
    # training noise that an untruncated fit magnifies on the new document.
    axes = torch.diag(torch.tensor([1., 0.001, 0.00001], dtype=dtype))
    x = torch.cat((axes, -axes))
    x[:, 0] += 4.  # keeps all six content rows below the sink threshold
    y = torch.cat((torch.eye(3), -torch.eye(3))).to(dtype)
    xt = x.clone()
    xt[:, 2] *= 1000
    yt = y.clone()
    yt[:, 2] = 0

    saved = [a.double().numpy() for a in (x, y, xt, yt)]
    a, b, c, d = saved
    rcond = np.sqrt(float(np.finfo(np.float32).eps))
    predicted = (c - a.mean(0)) @ np.linalg.pinv(a - a.mean(0), rcond=rcond) @ (b - b.mean(0))
    expected = 1 - np.square(d - b.mean(0) - predicted).sum() / np.square(d - b.mean(0)).sum()
    for copies in (1, 8):
        score, meta = heldout(x.repeat(copies, 1), y.repeat(copies, 1), xt, yt)
        assert score == pytest.approx(expected, abs=1e-12)
        assert meta["rank_relative_floor"] == pytest.approx(rcond, abs=1e-15)
        assert meta["pair_relative_cutoff"] == pytest.approx([rcond], abs=1e-15)
        assert meta["rank_selection"] == "numerical floor only"
        assert meta["pair_fit_rank"] == [2]


def test_zero_training_rank_uses_training_mean_and_records_rank_for_undefined_r2():
    x = torch.ones((4, 2))
    y = torch.arange(8.).reshape(4, 2)
    score, meta = heldout(x, y, 2*x, y)
    assert score == pytest.approx(0.)
    assert meta["pair_fit_rank"] == [0]
    score, meta = heldout(x, y, 2*x, y.mean(0).expand_as(y))
    assert np.isnan(score)
    assert meta["pair_fit_rank"] == [0]


@pytest.mark.parametrize("validation_sign, expected_rank", [(1, 2), (-1, 1)])
def test_training_documents_choose_cutoff_before_outer_refit(validation_sign, expected_rank):
    x = np.array([[5., 0.], [3., 0.], [4., 0.002], [4., -0.002]])
    y = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    ids = list(range(6))
    def chunks(docs):
        return DocList([torch.tensor(np.vstack((a[:1], a))) for a in docs], doc_ids=ids)

    for test_gain in (1, -100):
        targets = [y.copy() for _ in ids]
        targets[1][:, 1] *= validation_sign  # calibration parent
        for i in (0, 5):
            targets[i][:, 1] *= test_gain
        result = _document_heldout_profile({0: chunks([x] * 6), 1: chunks(targets)}, [0, 1], [0])
        meta = result["_meta_procrustes_heldout"]
        assert meta["rank_fit_documents"] == ["record:2", "record:3", "record:4"]
        assert meta["rank_validation_documents"] == ["record:1"]
        assert meta["test_documents"] == ["record:0", "record:5"]
        assert meta["pair_fit_rank"] == [expected_rank]
        cutoff, = meta["pair_relative_cutoff"]
        assert cutoff == pytest.approx(10**-2.5 if validation_sign < 0 else np.sqrt(2.**-23))
        # Refit the selected rank with all four outer training parents.
        a, b = np.tile(x, (4, 1)), np.vstack(targets[1:5])
        target = targets[0] - b.mean(0)
        predicted = (x - a.mean(0)) @ np.linalg.pinv(a - a.mean(0), rcond=cutoff) @ (b - b.mean(0))
        expected = 1 - np.square(target - predicted).sum() / np.square(target).sum()
        assert result["procrustes_linearity_heldout_r2_mean"] == pytest.approx(expected, abs=1e-12)
