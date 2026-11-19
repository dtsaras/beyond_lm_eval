"""One-point-per-sequence clouds: windows whose last token carries a massive
activation are counted, kept by default, and dropped at every depth with
massive_points: drop (docs/validation/decisions.yaml, conventions)."""
import pytest
import torch

from blme.tasks._windows import massive_window_mask, window_options, window_rows

N, D, L = 40, 8, 9


def _rows(massive_at=(), depths=range(3, 6), scale=50.0):
    g = torch.Generator().manual_seed(0)
    rows = {k: torch.randn(N, D, generator=g, dtype=torch.float64) for k in range(L)}
    for i in massive_at:
        for k in depths:
            rows[k][i] *= scale
    return rows


class _Cache:
    """Serves crafted rows through the shared-cache route of window_rows."""

    def __init__(self, rows):
        self.rows = rows

    def window_rows(self, dataset, *, n, opts, replace_final_norm):
        info = {"n_tokens": list(range(10, 10 + N)), "positions": list(range(9, 9 + N)),
                "last_is_block": True}
        return self.rows, "ws", dict(info, source="shared_window_cache")


def _call(rows, **cfg):
    return window_rows(None, None, [], _Cache(rows), n=N, opts=window_options(cfg),
                       replace_final_norm=True)


def test_default_keeps_every_window():
    assert window_options({})["massive_points"] == "keep"
    with pytest.raises(ValueError):
        window_options({"massive_points": "sometimes"})


def test_mask_flags_middle_depth_outliers_only():
    flag, ratio = massive_window_mask(_rows(massive_at=(3, 17)))
    assert flag.nonzero().flatten().tolist() == [3, 17]
    assert ratio > 10
    # A massive row at the first depth only (outside the middle) is not flagged.
    flag, _ = massive_window_mask(_rows(massive_at=(5,), depths=(0,)))
    assert not flag.any()


def test_keep_counts_and_leaves_the_rows():
    rows = _rows(massive_at=(3, 17))
    out, _, info = _call(rows)
    assert info["massive_meta"]["_meta_n_massive_points"] == 2
    assert info["massive_meta"]["_meta_massive_points"] == "keep"
    assert all(out[k] is rows[k] for k in rows)


def test_drop_removes_the_same_windows_at_every_depth():
    rows = _rows(massive_at=(3, 17))
    out, _, info = _call(rows, massive_points="drop")
    keep = [i for i in range(N) if i not in (3, 17)]
    for k in rows:
        assert torch.equal(out[k], rows[k][keep])
    assert info["positions"] == [9 + i for i in keep]
    assert info["n_tokens"] == [10 + i for i in keep]
    # The cached rows themselves are untouched.
    assert rows[4].shape[0] == N


def test_no_massive_window_is_a_no_op_for_drop():
    rows = _rows()
    out, _, info = _call(rows, massive_points="drop")
    assert info["massive_meta"]["_meta_n_massive_points"] == 0
    assert all(out[k] is rows[k] for k in rows)
