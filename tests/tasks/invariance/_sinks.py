"""Massive-row fixtures for the sink-policy (``exclude_sinks``) tests.

A first-period sink (pythia-70m/160m) is one massive row per document at a
fixed early position, massive at the middle depths only. ``inject_massive``
builds that on real states: one row of every document scaled x1000 at ONE
middle block. ``drop_row`` is its control: the same row deleted from every
entry, i.e. the corpus without that token. Under ``exclude_sinks`` the two must
give the same headline bit for bit (the row is dropped at every layer, not only
where it is massive); under ``exclude_pos0`` the injected row moves the
headline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS = Path(__file__).resolve().parents[2]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

import invariance as inv  # noqa: E402

FACTOR = 1000.0
POS = 3


def inject_massive(S: "inv.States", block: int, pos: int = POS, factor: float = FACTOR,
                   spike_dim=None):
    """Row ``pos`` of every document made massive at block ``block`` only.

    By default the row is scaled by ``factor``. With ``spike_dim`` it instead
    gains ``factor * ||row||`` on that one coordinate -- a massive activation
    as Sun et al. (2024) observe it (a few fixed feature dimensions), which
    also turns the row's direction; per-row scale-free metrics (correlation
    distance) see only that.
    """
    def edit(x):
        y = np.array(x, dtype=np.float64, copy=True)
        if spike_dim is None:
            y[pos] *= factor
        else:
            y[pos, spike_dim] += factor * np.linalg.norm(y[pos])
        return y
    blocks = dict(S.blocks)
    blocks[block] = [edit(x) for x in S.blocks[block]]
    return S.with_blocks(blocks)


def drop_row(S: "inv.States", pos: int = POS):
    """Row ``pos`` deleted from every document in every entry."""
    def cut(x):
        return np.delete(np.asarray(x, dtype=np.float64), pos, axis=0)
    return S.map(cut)


def headline(result, keys=None):
    keys = keys or [k for k, v in result.items() if not k.startswith("_meta_")
                    and isinstance(v, float)]
    assert keys
    return keys


def check_state_task(task, task_cls, S, monkeypatch, moving_key, keys=None,
                     config=None, block=None, spike_dim=None, max_new_over_old=0.1):
    """The four claims, through the real ``evaluate()`` on injected states:

    1. no massive row: ``exclude_sinks`` == ``exclude_pos0`` bit for bit;
    2. the injected row moves the ``exclude_pos0`` value of ``moving_key``;
    3. ``exclude_sinks`` on the injected states == on the row-deleted control,
       bit for bit, for every headline key (dropped at every layer);
    4. what is left of the move is the loss of one ordinary row, at most
       ``max_new_over_old`` of the old move.

    ``block`` (default ``L // 2``) is the middle block injected; it should be
    the layer the task reports ``_meta_massive_token_share`` on.
    """
    S_inj = inject_massive(S, S.n_layers // 2 if block is None else block,
                           spike_dim=spike_dim)

    def run(states, policy=None):
        with monkeypatch.context() as m:
            if policy is not None:
                m.setattr(task_cls, "TOKEN_POLICY", policy)
            return inv.run_task_on_states(task, states, config)

    new, old = run(S), run(S, "exclude_pos0")
    keys = headline(new, keys)
    assert new["_meta_sink_n_massive_dropped"] == 0
    inv.assert_invariant(old, new, keys, rtol=0.0)

    new_inj, old_inj = run(S_inj), run(S_inj, "exclude_pos0")
    assert new_inj["_meta_sink_n_massive_dropped"] == S.n_docs
    assert new_inj["_meta_massive_token_share"] > 0.5 > new["_meta_massive_token_share"]
    inv.assert_moves(old, old_inj, moving_key, 1e-2)
    inv.assert_invariant(run(drop_row(S)), new_inj, keys, rtol=0.0)
    d_old = abs(old_inj[moving_key] - old[moving_key])
    d_new = abs(new_inj[moving_key] - new[moving_key])
    assert d_new <= max_new_over_old * d_old, (moving_key, d_new, d_old)
    return new, new_inj


class massive_row_hook:
    """Context manager: block ``block``'s output at position ``pos`` (negative
    counts from the end) of every forward times ``factor`` -- a massive
    activation at one middle depth, for tasks that run their own forwards.

    The scaling is a multiplication by a constant tensor, so a backward pass
    through it is exact. In a causal LM the last position (``pos=-1``) is
    seen by no other token, so every other row stays bit-identical.
    """

    def __init__(self, model, block: int, pos: int = POS, factor: float = FACTOR):
        from blme.tasks.common import get_layers
        self.layer = get_layers(model)[block]
        self.pos, self.factor, self.handle = pos, float(factor), None

    def _hook(self, module, args, output):
        import torch
        h = output[0] if isinstance(output, tuple) else output
        T = h.shape[1]
        p = self.pos % T
        scale = torch.ones(T, dtype=h.dtype, device=h.device)
        scale[p] = self.factor
        h = h * scale[None, :, None]
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        return False
