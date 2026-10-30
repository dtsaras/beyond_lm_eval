"""SinkNormCollector (self-forwarding tasks) builds the masks that
sink_keep_masks builds on the full rows of a cache-fed task."""
from __future__ import annotations

import torch

from _family_c import TEXTS, inv, load
from _sinks import inject_massive

from blme.tasks._sink_norms import SinkNormCollector
from blme.tasks._tokens import middle_depth_layers, sink_keep_masks


def test_norm_columns_give_the_full_row_masks():
    """Real gpt2 states with a massive row at one middle block: masks and
    info from the (T, 1) norm columns equal those from the full rows, bit for
    bit, and the collector reads its shares on block L // 2."""
    S = inv.states_from_model(*load("gpt2"), TEXTS[:6])
    S = inject_massive(S, S.n_layers // 2)
    blocks = {k: [torch.as_tensor(x) for x in S.blocks[k]] for k in S.blocks}
    keys = middle_depth_layers(blocks)
    want, want_info = sink_keep_masks([blocks[k] for k in keys])
    col = SinkNormCollector()
    for i in range(S.n_docs):
        col.add([blocks[k][i] for k in sorted(blocks)])
    got, got_info = col.masks()
    assert col.keys == keys and want_info["n_massive_dropped"] == S.n_docs
    assert got_info == want_info
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert all(torch.equal(a, b) for a, b in zip(col.mid_docs, blocks[S.n_layers // 2]))
