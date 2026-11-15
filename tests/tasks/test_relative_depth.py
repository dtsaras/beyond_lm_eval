"""One peak-depth convention: HF hidden-state index / L (0 = embedding output)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blme.tasks.common import relative_depth  # noqa: E402
from invariance import driver as inv  # noqa: E402


def test_relative_depth_is_hidden_state_index_over_depth():
    assert relative_depth(0, 12) == 0.0 and relative_depth(12, 12) == 1.0
    assert relative_depth(3, 12) == 0.25
    assert math.isnan(relative_depth(-1, 12)) and math.isnan(relative_depth(3, 0))


def test_gride_peak_depth_counts_block_i_as_hidden_state_i_plus_1():
    rng = np.random.default_rng(0)
    L = 4
    # Block 1's output (hidden state 2) is the high-dimensional one.
    S = {li: [rng.standard_normal((6, 16)) * (1.0 if li != 1 else 1.0)
              @ (np.eye(16) if li == 1 else np.diag([1.0] * 2 + [1e-3] * 14))
              for _ in range(40)] for li in range(L)}
    r = inv.run_task_on_states("geometry_gride", S)
    peak_layer = r["_meta_gride_peak_layer"]
    assert 1 <= peak_layer <= L
    assert r["_meta_gride_peak_layer_frac"] == peak_layer / L
    assert np.nanargmax(r["gride_id_profile"]) + 1 == peak_layer
