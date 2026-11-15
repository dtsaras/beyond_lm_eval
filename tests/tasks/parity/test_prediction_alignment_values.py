"""prediction_alignment_mean / _std on an analytic anchor.

Output rows are the V one-hot vectors, which the task centres:
c_i = e_i - 1/V, so |c|^2 = 1 - 1/V and c_i . c_j = -1/V. Each hidden state is
h_t = a_t c_{y_t} + b_t 1, with 1 orthogonal to every c_i. The cosine with
the true row minus the cosine with ANY other row is then
a_t / (|c| |h_t|), whichever wrong token the frequency-matched derangement
picks, provided it never picks the true label (all labels in a document
are distinct, so the derangement has no collisions). The headline mean and
std are the mean and std of that margin.
"""
import numpy as np
import pytest
import torch

from blme.tasks.geometry import consistency as C


def test_margin_mean_and_std_match_the_closed_form(monkeypatch):
    V, T, n_docs = 16, 8, 3
    rng = np.random.default_rng(0)
    E = torch.eye(V, dtype=torch.float64)
    one = torch.ones(V, dtype=torch.float64)
    c = E - E.mean(dim=0, keepdim=True)
    hidden, labels, margins = [], [], []
    for _ in range(n_docs):
        y = torch.as_tensor(rng.permutation(V)[:T], dtype=torch.long)
        a = torch.as_tensor(rng.uniform(0.5, 2.0, T))
        b = torch.as_tensor(rng.uniform(0.1, 1.0, T))
        h = a[:, None] * c[y] + b[:, None] * one
        hidden.append(h.float())
        labels.append(y)
        cn = float(c[0].norm())
        margins += (a / (cn * h.norm(dim=1))).tolist()
    stats = {"hidden": hidden, "labels": labels}
    monkeypatch.setattr(C, "collect_prediction_stats", lambda *a, **k: (stats, None))
    monkeypatch.setattr(C, "_get_output_projection_weight", lambda m: E.float())
    monkeypatch.setattr(C, "_sink_masks", lambda *a, **k: (
        [torch.ones(T + 1, dtype=torch.bool) for _ in range(n_docs)], {}, None, "stub"))
    import blme.tasks.geometry.categories as cat
    monkeypatch.setattr(cat, "_n_emittable", lambda tok, n: n)
    task_cls = next(v for v in vars(C).values() if isinstance(v, type)
                    and "prediction_alignment_mean" in getattr(v, "HEADLINE_KEYS", ()))
    r = task_cls({"num_samples": n_docs}).evaluate(object(), object(), [{"text": "x"}] * n_docs)
    assert r["_meta_control_collision_rate"] == 0.0
    assert r["prediction_alignment_mean"] == pytest.approx(np.mean(margins), rel=1e-5)
    assert r["prediction_alignment_std"] == pytest.approx(np.std(margins), rel=1e-4)
