"""The pre-registered replacement candidates are emitted as ``_meta_`` keys.

RESOLVABILITY_PREREG §6 lists, for the BLME summaries whose resolvability is
decided in Amendment 10, alternatives measured in the same runs: the 95th/99th
percentiles of the same per-point LID estimates behind ``lid_max``, and the
mean of the beta_1 curve behind ``max_betti_1``. Each is checked against an
independent computation from the same values.
"""
import numpy as np

from blme.tasks.geometry import lid as lid_mod


def test_lid_quantiles_are_quantiles_of_the_per_point_estimates(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    seen = []
    real = lid_mod.np.array

    def spy(obj, *a, **k):
        out = real(obj, *a, **k)
        if isinstance(obj, list) and obj and isinstance(obj[0], float) and len(obj) == 400:
            seen.append(out)
        return out

    monkeypatch.setattr(lid_mod.np, "array", spy)
    res = lid_mod._compute_lid_for_matrix(X, 20)
    monkeypatch.undo()
    per_point = seen[-1]
    assert res["lid_max"] == float(per_point.max())
    assert res["_meta_lid_q95"] == float(np.quantile(per_point, 0.95))
    assert res["_meta_lid_q99"] == float(np.quantile(per_point, 0.99))
    assert res["lid_median"] <= res["_meta_lid_q95"] <= res["_meta_lid_q99"] <= res["lid_max"]


def test_betti_1_curve_mean(monkeypatch):
    from test_seed_offset import _texts, _tiny
    from invariance.driver import run_task_on_model
    model, tok = _tiny()
    res = run_task_on_model("topology_betti_curve", model, tok, _texts(60),
                            {"num_samples": 60}, use_cache=False)
    assert "error" not in res, res
    assert res["_meta_betti_1_curve_mean"] == float(np.mean(res["betti_1_curve"]))
