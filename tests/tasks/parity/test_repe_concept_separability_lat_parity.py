"""Parity: the LAT concept-reading headline vs the RELEASED RepE classes.

Runs ``repe.rep_readers.PCARepReader`` (get_rep_directions, get_signs,
transform) and ``repe_eval.tasks.utils.shuffle_all_train_choices`` from
andyzoujm/representation-engineering @5455d8a, with the evaluation loop of
``repe_eval/rep_reading_eval.py:67-87`` (choice accuracy per layer, first
maximum of the validation accuracy over hidden_layers = -1, -2, ...), on the
same per-layer states and splits the task uses. ``.cuda()`` is made a no-op so
the reference runs on CPU. Skips when the clone is absent (fails under
BLME_REQUIRE_REFS=1).
"""
import importlib.util
import os
import random
from itertools import islice

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

_REPO = os.path.join(os.environ.get("BLME_REFREPOS", os.path.expanduser("~/.cache/blme-refrepos")),
                     "representation-engineering")


def _load(rel, name):
    path = os.path.join(_REPO, rel)
    if not os.path.exists(path):
        if os.environ.get("BLME_REQUIRE_REFS") == "1":
            pytest.fail(f"reference not available: {path}")
        pytest.skip(f"reference not available: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ref(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    return (_load("repe/rep_readers.py", "repe_rep_readers_ref"),
            _load("repe_eval/tasks/utils.py", "repe_eval_task_utils_ref"))


def _reference_rotation(ref, states, hidden_layers, tr, va, te, seed):
    readers, utils = ref
    train_data = [[f"T{i}", f"F{i}"] for i in tr]
    train_labels = [[1, 0] for _ in tr]
    train_data, train_labels = utils.shuffle_all_train_choices(train_data, train_labels, seed)

    def rows(names, h):
        T, F = states[h]
        return np.stack([(T if n[0] == "T" else F)[int(n[1:])] for n in names])

    flat = [n for pair in train_data for n in pair]
    val, test = [], []
    for h in hidden_layers:
        H = rows(flat, h)
        rel = {h: H[::2] - H[1::2]}
        reader = readers.PCARepReader(n_components=1)
        reader.directions = reader.get_rep_directions(None, None, rel, [h])
        reader.directions[h] = reader.directions[h].astype(np.float32)
        reader.direction_signs = reader.get_signs({h: H}, train_labels, [h])
        for split, sink in ((va, val), (te, test)):
            names = [n for i in split for n in (f"T{i}", f"F{i}")]
            labels = [[1, 0] for _ in split]
            Ht = reader.transform({h: rows(names, h)}, [h], 0)[h]
            groups = [list(islice(Ht, 2 * i, 2 * i + 2)) for i in range(len(split))]
            pick = np.argmin if reader.direction_signs[h][0] == -1 else np.argmax
            sink.append(float(np.mean([labels[i].index(1) == pick(g) for i, g in enumerate(groups)])))
    best = val.index(max(val))
    return hidden_layers[best], val[best], test[best], val, test


def test_lat_reading_matches_released_pca_reader(ref):
    from sklearn.model_selection import GroupKFold
    from blme.tasks.repe.concept_separability import _fact_pair_groups, _FACT_PAIRS, _lat_reading

    rng = np.random.default_rng(0)
    n, d = len(_FACT_PAIRS), 24
    hidden_layers = [-1, -2, -3, -4]
    states = {}
    for k, h in enumerate(hidden_layers):
        concept = rng.standard_normal(d)
        base = rng.standard_normal((n, d)) * 2.0
        noise = rng.standard_normal((n, d)) * (1.0 + 1.5 * k)
        states[h] = ((base + 0.3 * concept).astype(np.float32),
                     (base - 0.3 * concept + noise).astype(np.float32))
    gid = _fact_pair_groups(_FACT_PAIRS)

    got = _lat_reading(states, hidden_layers, n, gid, seed=0)
    folds = [te for _, te in GroupKFold(n_splits=3).split(np.zeros(n), np.zeros(n), np.asarray(gid))]
    for r in range(3):
        layer, v, t, vc, tc = _reference_rotation(
            ref, states, hidden_layers, folds[(r + 1) % 3], folds[(r + 2) % 3], folds[r], 0)
        assert got[r]["val_curve"] == vc and got[r]["test_curve"] == tc
        assert (got[r]["layer"], got[r]["val"], got[r]["test"]) == (layer, v, t)
    # The fixture is not degenerate: layers differ and the best is above chance.
    assert len({x for r in got for x in r["test_curve"]}) > 2
    assert max(r["test"] for r in got) > 0.7
