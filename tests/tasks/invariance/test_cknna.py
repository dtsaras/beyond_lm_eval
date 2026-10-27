"""geometry_cknna: the platonic-rep feature protocol (per-text masked mean,
joint outlier clamp, embedding layer, 1024 texts), driven through the real
``evaluate()``."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from _family_c import TEXTS, inv, load

TASK = "geometry_cknna"


@pytest.fixture(scope="module")
def tiny():
    return load("hf-internal-testing/tiny-random-gpt2")


@pytest.fixture(scope="module")
def tiny_states(tiny):
    return inv.states_from_model(*tiny, TEXTS)


def _doc_means(docs):
    return torch.stack([torch.as_tensor(np.asarray(d, dtype=np.float64)).mean(0)
                        for d in docs]).float()


_REF = Path.home() / ".cache" / "blme-refrepos" / "platonic-rep" / "metrics.py"


def _reference_remove_outliers():
    """platonic-rep@dcd76ba ``remove_outliers``, extracted by AST (the module
    imports torchaudio, which the test environment does not need)."""
    if not _REF.exists():
        pytest.skip(f"reference not available: {_REF}")
    tree = ast.parse(_REF.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "remove_outliers")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_REF), "exec"), ns)
    return ns["remove_outliers"]


def _all_entries(S):
    """The raw HF tuple: embedding, blocks 0..L-2, then the post-norm entry."""
    return [S.embedding] + [S.blocks[i] for i in range(S.n_layers - 1)] + [S.final_norm]


def test_joint_clamp_matches_the_reference_bit_for_bit():
    """One bound over all layers, as measure_alignment.py:103 applies
    remove_outliers(exact=False) to the N x L x D tensor."""
    from blme.tasks.geometry.cknna import _remove_outliers_joint
    ref = _reference_remove_outliers()
    g = torch.Generator().manual_seed(0)
    F = {li: torch.randn(30, 17, generator=g) * (1 + 10 * li) for li in range(4)}
    got, bound = _remove_outliers_joint(F, [0, 1, 2, 3])
    want = ref(torch.stack([F[li] for li in range(4)], dim=1), q=0.95, exact=False)
    for li in range(4):
        assert torch.equal(got[li], want[:, li, :])


def test_points_are_per_text_means_with_one_joint_clamp(tiny_states):
    """Huh et al. score one vector per text (extract_features.py pool='avg'),
    clamped by one bound over all layers: early_late_cknna equals the
    reference preprocessing on per-document means of the HF entries."""
    from blme.tasks.geometry.cknna import _cknna, _l2_normalize
    ref = _reference_remove_outliers()
    r = inv.run_task_on_states(TASK, tiny_states, {"topk": 5})
    S = tiny_states
    entries = _all_entries(S)
    feats = torch.stack([_doc_means(e) for e in entries], dim=1)     # N x L x D
    clamped = ref(feats, q=0.95, exact=False)
    first, last = _l2_normalize(clamped[:, 0]), _l2_normalize(clamped[:, -1])
    want = _cknna(first, last, topk=5, unbiased=True)
    assert r["_meta_n_points"] == S.n_docs
    assert r["_meta_n_layers"] == S.n_layers + 1        # embedding included
    assert r["_meta_includes_embedding_layer"] is True
    assert r["_meta_pooling"] == "mean"
    assert r["_meta_outlier_clamp"] == "joint_over_layers"
    assert r["early_late_cknna"] == pytest.approx(want, rel=1e-6)
    # Known negative: the per-layer clamp the task used before scores differently.
    from blme.tasks.geometry.cknna import _remove_outliers
    per_layer = _cknna(_l2_normalize(_remove_outliers(feats[:, 0])),
                       _l2_normalize(_remove_outliers(feats[:, -1])), topk=5)
    assert abs(per_layer - want) > 1e-4


def test_reference_sample_size_and_no_silent_cache_cap(tiny):
    """Q3: 1024 texts is the default, and a shared cache holding fewer
    documents than requested cannot shrink N -- the task pools its own
    forward pass over the requested texts and reports the N it used."""
    from blme.cache import ModelOutputCache
    from blme.registry import get_task
    from blme.tasks.config_loader import resolve_task_config
    from invariance import limit_threads
    assert resolve_task_config(TASK, {})["num_samples"] == 1024
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    small = ModelOutputCache(model, tok, dataset=data[:6], num_samples=6)
    with limit_threads(1):
        small.populate(need_hidden=True, need_attentions=False, need_logits=False)
        r = get_task(TASK)(config={"num_samples": 12, "topk": 5}).evaluate(
            model, tok, dataset=data, cache=small)
    assert r["_meta_n_points"] == 12 == r["_meta_n_texts_requested"]
    assert r["_meta_features_source"] == "self_collected"


def test_pooled_forward_matches_the_cache_route(tiny):
    """The self-collected route pools per text on the fly and reproduces the
    shared-cache route bit for bit."""
    model, tok = tiny
    a = inv.run_task_on_model(TASK, model, tok, TEXTS, {"topk": 5}, use_cache=True)
    b = inv.run_task_on_model(TASK, model, tok, TEXTS, {"topk": 5}, use_cache=False)
    assert a["_meta_features_source"] == "shared_cache"
    assert b["_meta_features_source"] == "self_collected"
    keys = [k for k in a if isinstance(a[k], float)]
    inv.assert_invariant(a, b, keys, rtol=0.0)
