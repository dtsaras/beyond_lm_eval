"""Reference parity: BLME interpretability_attention_rollout vs the OFFICIAL
Abnar & Zuidema (2020) attention-rollout implementation.

WHAT BLME COMPUTES
    src/blme/tasks/interpretability/attention_rollout.py — the module-level
    kernel ``_attention_rollout(per_layer_attention)`` head-averages nothing
    (its input is already head-averaged (L, N, N)), augments each layer with
    the identity to model the residual connection, row-normalises, and takes
    the cumulative cross-layer matrix product
    ``rollout = Ã_L · Ã_{L-1} · ... · Ã_1``.

WHAT ABNAR & ZUIDEMA (2020, arXiv:2005.00928) ROLLOUT IS
    Identical: augment head-averaged layer attention with I, re-normalise,
    cumulative matmul across layers.

    OFFICIAL reference implementation: samiraabnar/attention_flow
        repo commit 8044f5312f4ced18d4cf66ffe28f6c045629b4ed (2021-09-08)
        attention_graph_util.py:104-119  compute_joint_attention()
    transcribed VERBATIM below as ``compute_joint_attention`` (only re-indented).

VERDICT: PARITY.
    ``_attention_rollout`` reproduces ``compute_joint_attention(add_residual=
    True)``'s final-layer joint matrix bit-for-bit (max abs diff 0.0 < 1e-9)
    on toy per-layer attention (3 layers × 5 tokens, fixed seed 1234).

ANCHORS (asserted on the BLME kernel):
    (a) single layer  -> rollout == row_normalize(0.5A+0.5I) == row_normalize(A+I)
    (b) every A_l = I -> rollout == I
    (c) rollout rows sum to 1 (row-stochastic)
    (d) rollout is non-negative
Also verified: 0.5A+0.5I ≡ A+I after row-normalisation (both end-to-end and
per-layer); and a distilgpt2 smoke run of ``evaluate()`` yields finite
features with the cache and no-cache paths agreeing bit-for-bit.
"""
import numpy as np
import pytest

from blme.tasks.interpretability import attention_rollout as ar
from blme.tasks.interpretability.attention_rollout import (
    _attention_rollout,
    _row_normalize as _blme_row_normalize,
)

REF_COMMIT = "8044f5312f4ced18d4cf66ffe28f6c045629b4ed"
TOL = 1e-9


# ---------------------------------------------------------------------------
# OFFICIAL Abnar & Zuidema rollout, transcribed VERBATIM from
# samiraabnar/attention_flow @ 8044f53  attention_graph_util.py:104-119
# ---------------------------------------------------------------------------
def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def _row_normalize(M):
    return M / M.sum(axis=-1, keepdims=True)


def _toy(seed=1234, n_layers=3, n_tokens=5):
    """Row-stochastic head-averaged per-layer attention: shape (L, N, N)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n_layers, n_tokens, n_tokens))
    return A / A.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# THE PARITY ASSERTION: BLME kernel == OFFICIAL reference, bit-for-bit.
# ---------------------------------------------------------------------------
def test_blme_rollout_matches_official_reference():
    """BLME ``_attention_rollout`` == OFFICIAL ``compute_joint_attention``[-1]."""
    A = _toy()
    ref = compute_joint_attention(A, add_residual=True)[-1]
    blme = _attention_rollout(A)
    assert blme.shape == (A.shape[1], A.shape[2])
    assert np.max(np.abs(ref - blme)) < TOL


def test_parity_multiple_seeds_and_sizes():
    """Parity holds across seeds and layer/token counts."""
    for seed in (0, 1, 7, 42, 1234):
        for L in (1, 2, 4):
            for N in (2, 3, 7):
                A = _toy(seed=seed, n_layers=L, n_tokens=N)
                ref = compute_joint_attention(A, add_residual=True)[-1]
                blme = _attention_rollout(A)
                assert np.max(np.abs(ref - blme)) < TOL, (seed, L, N)


# ---------------------------------------------------------------------------
# 0.5A+0.5I  ==  A+I  after row-normalisation.
# ---------------------------------------------------------------------------
def test_half_scaling_equiv_plus_identity_augmentation():
    """normalize(0.5A+0.5I) == normalize(A+I), per layer and end-to-end."""
    A = _toy()
    L, N, _ = A.shape
    half = np.stack([_row_normalize(0.5 * A[l] + 0.5 * np.eye(N)) for l in range(L)])
    plus = np.stack([_row_normalize(A[l] + np.eye(N)) for l in range(L)])
    assert np.max(np.abs(half - plus)) < TOL

    # End-to-end: the 0.5-form rollout equals the BLME (A+I-form) kernel.
    joint = half[0]
    for l in range(1, L):
        joint = half[l].dot(joint)
    assert np.max(np.abs(joint - _attention_rollout(A))) < TOL


# ---------------------------------------------------------------------------
# ANCHORS on the BLME kernel.
# ---------------------------------------------------------------------------
def test_anchor_a_single_layer_equals_normalized_residual():
    """(a) single layer -> rollout == normalize(A+I) == normalize(0.5A+0.5I)."""
    A1 = _toy(seed=7, n_layers=1, n_tokens=5)
    single = _attention_rollout(A1)
    expect_plus = _row_normalize(A1[0] + np.eye(5))
    expect_half = _row_normalize(0.5 * A1[0] + 0.5 * np.eye(5))
    assert np.max(np.abs(single - expect_plus)) < TOL
    assert np.max(np.abs(single - expect_half)) < TOL


def test_anchor_b_all_identity_gives_identity():
    """(b) every A_l = I -> rollout == I."""
    N = 5
    Aeye = np.stack([np.eye(N) for _ in range(3)])
    assert np.max(np.abs(_attention_rollout(Aeye) - np.eye(N))) < TOL


def test_anchor_c_rows_sum_to_one():
    """(c) rollout is row-stochastic."""
    rollout = _attention_rollout(_toy())
    assert np.max(np.abs(rollout.sum(axis=1) - 1.0)) < TOL


def test_anchor_d_nonnegative():
    """(d) rollout has no negative entries."""
    assert _attention_rollout(_toy()).min() >= 0.0


# ---------------------------------------------------------------------------
# Kernel robustness / guards.
# ---------------------------------------------------------------------------
def test_degenerate_zero_row_teleports_uniformly():
    """A fully-zeroed attention row is treated as uniform (no NaN)."""
    A = _toy(seed=3, n_layers=2, n_tokens=4)
    A[0][1] = 0.0  # zero out a row before augmentation
    rollout = _attention_rollout(A)
    assert np.all(np.isfinite(rollout))
    assert np.max(np.abs(rollout.sum(axis=1) - 1.0)) < TOL


def test_rejects_non_square_stack():
    with pytest.raises(ValueError):
        _attention_rollout(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        _attention_rollout(np.zeros((3, 5)))  # not 3-D


def test_blme_public_symbols_present():
    """The verified artifact and task class are importable module symbols."""
    assert hasattr(ar, "_attention_rollout")
    assert hasattr(ar, "AttentionRolloutTask")


def test_blme_module_has_no_reference_import():
    """BLME must gain NO dependency on the reference package. The provenance
    may be *documented* (repo/fn named in comments) but the reference must
    never be IMPORTED and its kernel must not be transcribed into the module.
    """
    import ast
    import inspect

    src = inspect.getsource(ar)
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = ("attention_flow", "attention_graph_util")
    offending = [m for m in imported if any(b in (m or "") for b in banned)]
    assert offending == [], f"BLME imports the reference package: {offending}"
    # The reference kernel is NOT re-defined inside the BLME module either.
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "compute_joint_attention" not in defined


# ---------------------------------------------------------------------------
# End-to-end evaluate() smoke on a real tiny model (skips if offline).
# ---------------------------------------------------------------------------
def test_evaluate_smoke_distilgpt2():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from blme.cache import ModelOutputCache

        tok = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", attn_implementation="eager"
        )
    except Exception as e:  # pragma: no cover - offline / no weights
        pytest.skip(f"distilgpt2 unavailable: {e}")

    model.eval()
    corpus = [
        "The quick brown fox jumps over the lazy dog near the river.",
        "Transformers use attention to route information between tokens.",
        "Residual connections carry a token's own state forward through layers.",
    ]
    cache = ModelOutputCache(model, tok, dataset=corpus, num_samples=3)
    cache.populate(need_attentions=True)

    task = ar.AttentionRolloutTask({"num_samples": 3})
    res = task.evaluate(model, tok, corpus, cache=cache)
    assert "error" not in res, res
    for k, v in res.items():
        if k.startswith("_meta_"):
            continue
        assert np.isfinite(v), (k, v)
    # Rollout influences are row-stochastic, so any mean influence is in [0, 1].
    assert 0.0 <= res["rollout_mean_influence_to_bos"] <= 1.0

    # Cache path and fallback (no-cache) path must agree bit-for-bit.
    res2 = task.evaluate(model, tok, corpus, cache=None)
    assert "error" not in res2, res2
    for k in res:
        if k.startswith("_meta_"):
            continue
        assert abs(res[k] - res2[k]) < 1e-9, (k, res[k], res2[k])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
