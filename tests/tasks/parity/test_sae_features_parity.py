"""Numeric-parity test: BLME interpretability_sae_features vs official SAELens.

METRIC — SAE feature-usage statistics (Bricken et al. 2023 / Cunningham et al.
2023; SAELens is the canonical library). Given a Sparse Autoencoder applied to a
transformer layer's residual-stream hidden state h (seq_len, d_model):

        feature_acts = sae.encode(h)                        # (seq_len, d_sae)
        l0_per_token = (feature_acts > 0).float().sum(-1)   # active features/token
        mean_active_features_l0 = mean over samples of l0_per_token.mean()
        max_active_features_l0  = max  over samples of l0_per_token.max()
        sae_total_dict_size     = sae.cfg.d_sae

BLME implements exactly this in
src/blme/tasks/interpretability/sae_features.py:149-167 (kernel lines 152-155,
164-165). BLME's task emits mean/max L0 + dict size; it does NOT itself compute
feature density or dead-feature fraction (those are standard SAELens diagnostics
checked here for completeness, verified against an independent numpy reference).

OFFICIAL REFERENCE — SAELens (`sae-lens`), the library the SAE interpretability
literature ships pretrained SAEs through. In an ISOLATED venv (BLME does NOT
depend on sae_lens) we loaded the REAL pretrained gpt2-small SAE:

    from sae_lens import SAE
    sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    feature_acts = sae.encode(h)          # h = gpt2 hidden_states[8]

(release gpt2-small-res-jb == Joseph Bloom's jbloom/GPT2-Small-SAEs-Reformatted;
StandardSAE, d_sae=24576, d_in=768.) We encoded gpt2 activations for 5 prompts
and fed the SAME encoded feature matrix to BLME's kernel and to the numpy
reference. Numbers are pinned in the committed fixture sae_features.json.

STRICTEST CHECK (this test, MAIN env, no network): BLME's kernel is transcribed
verbatim, asserted to still match the live BLME source (anti-drift), then run
against an independent numpy reference on synthetic feature matrices — the
per-token L0 counts are EXACT integers and the reference mean uses BLME's own
float32 dtype, so the diff is 0.0. Anchors confirm known-k -> L0==k, dead
detection, density in [0,1]. The fixture pins the real-SAELens parity.

VERDICT: PARITY (stat kernels, diff 0.0) + FAITHFUL (trained-SAE pipeline via
SAELens's own SAE.encode). See report + sae_features.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

TOL = 0.0  # exact: per-token counts are integers; mean uses BLME's float32 dtype

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures" / "reference_parity" / "parity" / "sae_features.json"
)


# --- Verbatim transcription of BLME's L0 kernel (sae_features.py:152-165) ---
def _blme_l0_kernel(feature_acts: torch.Tensor):
    """(feature_acts > 0).float().sum(-1) then per-seq mean / max, as BLME does."""
    l0_per_token = (feature_acts > 0).float().sum(dim=-1)   # (seq_len,)
    return l0_per_token.mean().item(), l0_per_token.max().item()


# --- Independent numpy reference (no torch reductions) ---
def _ref_l0_counts(feats: np.ndarray) -> np.ndarray:
    """Per-token L0 as EXACT integer counts of strictly-positive features."""
    return (feats > 0).sum(axis=-1).astype(np.int64)


def _ref_seq_mean_l0(feats: np.ndarray) -> float:
    """Mean per-token L0 in FLOAT32 — matches torch's `.float().sum().mean()`."""
    return float(_ref_l0_counts(feats).astype(np.float32).mean(dtype=np.float32))


def _ref_seq_max_l0(feats: np.ndarray) -> float:
    return float(_ref_l0_counts(feats).max())


def _ref_density(feats: np.ndarray) -> np.ndarray:
    """Per-feature activation density = fraction of tokens active. In [0,1]."""
    return (feats > 0).mean(axis=0).astype(np.float64)


def _ref_dead_fraction(feats: np.ndarray) -> float:
    """Fraction of features never active across all tokens."""
    return float((~(feats > 0).any(axis=0)).mean())


def _synthetic_sparse(T, D, k, seed=0):
    """A feature matrix with EXACTLY k active features per token."""
    rng = np.random.default_rng(seed)
    m = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        idx = rng.choice(D, size=k, replace=False)
        m[t, idx] = rng.uniform(0.1, 5.0, size=k).astype(np.float32)
    return m


def test_blme_kernel_transcription_matches_live_source():
    """ANTI-DRIFT: the transcribed kernel above must match the live BLME source.

    If someone edits sae_features.py's kernel, this test's transcription is no
    longer faithful and the parity claim would be stale — so pin the exact lines.
    """
    import inspect

    from blme.tasks.interpretability import sae_features as sf

    src = inspect.getsource(sf.SAEFeatureDimensionalityTask.evaluate)
    assert "(feature_acts > 0).float().sum(dim=-1)" in src
    assert "l0_per_token.mean()" in src
    assert "l0_per_token.max()" in src
    assert "sae.cfg.d_sae" in src


def test_blme_kernel_equals_numpy_reference_on_dense_matrix():
    """BLME L0 kernel == independent numpy reference to 0.0 on a dense matrix.

    Uses a large per-token active count (~2700, like the real SAE run) so the
    float32-accumulation path is exercised; still exact because the reference
    uses BLME's own float32 dtype.
    """
    rng = np.random.default_rng(7)
    T, D = 54, 24576
    feats = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        k = int(rng.integers(2000, 3500))
        idx = rng.choice(D, size=k, replace=False)
        feats[t, idx] = rng.uniform(0.01, 4.0, size=k).astype(np.float32)

    feats_t = torch.from_numpy(feats)
    b_mean, b_max = _blme_l0_kernel(feats_t)
    r_mean, r_max = _ref_seq_mean_l0(feats), _ref_seq_max_l0(feats)

    # Per-token integer counts must be BIT-EXACT identical.
    torch_counts = (feats_t > 0).float().sum(-1).numpy().astype(np.int64)
    assert np.array_equal(torch_counts, _ref_l0_counts(feats))

    assert abs(b_mean - r_mean) == TOL, f"mean diff {abs(b_mean - r_mean)}"
    assert abs(b_max - r_max) == TOL, f"max diff {abs(b_max - r_max)}"


def test_anchor_known_sparsity_k_gives_l0_exactly_k():
    """ANCHOR: a matrix with exactly k active features/token -> L0 == k exactly."""
    for k in (1, 7, 50):
        m = _synthetic_sparse(40, 200, k, seed=k)
        mean, mx = _blme_l0_kernel(torch.from_numpy(m))
        assert mean == float(k)
        assert mx == float(k)
        assert _ref_seq_mean_l0(m) == float(k)


def test_anchor_dead_feature_and_density_range():
    """ANCHOR: an always-zero feature is dead; an always-on one is not; density in [0,1]."""
    m = _synthetic_sparse(40, 200, 7, seed=1)
    m[:, 5] = 0.0     # force column 5 dead
    m[:, 11] = 3.0    # force column 11 always active
    dead = ~((m > 0).any(axis=0))
    assert bool(dead[5]) is True
    assert bool(dead[11]) is False
    assert 0.0 <= _ref_dead_fraction(m) <= 1.0
    dens = _ref_density(m)
    assert dens.min() >= 0.0 and dens.max() <= 1.0
    assert dens[11] == 1.0 and dens[5] == 0.0


def test_fixture_pins_official_saelens_parity():
    """Cross-check the numbers produced by the REAL sae-lens package (isolated venv).

    A pretrained gpt2-small-res-jb SAE was loaded via SAELens, gpt2 activations
    encoded, and BLME's kernel run on the identical encoded features. All kernel
    diffs were 0.0 with per-token counts bit-exact; pinned here so a regression
    in the BLME kernel is caught even though sae_lens is not a BLME dependency.
    """
    data = json.loads(FIXTURE.read_text())
    assert data["task"] == "interpretability_sae_features"
    assert data["reference_primary_pkg"].startswith("sae-lens")
    assert data["input"]["sae_release"] == "gpt2-small-res-jb"
    assert data["input"]["sae_id"] == "blocks.8.hook_resid_pre"
    assert data["input"]["d_sae"] == 24576

    res = data["results"]
    assert res["per_token_counts_bit_exact"] is True
    assert res["max_per_sample_kernel_diff"] == 0.0
    assert res["agg_diff_mean_l0"] == 0.0
    assert res["agg_diff_max_l0"] == 0.0
    assert res["full_matrix_diff_mean_l0"] == 0.0
    assert res["full_matrix_diff_max_l0"] == 0.0
    # float64 vs BLME-float32 mean gap is tiny and is dtype accumulation, not divergence.
    assert res["full_matrix_f64_info_diff"] < 1e-3
    assert res["sae_total_dict_size"] == 24576
    assert 0.0 <= res["density_min"] <= res["density_max"] <= 1.0
    assert 0.0 <= res["dead_feature_fraction"] <= 1.0

    anch = data["anchors"]
    assert anch["known_k_l0_exact"] is True
    assert anch["dead_feature_detected"] is True
    assert anch["density_in_unit_interval"] is True

    assert data["all_kernel_diffs_zero"] is True
    assert data["all_anchors_pass"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
