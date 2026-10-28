"""Formula-level parity tests for paper-derived scalar helpers.

These tests use checked-in external-reference fixtures instead of downloading
large reference repositories at test time. Each fixture records its paper or
reference repo source and, where available, the upstream HEAD observed during
the certification audit.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures/reference_parity/formula_fixtures.json"


def _fixtures():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _case(name: str) -> dict:
    return _fixtures()["cases"][name]


def test_reference_fixture_manifest_has_sources_and_cases():
    payload = _fixtures()

    assert payload["schema_version"] == 1
    assert "matrix_nuclear_norm" in payload["reference_sources"]
    assert payload["reference_sources"]["matrix_nuclear_norm"]["head"] == (
        "e7a9188eb30c146896e8cbc043caa7eb71a460ee"
    )
    assert "chain_of_embedding_three_points" in payload["cases"]


def test_linear_cka_matches_kornblith_invariances():
    from blme.tasks.geometry.cka import _linear_cka

    X = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    # Orthogonal feature transform preserves linear CKA.
    Q = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    assert _linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)

    Y = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ])
    got = _linear_cka(X, Y)
    assert 0.0 <= got <= 1.0


def test_normalized_linear_hsic_matches_centered_gram_trace():
    from blme.tasks.geometry.mutual_info import _normalized_linear_hsic

    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Y = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    assert _normalized_linear_hsic(X, Y) == pytest.approx(1.0, abs=1e-6)

    Z = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
    got = _normalized_linear_hsic(X, Z)
    assert 0.0 <= got <= 1.0


def test_linear_cka_and_hsic_match_kornblith_reference_value():
    """Reference-value parity for geometry_cka / geometry_hsic.

    Verified 2026-06-21 against an independent transcription of the
    official google-research cka.feature_space_linear_cka (Kornblith
    et al. 2019): on a fixed (4x3) pair the reference CKA is
    0.450953881199808. BLME's HSIC form matches to 3e-15; the
    _linear_cka form to 1.8e-8 (it casts to float32). This pins an
    actual reference value, complementing the invariance tests above.
    """
    from blme.tasks.geometry.cka import _linear_cka
    from blme.tasks.geometry.mutual_info import _normalized_linear_hsic

    case = _case("linear_cka_fixed_pair")
    X = torch.tensor(case["X"])
    Y = torch.tensor(case["Y"])
    assert _linear_cka(X, Y) == pytest.approx(case["expected"], abs=1e-6)
    assert _normalized_linear_hsic(X, Y) == pytest.approx(case["expected"], abs=1e-9)


def test_matrix_nuclear_norm_fixture_matches_reference_formula():
    from blme.tasks.geometry.schatten import _matrix_nuclear_norm_fast

    case = _case("matrix_nuclear_norm_diagonal")
    matrix = torch.tensor(case["matrix"])

    assert _matrix_nuclear_norm_fast(matrix) == pytest.approx(case["expected"])


def test_matrix_nuclear_norm_normalized_by_sequence_length_li2024():
    """Li et al. 2024 (arXiv:2410.10672) Eq. 11 defines the Matrix
    Nuclear-Norm as sum_i sqrt(sum_j X_ij^2) / L_input, where L_input is
    the SEQUENCE LENGTH (token/row count) — NOT d_model. Verified against
    the primary-source equation 2026-06-22. The schatten task previously
    divided by d_model (wrong by a factor T/d); this pins the row count.
    """
    from blme.tasks.geometry.schatten import _matrix_nuclear_norm_fast

    Z = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    T = Z.shape[0]  # L_input = sequence length = rows = 3
    raw = _matrix_nuclear_norm_fast(Z)
    assert raw == pytest.approx(6.0)            # 3 + 2 + 1
    assert raw / T == pytest.approx(2.0)        # Eq. 11 normalization (mnn / L_input)
    assert raw / Z.shape[1] != pytest.approx(raw / T)  # not the old /d_model


def test_schatten_and_rankme_fixture_matches_reference_formulas():
    from blme.tasks.geometry.schatten import _rankme, _schatten_p_norm

    case = _case("schatten_singular_values")
    singular_values = np.array(case["singular_values"])

    assert _schatten_p_norm(singular_values, p=1) == pytest.approx(case["expected_schatten_1"])
    assert _schatten_p_norm(singular_values, p=2) == pytest.approx(case["expected_schatten_2"])
    assert _schatten_p_norm(singular_values, p=4) == pytest.approx(case["expected_schatten_4"])
    assert _schatten_p_norm(singular_values, p=float("inf")) == pytest.approx(case["expected_schatten_inf"])
    assert _rankme(singular_values) == pytest.approx(case["expected_rankme"])


def test_attention_sink_fixture_matches_reference_formula():
    from blme.tasks.interpretability.activation_sinks import _sink_epsilon

    case = _case("attention_sink_all_queries_to_bos")
    T = case["tokens"]
    attn = torch.zeros(case["layers"], case["heads"], T, T)
    attn[:, :, :, 0] = 1.0
    mask = torch.tril(torch.ones(T, T))
    attn = attn * mask
    attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    assert _sink_epsilon(attn, epsilon=case["epsilon"]) == pytest.approx(case["expected"])


def test_chain_of_embedding_fixture_matches_reference_formula():
    from blme.tasks.dynamics.coe import _coe_from_chain

    case = _case("chain_of_embedding_three_points")
    chain = [torch.tensor(v) for v in case["chain"]]
    out = _coe_from_chain(chain)

    assert out["magnitudes"] == pytest.approx(case["expected_magnitudes"])
    assert out["angles"] == pytest.approx(case["expected_angles"])
    assert out["coe_r"] == pytest.approx(case["expected_coe_r"])
    assert out["coe_c"] == pytest.approx(case["expected_coe_c"])
    assert out["normalized_coe_c"] == pytest.approx(case["expected_normalized_coe_c"])


def test_calibration_ece_and_brier_match_guo_binning():
    from blme.tasks.consistency.calibration import _calibration_from_confidences

    case = _case("calibration_two_bins")
    confidences = np.array(case["confidences"])
    correct = np.array(case["correct"], dtype=bool)
    out = _calibration_from_confidences(confidences, correct, n_bins=case["n_bins"])

    assert out["ece"] == pytest.approx(case["expected_ece"])
    assert out["brier_score"] == pytest.approx(case["expected_brier"])


def test_weat_effect_size_matches_cohens_d_definition():
    from blme.tasks.consistency.bias import _weat_effect_size

    X = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    Y = [np.array([-1.0, 0.0]), np.array([-1.0, 0.0])]
    A = [np.array([1.0, 0.0])]
    B = [np.array([-1.0, 0.0])]

    expected = _case("weat_axis_aligned")["expected_effect_size"]
    assert _weat_effect_size(X, Y, A, B) == pytest.approx(expected)


def test_contextualization_intrasim_is_token_to_sentence_mean_ethayarajh():
    """Ethayarajh 2019 IntraSim_l(s) = mean_i cos(token_i, sentence_mean).

    Verified 2026-06-22 against the reference formula (kawine/contextual):
    BLME's _intra_sentence_mean_cosine matches the independent token-to-mean
    computation exactly, and differs from the (wrongly used) pairwise-mean.
    """
    from blme.tasks.geometry.contextualization import (
        _intra_sentence_mean_cosine, _cosine_pairwise_mean,
    )

    H = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0],
                  [0.0, 0.0, 3.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 0.0]])
    m = H.mean(0)
    ref = float(np.mean([np.dot(h, m) / (np.linalg.norm(h) * np.linalg.norm(m)) for h in H]))
    assert _intra_sentence_mean_cosine(H) == pytest.approx(ref, abs=1e-12)
    # And it is genuinely different from the pairwise form (SelfSim).
    assert abs(_intra_sentence_mean_cosine(H) - _cosine_pairwise_mean(H)) > 0.1


def test_isoscore_matches_rudman_reference_package():
    """geometry_isoscore vs official IsoScore package (Rudman et al. 2022).

    Verified 2026-06-21 against IsoScore==2.0.1 (bcbi-edu/p_eickhoff_isoscore):
    BLME's ``_isoscore`` agrees to <3e-8 on isotropic/anisotropic/N>d clouds.
    The reference L2-normalizes the covariance spectrum (``vector_norm``),
    refuting the AUDIT_V2 'L1-normalizes' lead. This fixture pins a fully
    hand-derivable symmetric case (cov eigenvalues [1.6, 0.4, 0.4] -> 0.5).
    """
    from blme.tasks.geometry.isotropy import _isoscore

    case = _case("isoscore_symmetric_3d")
    X = np.array(case["points"], dtype=np.float64)
    assert _isoscore(X) == pytest.approx(case["expected"], abs=case["tol"])

    # Live official-package cross-check (campaign-4 B5). Includes the N<d
    # zero-padding branch, which no test previously exercised — deleting
    # the padding passed the whole suite. Official IsoScore takes
    # (n_points, n_dims), same orientation as BLME.
    iso_mod = pytest.importorskip("IsoScore.IsoScore")
    rng = np.random.default_rng(0)
    Xa = rng.standard_normal((200, 16)) * np.linspace(1, 4, 16)[None, :]
    assert _isoscore(Xa) == pytest.approx(float(iso_mod.IsoScore(Xa)), abs=1e-6)
    Xw = rng.standard_normal((40, 128))          # N < d
    assert _isoscore(Xw) == pytest.approx(float(iso_mod.IsoScore(Xw)), abs=1e-6)


def test_twonn_intrinsic_dim_recovers_known_dimension_and_tracks_skdim():
    """geometry_intrinsic_dim (Two-NN, Facco et al. 2017) — formula-faithful.

    Verified 2026-06-21 with scikit-dimension 0.3.4: BLME's Two-NN
    recovers known intrinsic dimensions (5D->5.10, 3D-in-20D->3.07,
    10D->9.78, 2D->1.98 on 2000 pts) and agrees with skdim.id.TwoNN to
    ~0.3% (worst 0.016). It is intentionally NOT bit-exact: BLME pre-
    filters mu<=1 before the empirical CDF, shifting F=i/N slightly. Hence
    the conservative 'formula-faithful' (not 'parity-ready') label — this
    pins recovery accuracy + skdim agreement, not exact parity.
    """
    from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

    task = IntrinsicDimensionTask.__new__(IntrinsicDimensionTask)
    rng = np.random.default_rng(0)
    # 3-dim linear subspace embedded in 20-dim ambient space.
    X = (rng.standard_normal((2000, 3)) @ rng.standard_normal((3, 20))).astype(np.float32)
    blme = task._compute_id(X)["intrinsic_dimension"]
    assert abs(blme - 3.0) < 0.3, f"Two-NN failed to recover ID=3: {blme}"

    skdim = pytest.importorskip("skdim.id")
    ref = float(skdim.TwoNN().fit(X).dimension_)
    assert abs(blme - ref) < 0.05, f"Two-NN drifted from skdim: blme={blme} skdim={ref}"


def test_persistence_landscape_matches_persim_library():
    """Independent-library cross-check: BLME's _compute_landscape must
    equal scikit-tda persim's PersLandscapeApprox on the same diagram.

    Verified 2026-06-21 with persim 0.3.8: single bar [0,2] -> tent
    [0, 0.5, 1.0, 0.5, 0.0] from BOTH implementations. This is a second,
    independent reference beyond the analytic-tent fixture above.
    """
    persim = pytest.importorskip("persim")
    from blme.tasks.topology.persistence_landscape import _compute_landscape

    dgm = np.array([[0.0, 2.0]])
    blme = _compute_landscape(dgm, n_landscapes=1, n_points=5)[0]
    pl = persim.PersLandscapeApprox(
        dgms=[dgm], hom_deg=0, start=0.0, stop=2.0, num_steps=5
    )
    ref = np.asarray(pl.values[0], dtype=float)
    assert blme.tolist() == pytest.approx(ref.tolist(), abs=1e-9)


def test_persistence_landscape_multibar_k4_matches_persim():
    """Multi-bar, k>=2 persim parity: with overlapping bars the k-selection
    (descending sort of tent values at each t) actually matters — a
    sort-direction bug or missing max(0, .) clip is invisible on single-bar
    diagrams but changes L_2..L_4 here. Verified 2026-07: max diff vs
    persim 0.3.8 PersLandscapeApprox is ~8.9e-16 on this diagram.
    """
    persim = pytest.importorskip("persim")
    from blme.tasks.topology.persistence_landscape import _compute_landscape

    dgm = np.array([
        [0.0, 4.0],
        [1.0, 3.0],
        [2.0, 6.0],
        [2.5, 3.5],
        [5.0, 5.5],
    ])
    K, n_points = 4, 241
    blme = _compute_landscape(dgm, n_landscapes=K, n_points=n_points)
    pl = persim.PersLandscapeApprox(
        dgms=[dgm], hom_deg=0, start=0.0, stop=6.0, num_steps=n_points
    )
    ref = np.asarray(pl.values, dtype=float)  # (K_ref, n_points)
    for k in range(min(K, ref.shape[0])):
        assert blme[k].tolist() == pytest.approx(ref[k].tolist(), abs=1e-9), (
            f"landscape k={k + 1} diverged from persim"
        )
    # Landscapes beyond the persim-emitted depth must be identically zero.
    for k in range(ref.shape[0], K):
        assert not blme[k].any()
    # Sanity: the k-selection is non-trivial on this diagram (L2 != L1).
    assert np.max(np.abs(blme[0] - blme[1])) > 0.1


def test_calibration_ece_matches_torchmetrics_library():
    """Independent-library cross-check: BLME's ECE must equal
    torchmetrics' binary_calibration_error (l1, equal-width bins).

    Verified 2026-06-21 with torchmetrics 1.9.0: confidences
    [0.1,0.4,0.8,0.9], correct [F,T,T,F], 2 bins -> ECE 0.3 from BOTH.
    """
    tm = pytest.importorskip("torchmetrics.functional.classification")
    from blme.tasks.consistency.calibration import _calibration_from_confidences

    case = _case("calibration_two_bins")
    conf = np.array(case["confidences"])
    correct = np.array(case["correct"], dtype=bool)
    blme = _calibration_from_confidences(conf, correct, n_bins=case["n_bins"])

    ref = float(tm.binary_calibration_error(
        torch.tensor(conf), torch.tensor(correct.astype(int)),
        n_bins=case["n_bins"], norm="l1",
    ))
    assert blme["ece"] == pytest.approx(ref, abs=1e-9)


def test_attention_entropy_matches_shannon_reference():
    """interpretability_attention_entropy: per-distribution Shannon entropy
    (natural log) of the attention weights (Clark et al. 2019).

    Verified 2026-06-22: uniform attention over T keys -> log(T) exactly,
    and a fixed distribution matches scipy.stats.entropy (natural log).
    """
    from scipy.stats import entropy as scipy_entropy
    from blme.tasks.interpretability.attention import _attention_entropy

    # Uniform over T=5 keys -> log(5).
    unif = torch.full((1, 5), 0.2)
    assert float(_attention_entropy(unif)[0]) == pytest.approx(float(np.log(5)), abs=1e-6)

    # Fixed distribution -> matches scipy natural-log entropy.
    dist = [0.5, 0.3, 0.15, 0.05]
    got = float(_attention_entropy(torch.tensor([dist]))[0])
    assert got == pytest.approx(float(scipy_entropy(dist)), abs=1e-6)


def test_homology_lifespan_summary_matches_analytic_and_gudhi():
    """topology_homology: persistence-lifespan summary on a unit square has
    an analytically-known answer and matches an independent GUDHI backend.

    Unit square -> H0 finite lifespans [1,1,1] (mean=max=1); one H1 loop
    born at side=1, dies at diagonal=sqrt(2) -> mean_h1 = sqrt(2)-1; one loop.
    """
    from ripser import ripser
    from blme.tasks.topology.homology import _lifespan_summary

    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    summary = _lifespan_summary(ripser(square, maxdim=1)["dgms"])
    assert summary["mean_persistence_h0"] == pytest.approx(1.0, abs=1e-5)
    assert summary["max_persistence_h0"] == pytest.approx(1.0, abs=1e-5)
    assert summary["num_loops_h1"] == 1
    assert summary["mean_persistence_h1"] == pytest.approx(np.sqrt(2) - 1.0, abs=1e-5)

    # Independent backend: GUDHI Rips on the same cloud -> same H1 persistence.
    gudhi = pytest.importorskip("gudhi")
    st = gudhi.RipsComplex(points=square, max_edge_length=3.0).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    h1 = [(b, d) for b, d in st.persistence_intervals_in_dimension(1) if d != float("inf")]
    assert len(h1) == 1
    assert (h1[0][1] - h1[0][0]) == pytest.approx(np.sqrt(2) - 1.0, abs=1e-5)


def test_lid_mle_is_minus_k_variant_not_levina_bickel_parity():
    """geometry_lid uses the -k MLE variant: LID = -k / sum log(d_i/d_k).

    This is intentionally NOT the canonical Levina-Bickel estimator (which
    uses -(k-1)), so it is biased high by exactly k/(k-1) and is NOT bit-
    parity with skdim.id.MLE. Hence the conservative 'formula-faithful'
    (NOT 'parity-ready') label. This test pins the -k formula and the
    k/(k-1) relationship so the divergence is documented, not silent.
    """
    from blme.tasks.geometry.lid import _lid_mle

    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k = 5
    expected_minus_k = -k / np.sum(np.log(d / d[-1]))
    assert _lid_mle(d, k) == pytest.approx(expected_minus_k, abs=1e-9)

    levina_bickel = -(k - 1) / np.sum(np.log(d[:-1] / d[-1]))
    assert _lid_mle(d, k) == pytest.approx(levina_bickel * k / (k - 1), abs=1e-9)


def test_hubness_stats_from_occurrence_counts():
    from blme.tasks.geometry.hubness import _hubness_stats_from_occurrences

    out = _hubness_stats_from_occurrences(np.array([0, 1, 1, 2]), k=2)
    assert out["hubness_k2_max"] == 2
    assert out["hubness_k2_top1pct"] == pytest.approx(2 / 4)
    assert 0.0 <= out["hubness_k2_gini"] <= 1.0


def test_min_k_mean_logprob_matches_shi_definition():
    from blme.tasks.consistency.contamination import _min_k_mean_logprob

    case = _case("min_k_logprob")
    token_logprobs = np.array(case["token_logprobs"])

    assert _min_k_mean_logprob(token_logprobs, case["k_pct"]) == pytest.approx(case["expected"])
    # k is floored but at least one token.
    assert _min_k_mean_logprob(token_logprobs, 1) == pytest.approx(-3.0)


def test_distinct_n_matches_li_definition():
    from blme.tasks.dynamics.generation_diversity import _distinct_n

    case = _case("distinct_n_tokens")
    tokens = case["tokens"]
    assert _distinct_n(tokens, 1) == pytest.approx(case["expected_distinct_1"])
    assert _distinct_n(tokens, 2) == pytest.approx(case["expected_distinct_2"])


def test_self_bleu_single_matches_clipped_ngram_precision():
    from blme.tasks.dynamics.generation_diversity import _self_bleu_single

    case = _case("self_bleu_identical")
    hyp = case["hypothesis"]
    refs = case["references"]
    assert _self_bleu_single(hyp, refs, max_n=4) == pytest.approx(case["expected"])

    hyp = [7, 8, 9, 10]
    assert _self_bleu_single(hyp, refs, max_n=4) == pytest.approx(0.0)


def test_self_bleu_matches_texygen_nltk_smoothing():
    """Self-BLEU matches the Texygen reference: NLTK sentence_bleu with
    SmoothingFunction().method1 and uniform weights. On partial overlap the
    old hand-rolled estimator hard-zeroed; the fixed version matches NLTK.
    """
    pytest.importorskip("nltk.translate.bleu_score")
    from blme.tasks.dynamics.generation_diversity import _self_bleu_single
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    case = _case("self_bleu_partial_overlap")
    hyp = case["hypothesis"]
    refs = case["references"]
    n = case["max_n"]
    got = _self_bleu_single(hyp, refs, max_n=n)
    ref = float(sentence_bleu(
        refs, hyp, tuple(1.0 / n for _ in range(n)),
        smoothing_function=SmoothingFunction().method1,
    ))
    assert got == pytest.approx(ref, abs=1e-12)
    assert got == pytest.approx(case["expected"], abs=1e-9)


def test_persistence_entropy_matches_shannon_lifespans():
    from blme.tasks.topology.persistence_entropy import _persistence_entropy

    case = _case("persistence_entropy_lifespans")
    lifespans = case["lifespans"]

    assert _persistence_entropy(lifespans) == pytest.approx(case["expected"])
    assert _persistence_entropy([3.0]) == pytest.approx(0.0)

    # Independent base-2 reference: scipy.stats.entropy (== giotto-tda's
    # PersistenceEntropy, which calls scipy.stats.entropy base=2).
    from scipy.stats import entropy as _scipy_entropy
    assert _persistence_entropy(lifespans) == pytest.approx(
        float(_scipy_entropy(np.array(lifespans), base=2)), abs=1e-12
    )


def test_persistence_landscape_single_bar_tent_function():
    from blme.tasks.topology.persistence_landscape import _compute_landscape

    case = _case("persistence_landscape_single_bar")
    dgm = np.array(case["diagram"])
    landscape = _compute_landscape(
        dgm,
        n_landscapes=case["n_landscapes"],
        n_points=case["n_points"],
    )

    assert landscape.shape == (2, 5)
    assert landscape[0].tolist() == pytest.approx(case["expected_first_landscape"])
    assert landscape[1].tolist() == pytest.approx(case["expected_second_landscape"])



def test_contextualization_kernels_pin_values():
    """Campaign-4 B7: 3 of 4 contextualization kernels had zero value-pinning
    coverage (only IntraSim was pinned) — dropping MEV's centering or the
    pairwise normalization passed the whole suite."""
    from blme.tasks.geometry.contextualization import (
        _baseline_anisotropy,
        _cosine_pairwise_mean,
        _max_explainable_variance,
    )

    rng = np.random.default_rng(0)
    V = rng.standard_normal((30, 8)) + 0.5

    # MEV == the official kawine/contextual `variance_explained_by_pc`, which
    # is sklearn PCA's explained_variance_ratio_[0] -- i.e. CENTERED. The
    # repo's own released gpt2/variance_explained.csv gives raw MEV 0.286-0.351
    # over layers 2-11, reproducing the paper's "around 30% on average for
    # layers 2 to 11"; the uncentered convention reads ~0.99 there.
    #
    # This test briefly pinned the UNCENTERED value on 2026-09-03. That was a
    # regression against a parity previously verified to 3.3e-16, and the
    # mutant check below now guards the correct direction.
    sk = pytest.importorskip("sklearn.decomposition")
    ref_mev = float(sk.PCA(n_components=1).fit(V).explained_variance_ratio_[0])
    assert _max_explainable_variance(V) == pytest.approx(ref_mev, abs=1e-9)

    # Mutant discrimination: the UNCENTERED variant must differ, so a silent
    # removal of the centering fails here.
    S_un = np.linalg.svd(V, full_matrices=False)[1]
    uncentered_mev = float(S_un[0] ** 2 / (S_un ** 2).sum())
    assert abs(uncentered_mev - ref_mev) > 1e-3

    # SelfSim kernel (all-pairs regime) == independent full upper-triangle mean.
    unit = V / np.linalg.norm(V, axis=1, keepdims=True)
    sims = unit @ unit.T
    ref_pair = float(sims[np.triu_indices(len(V), k=1)].mean())
    assert _cosine_pairwise_mean(V) == pytest.approx(ref_pair, abs=1e-12)

    # Anisotropy baseline: rows all along ONE direction (varied magnitudes)
    # -> every pair cosine is exactly 1.
    ray = np.outer(np.linspace(0.5, 3.0, 40), np.ones(6))
    assert _baseline_anisotropy(ray) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Effective rank convention (Roy & Vetterli 2007, EUSIPCO).
#
# No test pinned this, which is how a change from raw sigma to sigma^2 --
# roughly a 150x shift in the spectrum, and a SIGN FLIP in EDG, which is
# derived from this profile -- passed the whole suite silently. Pin it.
# ---------------------------------------------------------------------------

def test_effective_rank_is_roy_vetterli_raw_sigma_not_squared():
    """R&V Sec 2.1 Eq. (1): p_k = sigma_k / ||sigma||_1, then erank = exp(H(p)).

    RAW singular values, L1-normalized. RankMe (Garrido et al. 2210.02885
    Eq. 1-2), which explicitly attributes to Roy & Vetterli, uses the same.
    """
    import numpy as np
    from blme.tasks.geometry.utils import effective_rank

    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 64))
    Xc = X - X.mean(axis=0)
    S = np.linalg.svd(Xc, compute_uv=False)

    # Independent transcription of Eq. (1) + Definition 1.
    p = S / np.sum(np.abs(S))
    p = p[p > 0]
    ref = float(np.exp(-np.sum(p * np.log(p))))

    assert effective_rank(S) == pytest.approx(ref, rel=1e-9)

    # And it must NOT be the squared convention.
    q = (S ** 2) / np.sum(S ** 2)
    q = q[q > 0]
    ref_sq = float(np.exp(-np.sum(q * np.log(q))))
    assert not np.isclose(ref, ref_sq, rtol=1e-3)

    # R&V Property 1: 1 <= erank(A) <= rank(A).
    assert 1.0 <= effective_rank(S) <= np.linalg.matrix_rank(Xc) + 1e-9


def test_collapse_task_reports_roy_vetterli_convention():
    """The task's erank_per_layer must use raw sigma; the squared spectrum is
    exposed separately and must be named for what it is."""
    import numpy as np
    import torch
    from scipy.linalg import svdvals
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from blme.tasks.geometry.collapse import RepresentationCollapseTask
    from blme.tasks.geometry.utils import collect_hidden_states

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=tok.vocab_size,
                     n_positions=64, n_ctx=64)
    model = GPT2LMHeadModel(cfg).eval()
    dataset = [
        {"text": "The quick brown fox jumps over the lazy dog and runs away."},
        {"text": "A second distinct sentence with different tokens entirely here now."},
        {"text": "Numbers one two three four five six seven eight nine ten eleven."},
    ]
    out = RepresentationCollapseTask(
        config={"num_samples": 3, "use_cache": False}
    ).evaluate(model, tok, dataset, cache=None)

    # Match the task's layer set: the deepest profile point is the hooked RAW
    # last-block output, not HF's ln_f(b_{L-1}) entry. Without this the
    # reference differs from the task in exactly one element -- the last.
    # TOKEN_POLICY exclude_sinks; this random model has no massive row, so the
    # policy drops row 0 of every document.
    acts_docs = collect_hidden_states(model, tok, dataset, num_samples=3,
                                      layer_idx="all", replace_final_norm=True,
                                      per_sample=True)
    assert out["_meta_sink_n_massive_dropped"] == 0
    acts = {li: torch.cat([x[1:] for x in docs]) for li, docs in acts_docs.items()}

    def ref(M, square):
        M = np.asarray(M, dtype=np.float64)
        M = M[np.all(np.isfinite(M), axis=1)]
        sv = svdvals(M - M.mean(axis=0))
        sv = sv ** 2 if square else sv
        sv = sv[sv > 0]
        q = sv / sv.sum()
        return float(np.exp(-(q * np.log(q)).sum()))

    keys = sorted(acts.keys())
    assert out["_meta_erank_per_layer"] == pytest.approx(
        [ref(acts[i].numpy(), False) for i in keys], rel=1e-6)
    assert out["_meta_erank_sigma2_per_layer"] == pytest.approx(
        [ref(acts[i].numpy(), True) for i in keys], rel=1e-6)
    # The legacy misnamed key must be gone.
    assert "erank_rawsigma_per_layer" not in out
