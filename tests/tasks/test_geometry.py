"""
Tests for all 19 geometry tasks.
Each test is parameterized over GPT2, Llama, and BERT via conftest.py.
"""
import pytest
import torch
import numpy as np
import json
import tempfile
import os


def test_positional_decay_annotations_are_py39_safe():
    """Campaign-3 release blocker: `float | None` (PEP 604) is evaluated eagerly
    at def-time and raises TypeError on Python 3.9 (the advertised minimum),
    crashing task registration. `from __future__ import annotations` keeps
    annotations lazy (strings), so the union is never evaluated."""
    from blme.tasks.geometry.positional_decay import _row_distance_correlation

    assert _row_distance_correlation.__annotations__.get("return") == "float | None", (
        "annotation must be a lazy string (from __future__ import annotations) "
        "so the PEP-604 union does not evaluate at import on py3.9"
    )


def test_matrix_entropy_default_max_length_matches_cache(mock_model, mock_tokenizer, monkeypatch):
    """Campaign-3 DISCREPANCY: the fresh path truncated at config max_length
    (default 128) while the shared-cache path truncates at 512 (cache.py:352),
    so the SAME config gave different entropy depending on whether a cache was
    populated. The fresh default must match the cache's 512."""
    import blme.tasks.geometry.matrix_entropy as me

    captured = {}

    def spy(model, tokenizer, samples, max_length):
        captured["max_length"] = max_length
        # (per_layer, post_norm_last) since 2026-09-05: the fresh collector
        # returns BOTH tensors, because the profile is hooked and the headline
        # follows the released reference's post-norm state.
        return {}, []

    monkeypatch.setattr(me, "_collect_per_sample_layer_tokens_fresh", spy)
    me.MatrixEntropyTask(config={"num_samples": 2}).evaluate(
        mock_model, mock_tokenizer, dataset=[{"text": "hello world"}])
    assert captured.get("max_length") == 512, (
        "fresh-path default max_length must be 512 to match the shared cache")


def test_trajectory_curvature_is_mean_of_per_sentence_means():
    """Campaign-3 DISCREPANCY: layer curvature pooled ALL angles across samples
    (length-weighted), over-weighting longer sentences. The paper (Hosseini &
    Fedorenko) means each sentence's angles first, then means over sentences."""
    from blme.tasks.geometry.trajectory_curvature import (
        _pooled_layer_curvature, _trajectory_angles,
    )

    rng = np.random.default_rng(3)
    long_chunk = np.vstack([[9.0, 9.0], rng.standard_normal((20, 2))])   # BOS + long
    short_chunk = np.vstack([[9.0, 9.0], rng.standard_normal((4, 2))])   # BOS + short
    aL = _trajectory_angles(long_chunk[1:])
    aS = _trajectory_angles(short_chunk[1:])
    pooled = float(np.mean(np.concatenate([aL, aS])))
    per_sentence = float(np.mean([np.mean(aL), np.mean(aS)]))
    assert abs(pooled - per_sentence) > 1e-3, "test cannot discriminate the two"

    got = _pooled_layer_curvature([torch.tensor(long_chunk), torch.tensor(short_chunk)],
                                  skip_first=1)
    assert abs(got - per_sentence) < 1e-6, (
        f"got {got:.6f} — expected per-sentence mean {per_sentence:.6f}, "
        f"not the length-weighted pool {pooled:.6f}"
    )


def test_svd_avg_cosine_uses_raw_states_not_centered():
    """Campaign-3 DISCREPANCY: avg_cosine_similarity (the Ethayarajh 2019
    anisotropy baseline) was computed on MEAN-CENTERED states, which reads ~0
    by construction. It must be computed on the RAW representations (Eq. 4)."""
    from blme.tasks.geometry.isotropy import _svd_metrics_for_layer

    rng = np.random.default_rng(0)
    # Strongly anisotropic cloud: a large shared offset + small noise, so raw
    # cosines are ~1 while centered cosines are ~0.
    X = np.ones((200, 16), dtype=np.float64) * 3.0 + rng.standard_normal((200, 16)) * 0.05
    r = _svd_metrics_for_layer(torch.tensor(X))
    assert r["avg_cosine_similarity"] > 0.8, (
        f"avg_cosine_similarity {r['avg_cosine_similarity']:.3f} — computed on "
        f"centered states (reads ~0) instead of the raw Ethayarajh baseline"
    )


def test_sharpness_top_eigenvalue_is_signed():
    """Campaign-3 DISCREPANCY: top_eigenvalue_estimate reported ||Hv|| (unsigned
    magnitude), so a dominant NEGATIVE-curvature direction (saddle) was
    misreported as positive. Like PyHessian, it must return the SIGNED
    largest-magnitude eigenvalue (Rayleigh quotient at the eigenvector)."""
    from blme.tasks.dynamics.sharpness import _power_iteration_top_eigenvalue

    diag = torch.tensor([-5.0, 3.0, 1.5])  # dominant-magnitude eigenvalue is -5

    def hvp(v):
        return [diag * v[0]]

    lam = _power_iteration_top_eigenvalue(hvp, [torch.tensor([1.0, 1.0, 1.0])], 100)
    assert abs(lam - (-5.0)) < 1e-3, f"expected signed -5.0 (PyHessian), got {lam}"


def test_perplexity_bpc_correct_with_bos_tokenizer():
    """Campaign-3 BUG: BPC re-tokenised with add_special_tokens=False while
    scoring used the default (True), so on a BOS-adding tokenizer the recount
    dropped the BOS and undercounted tokens by one per doc -> BPC ~13% low.
    With aligned tokenisation, BPC must equal the textbook nll_bits / chars."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from blme.tasks.geometry.perplexity import RarePPLTask

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_bos_token = True  # simulate a Llama/Gemma-style BOS tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    docs = [
        "The capital of France is Paris and it is a beautiful city.",
        "Machine learning models can memorize training data in surprising ways.",
        "Photosynthesis converts sunlight into chemical energy inside plant cells.",
    ]
    dataset = [{"text": d} for d in docs]

    task = RarePPLTask(config={"num_samples": 3, "use_cache": False})
    r = task.evaluate(model, tok, dataset)
    bpc = r["bits_per_char"]

    # Independent textbook BPC under the protocol actually used: ONE
    # "\n\n"-joined stream scored in a single pass, total NLL in bits over
    # every predicted token, divided by the characters of that same span (the
    # joined text minus the first token, which is never predicted).
    #
    # This test originally summed per-document NLL and per-document chars,
    # matching the old per-document scoring. That protocol is gone: documents
    # are no longer scored from a cold start, no longer truncated at 512, and
    # BPC no longer re-tokenizes anything -- so the BOS misalignment this test
    # guards is now structurally impossible rather than merely fixed. The
    # assertion below still catches any drift between the scored token window
    # and the character denominator, which is what the bug actually was.
    import torch
    import torch.nn.functional as F
    corpus = "\n\n".join(docs)
    with torch.no_grad():
        ids = tok(corpus, return_tensors="pt")["input_ids"]
        logits = model(ids).logits[:, :-1, :]
        labels = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              labels.reshape(-1), reduction="sum").item()
    # SOURCE-OFFSET span, not `len(decode(first_id))`.
    #
    # This reference used to subtract the DECODED length of the first token,
    # which on a BOS tokenizer is `'<|endoftext|>'` -- 13 characters that are
    # not in the corpus at all. It therefore encoded the very defect the task
    # was fixed for (adversarial review round 2, 2026-09-04). The correct
    # denominator is the source span the SCORED targets actually cover, which
    # is what `strided_perplexity` now derives from the offset mapping.
    _off = tok(corpus, return_offsets_mapping=True)["offset_mapping"]
    _scored_chars = sum(b - a for (a, b) in _off[1:] if b > a)
    expected = (nll / np.log(2)) / _scored_chars
    assert abs(bpc - expected) < 1e-3, (
        f"BPC {bpc:.5f} != textbook {expected:.5f} — scored token window and "
        f"character denominator misaligned on a BOS tokenizer"
    )

    # The whole corpus fits in one window here, so the strided pass must have
    # scored exactly (n_tokens - 1) tokens: no double-count, no skip.
    assert r["_meta_n_tokens_strided"] == ids.shape[1] - 1


def test_perplexity_survives_empty_text_sample():
    """Campaign-3 BUG: an empty-text doc crashed the scoring forward pass
    ('cannot reshape tensor of 0 elements'). Empty rows (common in wikitext)
    must be skipped, not crash the task."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from blme.tasks.geometry.perplexity import RarePPLTask

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    dataset = [{"text": ""}, {"text": "A full sentence with several tokens here."},
               {"text": "Another complete sentence for the perplexity estimate."}]
    task = RarePPLTask(config={"num_samples": 3, "use_cache": False})
    r = task.evaluate(model, tok, dataset)
    assert np.isfinite(r["ppl_overall"]) and r["ppl_overall"] > 1.0


def test_weight_norms_handles_3d_fused_params():
    """Campaign-3 BUG: a >2-D parameter (fused MoE experts, shape (E, in, out))
    passed the ndim<2 filter, then float(S[0].item()) crashed on the batched
    singular-value vector and fell back to spectral=Frobenius (3x high) and
    stable_rank=1.0. Each expert must be treated as its own matrix."""
    from blme.tasks.geometry.weight_norms import _weight_matrix_stats

    torch.manual_seed(0)
    W = torch.randn(4, 8, 8)  # 4 experts, 8x8 each
    # _weight_matrix_stats also returns the two dimensionless companions
    # (stable_rank/min(m,n) and ||W||_F/sqrt(mn)), which divide the matrix-shape
    # term out of quantities that otherwise track d_model at rho +0.683.
    frobs, specs, sranks, sranks_norm, rms = _weight_matrix_stats(W)
    assert len(specs) == 4, "each fused expert must yield its own spectral norm"
    for i in range(4):
        true_sigma = float(torch.linalg.svdvals(W[i])[0])
        assert abs(specs[i] - true_sigma) < 1e-4, "spectral must be the real sigma_max"
        assert specs[i] < frobs[i], "sigma_max < Frobenius for this matrix"
        assert sranks[i] > 1.5, f"stable_rank must be real, not the fabricated 1.0 ({sranks[i]})"

    # 2-D input is unchanged (single matrix)
    W2 = torch.randn(6, 10)
    
    # The companions must be per-matrix and correctly bounded.
    import numpy as _np
    assert len(sranks_norm) == len(sranks) and len(rms) == len(frobs)
    for _sr, _srn, _fr, _rms, _M in zip(sranks, sranks_norm, frobs, rms,
                                        (W.reshape(-1, W.shape[-2], W.shape[-1])
                                         if W.ndim > 2 else W.unsqueeze(0))):
        _m, _n = int(_M.shape[-2]), int(_M.shape[-1])
        assert 0.0 <= _srn <= 1.0 + 1e-9, f"normalized stable rank out of [0,1]: {_srn}"
        assert _srn == pytest.approx(_sr / min(_m, _n), rel=1e-9)
        assert _rms == pytest.approx(_fr / _np.sqrt(_m * _n), rel=1e-9)

    f2, s2, r2, rn2, rms2 = _weight_matrix_stats(W2)
    assert len(s2) == 1 and abs(s2[0] - float(torch.linalg.svdvals(W2)[0])) < 1e-4


def test_mahalanobis_default_id_pool_is_distinct():
    """Campaign-3 BUG: the dataset=None fallback built dataset_id by cycling
    5 sentences (id_texts[i % 5]), so the index-split 'held-out' ID score rows
    were exact duplicates of the fit rows -> artificial ~0 ID Mahalanobis and a
    spurious AUROC=1.0. The default ID pool must supply enough DISTINCT
    sentences that a 50/50 split is genuinely held out (no duplicated rows)."""
    from blme.tasks.geometry.mahalanobis import _default_id_dataset

    texts = [d["text"] for d in _default_id_dataset(40)]
    assert len(texts) >= 20, "need >=20 samples for a stable held-out split"
    assert len(set(texts)) == len(texts), (
        "default ID texts must be distinct (cycling duplicates fit into the "
        "held-out score split)"
    )


def test_mp_ks_distance_handles_zero_atom():
    """Campaign-3 BUG: for gamma>1 the empirical spectrum has ~(D-N) zero
    eigenvalues (the rank-deficiency atom). _ks_distance compared F(0)=point
    mass 1-1/gamma against 0 at each of them and returned ~1-1/gamma regardless
    of fit. iid-MP data must yield a SMALL KS distance."""
    from blme.tasks.geometry.rmt_bulk import _ks_distance

    rng = np.random.default_rng(0)
    N, D = 200, 600  # gamma = D/N = 3  -> ~400 zero eigenvalues
    X = rng.standard_normal((N, D))
    Z = (X - X.mean(0)) / X.std(0)
    C = Z.T @ Z / N
    eigs = np.linalg.eigvalsh(C)
    ks = _ks_distance(eigs, gamma=D / N)
    assert ks < 0.06, (
        f"KS distance {ks:.4f} for iid-MP data — the zero atom is mishandled "
        f"(old code returns ~1-1/gamma = {1 - 1/3:.4f})"
    )


def test_mp_cdf_accurate_at_gamma_one():
    """Campaign-3 BUG: at gamma==1 the MP density ~ t^-1/2 at t=0; the uniform
    trapezoid grid put a huge spurious spike at the first node (max CDF error
    0.593). The CDF must match a direct scipy.quad integration."""
    from scipy.integrate import quad

    from blme.tasks.geometry.rmt_bulk import _mp_cdf, _mp_bulk_edge, _mp_lower_edge

    gamma = 1.0
    a, b = _mp_lower_edge(gamma), _mp_bulk_edge(gamma)  # 0, 4

    def dens(t):
        return np.sqrt(max((b - t) * (t - a), 0.0)) / (2 * np.pi * gamma * t) if t > 1e-15 else 0.0

    xs = np.linspace(0.05, b, 40)
    ref = np.array([quad(dens, a, x, limit=400)[0] for x in xs])
    got = _mp_cdf(xs, gamma)
    err = float(np.max(np.abs(got - ref)))
    assert err < 1e-3, f"gamma=1 MP CDF max error {err:.4f} (singularity mishandled)"


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_svd_isotropy(mock_model, mock_tokenizer):
    from blme.tasks.geometry.isotropy import SVDIsotropyTask

    task = SVDIsotropyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "_meta_svd_auc" in results
    assert "cond_number" in results
    assert results["_meta_svd_auc"] > 0
    assert results["_meta_svd_auc"] <= 1.0
    assert results["cond_number"] >= 1.0


def test_isoscore_matches_rudman_formula_on_known_spectrum():
    """Rudman et al. Eq. 4.1 uses L2-normalised PCA variances and maps
    pure one-dimensional use to 0. The legacy sum-normalised formula gives
    a different value on non-uniform spectra."""
    from blme.tasks.geometry.isotropy import _isoscore

    variances = np.array([9.0, 4.0, 1.0], dtype=np.float64)
    dim = len(variances)
    rows = []
    n_rows = 2 * dim
    for i, var in enumerate(variances):
        row = np.zeros(dim, dtype=np.float64)
        row[i] = np.sqrt(var * (n_rows - 1) / 2.0)
        rows.extend([row, -row])
    X = np.stack(rows, axis=0)

    sigma_hat = np.sqrt(dim) * variances / np.linalg.norm(variances)
    defect = np.linalg.norm(sigma_hat - 1.0) / np.sqrt(2.0 * (dim - np.sqrt(dim)))
    expected = ((dim - defect ** 2 * (dim - np.sqrt(dim))) ** 2 - dim) / (
        dim * (dim - 1)
    )

    assert _isoscore(X) == pytest.approx(expected, abs=1e-12)


def test_consistency(mock_model, mock_tokenizer):
    from blme.tasks.geometry.consistency import PredictionAlignmentTask

    task = PredictionAlignmentTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "prediction_alignment_mean" in results
    assert "prediction_alignment_std" in results
    mean = results["prediction_alignment_mean"]
    assert -1.0 <= mean <= 1.0


def test_categories(mock_model, mock_tokenizer):
    from blme.tasks.geometry.categories import CategoryGeometryTask

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump({"test_cat": ["A", "B", "C"]}, tmp)
        tmp_path = tmp.name

    try:
        task = CategoryGeometryTask(config={"categories_path": tmp_path})
        results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
        assert isinstance(results, dict)

        task_proj = CategoryGeometryTask(
            config={"categories_path": tmp_path, "projection_method": "pca"}
        )
        results_proj = task_proj.evaluate(mock_model, mock_tokenizer, dataset=None)

        if "projection_points" in results_proj:
            pts = results_proj["projection_points"]
            if len(pts) > 0:
                assert "x" in pts[0]
                assert "y" in pts[0]
    finally:
        os.remove(tmp_path)


def test_hubness(mock_model, mock_tokenizer):
    from blme.tasks.geometry.hubness import GlobalHubnessTask

    task = GlobalHubnessTask(config={"k_values": [5], "batch_size": 50})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "hubness_k5_skew" in results
    assert "hubness_k5_max" in results
    assert "hubness_k5_gini" in results
    assert 0 <= results["hubness_k5_gini"] <= 1.0


def test_intrinsic_dim(mock_model, mock_tokenizer):
    from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

    task = IntrinsicDimensionTask(config={})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "intrinsic_dimension" in results
    assert results["intrinsic_dimension"] >= 0


def test_unembedding(mock_model, mock_tokenizer):
    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask

    task = UnembeddingDiagnosticsTask(config={"n_sample": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "unembedding_is_tied" in results
    assert "unembedding_eff_rank" in results
    assert "unembedding_purity_mean" in results


def test_unembedding_purity_nonzero_when_vocab_large_vs_labels(tmp_path):
    """Historic bug: ``purity`` was computed by sampling random vocab
    ids and only keeping those that happen to be labelled. With ~100
    labelled tokens in a 50k vocab, the sample almost never landed on
    a labelled token and ``purity_mean`` was 0 for all 32 study models.
    The fix iterates ``cat_labels.keys()`` directly so every labelled
    token becomes a query regardless of the vocab/label ratio.
    """
    import json
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    # Large vocab, tiny label set — reproduces the real-world ratio.
    config = GPT2Config(
        vocab_size=10_000, n_positions=32, n_embd=16, n_layer=1, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()

    with torch.no_grad():
        W = model.lm_head.weight.data
        W.normal_(0.0, 0.01)
        # Cluster A at ids 10-19, Cluster B at 20-29.
        W[10:20, 0] += 5.0
        W[20:30, 1] += 5.0

    class ManualTok:
        vocab_size = 10_000
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False, **_):
            table = {f"A{i}": [10 + i] for i in range(10)}
            table.update({f"B{i}": [20 + i] for i in range(10)})
            table.update({f" A{i}": [10 + i] for i in range(10)})
            table.update({f" B{i}": [20 + i] for i in range(10)})
            return table.get(text, [0])
    tokenizer = ManualTok()

    cats = tmp_path / "cats.json"
    cats.write_text(json.dumps({
        "cluster_a": [f"A{i}" for i in range(10)],
        "cluster_b": [f"B{i}" for i in range(10)],
    }))

    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask
    task = UnembeddingDiagnosticsTask(config={
        "k": 5,
        "num_samples": 50,  # far fewer than vocab_size
        "categories_path": str(cats),
    })
    results = task.evaluate(model, tokenizer, dataset=None)

    assert "error" not in results
    # With clustered rows the within-cluster top-5 NNs should almost
    # entirely match the query's category.
    assert results["unembedding_purity_mean"] > 0.5, (
        f"purity still collapses on large vocab: "
        f"{results['unembedding_purity_mean']}"
    )


def test_intrinsic_dim_recovers_known_id_subspace():
    """On a 5-dim random subspace embedded in 32-dim ambient space,
    Two-NN should recover ID close to 5 (not 32)."""
    import numpy as np
    from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

    rng = np.random.default_rng(0)
    D, d_true = 32, 5
    basis = rng.standard_normal((d_true, D))
    X = rng.standard_normal((2000, d_true)) @ basis

    task = IntrinsicDimensionTask(config={})
    out = task._compute_id(X)
    id_est = out["intrinsic_dimension"]
    # Loose tolerance since Two-NN is noisy on small samples; enough to
    # distinguish from 32 (ambient) and 1 (degenerate lower bound).
    assert 3.0 <= id_est <= 8.0, (
        f"Two-NN should recover ID ≈ {d_true}, got {id_est}"
    )


def test_intrinsic_dim_robust_to_outlier_mu():
    """Construct a pathological μ distribution (heavy-tailed) that
    sends the naive MLE ``d = N / Σ log μ`` to tiny, near-zero values.
    The linear fit with top-10% tail trimming (Facco et al. 2017,
    step 5) should produce a substantially larger, more stable estimate
    — but we do NOT floor at 1 since the paper does not (Fig. 3 shows
    d < 1 on degenerate geometries as a valid diagnostic signal)."""
    import numpy as np
    from blme.tasks.geometry.intrinsic_dim import _twonn_linear_fit

    rng = np.random.default_rng(0)
    small = 1.0 + rng.uniform(1e-4, 1e-3, size=500)
    heavy = np.exp(rng.uniform(5, 10, size=50))   # μ in [e^5, e^10]
    mus = np.concatenate([small, heavy])

    d_est = _twonn_linear_fit(mus)
    assert np.isfinite(d_est)
    # Naive MLE here gives d ≪ 1 (heavy tail dominates). The trimmed
    # linear fit must do substantially better.
    naive_mle = len(mus) / np.sum(np.log(mus))
    assert d_est > 10 * naive_mle, (
        f"linear fit no more robust than naive MLE: "
        f"linear={d_est:.4f}, mle={naive_mle:.4f}"
    )


def test_intrinsic_dim_never_below_one():
    """Two-NN intrinsic dimension (Facco et al. 2017) is bounded below
    by 1 for any non-degenerate data — it counts the effective degrees
    of freedom in the embedding manifold. The historic MLE formula
    ``d = N / Σ log(μ_i)`` blew up when the μ distribution had heavy
    outliers (a single μ >> 1 dominated the sum), producing IDs < 1 —
    e.g. 0.11 for gpt2-small and 0.18 for gpt2-xl in the 32-model study.

    We exercise the failure path directly on a real GPT-2 input
    embedding matrix, which the aggregated study ran on.
    """
    import numpy as np
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.geometry.intrinsic_dim import IntrinsicDimensionTask

    config = GPT2Config(
        vocab_size=2000, n_positions=32, n_embd=32, n_layer=1, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()

    task = IntrinsicDimensionTask(config={})
    out = task.evaluate(model, tokenizer=None, dataset=None)
    d = out["intrinsic_dimension"]
    assert np.isfinite(d), f"ID is not finite: {d}"
    # The defining sanity check — Two-NN is bounded below by 1 by
    # construction on any non-degenerate point cloud. Values < 1 on
    # real embedding matrices signal the numerical failure.
    # Note: Facco et al. 2017 does not strictly bound Two-NN at 1 —
    # d ∈ (0, ∞) for the Pareto model — but for non-degenerate LLM
    # embeddings the estimate should comfortably exceed 1.
    assert d >= 1.0, (
        f"ID fell below 1 on a {config.n_embd}-dim embedding — "
        f"got {d:.4f}. The Two-NN estimator is underflowing."
    )


def test_unembedding_alignment_is_meaningful_when_tied():
    """When ``lm_head`` is tied to ``wte`` the raw ``E vs W_out`` cosine
    is identically 1 and the field conveys no information beyond
    ``is_tied``. The fix runs each embedding row through the final layer
    norm before comparison — the LayerNorm/RMSNorm transform makes the
    alignment non-trivial even for tied weights."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=200, n_positions=32, n_embd=16, n_layer=2, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()
    # Perturb the final LayerNorm so it isn't trivially close to identity
    # (default GPT-2 init gives weight=1, bias=0).
    with torch.no_grad():
        model.transformer.ln_f.weight.normal_(1.0, 0.3)
        model.transformer.ln_f.bias.normal_(0.0, 0.3)
    assert torch.allclose(
        model.lm_head.weight, model.transformer.wte.weight
    )

    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask
    task = UnembeddingDiagnosticsTask(config={"num_samples": 10})
    results = task.evaluate(model, tokenizer=None, dataset=None)

    assert results["unembedding_is_tied"] in (True, 1, 1.0)
    assert np.isfinite(results["embedding_alignment_mean"])
    assert results["embedding_alignment_mean"] < 0.999, (
        "alignment is tautologically 1 for tied model: "
        f"{results['embedding_alignment_mean']}"
    )


# ---------------------------------------------------------------------------
# New tests for remaining 13 geometry tasks
# ---------------------------------------------------------------------------

def test_spectral(mock_model, mock_tokenizer):
    """Spectral analysis of weight matrices (Stable Rank, Power Law)."""
    from blme.tasks.geometry.spectral import WeightSpectralTask

    task = WeightSpectralTask(config={})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    assert "avg_stable_rank" in results
    assert results["avg_stable_rank"] > 0


def test_spectral_gpt2_covers_conv1d():
    """GPT-2 uses transformers.pytorch_utils.Conv1D (not nn.Conv1d) for
    attention and MLP projections. The spectral task must analyze those
    matrices, not only the final nn.Linear lm_head — otherwise std_alpha
    is 0 and avg_alpha is computed from a single matrix.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    from blme.tasks.geometry.spectral import WeightSpectralTask

    config = GPT2Config(
        vocab_size=1000, n_positions=128, n_embd=32, n_layer=2, n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()

    task = WeightSpectralTask(config={})
    results = task.evaluate(model, tokenizer=None, dataset=None)

    assert "error" not in results
    # GPT-2 small has many Conv1D projections per layer (c_attn, c_proj in
    # attention, c_fc, c_proj in MLP) plus the lm_head nn.Linear. With
    # 2 layers we expect at least 4 Conv1Ds + 1 Linear = 5 matrices → the
    # per-matrix alpha values cannot all be identical.
    assert results["std_alpha"] > 0.0, (
        "std_alpha=0 means only a single weight matrix was analysed — "
        "GPT-2 Conv1D projections are being skipped."
    )


def test_lipschitz(mock_model, mock_tokenizer):
    """Lipschitz continuity estimation between layers."""
    from blme.tasks.geometry.lipschitz import LipschitzContinuityTask

    task = LipschitzContinuityTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    # `lipschitz_*` are now `_meta_`-prefixed aliases: they were
    # bit-identical duplicates of `relative_change_*` (eight keys, four
    # values), and the name was wrong -- this is the relative-change
    # ratio, not an operator Lipschitz constant.
    assert "relative_change_max" in results
    assert "_meta_lipschitz_max" in results
    assert "_meta_lipschitz_mean" in results
    assert results["_meta_lipschitz_max"] >= 0


def test_cka(mock_model, mock_tokenizer):
    """Centered Kernel Alignment between layers — returns scalar
    summaries, not the raw matrix (which the aggregator silently
    dropped while misinterpreting the `layers` list as data)."""
    from blme.tasks.geometry.cka import CKATask

    task = CKATask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    assert "error" not in results
    # The raw matrix + layer-index list used to pollute the aggregate
    # with stats of [0..N-1]; the rewrite intentionally drops them.
    assert "cka_matrix" not in results
    assert "layers" not in results
    # Depth-invariant headline keys, their debiased twins, and the uniform
    # layer summaries as _meta_ profiles (tests/tasks/invariance/test_cka.py).
    for k in ("early_late_cka", "min_offdiag_cka", "min_adjacent_cka",
              "early_late_cka_debiased", "_meta_avg_adjacent_cka",
              "_meta_mean_offdiag_cka", "_meta_first_middle_cka", "_meta_n_layers"):
        assert k in results, f"missing {k}"
    assert 0 <= results["_meta_avg_adjacent_cka"] <= 1.0


def test_collapse(mock_model, mock_tokenizer):
    """Representation collapse detection via effective rank."""
    from blme.tasks.geometry.collapse import RepresentationCollapseTask

    task = RepresentationCollapseTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "erank_per_layer" in results or "collapse_ratio" in results


def test_correlation_dimension(mock_model, mock_tokenizer):
    """Grassberger-Procaccia correlation dimension."""
    from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask

    task = CorrelationDimensionTask(config={"num_samples": 10})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "correlation_dimension" in results
        assert results["correlation_dimension"] > 0


def test_correlation_dimension_reframes_hidden_state_gp_method():
    """This task computes Grassberger-Procaccia dimension on hidden-state
    point clouds, not the cited log-prob trajectory method. The result must
    say so while keeping the legacy key for compatibility."""
    import torch.nn as nn
    from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, **kwargs):
            idx = float(input_ids[0, 0].item())
            t = torch.arange(input_ids.shape[1], dtype=torch.float32)
            base = torch.stack([
                idx + t * 0.1,
                (idx + 1.0) ** 0.5 + t * 0.01,
                torch.sin(torch.tensor(idx * 0.2)) + t * 0.03,
            ], dim=1).unsqueeze(0)

            class Out:
                hidden_states = (base * 0.5, base)

            return Out()

    class Tok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            idx = int(str(text).split()[-1])

            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)

            ids = torch.full((1, 4), idx, dtype=torch.long)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    dataset = [{"text": f"sample {i}"} for i in range(25)]
    result = CorrelationDimensionTask(config={"num_samples": 25}).evaluate(
        StubModel(), Tok(), dataset=dataset,
    )

    assert "error" not in result, result
    assert "correlation_dimension" in result
    # NaN-aware: the two are ALIASES, and with only 25 stub points the
    # scaling-region fit is legitimately censored to NaN -- and `nan == nan` is
    # False, so a bare equality check fails on a correct result.
    _a = result["_meta_hidden_state_correlation_dimension"]
    _b = result["correlation_dimension"]
    assert (_a == _b) or (np.isnan(_a) and np.isnan(_b)), (_a, _b)
    # The primary key is now the SCALING-REGION fit -- the old 5th-95th
    # percentile window sat in the saturation region and missed GP 1983's
    # published D2 = 2.05 by -27.7% on a Lorenz attractor.
    assert result["correlation_dimension_method"] == (
        "hidden_state_grassberger_procaccia_scaling_region")
    assert result["correlation_dimension_point_space"] == "mean_pooled_hidden_states"


def test_information_geometry(mock_model, mock_tokenizer):
    """Representation sensitivity (gradient norm w.r.t. hidden states)."""
    from blme.tasks.geometry.information_geometry import RepresentationSensitivityTask

    task = RepresentationSensitivityTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "representation_sensitivity" in results
        assert results["representation_sensitivity"] >= 0


def test_lid(mock_model, mock_tokenizer):
    """Local Intrinsic Dimensionality estimation."""
    from blme.tasks.geometry.lid import LocalIntrinsicDimensionalityTask

    task = LocalIntrinsicDimensionalityTask(config={"num_samples": 5, "k": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "lid_mean" in results
        assert "lid_std" in results


def test_mahalanobis(mock_model, mock_tokenizer):
    """Mahalanobis distance OOD detection."""
    from blme.tasks.geometry.mahalanobis import MahalanobisOODTask

    task = MahalanobisOODTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Mock tokenizer returns random tokens — Mahalanobis may encounter
    # singular covariance with tiny hidden dimensions. Accept error or valid.
    if "error" not in results:
        assert (
            "mean_mahalanobis_id" in results
            or results.get("insufficient_holdout") is True
        )


def test_mutual_info(mock_model, mock_tokenizer):
    """HSIC dependence estimation between layers."""
    from blme.tasks.geometry.mutual_info import HSICDependenceTask

    task = HSICDependenceTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "min_adjacent_hsic" in results
        assert "_meta_avg_adjacent_hsic" in results


def test_perplexity_rare_freq(mock_model, mock_tokenizer):
    """Perplexity analysis on rare vs frequent tokens."""
    from blme.tasks.geometry.perplexity import RarePPLTask

    task = RarePPLTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert any("perplexity" in k or "ppl" in k.lower() for k in results.keys())


def test_positional_decay(mock_model, mock_tokenizer):
    """Positional encoding integrity via attention decay."""
    from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask

    task = PositionalAttentionDecayTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    # Requires output_attentions; may error if SDPA attention is used
    if "error" not in results:
        assert any("corr" in k or "decay" in k for k in results.keys())


def test_positional_decay_uniform_causal_attention_is_null():
    """Uniform causal attention has no within-row distance preference.
    Pooling all triangular entries together invents a negative decay
    because later rows have smaller uniform probabilities."""
    import torch.nn as nn
    from blme.tasks.geometry.positional_decay import PositionalAttentionDecayTask

    class UniformCausalAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, output_attentions=False, **kwargs):
            seq_len = input_ids.shape[1]
            # WELL-FORMED causal attention: every row sums to 1 over keys
            # 0..i INCLUSIVE. The previous fixture wrote `attn[i, :i] = 1/i`,
            # which left the DIAGONAL at 0 and row 0 entirely empty, so no row
            # summed to 1 -- it was not an attention pattern at all, and the
            # number this test reported for the pooled estimand was measured on
            # that malformed input (independent Opus review, 2026-09-04).
            attn = torch.zeros(1, 2, seq_len, seq_len)
            for i in range(seq_len):
                attn[:, :, i, :i + 1] = 1.0 / (i + 1)

            class Out:
                attentions = (attn,)

            return Out()

    class Tok:
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=256):
            ids = torch.arange(8, dtype=torch.long).unsqueeze(0)

            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)

            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    result = PositionalAttentionDecayTask(config={"num_samples": 1}).evaluate(
        UniformCausalAttentionModel(), Tok(), dataset=[{"text": "x"}],
    )

    assert "error" not in result, result
    # The HEADLINE (rowwise) must be exactly 0 on this null.
    assert result["mean_positional_decay_correlation"] == pytest.approx(0.0, abs=1e-12)
    # And the MANUSCRIPT'S pooled estimand must NOT be, which is why the
    # headline is the rowwise one. Pinned so the evidence for overriding the
    # paper survives: the pooled null bias converges to about -0.467 as the
    # context grows (-0.4344766895 at L=8, -0.4666215371 at L=256).
    _pooled = result["mean_positional_decay_correlation_pooled"]
    assert _pooled < -0.3, (
        f"pooled estimand no longer shows its null bias ({_pooled}); if this "
        f"changed, revisit which estimand should be the headline")
    assert result["layer_positional_decay"]["layer_0"] == pytest.approx(0.0, abs=1e-12)


def test_rsa(mock_model, mock_tokenizer):
    """Representational Similarity Analysis across layers."""
    from blme.tasks.geometry.rsa import RepresentationalSimilarityTask

    task = RepresentationalSimilarityTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert any("rsa" in k for k in results.keys())


def test_matrix_entropy(mock_model, mock_tokenizer):
    """Von Neumann spectral entropy of covariance matrices."""
    from blme.tasks.geometry.matrix_entropy import MatrixEntropyTask

    # Give the task an in-memory corpus so it doesn't depend on network-
    # accessible WikiText, and so we have enough rows for a non-degenerate
    # covariance matrix at hidden_dim=32.
    dataset = [{"text": f"passage number {i}"} for i in range(4)]
    task = MatrixEntropyTask(config={"num_samples": 4})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    assert "error" not in results, f"matrix_entropy failed: {results}"
    assert "mean_matrix_entropy" in results
    # Von Neumann entropy must be finite and non-negative. NaN/0 means
    # the covariance collapsed (single row / all-zero) — the historic bug.
    me = results["mean_matrix_entropy"]
    assert np.isfinite(me), f"mean_matrix_entropy is not finite: {me}"
    assert me > 0.0, f"mean_matrix_entropy collapsed to zero: {me}"


def test_matrix_entropy_normalized_by_log_dim():
    """Wei et al. 2024 Def. 4.3 divides the per-sentence entropy by
    ``log d`` (the ambient hidden dimension) so the metric is
    comparable across model widths. Without the normalisation the
    headline number grows with model size even when the intrinsic
    compression is constant.
    """
    import math
    import numpy as np
    from blme.tasks.geometry.matrix_entropy import _matrix_entropy

    rng = np.random.default_rng(0)
    # Random N × D tokens; entropy should be ≤ log D, so normalised
    # entropy must be ≤ 1 + eps.
    for D in (16, 64, 128):
        X = rng.standard_normal((200, D)).astype(np.float32)
        import torch
        res = _matrix_entropy(torch.from_numpy(X))
        assert np.isfinite(res["entropy"])
        assert 0 <= res["entropy_normalized"] <= 1.01, (
            f"D={D} entropy_normalized={res['entropy_normalized']} "
            "out of [0, 1]"
        )


def test_neural_collapse_nc1_uses_sigma_b_subspace():
    """Papyan et al. 2020 define NC1 as within-class variability
    measured in the subspace spanned by class-mean differences
    (``Σ_B``). Naive ``tr(Σ_W · pinv(Σ_B))`` on a D-dim hidden space
    with only K classes has Σ_B of rank ≤ K-1 — and ``np.linalg.pinv``
    with default ``rcond`` will promote any near-zero eigenvalue to
    ~1/rcond. On real LLM hidden states (with small-but-nonzero
    cross-class noise leakage) this produced NC1 ≈ 3×10⁷ for
    pythia-70m in the 32-model study.

    The fix projects everything onto Σ_B's top-(K−1) eigenvectors —
    the metric is mathematically unchanged on the clean case but
    skips the numerically explosive off-subspace directions.
    """
    import numpy as np
    from blme.tasks.geometry.neural_collapse import _neural_collapse_metrics

    rng = np.random.default_rng(0)
    K = 5
    n_per_class = 8
    D = 2048

    # Class means in a clean (K-1)-dim subspace; tight intra-class
    # clusters. NC1 (restricted to the between-class subspace) should
    # be a small positive number proportional to (intra/inter)².
    class_centers = np.zeros((K, D))
    class_centers[:, :K - 1] = rng.standard_normal((K, K - 1)) * 5.0
    features = np.concatenate([
        class_centers[k] + rng.standard_normal((n_per_class, D)) * 0.1
        for k in range(K)
    ], axis=0)
    labels = np.concatenate([
        np.full(n_per_class, k) for k in range(K)
    ])

    res = _neural_collapse_metrics(features, labels)
    nc1 = res["nc1_within_class_collapse"]
    assert np.isfinite(nc1)
    # Within/between squared-ratio is ≈ 4×10⁻⁴; proper NC1 must match.
    assert nc1 < 1.0, f"NC1 unexpectedly large: {nc1}"

    # The fix must expose the subspace rank it actually used, so
    # reviewers can verify the projection happened.
    assert "nc1_subspace_rank" in res
    assert res["nc1_subspace_rank"] <= K - 1


def test_matrix_entropy_bounded_and_headline_is_last_layer():
    """Two structural checks on the Wei et al. 2024 implementation:

    1. Normalised entropy is always in [0, 1] since ``H ≤ log d``
       (entropy of a uniform distribution on ``d`` eigenvalues).
    2. The headline ``matrix_entropy`` field is explicitly the
       last-layer value (the Wei et al. paper reports that number, not
       an aggregate across layers).
    """
    import numpy as np
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.geometry.matrix_entropy import MatrixEntropyTask

    cfg = GPT2Config(
        vocab_size=300, n_positions=16, n_embd=16, n_layer=3, n_head=2,
    )
    cfg._attn_implementation = "eager"
    model = GPT2LMHeadModel(cfg).eval()

    class Tok:
        vocab_size = 300
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=16, **kw):
            ids = torch.randint(0, 300, (1, 10))
            class B(dict):
                def to(self, dev): return self
                def __getattr__(self, n):
                    try: return self[n]
                    except KeyError: raise AttributeError(n)
            return B({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    dataset = [{"text": f"sample {i}"} for i in range(4)]
    res = MatrixEntropyTask(config={"num_samples": 4}).evaluate(
        model, Tok(), dataset=dataset,
    )
    assert "error" not in res

    # Normalised entropy must be in [0, 1].
    # `_meta_` prefixed: the H / log(d) twin was demoted on 2026-09-04 because
    # zero-padding d=512 -> 4096 (an exact isometry) moves it 0.721317 ->
    # 0.540988 and it makes the archive width association WORSE than the raw
    # value. Still emitted, still bounded, just not feature-eligible.
    # `matrix_entropy` IS the normalised H (Wei et al. Def. 4.3); the
    # `_meta_..._normalized` aliases were removed on 2026-09-05 because they
    # carried the IDENTICAL float under a second name -- the "demotion" they
    # documented never took effect (independent Opus review).
    assert 0 <= res["matrix_entropy"] <= 1.01, (
        f"H_normalized out of [0, 1]: {res['matrix_entropy']}"
    )
    assert 0 <= res["mean_matrix_entropy"] <= 1.01
    assert "_meta_matrix_entropy_normalized" not in res
    assert res["matrix_entropy_raw"] >= 0.0

    # Headline == last-layer entropy, in the NORMALISED convention.
    #
    # `matrix_entropy` now holds H / log d, which is Wei et al.'s Definition
    # 4.3 and the quantity their tables report: on their protocol pythia-70m's
    # published 0.7003 matches the normalised 0.6659 (-4.9%), against a raw
    # H of 4.1542 (+493%). The raw value is retained under a _meta_ key.
    last_key = max(
        res["layer_matrix_entropies"].keys(),
        key=lambda k: int(k.split("_")[1]),
    )
    # THE HEADLINE IS NO LONGER THE PROFILE'S LAST ENTRY, deliberately.
    #
    # Wei et al.'s released recipe is `AutoModel(...)[0]` == `hidden_states[-1]`
    # == `ln_f(b_{L-1})`, so the headline is a single-layer estimand the
    # REFERENCE defines as the post-norm state, while the per-layer PROFILE is
    # hooked so it is homogeneous. They are different tensors on purpose
    # (2026-09-05). Asserting they are equal is asserting the defect that made
    # this task RED: hooking the headline moved gpt2 +93.3% off the reference
    # and turned a published 70m-vs-160m gap from +6.3% into +105.5%.
    assert res["_meta_headline_is_post_final_norm"] is True
    # The profile's deepest entry is the HOOKED one, so it must differ from the
    # headline on any model with a real final norm -- if these ever coincide,
    # the substitution silently stopped happening.
    assert res["matrix_entropy"] != res["layer_matrix_entropies_normalized"][last_key]
    assert res["matrix_entropy_raw"] != res["layer_matrix_entropies"][last_key]
    # The headline must be the normalised one, not the raw one.
    assert 0 <= res["matrix_entropy"] <= 1.01


def test_matrix_entropy_uses_per_sample_cache():
    """When a populated ModelOutputCache is passed, matrix_entropy must
    not collapse per-layer hidden states down to a single mean row (the
    historic cache path did that, producing NaN for every model).
    """
    import torch
    from unittest.mock import MagicMock
    from blme.cache import ModelOutputCache
    from blme.tasks.geometry.matrix_entropy import MatrixEntropyTask

    # Hand-roll a tiny fake cache backed by real tensors so we can assert
    # the SVD has real rank, not rank 1.
    n_layers, hidden_dim, n_samples, T = 2, 8, 3, 4

    def make_hidden_states():
        return tuple(torch.randn(1, T, hidden_dim) for _ in range(n_layers + 1))

    tokenizer = MagicMock()
    def tokenize(text, **kwargs):
        r = MagicMock()
        ids = torch.zeros((1, T), dtype=torch.long)
        r.__getitem__ = lambda self, k: ids if k == "input_ids" else MagicMock()
        r.__contains__ = lambda self, k: k == "input_ids"
        r.to.return_value = r
        r["input_ids"] = ids
        return r
    tokenizer.side_effect = tokenize

    model = MagicMock()
    model.config = MagicMock(vocab_size=100)
    param = MagicMock(); param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])

    def forward_fn(**kwargs):
        out = MagicMock()
        out.hidden_states = make_hidden_states() if kwargs.get("output_hidden_states") else None
        out.attentions = None
        out.logits = torch.randn(1, T, 100)
        return out
    model.side_effect = forward_fn

    cache = ModelOutputCache(model, tokenizer,
                              dataset=[{"text": "x"}] * n_samples,
                              num_samples=n_samples)
    cache.populate(need_hidden=True)

    task = MatrixEntropyTask(config={"num_samples": n_samples})
    results = task.evaluate(model, tokenizer,
                             dataset=[{"text": "x"}] * n_samples, cache=cache)

    assert "error" not in results
    me = results["mean_matrix_entropy"]
    assert np.isfinite(me), f"cache path produced non-finite matrix entropy: {me}"
    assert me > 0.0, "cache path collapsed hidden states to a single row"


def test_effective_rank_consistent_convention():
    """Three geometry tasks compute an "effective rank": collapse
    (per-layer hidden states), isotropy (static embeddings), and
    unembedding (lm_head weights). They must use the same Roy-Vetterli
    formula on raw singular values: ``p_i = σ_i / Σσ``. Covariance
    effective rank on ``σ²`` is a different convention and must not be
    mislabeled as Roy-Vetterli."""
    import numpy as np
    from blme.tasks.geometry.utils import effective_rank

    # Orthonormal matrix: every σ == 1, erank == D (full-rank uniform).
    S = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert abs(effective_rank(S) - 4.0) < 1e-9

    # Single non-zero singular value: erank == 1.
    S = np.array([3.0, 0.0, 0.0], dtype=np.float64)
    assert abs(effective_rank(S) - 1.0) < 1e-9

    # Geometrically decaying: erank < min(N, D).
    S = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
    val = effective_rank(S)
    assert 1.0 < val < 4.0

    # Linearly-spread σ: Roy-Vetterli raw-σ normalisation is strictly
    # larger than covariance/eigenvalue σ² normalisation.
    S = np.array([1.0, 0.5, 0.1, 0.01], dtype=np.float64)
    got = effective_rank(S)
    raw_p = S / S.sum()
    raw_form = float(np.exp(-np.sum(raw_p * np.log(raw_p))))
    sq = S * S
    sq_p = sq / sq.sum()
    sq_form = float(np.exp(-np.sum(sq_p * np.log(sq_p))))
    assert got == pytest.approx(raw_form, rel=1e-12)
    assert got > sq_form, (
        f"Roy-Vetterli σ erank ({got}) should be > covariance σ² erank "
        f"({sq_form}) for decreasing spectra."
    )


# ---------------------------------------------------------------------------
# Trajectory curvature (Hosseini & Fedorenko, NeurIPS 2023)
# ---------------------------------------------------------------------------

def test_trajectory_curvature_collinear_is_zero():
    """Collinear points x_t = t·u: every difference vector is u, so
    every angle is exactly 0."""
    from blme.tasks.geometry.trajectory_curvature import _mean_curvature

    u = np.array([1.0, 2.0, -0.5, 3.0])
    X = np.stack([t * u for t in range(10)])
    assert np.isclose(_mean_curvature(X), 0.0, atol=1e-6)


def test_trajectory_curvature_right_angle_zigzag():
    """Planar right-angle zigzag: consecutive difference vectors
    alternate between (1,0) and (0,1) — every angle is pi/2."""
    from blme.tasks.geometry.trajectory_curvature import (
        _mean_curvature, _trajectory_angles,
    )

    X = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=np.float64)
    angles = _trajectory_angles(X)
    assert angles.shape == (3,)
    assert np.allclose(angles, np.pi / 2, atol=1e-9)
    assert np.isclose(_mean_curvature(X), np.pi / 2, atol=1e-9)


def test_trajectory_curvature_hexagon():
    """Regular hexagon vertices walked in order: each step rotates the
    difference vector by the exterior angle pi/3."""
    from blme.tasks.geometry.trajectory_curvature import _trajectory_angles

    thetas = np.arange(7) * np.pi / 3  # close the loop: 7 points, 6 edges
    X = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    angles = _trajectory_angles(X)
    assert angles.shape == (5,)
    assert np.allclose(angles, np.pi / 3, atol=1e-9)


def test_trajectory_curvature_degenerate_no_nan():
    """Repeated identical points produce zero-norm difference vectors —
    the touching angles must be skipped (not NaN), and a T=2 sample is
    skipped cleanly at the pooling level."""
    import torch
    from blme.tasks.geometry.trajectory_curvature import (
        _pooled_layer_curvature, _trajectory_angles,
    )

    # Trajectory with stalls: diffs are [0, u, u, 0, u] — only the
    # (u, u) pair forms a valid angle (= 0).
    X = np.array(
        [[0, 0], [0, 0], [1, 0], [2, 0], [2, 0], [3, 0]], dtype=np.float64,
    )
    angles = _trajectory_angles(X)
    assert angles.size == 1
    assert np.all(np.isfinite(angles))

    # All-identical trajectory yields no valid angle.
    assert _trajectory_angles(np.zeros((8, 4))).size == 0

    # Pooling: degenerate + too-short samples are skipped; the hexagon
    # contributes the only angles, so the result is finite (pi/3).
    thetas = np.arange(7) * np.pi / 3
    hexagon = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    chunks = [
        torch.zeros(6, 2),            # repeated identical points
        torch.randn(2, 2),            # T=2: fewer than skip+3 tokens
        torch.from_numpy(np.vstack([[5.0, 5.0], hexagon])),  # BOS + hexagon
    ]
    pooled = _pooled_layer_curvature(chunks, skip_first=1)
    assert np.isfinite(pooled)
    assert np.isclose(pooled, np.pi / 3, atol=1e-6)


def test_trajectory_curvature_integration(mock_model, mock_tokenizer):
    """End-to-end on the conftest toy models: finite floats, curvature
    in [0, pi], finite straightening ratio."""
    from blme.tasks.geometry.trajectory_curvature import TrajectoryCurvatureTask

    dataset = [{"text": f"sample passage number {i}"} for i in range(4)]
    task = TrajectoryCurvatureTask(config={"num_samples": 4})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    assert "error" not in results, f"trajectory curvature failed: {results}"
    for key in (
        "curvature_mean_first_layer", "curvature_mean_mid_layer",
        "curvature_mean_last_layer", "curvature_overall_mean",
        "straightening_ratio", "curvature_slope",
        "curvature_q25", "curvature_q50", "curvature_q75",
    ):
        assert key in results, f"missing {key}"
        assert np.isfinite(results[key]), f"{key} not finite: {results[key]}"
    for key in (
        "curvature_mean_first_layer", "curvature_mean_mid_layer",
        "curvature_mean_last_layer", "curvature_overall_mean",
    ):
        assert 0.0 <= results[key] <= np.pi


# ---------------------------------------------------------------------------
# Marchenko-Pastur bulk deviation (Marchenko & Pastur 1967; BBP 2005)
# ---------------------------------------------------------------------------

def test_mp_bulk_iid_null():
    """iid standard normal activations: essentially no eigenvalue
    escapes the (tolerance-buffered) MP bulk and the spectrum matches
    the MP law closely."""
    from blme.tasks.geometry.rmt_bulk import _mp_layer_metrics

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 200))
    m = _mp_layer_metrics(X, edge_tol=0.05, fixed_gamma=None)

    assert "error" not in m
    assert m["outlier_frac"] <= 0.01, f"iid null produced outliers: {m}"
    assert m["ks_distance"] < 0.05, f"KS too large for iid null: {m}"
    # gamma = D/(N-1), not D/N: centering removes one degree of freedom, so
    # the centered matrix has rank <= N-1 and D/(N-1) is the ratio that
    # actually governs the spectrum. Using D/N misplaces the bulk edge, and
    # just below gamma = 2 the off-by-one gave a ~2x error in the fitted
    # sigma^2 (0.49 for a truth of 1.0).
    assert np.isclose(m["gamma"], 200 / 1999)


def test_mp_bulk_planted_spike_detected():
    """BBP-supercritical rank-1 spike: X = Z + 5·outer(u, v) with unit
    v gives a population spike of strength 25 ≫ sqrt(gamma), so at
    least one eigenvalue must exit the bulk with positive energy."""
    from blme.tasks.geometry.rmt_bulk import _mp_layer_metrics

    rng = np.random.default_rng(1)
    N, D = 2000, 200
    Z = rng.standard_normal((N, D))
    u = rng.standard_normal(N)
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v)
    X = Z + 5.0 * np.outer(u, v)

    m = _mp_layer_metrics(X, edge_tol=0.05, fixed_gamma=None)
    assert "error" not in m
    assert m["outlier_frac"] >= 1.0 / D, f"planted spike not detected: {m}"
    assert m["spike_energy"] > 0.0


def test_mp_bulk_constant_dimension_dropped():
    """A zero-variance dimension cannot be z-scored — it must be
    dropped (and counted) without poisoning any metric with NaN."""
    from blme.tasks.geometry.rmt_bulk import _mp_layer_metrics

    rng = np.random.default_rng(2)
    X = rng.standard_normal((1000, 100))
    X[:, 7] = 3.14  # constant dimension

    m = _mp_layer_metrics(X, edge_tol=0.05, fixed_gamma=0.25)
    assert "error" not in m
    assert m["n_dims_dropped"] == 1
    for key in ("outlier_frac", "spike_energy", "ks_distance", "gamma",
                "outlier_frac_g25", "spike_energy_g25", "ks_distance_g25"):
        assert np.isfinite(m[key]), f"{key} is NaN after dropping dim: {m}"
    # gamma = D/(N-1), not D/N: centering removes one degree of freedom, so
    # the centered matrix has rank <= N-1 and D/(N-1) is the ratio that
    # actually governs the spectrum. Using D/N misplaces the bulk edge, and
    # just below gamma = 2 the off-by-one gave a ~2x error in the fitted
    # sigma^2 (0.49 for a truth of 1.0).
    assert np.isclose(m["gamma"], 99 / 999)


def test_mp_bulk_edge_and_cdf_sanity():
    """gamma = 0.25 ⇒ bulk edge (1 + 0.5)² = 2.25; the numerically
    integrated MP CDF must vanish below the bulk, reach 1 above it,
    and be monotone."""
    from blme.tasks.geometry.rmt_bulk import (
        _mp_bulk_edge, _mp_cdf, _mp_layer_metrics, _mp_lower_edge,
    )

    assert np.isclose(_mp_bulk_edge(0.25), 2.25, atol=1e-12)
    assert np.isclose(_mp_lower_edge(0.25), 0.25, atol=1e-12)

    grid = np.linspace(0.0, 3.0, 301)
    cdf = _mp_cdf(grid, 0.25)
    assert np.isclose(cdf[0], 0.0, atol=1e-6)
    assert np.isclose(cdf[-1], 1.0, atol=1e-3)
    assert np.all(np.diff(cdf) >= -1e-12), "MP CDF must be monotone"
    # No point mass for gamma <= 1.
    assert _mp_cdf(np.array([0.2]), 0.25)[0] < 1e-6

    # With N=800, D=200 the gamma actually used is 0.25, so the bulk
    # edge applied inside the metric equals (1 + sqrt(0.25))².
    rng = np.random.default_rng(3)
    m = _mp_layer_metrics(rng.standard_normal((800, 200)),
                          edge_tol=0.05, fixed_gamma=None)
    # gamma = D/(N-1), not D/N: centering removes one degree of freedom, so
    # the centered matrix has rank <= N-1 and D/(N-1) is the ratio that
    # actually governs the spectrum. Using D/N misplaces the bulk edge, and
    # just below gamma = 2 the off-by-one gave a ~2x error in the fitted
    # sigma^2 (0.49 for a truth of 1.0).
    assert np.isclose(m["gamma"], 200 / 799)
    # (1 + sqrt(gamma))^2 at the corrected gamma.
    assert np.isclose(_mp_bulk_edge(m["gamma"]),
                      (1 + (200 / 799) ** 0.5) ** 2, atol=1e-9)


def test_mp_bulk_integration(mock_model, mock_tokenizer):
    """End-to-end on the conftest toy models: per-tag scalars present
    and the gamma-matched metrics finite."""
    from blme.tasks.geometry.rmt_bulk import MPBulkDeviationTask

    dataset = [{"text": f"sample passage number {i}"} for i in range(8)]
    task = MPBulkDeviationTask(config={"num_samples": 8})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=dataset)

    assert isinstance(results, dict)
    assert "error" not in results, f"mp_bulk_deviation failed: {results}"
    for tag in ("first", "mid", "last"):
        for stem in ("mp_outlier_frac", "mp_spike_energy", "mp_ks_distance"):
            key = f"{stem}_{tag}"
            assert key in results, f"missing {key}"
            assert np.isfinite(results[key]), f"{key} not finite"
        assert 0.0 <= results[f"mp_outlier_frac_{tag}"] <= 1.0
        assert results[f"_meta_mp_gamma_{tag}"] > 0
        assert results[f"_meta_mp_n_tokens_{tag}"] > 0
    assert np.isfinite(results["mp_ks_distance_mean"])


# ---------------------------------------------------------------------------
# Campaign-4 B4: numeric pins for the weights-and-spectra kernels that were
# previously only key-presence / bound-checked.
# ---------------------------------------------------------------------------

class _EmbOnlyModel(torch.nn.Module):
    """Fake model exposing just an input-embedding matrix."""

    def __init__(self, E):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(E.clone(), freeze=True)

    def get_input_embeddings(self):
        return self.emb


def test_hubness_evaluate_matches_independent_knn_reference():
    """Regression (campaign-4 B4): the evaluate() k-occurrence kernel was
    never numerically pinned — removing the self-mask changed skew/gini and
    every test still passed. Full independent numpy reference: cosine kNN,
    self excluded, occurrence counts -> scipy skew + sorted-rank Gini."""
    from scipy.stats import skew as scipy_skew
    from blme.tasks.geometry.hubness import GlobalHubnessTask

    torch.manual_seed(0)
    V, d, k = 257, 24, 5
    E = torch.randn(V, d)
    task = GlobalHubnessTask(config={"k_values": [k], "batch_size": 40})
    out = task.evaluate(_EmbOnlyModel(E), None, dataset=None)
    assert "error" not in out

    # Independent reference (numpy float64, full argsort — no argpartition).
    En = E.double().numpy()
    En = En / np.linalg.norm(En, axis=1, keepdims=True)
    sims = En @ En.T
    np.fill_diagonal(sims, -np.inf)               # self-mask
    counts = np.zeros(V, dtype=np.int64)
    for i in range(V):
        for j in np.argsort(sims[i])[-k:]:
            counts[j] += 1
    ref_skew = float(scipy_skew(counts))
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    ref_gini = float(
        (2.0 * np.arange(1, n + 1) - n - 1.0) @ x / (n * x.sum())
    )

    assert out[f"hubness_k{k}_skew"] == pytest.approx(ref_skew, abs=1e-6)
    assert out[f"hubness_k{k}_gini"] == pytest.approx(ref_gini, abs=1e-6)
    assert out[f"hubness_k{k}_max"] == counts.max()


def test_schatten_per_sentence_matches_official_mnn_pipeline():
    """Regression (campaign-4 B4): nothing pinned _per_sentence_measurements
    (preprocessing + MNN + the 2026-06-22 '/T' fix) — reverting mnn/T to
    mnn/d passed all 31 schatten tests. Reference: verbatim transcription of
    MLGroupJLU/MatrixNuclearNorm MNN.py (normalize L32-38, MNN L41-49,
    per-token division L195) @ e7a9188."""
    from blme.tasks.geometry.schatten import (
        _normalise_matrix,
        _per_sentence_measurements,
    )

    torch.manual_seed(0)
    Z = torch.randn(24, 16)

    # --- verbatim official reference ---
    def official_normalize(R):
        R = R - R.mean(dim=0, keepdim=True)
        return R / R.norm(dim=1, keepdim=True)

    def official_mnn(X):
        D = min(X.shape)
        col = X.norm(dim=0)
        return float(col.sort(descending=True).values[:D].sum())

    Zn_ref = official_normalize(Z.clone())
    ref_mnn_over_T = official_mnn(Zn_ref) / Z.shape[0]

    # Preprocessing must be BIT-identical to the official normalize().
    assert torch.equal(_normalise_matrix(Z.clone()), Zn_ref)

    out = _per_sentence_measurements(Z.clone())
    assert out["row_normalized_matrix_nuclear_norm"] == pytest.approx(
        ref_mnn_over_T, rel=1e-6
    )


def test_unembedding_purity_exact_on_orthogonal_clusters(tmp_path):
    """Analytic anchor (campaign-4 B4): two orthogonal category clusters in
    the unembedding matrix -> top-k purity EXACTLY 1.0 (purity was only
    bound-checked before; formula drift like match/(k+1) was invisible)."""
    import json as _json
    from blme.tasks.geometry.unembedding import UnembeddingDiagnosticsTask

    words_a = ["aa", "bb", "cc", "dd"]
    words_b = ["ee", "ff", "gg", "hh"]
    vocab = {w: i + 2 for i, w in enumerate(words_a + words_b)}  # ids 2..9

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            t = text.strip()
            if t in vocab:
                return [vocab[t]]
            return [0, 1]  # multi-token -> skipped by the label loader

    V, d = 12, 8
    W = torch.zeros(V, d)
    for w in words_a:
        W[vocab[w], 0] = 1.0          # category A -> e1
    for w in words_b:
        W[vocab[w], 1] = 1.0          # category B -> e2
    W[0, 2] = 1.0                      # unlabeled distractors, orthogonal
    W[1, 3] = 1.0
    W[10, 4] = 1.0
    W[11, 5] = 1.0

    class _HeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(d, V, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(W)

    cats = tmp_path / "categories.json"
    cats.write_text(_json.dumps({"cat_a": words_a, "cat_b": words_b}))

    # k = cluster size - 1: every neighbor slot must be same-category.
    task = UnembeddingDiagnosticsTask(config={
        "categories_path": str(cats), "k": 3, "num_samples": 0,
    })
    out = task.evaluate(_HeadModel(), _Tok(), dataset=None)

    assert out["_meta_unembedding_n_category_tokens"] == 8
    assert out["unembedding_purity_mean"] == 1.0


def test_neural_collapse_evaluate_end_to_end(mock_model, mock_tokenizer):
    """Behavioral coverage (campaign-4 B6): no test previously ran
    NeuralCollapseTask.evaluate() on a model — the double-norm regression
    guard was a string match on one exact spelling. Drive the real task and
    assert finite, plausible outputs."""
    from blme.tasks.geometry.neural_collapse import NeuralCollapseTask

    task = NeuralCollapseTask(config={"num_samples": 4})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in results:
        pytest.skip(f"unsupported architecture: {results['error']}")

    # The dummy tokenizer's 4 x 8 random ids give no next-token class with
    # min_per_class samples: the CDNV keys are then NaN WITH a reason, and
    # every other float (the topic proxy included) must be finite.
    explained = {"nc1_cdnv", "nc1_cdnv_galanti"} if "_meta_nc_reason" in results else set()
    for key, val in results.items():
        if isinstance(val, float) and key not in explained:
            assert np.isfinite(val), f"{key} not finite"
    assert any("nc1" in k for k in results), results.keys()
    assert np.isfinite(results["_meta_topic_nc1_within_class_collapse"])
    r2 = NeuralCollapseTask(config={"num_samples": 4}).evaluate(
        mock_model, mock_tokenizer, dataset=None)
    for key, val in results.items():
        if isinstance(val, float) and key not in explained:
            assert val == pytest.approx(r2[key], abs=0.0), f"{key} not deterministic"


def test_mahalanobis_pinv_fallback_matches_closed_form(monkeypatch):
    """Campaign-4 B7: the HAS_SKLEARN=False pinv fallback had zero coverage —
    dropping the ridge or the ddof would pass the suite. Analytic anchor:
    symmetric 4-point cloud has a DIAGONAL ddof-1 covariance, so the
    Mahalanobis distance has a closed form."""
    import blme.tasks.geometry.mahalanobis as mah

    monkeypatch.setattr(mah, "HAS_SKLEARN", False)
    a, b = 1.5, 0.5
    X_train = np.array([[a, 0.0], [-a, 0.0], [0.0, b], [0.0, -b]])
    query = np.array([[a, b]])
    d = mah._compute_mahalanobis_distances(X_train, query)

    var_x = 2 * a * a / 3.0     # ddof=1 over 4 zero-mean points
    var_y = 2 * b * b / 3.0
    ridge = 1e-6
    expected = np.sqrt(a * a / (var_x + ridge) + b * b / (var_y + ridge))
    assert d[0] == pytest.approx(float(expected), rel=1e-9)


def test_prediction_alignment_uses_output_projection_when_untied():
    """Campaign-4 B7: the untied-head behavior (the task's central advertised
    claim) was guarded only by a source grep — a regression to
    `return get_embeddings(model)` passed. Pin it numerically: on an untied
    model with a re-randomized head, the projection weight must be the
    lm_head tensor and must NOT be the input embedding table."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.geometry.consistency import _get_output_projection_weight

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=97,
                     n_positions=32, tie_word_embeddings=False)
    model = GPT2LMHeadModel(cfg).eval()
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.3)   # decouple from wte for sure

    W = _get_output_projection_weight(model)
    assert torch.equal(W, model.lm_head.weight.detach())
    assert not torch.allclose(W, model.transformer.wte.weight.detach())


def test_categories_relation_hubness_and_analogy_kernels():
    """Campaign-4 B7: the relation-consistency and analogy kernels never
    executed in tests, and hub_skew/hub_max were computed but unasserted.
    Constructed embedding: pairs relate by an EXACT constant vector r, so
    consistency == 1.0 and analogy top-1 == 1.0 analytically; hubness is
    recomputed independently."""
    from scipy.stats import skew as scipy_skew
    from blme.tasks.geometry.categories import CategoryGeometryTask

    rng = np.random.default_rng(0)
    V, d = 40, 12
    E = rng.standard_normal((V, d)) * 5.0
    r = np.zeros(d); r[0] = 30.0                 # exact relation vector
    pairs = []
    for i in range(6):                           # ids 0..5 -> 20..25
        a_id, b_id = i, 20 + i
        E[b_id] = E[a_id] + r
        pairs.append((a_id, b_id))

    task = CategoryGeometryTask(config={})
    out = task._compute_relation_consistency(E, {"rel": pairs}, tokenizer=None)

    assert out["rel_consistency"] == pytest.approx(1.0, abs=1e-9)
    assert out["_meta_rel_analogy_acc_top1"] == pytest.approx(1.0, abs=1e-12)
    assert out["_meta_rel_analogy_acc_top5"] == pytest.approx(1.0, abs=1e-12)

    # Hubness kernel vs an independent recomputation on a category.
    tids = list(range(10, 20))
    hub = task._compute_hubness(E, None, {"cat": tids}, k_cat=3)
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    Ec = E_norm[np.array(tids)]
    n_occ = np.zeros(len(tids))
    for i in range(len(tids)):
        sims = Ec @ Ec[i]
        sims[i] = -np.inf
        for idx in np.argsort(sims)[-3:]:
            n_occ[idx] += 1
    assert hub["cat_hub_max"] == int(n_occ.max())
    ref_skew = 0.0 if n_occ.std() == 0 else float(scipy_skew(n_occ))
    assert hub["cat_hub_skew"] == pytest.approx(ref_skew, abs=1e-9)


def test_representation_sensitivity_bias_head_matches_autograd():
    """Campaign-4 B7: the lm_head-bias branch (logits += bias) had zero
    coverage — both existing tests use GPT-2's bias-free head, so flipping
    the '+' to '-' passed the suite. Cross-check the closed-form gradient
    against autograd on a model whose head HAS a bias."""
    from transformers import GPT2Config, GPT2LMHeadModel
    from blme.tasks.geometry.information_geometry import (
        RepresentationSensitivityTask,
    )

    torch.manual_seed(0)
    vocab = 211
    cfg = GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=vocab,
                     n_positions=32, tie_word_embeddings=False)
    model = GPT2LMHeadModel(cfg).eval()
    model.lm_head = torch.nn.Linear(16, vocab, bias=True)
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.2)
        model.lm_head.bias.normal_(std=0.5)   # non-trivial bias

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, text, return_tensors="pt", **kw):
            import zlib
            g = torch.Generator().manual_seed(zlib.crc32(text.encode()))
            ids = torch.randint(2, vocab, (1, 10), generator=g)
            class BD(dict):
                def to(self, dev): return self
            return BD({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    texts = [{"text": f"probe sentence {i}"} for i in range(3)]
    task = RepresentationSensitivityTask(config={"num_samples": 3})
    out = task.evaluate(model, _Tok(), dataset=texts)
    if "error" in out:
        pytest.skip(f"unsupported: {out['error']}")
    blme_val = out["representation_sensitivity"]

    # Independent autograd reference on the same texts/ids/hidden states.
    W = model.lm_head.weight.detach().double()
    b = model.lm_head.bias.detach().double()
    sq_norms = []
    with torch.no_grad():
        pass
    for row in texts:
        enc = _Tok()(row["text"])
        with torch.no_grad():
            hs = model(input_ids=enc["input_ids"],
                       output_hidden_states=True).hidden_states[-1][0]
        ids = enc["input_ids"][0]
        for pos in range(hs.shape[0] - 1):
            h = hs[pos].detach().double().requires_grad_(True)
            logp = torch.log_softmax(W @ h + b, dim=-1)[ids[pos + 1]]
            (grad,) = torch.autograd.grad(logp, h)
            sq_norms.append(float((grad ** 2).sum()))
    ref = float(np.mean(sq_norms))
    assert blme_val == pytest.approx(ref, rel=1e-4), (
        f"bias-branch closed form diverged from autograd: {blme_val} vs {ref}"
    )


def test_every_geometry_task_actually_runs():
    """Smoke test: each registered geometry task must EXECUTE.

    `geometry_vendi_score` once shipped a `NameError` on every call -- a
    reference to an out-of-scope loop variable -- while its parity suite stayed
    9/9 green, because no test invoked `evaluate()`. A task that cannot run is
    the most severe failure there is, and nothing caught it.
    """
    import blme.tasks.geometry  # noqa: F401
    from blme.registry import list_tasks, get_task

    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    import torch
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        n_layer=3, n_head=2, n_embd=32, vocab_size=tok.vocab_size,
        n_positions=64, n_ctx=64)).eval()
    dataset = [{"text": "The quick brown fox jumps over the lazy dog."},
               {"text": "A second sentence with entirely different tokens here."},
               {"text": "Numbers one two three four five six seven eight nine."}]

    failures = []
    for name in sorted(t for t in list_tasks() if t.startswith("geometry_")):
        cls = get_task(name)
        if cls is None:
            continue
        try:
            cls(config={"num_samples": 3, "use_cache": False}).evaluate(
                model, tok, dataset, cache=None)
        except Exception as exc:  # noqa: BLE001 - we want the type and message
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "geometry tasks that raise on a minimal input:\n" + "\n".join(failures)
