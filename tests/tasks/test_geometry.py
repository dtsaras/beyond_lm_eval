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


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_svd_isotropy(mock_model, mock_tokenizer):
    from blme.tasks.geometry.isotropy import SVDIsotropyTask

    task = SVDIsotropyTask(config={"num_samples": 5})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert "svd_auc" in results
    assert "cond_number" in results
    assert results["svd_auc"] > 0
    assert results["svd_auc"] <= 1.0
    assert results["cond_number"] >= 1.0


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
    assert "lipschitz_max" in results
    assert "lipschitz_mean" in results
    assert results["lipschitz_max"] >= 0


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
    # Paper-faithful scalar summaries.
    for k in ("avg_adjacent_cka", "mean_offdiag_cka", "early_late_cka",
              "first_middle_cka", "n_layers"):
        assert k in results, f"missing {k}"
    assert 0 <= results["avg_adjacent_cka"] <= 1.0


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
        assert "mean_mahalanobis_id" in results


def test_mutual_info(mock_model, mock_tokenizer):
    """HSIC dependence estimation between layers."""
    from blme.tasks.geometry.mutual_info import HSICDependenceTask

    task = HSICDependenceTask(config={"num_samples": 3})
    results = task.evaluate(mock_model, mock_tokenizer, dataset=None)

    assert isinstance(results, dict)
    if "error" not in results:
        assert "avg_adjacent_hsic" in results


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
    assert 0 <= res["matrix_entropy_normalized"] <= 1.01, (
        f"H_normalized out of [0, 1]: {res['matrix_entropy_normalized']}"
    )
    assert 0 <= res["mean_matrix_entropy_normalized"] <= 1.01

    # Headline == last-layer entropy.
    last_key = max(
        res["layer_matrix_entropies"].keys(),
        key=lambda k: int(k.split("_")[1]),
    )
    assert res["matrix_entropy"] == res["layer_matrix_entropies"][last_key]


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
    unembedding (lm_head weights). They must use the same canonical
    Roy-Vetterli formula on ``σ²``. The historic code used ``σ``
    without squaring — numerically different and inconsistent with
    the paper."""
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

    # Linearly-spread σ: erank with σ² normalisation is strictly
    # smaller than erank with σ normalisation (the historic
    # miscomputation). This cross-checks the σ²-vs-σ convention.
    S = np.array([1.0, 0.5, 0.1, 0.01], dtype=np.float64)
    sq_form = effective_rank(S)
    lin_p = S / S.sum()
    lin_form = float(np.exp(-np.sum(lin_p * np.log(lin_p))))
    assert sq_form < lin_form, (
        f"σ² erank ({sq_form}) should be < σ erank ({lin_form}) "
        "for decreasing spectra; if equal, the helper isn't squaring."
    )
