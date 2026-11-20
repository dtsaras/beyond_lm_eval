"""GPU linear-algebra paths (geometry/utils.py): same quantities as the CPU path.

knn_sorted_batched must reproduce the stable lexsort loop the tasks used
(nearest first, ties to the lower index, the query excluded), including on
exact duplicate rows; safe_svdvals and the svd/eigh wrappers on the GPU must
agree with the CPU to float64 round-off. BLME_CPU_LINALG=1 must force the CPU.
"""
import numpy as np
import pytest
import torch

from blme.tasks.geometry import utils as U

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _lexsort_knn(E, i, k):
    d2 = ((E - E[i]) ** 2).sum(axis=1)
    d2[i] = np.inf
    return np.lexsort((np.arange(len(d2)), d2))[:k]


def _cloud(n=3000, d=48, n_dup=40, seed=0):
    g = np.random.default_rng(seed)
    E = g.standard_normal((n, d))
    E[n - n_dup:] = E[7]                       # exact ties with row 7
    return U.safe_unit_rows(E)


def test_knn_matches_lexsort_on_cpu(monkeypatch):
    monkeypatch.setenv("BLME_CPU_LINALG", "1")
    E = _cloud()
    q = [0, 7, 100, 2999, 1500]
    got = U.knn_sorted_batched(E, q, 50, batch=2)
    for row, i in zip(got, q):
        assert row.tolist() == _lexsort_knn(E, i, 50).tolist()


@cuda
def test_knn_matches_lexsort_on_gpu(monkeypatch):
    monkeypatch.delenv("BLME_CPU_LINALG", raising=False)
    assert U.gpu_linalg_device() is not None
    E = _cloud()
    q = list(range(0, 3000, 97)) + [7, 2999]
    got = U.knn_sorted_batched(E, q, 50, batch=8)
    for row, i in zip(got, q):
        assert row.tolist() == _lexsort_knn(E, i, 50).tolist(), i


def test_cpu_escape_hatch(monkeypatch):
    monkeypatch.setenv("BLME_CPU_LINALG", "1")
    assert U.gpu_linalg_device() is None
    X = torch.randn(400, 300)
    (Y,), moved = U.to_gpu_linalg(X)
    assert not moved and Y is X


@cuda
def test_safe_svdvals_gpu_matches_cpu(monkeypatch):
    g = np.random.default_rng(1)
    X = g.standard_normal((2000, 300)) * np.linspace(0.01, 10, 300)
    monkeypatch.setenv("BLME_CPU_LINALG", "1")
    cpu = U.safe_svdvals(X)
    monkeypatch.delenv("BLME_CPU_LINALG")
    gpu = U.safe_svdvals(X)
    assert isinstance(gpu, np.ndarray)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12 * cpu[0])
    # Torch input on the CPU comes back as a CPU tensor.
    t = U.safe_svdvals(torch.as_tensor(X))
    assert t.device.type == "cpu"


@cuda
def test_svd_and_eigh_wrappers_match_cpu(monkeypatch):
    X = torch.randn(1500, 200, dtype=torch.float64)
    monkeypatch.delenv("BLME_CPU_LINALG", raising=False)
    u, s, vh = U.gpu_svd(X)
    assert u.device.type == "cpu"
    np.testing.assert_allclose(s.numpy(), torch.linalg.svdvals(X).numpy(), rtol=1e-12)
    np.testing.assert_allclose((u * s) @ vh, X.numpy(), atol=1e-10)
    A = X.T @ X
    w, v = U.gpu_eigh(A)
    np.testing.assert_allclose(w.numpy(), torch.linalg.eigvalsh(A).numpy(), rtol=1e-10, atol=1e-8)


@cuda
def test_linear_cka_gpu_path_is_float64_exact():
    from blme.tasks.geometry.cka import _linear_cka
    g = torch.Generator().manual_seed(3)
    X = torch.randn(20000, 256, generator=g) + 3.0
    Y = X @ torch.randn(256, 256, generator=g) * 0.1 + torch.randn(20000, 256, generator=g)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    exact = _linear_cka(X.double(), Y.double(), centered=True)
    gpu = _linear_cka(X.cuda(), Y.cuda(), centered=True)
    cpu32 = _linear_cka(X, Y, centered=True)
    assert abs(gpu - exact) / exact < 1e-12
    # The float32 CPU path is the less accurate one.
    assert abs(cpu32 - exact) >= abs(gpu - exact)



def _massive_cloud(n=1500, d=1024, seed=5):
    """Centred token cloud with two massive coordinates (Sun et al. 2024 scale)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    X[:, :2] *= 120.0
    return X - X.mean(0)


@cuda
def test_cached_norms_share_the_gpu_numerator_arithmetic():
    """evaluate() caches ||X^T X||_F per layer and passes it to _linear_cka; on the
    GPU the numerator is float64, so the cached norm must be too. A float32 norm
    there put CKA(X, X) up to 1e-2 above 1 on 12-34B models (BLME 0.2.1-0.2.2)."""
    from blme.tasks.geometry.cka import _linear_cka, _max_relative_gram_norm
    X = _massive_cloud().cuda()
    nx = _max_relative_gram_norm(X)
    assert abs(_linear_cka(X, X, norm_x=nx, norm_y=nx, centered=True) - 1.0) < 1e-12
    Y = X + 0.5 * torch.randn_like(X)
    exact = _linear_cka(X.double().cpu(), Y.double().cpu(), centered=True)
    got = _linear_cka(X, Y, norm_x=nx, norm_y=_max_relative_gram_norm(Y), centered=True)
    assert abs(got - exact) / exact < 1e-12


def _massive_states(n_docs=24, T=40, d=384, n_layers=4, seed=6):
    g = torch.Generator().manual_seed(seed)
    base = [torch.randn(T, d, generator=g) for _ in range(n_docs)]
    states = {}
    for layer in range(n_layers):
        docs = []
        for x in base:
            y = x + 0.3 * layer * torch.randn(T, d, generator=g)
            y[:, :2] *= 120.0
            docs.append(y)
        states[layer] = docs
    return states


@cuda
def test_cka_evaluate_self_similarity_is_one_on_the_gpu():
    """The real evaluate(): the derived flag reads the matrix diagonal, which goes
    through the cached norms, so it is ~1e-16 only if they match the numerator."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from invariance.driver import run_task_on_states
    r = run_task_on_states("geometry_cka", _massive_states())
    assert r["_meta_linalg_device"] == "cuda"
    assert r["_meta_max_abs_self_cka_minus_one"] < 1e-12
    assert r["min_adjacent_cka"] <= 1.0 and r["_meta_max_adjacent_cka"] <= 1.0 + 1e-12


def test_cka_evaluate_self_similarity_is_one_on_the_cpu(monkeypatch):
    """The CPU path is float32 throughout, numerator and norms alike: exactly 1."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from invariance.driver import run_task_on_states
    monkeypatch.setenv("BLME_CPU_LINALG", "1")
    r = run_task_on_states("geometry_cka", _massive_states())
    assert r["_meta_linalg_device"] == "cpu"
    assert r["_meta_max_abs_self_cka_minus_one"] == 0.0

def test_eigvalsh_external_helper_round_trip(tmp_path):
    """The helper protocol: (n, n) float64 .npy in, n float64 eigenvalues out."""
    helper = tmp_path / "helper.py"
    helper.write_text("import sys, numpy as np\n"
                      "np.save(sys.argv[2], np.linalg.eigvalsh(np.load(sys.argv[1])))\n")
    g = np.random.default_rng(4)
    X = g.standard_normal((300, 120))
    M = torch.as_tensor(X.T @ X)
    import sys
    w = U._external_eigvalsh(M, f"{sys.executable} {helper}")
    np.testing.assert_allclose(w.numpy(), torch.linalg.eigvalsh(M).numpy(), rtol=1e-12, atol=1e-10)
    bad = tmp_path / "bad.py"
    bad.write_text("import sys, numpy as np\nnp.save(sys.argv[2], np.zeros(3))\n")
    with pytest.raises(ValueError):
        U._external_eigvalsh(M, f"{sys.executable} {bad}")


@cuda
def test_eigvalsh_f64_falls_back_on_a_cuda_failure(monkeypatch, tmp_path):
    """A cuSOLVER rejection goes to the helper, then to the CPU; never an error."""
    import sys
    g = np.random.default_rng(5)
    X = g.standard_normal((400, 150))
    M = torch.as_tensor(X.T @ X).cuda()
    want = torch.linalg.eigvalsh(M.cpu())
    real = torch.linalg.eigvalsh

    def eig(A, *a, **k):
        if A.is_cuda:
            raise RuntimeError("cusolver error: CUSOLVER_STATUS_INVALID_VALUE (simulated)")
        return real(A, *a, **k)
    monkeypatch.setattr(torch.linalg, "eigvalsh", eig)
    monkeypatch.delenv("BLME_EIGVALSH_HELPER", raising=False)
    w, solver = U.eigvalsh_f64(M)
    assert solver == "cpu" and w.is_cuda
    np.testing.assert_allclose(w.cpu().numpy(), want.numpy(), rtol=1e-12, atol=1e-10)
    helper = tmp_path / "helper.py"
    helper.write_text("import sys, numpy as np\n"
                      "np.save(sys.argv[2], np.linalg.eigvalsh(np.load(sys.argv[1])))\n")
    monkeypatch.setenv("BLME_EIGVALSH_HELPER", f"{sys.executable} {helper}")
    w, solver = U.eigvalsh_f64(M)
    assert solver == "external"
    np.testing.assert_allclose(w.cpu().numpy(), want.numpy(), rtol=1e-12, atol=1e-10)


@cuda
def test_safe_svdvals_cuda_input_falls_back_to_the_cpu(monkeypatch):
    """A weight already on the GPU: an out-of-memory there recomputes on the CPU
    (it used to raise, and geometry_spectral left the module out)."""
    W = torch.randn(600, 200, dtype=torch.float32).cuda()
    monkeypatch.setenv("BLME_CPU_LINALG", "1")
    want = U.safe_svdvals(W.cpu())
    monkeypatch.delenv("BLME_CPU_LINALG")
    real = torch.linalg.svdvals

    def sv(A, *a, **k):
        if A.is_cuda:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(A, *a, **k)
    monkeypatch.setattr(torch.linalg, "svdvals", sv)
    got = U.safe_svdvals(W)
    assert got.is_cuda
    np.testing.assert_allclose(got.cpu().numpy(), want.numpy(), rtol=0, atol=0)
