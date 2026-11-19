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
