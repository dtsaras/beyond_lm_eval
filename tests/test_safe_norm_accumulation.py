"""Norms over parameter-sized float32 vectors accumulate in float64 (blind
review D: SAM/ASAM on gpt2 were 0.1-0.15% off an independent computation
because a float32 norm() of the 7M-coordinate gradient is 4e-4 off). torch's
sum() is already cascade-accurate, so sharpness's _dot is guarded, not changed."""
import numpy as np
import torch

from blme.tasks.dynamics.sharpness import _dot, _sam_sharpness
from blme.tasks.geometry.utils import safe_norm

N = 7_000_000


def _vec(seed, n=N):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * torch.rand(n, generator=g) ** 4


def test_safe_norm_of_a_large_float32_vector_is_float64_accurate():
    x = _vec(0)
    ref = float(np.sqrt(np.sum(x.double().numpy() ** 2)))
    assert abs(safe_norm(x) - ref) / ref < 1e-6
    # float32 accumulation, the old path, is far off at this size.
    assert abs(float(x.norm()) - ref) / ref > 1e-5


def test_safe_norm_keeps_tiny_and_huge_scales():
    tiny = torch.full((8,), 1.250000039e-23)
    assert safe_norm(tiny) == np.float32(1.250000039e-23) * np.sqrt(8.0) or \
        abs(safe_norm(tiny) - 3.535534017e-23) / 3.535534017e-23 < 1e-6
    huge = torch.full((4,), 3e38)
    assert np.isfinite(safe_norm(huge)) and abs(safe_norm(huge) - 6e38) / 6e38 < 1e-6


def test_dot_of_large_float32_lists_is_float64_accurate():
    a, b = [_vec(1, N // 2), _vec(2, N // 2)], [_vec(3, N // 2), _vec(4, N // 2)]
    ref = sum(float(np.dot(x.double().numpy(), y.double().numpy())) for x, y in zip(a, b))
    assert abs(_dot(a, b) - ref) / abs(ref) < 1e-6


def test_sam_gradient_norm_is_float64_accurate():
    g = torch.Generator().manual_seed(5)
    target = torch.randn(4_000_000, generator=g)
    theta = torch.nn.Parameter(torch.zeros(4_000_000))

    def loss():
        return 0.5 * ((theta - target) ** 2).mean()

    base = float(loss())
    _, _, grad_norm = _sam_sharpness(loss, [theta], 0.05, base)
    want = float(np.linalg.norm(target.double().numpy())) / target.numel()
    assert abs(grad_norm - want) / want < 1e-6
