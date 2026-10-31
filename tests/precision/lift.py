"""Run a task's analysis in float64: the same algorithm at higher precision.

``lift()`` makes every float32/float16/bfloat16 request inside its block a
float64 one:

- torch: a ``TorchDispatchMode`` rewrites ``dtype=`` on every aten op
  (``.float()``, ``.to()``, factories, ``sum(dtype=)``) and upcasts narrow
  inputs of out-of-place ops; the default dtype is float64.
- numpy: each ``blme`` module's ``np``/``numpy`` binding becomes a proxy whose
  ``float32``/``float16``/``single``/``half`` are ``float64``.

Dtype-derived constants (``torch.finfo``/``np.finfo``) are replayed at the
values the shipped run saw, call by call, so the lift changes precision and
not tolerances. ``record()`` runs the shipped analysis under the same
instrumentation without changing anything.

A lift is certified (``Certificate.ok``) when
1. no aten op inside it produced a float tensor below float64,
2. its aten op sequence equals the shipped run's, conversions aside (the
   same algorithm),
3. it made the same finfo calls as the shipped run, and
4. every array passed to a wrapped library entry point (ripser, sklearn
   estimators) was float64, and every entry point that computes at a fixed
   precision of its own (ripser: float32) was replaced by a float64
   reference (GUDHI).
"""
from __future__ import annotations

import contextlib
import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

LOW = (torch.float32, torch.float16, torch.bfloat16)
_NP_LOW = ("float32", "float16", "single", "half")
# Conversions: present or absent depending on the input dtype, never part of
# the algorithm.
_CONVERSIONS = {"_to_copy", "lift_fresh", "lift_fresh_copy", "detach", "alias"}
# Random ops draw different values at a different dtype from the same seed, so
# the lift draws at the dtype the shipped run used and then upcasts.
_RANDOM = {"randn", "rand", "randn_like", "rand_like", "normal", "uniform",
           "bernoulli", "normal_", "uniform_", "bernoulli_", "exponential_",
           "cauchy_", "log_normal_", "geometric_", "multinomial"}
# Entry points that compute at a fixed precision whatever they are given. The
# lift replaces each with a float64 reference whose agreement on exactly
# representable inputs is tested (test_lift.py); without one it is listed.
FIXED_PRECISION = {"ripser.ripser": "casts distances to float32 (ripser.py:315,324)"}


def _ripser_fp64(X, maxdim=1, thresh=np.inf, coeff=2, distance_matrix=False, **_):
    """``ripser.ripser``'s ``dgms`` from GUDHI, in float64. H0 is ordered by
    death as ripser orders it; every BLME use of a diagram is order-free."""
    import gudhi
    X = np.asarray(X, dtype=np.float64)
    rc = (gudhi.RipsComplex(distance_matrix=X, max_edge_length=thresh) if distance_matrix
          else gudhi.RipsComplex(points=X, max_edge_length=thresh))
    st = rc.create_simplex_tree(max_dimension=maxdim + 1)
    st.compute_persistence(homology_coeff_field=coeff, min_persistence=0.0)
    dgms = []
    for d in range(maxdim + 1):
        a = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64).reshape(-1, 2)
        dgms.append(a[np.lexsort((a[:, 0], a[:, 1]))])
    return {"dgms": dgms}


REFERENCES = {"ripser.ripser": _ripser_fp64}


@dataclasses.dataclass
class Trace:
    ops: List[str] = dataclasses.field(default_factory=list)
    low_ops: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    finfo: List[str] = dataclasses.field(default_factory=list)
    random: List[str] = dataclasses.field(default_factory=list)
    boundaries: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    substituted: List[str] = dataclasses.field(default_factory=list)

    def algorithm(self) -> List[str]:
        return [o for o in self.ops if o not in _CONVERSIONS]


@dataclasses.dataclass
class Certificate:
    low_ops: List[Tuple[str, str]]
    same_algorithm: bool
    first_divergence: Optional[Tuple[int, str, str]]
    same_finfo_calls: bool
    narrow_boundaries: List[Tuple[str, str]]
    fixed_precision: List[str]
    substituted: List[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (not self.low_ops and self.same_algorithm and self.same_finfo_calls
                and not self.narrow_boundaries
                and set(self.fixed_precision) <= set(self.substituted))


def _is_low(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in LOW


class _Mode(TorchDispatchMode):
    def __init__(self, trace: Trace, lift: bool, replay_random: Optional[List[str]]):
        super().__init__()
        self.trace, self.lift, self.replay_random = trace, lift, replay_random

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        name = func.overloadpacket.__name__
        if name in _RANDOM:
            out = self._random(func, name, args, kwargs)
        else:
            if self.lift:
                if kwargs.get("dtype") in LOW:
                    kwargs["dtype"] = torch.float64
                if not (name.endswith("_") or "out" in kwargs):
                    args = tree_map(lambda t: t.double() if _is_low(t) else t, args)
            out = func(*args, **kwargs)
        self.trace.ops.append(name)
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if _is_low(t):
                self.trace.low_ops.append((name, str(t.dtype)))
        return out

    def _random(self, func, name, args, kwargs):
        i = len(self.trace.random)
        if not self.lift or self.replay_random is None or i >= len(self.replay_random):
            out = func(*args, **kwargs)
            first = out[0] if isinstance(out, (tuple, list)) else out
            self.trace.random.append(str(first.dtype))
            return out
        drawn = getattr(torch, self.replay_random[i].replace("torch.", ""))
        self.trace.random.append(self.replay_random[i])
        if name.endswith("_"):  # fill in place: draw into a tensor of the shipped dtype
            dest = args[0]
            tmp = torch.empty(dest.shape, dtype=drawn)
            func(tmp, *args[1:], **kwargs)
            return dest.copy_(tmp)
        if "dtype" in kwargs or name in ("randn", "rand", "randn_like", "rand_like"):
            kwargs["dtype"] = drawn
        out = func(*args, **kwargs)
        return out.double() if out.is_floating_point() and out.dtype in LOW else out


class _FinfoReplay:
    """``finfo`` that records its argument (shipped) or answers each call with
    the dtype the shipped run asked for at the same position (lift)."""

    def __init__(self, real: Callable, trace: Trace, replay: Optional[List[str]], kind: str):
        self.real, self.trace, self.replay, self.kind = real, trace, replay, kind

    def __call__(self, dtype=None):
        if dtype is None:
            dtype = torch.get_default_dtype() if self.kind == "torch" else np.float64
        i = len(self.trace.finfo)
        self.trace.finfo.append(f"{self.kind}:{dtype}")
        if self.replay is not None and i < len(self.replay):
            kind, recorded = self.replay[i].split(":", 1)
            if kind == self.kind:
                return self.real(_parse_dtype(recorded, kind))
        return self.real(dtype)


def _parse_dtype(s: str, kind: str):
    if kind == "torch":
        return getattr(torch, s.replace("torch.", ""))
    return np.dtype(s.replace("<class 'numpy.", "").replace("'>", ""))


class _NumpyProxy(types.ModuleType):
    def __init__(self, finfo, lift: bool):
        super().__init__("numpy")
        self._finfo, self._lift = finfo, lift

    def __getattr__(self, name):
        if self._lift and name in _NP_LOW:
            return np.float64
        if name == "finfo":
            return self._finfo
        return getattr(np, name)


class _TorchProxy(types.ModuleType):
    """``torch`` with a replaying ``finfo``; lifted, ``torch.float32`` is
    float64, so ``x.dtype == torch.float32`` takes the shipped branch."""

    def __init__(self, finfo, lift: bool):
        super().__init__("torch")
        self._finfo, self._lift = finfo, lift

    def __getattr__(self, name):
        if name == "finfo":
            return self._finfo
        if self._lift and name in ("float32", "float"):
            return torch.float64
        return getattr(torch, name)


def _import_blme():
    """Import every ``blme`` module first: one imported inside the block would
    keep the real bindings and escape the instrumentation."""
    import importlib
    import pkgutil
    import blme
    for info in pkgutil.walk_packages(blme.__path__, "blme."):
        try:
            importlib.import_module(info.name)
        except ImportError:  # an optional dependency; its task cannot run either
            pass


def _blme_modules():
    return [m for n, m in list(sys.modules.items())
            if m is not None and (n == "blme" or n.startswith("blme."))]


@contextlib.contextmanager
def _swap_bindings(np_proxy, torch_proxy):
    saved = []
    for mod in _blme_modules():
        for attr, val in vars(mod).copy().items():
            if val is np:
                saved.append((mod, attr, val))
                setattr(mod, attr, np_proxy)
            elif val is torch:
                saved.append((mod, attr, val))
                setattr(mod, attr, torch_proxy)
    try:
        yield
    finally:
        for mod, attr, val in saved:
            setattr(mod, attr, val)


def _wrap_boundaries(trace: Trace, lift: bool):
    """Record the dtype of every array passed to ripser and to sklearn
    estimators' fit/predict/transform/score methods; lifted, a fixed-precision
    entry point with a float64 reference is replaced by it. A function is
    rebound wherever a ``blme`` module imported it by name."""
    patches = []

    def dtypes(a, kw):
        out = []
        for x in list(a) + list(kw.values()):
            if isinstance(x, np.ndarray) and x.dtype.kind == "f":
                out.append(str(x.dtype))
            elif isinstance(x, torch.Tensor) and x.is_floating_point():
                out.append(str(x.dtype))
        return out

    def wrap(owner, attr, label):
        real = getattr(owner, attr)
        target = REFERENCES[label] if lift and label in REFERENCES else real
        if target is not real:
            trace.substituted.append(label)

        def f(*a, **kw):
            for d in dtypes(a, kw):
                trace.boundaries.append((label, d))
            return target(*a, **kw)
        patches.append((owner, attr, real))
        setattr(owner, attr, f)
        if not isinstance(owner, type):
            for mod in _blme_modules():
                for name, val in vars(mod).copy().items():
                    if val is real:
                        patches.append((mod, name, real))
                        setattr(mod, name, f)

    try:
        import ripser as _rip
        wrap(_rip, "ripser", "ripser.ripser")
    except ImportError:
        pass
    try:
        from sklearn.base import BaseEstimator
        import sklearn.linear_model, sklearn.neighbors, sklearn.svm, sklearn.decomposition  # noqa: E401,F401
        for cls in _subclasses(BaseEstimator):
            for meth in ("fit", "predict", "predict_proba", "decision_function",
                         "transform", "fit_transform", "score", "kneighbors"):
                if meth in vars(cls):
                    wrap(cls, meth, f"sklearn.{cls.__name__}.{meth}")
    except ImportError:
        pass
    return patches


def _subclasses(cls):
    seen, todo = set(), [cls]
    while todo:
        for s in todo.pop().__subclasses__():
            if s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


@contextlib.contextmanager
def _instrument(trace: Trace, lift: bool, shipped: Optional[Trace]):
    _import_blme()
    replay = shipped.finfo if shipped is not None else None
    tf = _FinfoReplay(torch.finfo, trace, replay, "torch")
    nf = _FinfoReplay(np.finfo, trace, replay, "numpy")
    patches = _wrap_boundaries(trace, lift)
    old_default = torch.get_default_dtype()
    try:
        if lift:
            torch.set_default_dtype(torch.float64)
        with _swap_bindings(_NumpyProxy(nf, lift), _TorchProxy(tf, lift)), \
                _Mode(trace, lift, shipped.random if shipped is not None else None):
            yield
    finally:
        torch.set_default_dtype(old_default)
        for owner, attr, real in reversed(patches):
            setattr(owner, attr, real)


def record(fn: Callable[[], Any]) -> Tuple[Any, Trace]:
    """Run ``fn`` as shipped, recording ops, finfo calls and boundaries."""
    trace = Trace()
    with _instrument(trace, lift=False, shipped=None):
        out = fn()
    return out, trace


def lift(fn: Callable[[], Any], shipped: Trace) -> Tuple[Any, Trace]:
    """Run ``fn`` lifted to float64, replaying ``shipped``'s finfo calls."""
    trace = Trace()
    with _instrument(trace, lift=True, shipped=shipped):
        out = fn()
    return out, trace


def certify(shipped: Trace, lifted: Trace) -> Certificate:
    a, b = shipped.algorithm(), lifted.algorithm()
    div = None
    if a != b:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        div = (i, a[i] if i < len(a) else "<end>", b[i] if i < len(b) else "<end>")
    labels = {lbl for lbl, _ in lifted.boundaries}
    return Certificate(
        low_ops=sorted(set(lifted.low_ops)),
        same_algorithm=a == b,
        first_divergence=div,
        same_finfo_calls=(len(shipped.finfo) == len(lifted.finfo)
                          and shipped.random == lifted.random),
        narrow_boundaries=sorted({(l, d) for l, d in lifted.boundaries
                                  if d != "float64" and d != "torch.float64"
                                  and l not in FIXED_PRECISION}),
        fixed_precision=sorted(labels & set(FIXED_PRECISION)),
        substituted=sorted(set(lifted.substituted)),
    )


def shipped_and_lifted(fn_shipped: Callable[[], Any], fn_lifted: Callable[[], Any]
                       ) -> Tuple[Any, Any, Certificate]:
    """``fn_shipped`` and ``fn_lifted`` run the same analysis on the same
    values; the lifted one is handed float64 storage of them."""
    out_s, tr_s = record(fn_shipped)
    out_l, tr_l = lift(fn_lifted, tr_s)
    return out_s, out_l, certify(tr_s, tr_l)
