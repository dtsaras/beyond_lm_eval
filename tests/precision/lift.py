"""Run a task in float64: the same algorithm at higher precision.

``lift()`` makes every float32/float16/bfloat16 request inside its block a
float64 one:

- torch: a ``TorchDispatchMode`` rewrites ``dtype=`` on every aten op
  (``.float()``, ``.to()``, factories, ``sum(dtype=)``) and upcasts narrow
  inputs of out-of-place ops (parameters once, memoised); the default dtype
  is float64.
- numpy/torch names: each ``blme`` module's ``np``/``torch`` binding becomes a
  proxy whose ``float32`` (numpy also ``float16``/``single``/``half``) is
  ``float64``, so ``x.dtype == torch.float32`` takes the shipped branch.
- ``fp32_compute`` promotes to float64, and its restore really returns each
  parameter and buffer to its storage dtype.
- ripser, which computes in float32 whatever it is given, is replaced by a
  GUDHI float64 reference.

Two things are replayed from the shipped run, keyed by the call site in
``blme`` code, so that only precision changes: ``finfo`` answers (tolerances)
and the dtype of every random draw. ``record()`` runs the shipped code under
the same instrumentation without changing anything.

``certify`` passes a lift when
1. no aten op inside it produced a float tensor below float64,
2. every finfo call and random draw it made was replayed from a site the
   shipped run also reached,
3. every array passed to a wrapped library entry point (ripser, sklearn
   estimators) was float64, and every fixed-precision entry point was
   replaced by its reference, and
4. (``strict``) its aten op sequence equals the shipped run's, conversions
   aside. An analysis-only lift on identical inputs must be the same
   algorithm; an end-to-end run may legitimately branch differently (a
   tolerance loop stopping one step earlier, greedy decoding choosing another
   token), which is part of the effect being measured and is reported.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
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
# representable inputs is tested (test_lift.py).
FIXED_PRECISION = {"ripser.ripser": "casts distances to float32 (ripser.py:315,324)"}
_BLME_DIR = os.sep + "blme" + os.sep


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
    finfo: Dict[str, str] = dataclasses.field(default_factory=dict)      # site -> dtype
    random: Dict[str, str] = dataclasses.field(default_factory=dict)     # site -> dtype
    unmatched: List[str] = dataclasses.field(default_factory=list)
    boundaries: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    substituted: List[str] = dataclasses.field(default_factory=list)

    def algorithm(self) -> List[str]:
        return [o for o in self.ops if o not in _CONVERSIONS]


@dataclasses.dataclass
class Certificate:
    low_ops: List[Tuple[str, str]]
    unmatched_sites: List[str]
    narrow_boundaries: List[Tuple[str, str]]
    fixed_precision: List[str]
    substituted: List[str]
    same_algorithm: bool
    first_divergence: Optional[Tuple[int, str, str]]
    strict: bool = True

    @property
    def ok(self) -> bool:
        return (not self.low_ops and not self.unmatched_sites and not self.narrow_boundaries
                and set(self.fixed_precision) <= set(self.substituted)
                and (self.same_algorithm or not self.strict))


def _is_low(t) -> bool:
    return isinstance(t, torch.Tensor) and t.dtype in LOW


def _site() -> str:
    """The innermost ``blme`` source line on the stack (else the outermost
    caller), identifying a finfo call or random draw across the two runs."""
    f, last = sys._getframe(2), None
    while f is not None:
        fn = f.f_code.co_filename
        if _BLME_DIR in fn and "precision" + os.sep + "lift" not in fn:
            return f"{fn.rsplit(_BLME_DIR, 1)[1]}:{f.f_lineno}"
        last = last or f"{os.path.basename(fn)}:{f.f_lineno}"
        f = f.f_back
    return last or "<unknown>"


class _Mode(TorchDispatchMode):
    def __init__(self, trace: Trace, lift: bool, shipped: Optional[Trace]):
        super().__init__()
        self.trace, self.lift, self.shipped = trace, lift, shipped
        self._memo: Dict[int, Tuple[torch.Tensor, int, torch.Tensor]] = {}

    def _up(self, t):
        if not _is_low(t):
            return t
        if isinstance(t, torch.nn.Parameter) or t.is_leaf and t.requires_grad:
            hit = self._memo.get(id(t))
            if hit is not None and hit[0] is t and hit[1] == t._version:
                return hit[2]
            up = t.double()
            self._memo[id(t)] = (t, t._version, up)
            return up
        return t.double()

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
                    args = tree_map(self._up, args)
            out = func(*args, **kwargs)
        self.trace.ops.append(name)
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if _is_low(t):
                self.trace.low_ops.append((name, str(t.dtype)))
        return out

    def _random(self, func, name, args, kwargs):
        site = f"{_site()}:{name}"
        if not self.lift:
            out = func(*args, **kwargs)
            first = out[0] if isinstance(out, (tuple, list)) else out
            self.trace.random[site] = str(first.dtype)
            return out
        recorded = (self.shipped.random if self.shipped else {}).get(site)
        if recorded is None:
            self.trace.unmatched.append(f"random {site}")
            return func(*args, **kwargs)
        self.trace.random[site] = recorded
        drawn = getattr(torch, recorded.replace("torch.", ""))
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
    """``finfo`` that records the dtype asked for at each call site (shipped)
    or answers with the one recorded there (lift)."""

    def __init__(self, real: Callable, kind: str, trace: Trace, shipped: Optional[Trace]):
        self.real, self.kind, self.trace, self.shipped = real, kind, trace, shipped

    def __call__(self, dtype=None):
        if dtype is None:
            dtype = torch.get_default_dtype() if self.kind == "torch" else np.float64
        site = f"{self.kind}:{_site()}"
        if self.shipped is None:
            self.trace.finfo[site] = _dtype_name(dtype, self.kind)
            return self.real(dtype)
        recorded = self.shipped.finfo.get(site)
        if recorded is None:
            self.trace.unmatched.append(f"finfo {site}")
            return self.real(dtype)
        self.trace.finfo[site] = recorded
        return self.real(getattr(torch, recorded) if self.kind == "torch" else np.dtype(recorded))


def _dtype_name(dtype, kind) -> str:
    return str(dtype).replace("torch.", "") if kind == "torch" else np.dtype(dtype).name


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


def _rebind(real, replacement, patches):
    """Point every ``blme`` module global bound to ``real`` at ``replacement``."""
    for mod in _blme_modules():
        for name, val in vars(mod).copy().items():
            if val is real:
                patches.append((mod, name, real))
                setattr(mod, name, replacement)


def _fp32_compute_lifted(real):
    """``fp32_compute`` for the lift: inside it promotes to float64 (the
    dispatch rewrite), and on exit every parameter, gradient and buffer goes
    back to the storage dtype it had on entry, which the rewrite would
    otherwise turn into another float64 cast."""
    @contextlib.contextmanager
    def cm(model, enabled: bool = True):
        params = [(p, p.dtype) for p in model.parameters()]
        buffers = [(mod, name, buf.dtype) for mod in model.modules()
                   for name, buf in mod._buffers.items() if buf is not None]
        try:
            with real(model, enabled) as info:
                yield info
        finally:
            with _disable_current_modes(), torch.no_grad():
                for p, dt in params:
                    if p.dtype != dt:
                        p.data = p.data.to(dt)
                    if p.grad is not None and p.grad.dtype != dt:
                        p.grad = p.grad.to(dt)
                for mod, name, dt in buffers:
                    b = mod._buffers.get(name)
                    if b is not None and b.dtype != dt:
                        mod._buffers[name] = b.to(dt)
    return cm


def _patch(trace: Trace, lift: bool):
    """Wrap the library entry points (recording array dtypes) and, lifted,
    substitute references and the float64 ``fp32_compute``."""
    patches = []

    def dtypes(a, kw):
        out = []
        for x in list(a) + list(kw.values()):
            if isinstance(x, np.ndarray) and x.dtype.kind == "f":
                out.append(str(x.dtype))
            elif isinstance(x, torch.Tensor) and x.is_floating_point():
                out.append(str(x.dtype).replace("torch.", ""))
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
            _rebind(real, f, patches)

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
    if lift:
        from blme.tasks import _precision
        real = _precision.fp32_compute
        lifted = _fp32_compute_lifted(real)
        _rebind(real, lifted, patches)
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
    tf = _FinfoReplay(torch.finfo, "torch", trace, shipped)
    nf = _FinfoReplay(np.finfo, "numpy", trace, shipped)
    old_default = torch.get_default_dtype()
    patches = []
    swapped = []
    try:
        patches = _patch(trace, lift)
        for mod in _blme_modules():
            for attr, val in vars(mod).copy().items():
                if val is np or val is torch:
                    swapped.append((mod, attr, val))
                    setattr(mod, attr, _NumpyProxy(nf, lift) if val is np
                            else _TorchProxy(tf, lift))
        if lift:
            torch.set_default_dtype(torch.float64)
        with _Mode(trace, lift, shipped):
            yield
    finally:
        torch.set_default_dtype(old_default)
        for owner, attr, real in reversed(swapped + patches):
            setattr(owner, attr, real)


def record(fn: Callable[[], Any]) -> Tuple[Any, Trace]:
    """Run ``fn`` as shipped, recording ops, finfo calls, draws and boundaries."""
    trace = Trace()
    with _instrument(trace, lift=False, shipped=None):
        out = fn()
    return out, trace


def lift(fn: Callable[[], Any], shipped: Trace) -> Tuple[Any, Trace]:
    """Run ``fn`` lifted to float64, replaying ``shipped``'s finfo and draws."""
    trace = Trace()
    with _instrument(trace, lift=True, shipped=shipped):
        out = fn()
    return out, trace


def certify(shipped: Trace, lifted: Trace, strict: bool = True) -> Certificate:
    a, b = shipped.algorithm(), lifted.algorithm()
    div = None
    if a != b:
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        div = (i, a[i] if i < len(a) else "<end>", b[i] if i < len(b) else "<end>")
    labels = {lbl for lbl, _ in lifted.boundaries}
    return Certificate(
        low_ops=sorted(set(lifted.low_ops)),
        unmatched_sites=sorted(set(lifted.unmatched)),
        narrow_boundaries=sorted({(lbl, d) for lbl, d in lifted.boundaries
                                  if d != "float64" and lbl not in FIXED_PRECISION}),
        fixed_precision=sorted(labels & set(FIXED_PRECISION)),
        substituted=sorted(set(lifted.substituted)),
        same_algorithm=a == b,
        first_divergence=div,
        strict=strict,
    )


def shipped_and_lifted(fn_shipped: Callable[[], Any], fn_lifted: Callable[[], Any],
                       strict: bool = True) -> Tuple[Any, Any, Certificate]:
    """``fn_shipped`` and ``fn_lifted`` run the same task on the same values;
    the lifted one may be handed float64 storage of them."""
    out_s, tr_s = record(fn_shipped)
    out_l, tr_l = lift(fn_lifted, tr_s)
    return out_s, out_l, certify(tr_s, tr_l, strict)
