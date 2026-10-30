"""A task that fails must not leave the model modified.

`dynamics_perturbation_growth` registers a forward hook on `layers[-1]` and
removed it on the straight-line path only. The first fix moved the removal above
the two early returns, closing the "no usable samples" and "degenerate profile"
paths — and leaving the EXCEPTION path wide open. An independent review measured
it: raise inside the collection loop and `layers[-1]._forward_hooks` goes 0 -> 1
and STAYS there for the life of the process.

That is worse than it sounds in a harness that runs many tasks against one
model object: every LATER task then pays the hook on every forward, and the hook
closure pins the last activation tensor alive. The sibling `dynamics_coe`
already used try/finally.

This test covers every task that hooks a module, so the next one to grow a hook
is covered by construction rather than by someone remembering.
"""
import pytest
import torch

import ast
import os
import pathlib

import blme
from blme.tasks.common import get_layers
from blme.registry import _TASK_REGISTRY
from blme.tasks import geometry, dynamics, consistency, interpretability  # noqa: F401
from blme.tasks import causality, topology, representation_engineering  # noqa: F401

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "blme"


def _tasks_that_register_hooks():
    """Every registered task whose module calls `register_forward*_hook`.

    DISCOVERED, not hand-listed. The first version of this file named two tasks
    and its own docstring claimed to cover "every task that hooks a module". It
    did not: `geometry_matrix_entropy` hooks `layers[-1]`, leaked it on the
    exception path, and was not in the list -- found by an independent review,
    not by this test (2026-09-05). A hand-maintained list of things to check is
    the same failure mode as a hand-sampled review.
    """
    hooking = set()
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        found = any(
            isinstance(n, ast.Attribute)
            and n.attr in ("register_forward_hook", "register_forward_pre_hook")
            for n in ast.walk(tree))
        if found:
            hooking.add(path.resolve())

    # SHARED HELPERS COUNT TOO, and missing that hid the worst instance.
    #
    # Mapping `inspect.getfile(cls)` over the task registry finds only tasks
    # that hook in their OWN module. `cache.py` registers a hook on
    # `layers[-1]` inside `ModelOutputCache._run_forward_pass` and defines no
    # task class, so it could never appear -- and `core.py` populates one shared
    # cache for a whole run, making it the most consequential hook site in the
    # harness. An independent review found it leaking there while this test
    # passed (2026-09-06). The check written to replace a hand-maintained list
    # had a hand-list-shaped hole one indirection further out.
    #
    # A task that hooks only through a shared helper is included via
    # SHARED_HOOKERS, so its exception path is exercised.
    SHARED_HOOKERS = {
        (SRC / "cache.py").resolve(),
        (SRC / "tasks" / "geometry" / "utils.py").resolve(),
    }
    shared_hooking = hooking & SHARED_HOOKERS

    import inspect
    out = []
    for name, cls in sorted(_TASK_REGISTRY.items()):
        try:
            f = pathlib.Path(inspect.getfile(cls)).resolve()
        except Exception:  # pragma: no cover
            continue
        if f in hooking:
            out.append((name, cls))
        elif shared_hooking and _uses_shared_helper(cls):
            out.append((name, cls))
    return out


def _uses_shared_helper(cls):
    """True if the task's module reaches a hook-registering shared helper."""
    import inspect
    try:
        src = pathlib.Path(inspect.getfile(cls)).read_text()
    except Exception:  # pragma: no cover
        return False
    return any(k in src for k in ("ModelOutputCache", "collect_hidden_states",
                                  "get_hidden_states"))


HOOKING_TASKS = _tasks_that_register_hooks()


_BLME_PKG_DIR = os.path.dirname(os.path.abspath(blme.__file__))


def _is_blme_hook(fn):
    """True for a hook this repo registered.

    The distinction matters, and finding it out is why this helper exists.
    On a Llama-shaped model, `output_hidden_states=True` makes **transformers
    itself** register `install_output_capuring_hook.<locals>.output_capturing_hook`
    on every block and remove it after the forward returns — so when the
    forward RAISES, HF leaks one hook per layer. That is upstream behaviour,
    not BLME's, and asserting on the raw count would have this test failing for
    something the repo does not control and cannot fix. It would also be a
    false accusation in a campaign whose whole point is not making those.
    """
    mod = getattr(fn, "__module__", "") or ""
    if mod == "blme" or mod.startswith("blme."):
        return True
    # Ownership is decided by the real package directory, never by a substring:
    # a venv named `blme` put transformers' own `output_capturing_hook` under a
    # path containing "blme" and produced 690 false failures.
    filename = getattr(getattr(fn, "__code__", None), "co_filename", "") or ""
    return bool(filename) and os.path.abspath(filename).startswith(_BLME_PKG_DIR + os.sep)


def test_hook_ownership_is_decided_by_package_path_not_substring():
    """Known positive and known negative for the helper above."""
    from blme.tasks import common as _common
    assert _is_blme_hook(_common.get_layers)
    ns = {}
    exec(compile("def output_capturing_hook(*a):\n    return None\n",
                 "/venvs/blme/lib/python3.12/site-packages/transformers/x.py", "exec"), ns)
    ns["output_capturing_hook"].__module__ = "transformers.utils.output_capturing"
    assert not _is_blme_hook(ns["output_capturing_hook"])


def _hook_counts(model):
    """Count BLME hooks over EVERY module, not just the top-level blocks.

    This walked `get_layers(model)` only, so a hook on `layers[0].mlp`, on an
    attention submodule, or on the embedding was invisible -- both such cases
    measured [0,0] -> [0,0] under the old counter, and **9 of the repo's hook
    sites target exactly those** (independent Opus review, 2026-09-06).
    `model.modules()` is the whole tree, so nothing can hide below a block.
    """
    if get_layers(model) is None:
        return None
    total = 0
    for m in model.modules():
        total += sum(1 for f in m._forward_hooks.values() if _is_blme_hook(f))
        total += sum(1 for f in m._forward_pre_hooks.values() if _is_blme_hook(f))
    return total


# Unhooked forwards a task runs before it registers its first hook, on the
# default dataset (4 passages): gradient_flow's no-grad sink-mask pass. The
# failure is injected after them, inside the hooked region.
_FORWARDS_BEFORE_FIRST_HOOK = {"dynamics_gradient_flow": 4}


@pytest.mark.parametrize("task_name,TaskCls", HOOKING_TASKS,
                         ids=[n for n, _ in HOOKING_TASKS])
def test_hooks_are_removed_when_the_forward_raises(
        mock_model_tokenizer, task_name, TaskCls):
    model, tokenizer = mock_model_tokenizer
    before = _hook_counts(model)
    if before is None:
        pytest.skip("architecture not supported by get_layers")

    class _Boom(RuntimeError):
        pass

    calls = {"n": 0}
    real_forward = model.forward
    n_ok = 1 + _FORWARDS_BEFORE_FIRST_HOOK.get(task_name, 0)

    def _explode(*a, **k):
        calls["n"] += 1
        if calls["n"] > n_ok:    # let one hooked forward succeed, then fail
            raise _Boom("simulated CUDA failure mid-collection")
        return real_forward(*a, **k)

    # Did the task actually REGISTER a hook during this run? A well-behaved
    # task has before == after == 0 whether it hooked and cleaned up or never
    # hooked at all, so the counts cannot distinguish real coverage from
    # vacuous coverage. Patching the registration API can.
    _saw_hook = {"any": False}
    _orig_fh = torch.nn.Module.register_forward_hook
    _orig_pre = torch.nn.Module.register_forward_pre_hook

    def _spy_fh(self, fn, *a, **k):
        if _is_blme_hook(fn):
            _saw_hook["any"] = True
        return _orig_fh(self, fn, *a, **k)

    def _spy_pre(self, fn, *a, **k):
        if _is_blme_hook(fn):
            _saw_hook["any"] = True
        return _orig_pre(self, fn, *a, **k)

    # PATCH INSIDE THE try. These are GLOBAL attributes on `torch.nn.Module`;
    # patching them before the try leaves a window in which an exception skips
    # the restore and every later test in the process sees a spy that closes
    # over THIS test's model. A test that can corrupt the rest of the run is a
    # worse defect than the one it was written to catch.
    model.forward = _explode
    try:
        torch.nn.Module.register_forward_hook = _spy_fh
        torch.nn.Module.register_forward_pre_hook = _spy_pre
        # The task may PROPAGATE the failure or handle it and return an error
        # dict -- both are legitimate, and which one it does is not what this
        # test is about. Requiring a raise made three well-behaved tasks fail
        # for catching their own errors.
        TaskCls({"num_samples": 4}).evaluate(model, tokenizer, dataset=None)
    except Exception:
        pass
    finally:
        model.forward = real_forward
        torch.nn.Module.register_forward_hook = _orig_fh
        torch.nn.Module.register_forward_pre_hook = _orig_pre

    after = _hook_counts(model)
    # VACUOUS COVERAGE IS A FAILURE, not a pass. Six of the discovered tasks
    # registered ZERO hooks under this test's conditions -- including one of the
    # ten wrapped the round before -- so their exception path was never
    # exercised and the green tick meant nothing (independent Opus review,
    # 2026-09-06). If a task the AST says hooks never actually does so here, the
    # test says so out loud instead of passing quietly.
    if before == 0 and after == 0 and not _saw_hook["any"]:
        pytest.skip(
            f"{task_name} registered no hook under this fixture — coverage "
            f"would be vacuous; give it a fixture that reaches its hook path")
    assert after == before, (
        f"{TaskCls.__name__} leaked a BLME forward hook when the forward "
        f"raised: per-layer counts {before} -> {after}. A task that fails must "
        f"leave the model exactly as it found it — the next task in the run "
        f"shares this model object, pays the hook on every forward, and the "
        f"closure pins the last activation tensor alive.")


@pytest.mark.parametrize("task_name,TaskCls", HOOKING_TASKS,
                         ids=[n for n, _ in HOOKING_TASKS])
def test_hooks_are_removed_on_the_happy_path(
        mock_model_tokenizer, task_name, TaskCls):
    model, tokenizer = mock_model_tokenizer
    before = _hook_counts(model)
    if before is None:
        pytest.skip("architecture not supported by get_layers")
    TaskCls({"num_samples": 4}).evaluate(model, tokenizer, dataset=None)
    assert _hook_counts(model) == before
