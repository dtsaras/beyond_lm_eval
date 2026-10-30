"""Every paper-core task declares its method provenance on the class.

The declarations are TOKEN_POLICY, SOURCES, ADAPTATION_NOTE, HEADLINE_KEYS
and SIZE_INVARIANT. Each rule is one test, parametrised by task, so a new or
edited task fails on the declaration it got wrong.
"""
import ast
import inspect
import json
import pathlib
import re
import sys

import pytest

from blme.core import _register_all_tasks
from blme.registry import get_task
from blme.task_metadata import TASK_CERTIFICATION

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER_CORE = json.loads((ROOT / "tests" / "fixtures" / "paper_core_tasks.json").read_text())

PLAIN_POLICIES = {"exclude_pos0", "exclude_sinks", "last_per_sequence", "not_token_based"}
# Policies that select token rows, so the task must read the attribute: a
# central policy change then reaches every task that declares it.
SELECTING_POLICIES = {"exclude_pos0", "exclude_sinks", "last_per_sequence"}
ADAPTED_TIERS = {"refined-adaptation", "proxy-only"}
# Tier 1 in the methods dossier (they follow their paper's protocol) while the
# certification label still says refined-adaptation; the label is settled in
# the dual review, not here.
LABEL_PENDING_REVIEW = {"dynamics_fractal", "geometry_contextualization",
                        "geometry_spectral", "topology_zigzag_persistence"}

_register_all_tasks()


def _cls(task):
    cls = get_task(task)
    assert cls is not None, f"{task} is in the paper core but not registered"
    return cls


@pytest.mark.parametrize("task", PAPER_CORE)
def test_token_policy_is_a_known_policy(task):
    tp = getattr(_cls(task), "TOKEN_POLICY", None)
    assert isinstance(tp, str), f"{task}: TOKEN_POLICY missing"
    if tp.startswith("paper_protocol:"):
        assert tp.split(":", 1)[1].strip(), f"{task}: paper_protocol needs a citation"
    else:
        assert tp in PLAIN_POLICIES, f"{task}: unknown TOKEN_POLICY {tp!r}"


@pytest.mark.parametrize("task", PAPER_CORE)
def test_selecting_policy_is_read_by_the_task(task):
    cls = _cls(task)
    if cls.TOKEN_POLICY not in SELECTING_POLICIES:
        pytest.skip(f"{cls.TOKEN_POLICY} selects no rows by the shared policy")
    tree = ast.parse(inspect.getsource(sys.modules[cls.__module__]))
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and n.attr == "TOKEN_POLICY" and isinstance(n.ctx, ast.Load)]
    assert reads, f"{task} declares {cls.TOKEN_POLICY!r} but its module never reads TOKEN_POLICY"


@pytest.mark.parametrize("task", PAPER_CORE)
def test_sources_name_paper_code_and_precedent(task):
    src = getattr(_cls(task), "SOURCES", None)
    assert isinstance(src, dict) and set(src) == {"paper", "code", "llm_precedent"}, (
        f"{task}: SOURCES must have exactly paper / code / llm_precedent, got {src!r}")
    assert isinstance(src["paper"], str) and src["paper"].strip(), f"{task}: SOURCES['paper'] empty"
    for k in ("code", "llm_precedent"):
        assert src[k] is None or (isinstance(src[k], str) and src[k].strip()), (
            f"{task}: SOURCES[{k!r}] must be a non-empty string or None")


@pytest.mark.parametrize("task", PAPER_CORE)
def test_adapted_tasks_carry_a_short_adaptation_note(task):
    note = getattr(_cls(task), "ADAPTATION_NOTE", None)
    if TASK_CERTIFICATION[task].status in ADAPTED_TIERS and task not in LABEL_PENDING_REVIEW:
        assert isinstance(note, str) and note.strip(), (
            f"{task} is {TASK_CERTIFICATION[task].status} and needs an ADAPTATION_NOTE")
    if note:
        n = len(re.findall(r"[.!?](?:\s|$)", note.strip()))
        assert n <= 8, f"{task}: ADAPTATION_NOTE has {n} sentences (at most 8)"


@pytest.mark.parametrize("task", PAPER_CORE)
def test_headline_keys_are_reported_keys(task):
    hk = getattr(_cls(task), "HEADLINE_KEYS", None)
    assert isinstance(hk, tuple) and hk, f"{task}: HEADLINE_KEYS must be a non-empty tuple"
    assert all(isinstance(k, str) and k and not k.startswith("_meta_") for k in hk), (
        f"{task}: a headline key is empty or _meta_: {hk}")
    assert len(set(hk)) == len(hk), f"{task}: duplicate headline keys"


def _defines(path, node_id):
    """Whether ``path`` defines the test named by ``node_id`` (``f`` or ``C::f``)."""
    parts = node_id.split("::")
    body = ast.parse(path.read_text()).body
    for name in parts[:-1]:
        body = next((n.body for n in body if isinstance(n, ast.ClassDef) and n.name == name), None)
        if body is None:
            return False
    leaf = parts[-1].split("[")[0]
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == leaf
               for n in body)


@pytest.mark.parametrize("task", PAPER_CORE)
def test_size_invariant_entries_point_at_real_tests(task):
    si = getattr(_cls(task), "SIZE_INVARIANT", None)
    assert isinstance(si, dict), f"{task}: SIZE_INVARIANT must be a dict (empty when none passes)"
    for key, entry in si.items():
        assert isinstance(key, str) and not key.startswith("_meta_"), f"{task}: SIZE_INVARIANT key {key!r}"
        assert isinstance(entry, tuple) and len(entry) == 2, f"{task}[{key}]: want (transforms, node id)"
        transforms, node = entry
        assert (isinstance(transforms, tuple) and transforms
                and all(isinstance(t, str) and t for t in transforms)), f"{task}[{key}]: transforms {transforms!r}"
        assert isinstance(node, str) and "::" in node, f"{task}[{key}]: node id {node!r}"
        file, test = node.split("::", 1)
        path = ROOT / file
        assert path.is_file(), f"{task}[{key}]: {file} does not exist"
        assert _defines(path, test), f"{task}[{key}]: {file} defines no {test}"
