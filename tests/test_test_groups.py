"""Every rule in tests/_plugins/blme_test_groups.py still names a real file or test.

The markers are assigned by path, so a rename would otherwise drop a test out
of its group without a sound: the release check would quietly rejoin the
per-commit run, or a guard would leave ``-m population_guard``.
"""
import ast
from pathlib import Path

import pytest

import blme_test_groups as groups

ROOT = Path(__file__).resolve().parents[1]


def _defined_tests(path: Path) -> set[str]:
    """``name`` and ``Class::name`` for every test function in a file."""
    found = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            found |= {f"{node.name}::{f.name}" for f in node.body
                      if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return found


@pytest.mark.parametrize("group,prefix", [(g, p) for g, ps in groups.PATH_GROUPS.items() for p in ps])
def test_path_rule_names_an_existing_path(group, prefix):
    path = ROOT / prefix
    exists = path.is_dir() if prefix.endswith("/") else path.is_file()
    assert exists, f"marker {group!r}: {prefix} no longer exists"


@pytest.mark.parametrize("group,node", [(g, n) for g, ns in groups.NODE_GROUPS.items() for n in ns])
def test_node_rule_names_an_existing_test(group, node):
    file, _, name = node.partition("::")
    assert (ROOT / file).is_file(), f"marker {group!r}: {file} no longer exists"
    assert name in _defined_tests(ROOT / file), f"marker {group!r}: {node} no longer exists"


def test_the_release_check_is_marked_when_collected(request):
    target = groups.NODE_GROUPS["release"][0]
    items = {item.nodeid: item for item in request.session.items}
    if target not in items:
        pytest.skip("the metric-status check is not part of this session")
    assert items[target].get_closest_marker("release") is not None


def test_skip_reasons_mark_gpu_and_slow_tests():
    assert groups.groups_for("tests/x.py", "tests/x.py::t", ["needs cuda for the device test"]) == {"gpu"}
    assert groups.groups_for("tests/x.py", "tests/x.py::t", ["slow; set blme_slow_invariance=1"]) == {"slow"}
    assert groups.groups_for("tests/tasks/parity/test_x.py", "tests/tasks/parity/test_x.py::t", []) == {"parity"}
