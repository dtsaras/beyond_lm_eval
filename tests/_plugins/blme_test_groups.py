"""Assign the suite's test-group markers by path, outside the test files.

The validation receipts pin test files and shared test support by hash
(``docs/validation/receipt-*.json``), so a marker added inside a pinned file
would put every task that links it under review. This plugin, loaded from
``pyproject.toml`` (``-p blme_test_groups``), adds the markers instead. The
marker names are registered in ``[tool.pytest.ini_options]``.

Selecting a group: ``pytest -m "not release"`` (the per-commit CI run),
``pytest -m release`` (the release check), ``pytest -m "not slow"``.

``tests/test_test_groups.py`` fails when a rule below stops matching a file or
test, so a rename cannot silently drop a test out of its group.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Path prefixes relative to the repository root; a trailing "/" is a directory.
PATH_GROUPS = {
    # Released-reference and published-value parity (tests/tasks/parity/).
    "parity": ("tests/tasks/parity/",),
    # Invariance harness: model transforms that must leave a readout unchanged.
    "invariance": ("tests/tasks/invariance/", "tests/test_invariance_harness.py"),
    # Precision lifting and dtype restore; the loader's dtype contract and the
    # precision check (0.2.7).
    "precision": ("tests/precision/", "tests/test_precision.py",
                  "tests/test_precision_restore_after_failure.py",
                  "tests/test_dtype_contract.py", "tests/test_precision_check.py"),
    # The golden-snapshot tool (tests/golden/golden_snapshot.py).
    "golden": ("tests/test_golden_configs.py",),
    # Whole-population guards: one assertion swept over every task or site.
    "population_guard": ("tests/test_cache_population_guard.py",
                         "tests/test_offload_parity_guard.py",
                         "tests/test_defect_class_guards.py",
                         "tests/test_reference_parity_guard.py"),
    # Files whose pytest session took at least 60 s in each of the 2026-11-19
    # full-suite runs (one process per file, 10-12 files at once, CPU only,
    # transformers 5.2.0 and 5.10.2); the two guards took 6 and 13 minutes.
    "slow": ("tests/test_offload_parity_guard.py",
             "tests/test_cache_population_guard.py",
             "tests/tasks/test_forward_hook_leaks.py",
             "tests/tasks/invariance/test_geometry_alpha_req.py",
             "tests/tasks/parity/test_prompt_id_parity.py",
             "tests/tasks/parity/test_ww_randomized_parity.py",
             "tests/tasks/test_geometry.py",
             "tests/tasks/parity/test_farms_parity.py",
             "tests/tasks/invariance/test_end_of_stack_identity_blocks.py",
             "tests/tasks/test_dynamics.py",
             "tests/tasks/test_repe.py"),
}

# Individual tests (node IDs without parameters).
NODE_GROUPS = {
    # Fails on every commit that changes a pinned input until a new receipt is
    # recorded, so it runs at release time with scripts/metric_status.py --check.
    "release": ("tests/test_metric_status.py::test_published_reviews_cover_current_registry_and_inputs",),
}

# A skip condition whose reason names these marks a test as needing them.
REASON_GROUPS = {
    "gpu": ("gpu", "cuda"),
    "slow": ("blme_slow_invariance",),
}


def _relative(item, root: Path) -> str | None:
    try:
        return Path(item.path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def _skip_reasons(item):
    for mark in item.iter_markers("skipif"):
        reason = mark.kwargs.get("reason", "")
        if reason:
            yield reason.lower()


def groups_for(relpath: str, base_nodeid: str, reasons) -> set[str]:
    """The groups for one test, from its path, its node ID and its skip reasons."""
    found = {group for group, prefixes in PATH_GROUPS.items()
             if any(relpath == p or (p.endswith("/") and relpath.startswith(p)) for p in prefixes)}
    found |= {group for group, nodes in NODE_GROUPS.items() if base_nodeid in nodes}
    reasons = list(reasons)
    found |= {group for group, words in REASON_GROUPS.items()
              if any(word in reason for reason in reasons for word in words)}
    return found


@pytest.hookimpl(tryfirst=True)  # before `-m` deselects on these markers
def pytest_collection_modifyitems(config, items):
    root = Path(config.rootpath).resolve()
    for item in items:
        relpath = _relative(item, root)
        if relpath is None:
            continue
        # Node IDs are relative to the rootdir; drop the parametrization.
        base = item.nodeid.split("[", 1)[0]
        for group in sorted(groups_for(relpath, base, _skip_reasons(item))):
            item.add_marker(getattr(pytest.mark, group))
